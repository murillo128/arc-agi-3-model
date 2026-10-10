#!/usr/bin/env python3
"""Play committed games locally, or run a bounded random SDK plumbing check."""

import argparse
import json
import logging
from pathlib import Path
import random
import sys
from threading import Lock

from arc_agi.rendering import COLOR_MAP
from arcengine import GameAction, GameState
from flask import jsonify, request, send_file

from arc3.envs.sdk import open_arcade

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {GameState.WIN, GameState.GAME_OVER}


def require_frame(frame, operation):
    if frame is None or not frame.frame:
        raise RuntimeError(f"SDK {operation} returned no frame")
    return frame


def run_sdk(env, seed, max_actions):
    rng = random.Random(seed)
    frame = require_frame(env.observation_space, "make/reset")
    actions = 0
    while actions < max_actions and frame.state not in TERMINAL:
        choices = [a for a in frame.available_actions if a in (1, 2, 3, 4)]
        if not choices:
            raise RuntimeError("Game exposes no directional actions")
        frame = require_frame(env.step(GameAction.from_id(rng.choice(choices))), "step")
        actions += 1
    print(json.dumps({"game": frame.game_id, "seed": seed, "state": frame.state.name,
                      "levels_completed": frame.levels_completed, "win_levels": frame.win_levels,
                      "actions": actions, "stop": "terminal" if frame.state in TERMINAL else "max-actions"}))


def web_routes(env, seed):
    """One local shared play session; the lock also serializes multiple tabs."""
    lock = Lock()
    frame = require_frame(env.observation_space, "make/reset")

    def payload():
        return {"frame": frame.frame[-1].tolist(), "state": frame.state.name,
                "levels_completed": frame.levels_completed, "win_levels": frame.win_levels,
                "game": frame.game_id, "seed": seed,
                "palette": [COLOR_MAP[i] for i in range(16)]}

    def register(arcade, app):
        @app.get("/")
        def index():
            return send_file(ROOT / "scripts" / "play_game.html")

        @app.get("/play/state")
        def state():
            with lock:
                return jsonify(payload())

        @app.post("/play/action")
        def action():
            nonlocal frame
            data = request.get_json(silent=True)
            name = data.get("action") if isinstance(data, dict) else None
            if not isinstance(name, str) or name not in {"RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
                return jsonify(error="Choose RESET or ACTION1–ACTION4"), 400
            with lock:
                if name != "RESET" and frame.state in TERMINAL:
                    return jsonify(error="Game finished; restart to play again"), 409
                try:
                    result = env.reset() if name == "RESET" else env.step(GameAction[name])
                    frame = require_frame(result, name)
                except Exception as exc:
                    app.logger.exception("Local play action failed")
                    return jsonify(error=f"SDK action failed: {exc}"), 500
                return jsonify(payload())

    return register


def main(argv=None):
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)
    logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("web", "sdk"), default="web")
    parser.add_argument("--game", default="sc01-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-actions", type=int, default=100)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--render", action="store_true", help="Render SDK smoke frames in the terminal")
    args = parser.parse_args(argv)
    if args.max_actions < 0:
        parser.error("--max-actions must be nonnegative")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    arcade = None
    card = None
    try:
        arcade = open_arcade("offline", str(ROOT / "games"))
        card = arcade.open_scorecard()
        env = arcade.make(args.game, seed=args.seed, scorecard_id=card, save_recording=False,
                          render_mode="terminal" if args.render and args.mode == "sdk" else None)
        if env is None:
            raise RuntimeError(f"SDK could not make {args.game}; check games/ metadata")
        require_frame(env.observation_space, "make/reset")
        if args.mode == "sdk":
            run_sdk(env, args.seed, args.max_actions)
        else:
            host = f"[{args.host}]" if ":" in args.host else args.host
            print(f"Local play UI: http://{host}:{args.port}/", flush=True)
            arcade.listen_and_serve(host=args.host, port=args.port, use_reloader=False,
                                    extra_api_routes=web_routes(env, args.seed))
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"play_game: {exc}", file=sys.stderr)
        return 1
    finally:
        if arcade is not None and card is not None:
            arcade.close_scorecard(card)
        # OFFLINE Arcade owns no HTTP session; recordings and render windows are disabled.


if __name__ == "__main__":
    raise SystemExit(main())
