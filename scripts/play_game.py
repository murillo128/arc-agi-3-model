#!/usr/bin/env python3
"""Play SDK-discovered games locally, or run a bounded random SDK plumbing check."""

import argparse
from contextlib import ExitStack
import json
import logging
from pathlib import Path
import random
import sys
from threading import Lock
from uuid import uuid4

from arc_agi.rendering import COLOR_MAP
from arcengine import GameAction, GameState
from flask import Flask, jsonify, request, send_file

from arc3.envs.sdk import open_arcade

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {GameState.WIN, GameState.GAME_OVER}


def require_frame(frame, operation):
    if frame is None or not frame.frame:
        raise RuntimeError(f"SDK {operation} returned no frame")
    if any(image.shape != (64, 64) for image in frame.frame):
        raise RuntimeError(f"SDK {operation} returned a non-64×64 frame")
    return frame


def run_sdk(env, seed, max_actions):
    rng = random.Random(seed)
    frame = require_frame(env.observation_space, "make/reset")
    actions = 0
    while actions < max_actions and frame.state not in TERMINAL:
        choices = [a for a in frame.available_actions if a in range(1, 8)]
        if not choices:
            raise RuntimeError("Game exposes no playable actions")
        action = GameAction.from_id(rng.choice(choices))
        data = {"x": rng.randrange(64), "y": rng.randrange(64)} if action == GameAction.ACTION6 else None
        frame = require_frame(env.step(action, data=data), "step")
        actions += 1
    print(json.dumps({"game": frame.game_id, "seed": seed, "state": frame.state.name,
                      "levels_completed": frame.levels_completed, "win_levels": frame.win_levels,
                      "actions": actions, "stop": "terminal" if frame.state in TERMINAL else "max-actions"}))


class PlaySession:
    """Separate SDK providers and one shared, epoch-guarded active environment."""

    def __init__(self, sources, game, seed, render_mode=None):
        self.sources = sources  # source -> (Arcade, scorecard ID)
        self.seed = seed
        self.render_mode = render_mode
        self.lock = Lock()
        self.games = [{"source": source, "id": info.game_id, "title": info.title}
                      for source, (arcade, _) in sources.items()
                      for info in arcade.get_environments()]
        exact = [item for item in self.games if item["id"] == game]
        matches = exact or [item for item in self.games if item["id"].split("-")[0] == game]
        if len(matches) != 1:
            raise ValueError(f"SDK could not make {game}: choose a unique discovered game ID")
        self.selected = matches[0]
        self.env, self.frame = self.make(self.selected)
        self.epoch = uuid4().hex

    def make(self, item):
        arcade, card = self.sources[item["source"]]
        env = arcade.make(item["id"], seed=self.seed, scorecard_id=card,
                          save_recording=False, render_mode=self.render_mode)
        if env is None:
            raise RuntimeError(f"SDK could not make {item['id']}")
        return env, require_frame(env.observation_space, "make/reset")

    def payload(self, animate=False):
        frame = self.frame
        data = {"frame": frame.frame[-1].tolist(), "state": frame.state.name,
                "levels_completed": frame.levels_completed, "win_levels": frame.win_levels,
                "game": self.selected["id"], "source": self.selected["source"], "seed": self.seed,
                "epoch": self.epoch, "available_actions": list(frame.available_actions),
                "palette": [COLOR_MAP[i] for i in range(16)]}
        if animate and len(frame.frame) > 1:
            data["frames"] = [image.tolist() for image in frame.frame]
        return data


def web_routes(session):
    def register(arcade, app):
        @app.get("/")
        def index():
            return send_file(ROOT / "scripts" / "play_game.html")

        @app.after_request
        def no_cache(response):
            response.headers["Cache-Control"] = "no-store"
            return response

        @app.get("/play/games")
        def games():
            return jsonify(groups=[{"source": source, "label": label,
                                    "games": [g for g in session.games if g["source"] == source]}
                                   for source, label in (("synthetic", "Synthetic"),
                                                         ("original", "Original ARC-AGI-3"))])

        @app.get("/play/state")
        def state():
            with session.lock:
                return jsonify(session.payload())

        @app.post("/play/select")
        def select():
            data = request.get_json(silent=True)
            if not isinstance(data, dict) or set(data) != {"game", "source", "epoch"}:
                return jsonify(error="Choose a cataloged game and source with its session epoch"), 400
            item = next((g for g in session.games
                         if g["id"] == data["game"] and g["source"] == data["source"]), None)
            if item is None:
                return jsonify(error="Unknown game or source"), 400
            with session.lock:
                if data["epoch"] != session.epoch:
                    return jsonify(error="Session changed in another tab; live state restored"), 409
                if item != session.selected:
                    try:
                        env, frame = session.make(item)
                    except Exception:
                        app.logger.exception("Local play selection failed")
                        return jsonify(error="SDK could not load this game; previous game retained. Check local cache or SDK access"), 500
                    session.env, session.frame, session.selected = env, frame, item
                    session.epoch = uuid4().hex
                return jsonify(session.payload())

        @app.post("/play/action")
        def action():
            data = request.get_json(silent=True)
            name = data.get("action") if isinstance(data, dict) else None
            if not isinstance(name, str) or name not in {"RESET", *(f"ACTION{i}" for i in range(1, 8))}:
                return jsonify(error="Choose RESET or ACTION1–ACTION7"), 400
            expected_keys = {"action", "epoch", "data"} if name == "ACTION6" else {"action", "epoch"}
            if set(data) != expected_keys:
                return jsonify(error="Invalid action payload or missing session epoch"), 400
            coordinates = data.get("data")
            if name == "ACTION6" and (not isinstance(coordinates, dict) or set(coordinates) != {"x", "y"}
                                     or any(type(v) is not int or not 0 <= v <= 63 for v in coordinates.values())):
                return jsonify(error="CLICK requires exactly integer x,y coordinates in 0..63"), 400
            with session.lock:
                if data["epoch"] != session.epoch:
                    return jsonify(error="Session changed in another tab; live state restored"), 409
                if name != "RESET":
                    if session.frame.state in TERMINAL:
                        return jsonify(error="Game finished; restart to play again"), 409
                    if GameAction[name].value not in session.frame.available_actions:
                        return jsonify(error="Action is not available in this observation"), 400
                try:
                    result = session.env.reset() if name == "RESET" else session.env.step(GameAction[name], data=coordinates)
                    session.frame = require_frame(result, name)
                    if name == "RESET":
                        session.epoch = uuid4().hex
                except Exception:
                    app.logger.exception("Local play action failed")
                    return jsonify(error="SDK action failed; try Restart to recover"), 500
                return jsonify(session.payload(animate=True))

    return register


def main(argv=None):
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)
    logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("web", "sdk"), default="web")
    parser.add_argument("--game", default="sc01-v1")
    parser.add_argument("--originals-mode", choices=("offline", "normal"), default="offline",
                        help="Original games only: normal explicitly enables SDK network access (may require an authorized API key)")
    parser.add_argument("--originals-dir", type=Path, default=ROOT / "environment_files",
                        help="Local public original SDK game cache (default: repository environment_files/)")
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
    try:
        with ExitStack() as cleanup:
            sources = {}
            for source, mode, path in (("synthetic", "offline", ROOT / "games"),
                                       ("original", args.originals_mode, args.originals_dir.resolve())):
                arcade = open_arcade(mode, str(path))
                card = arcade.open_scorecard()
                cleanup.callback(arcade.close_scorecard, card)
                sources[source] = (arcade, card)
            session = PlaySession(sources, args.game, args.seed,
                                  "terminal" if args.render and args.mode == "sdk" else None)
            if args.mode == "sdk":
                run_sdk(session.env, args.seed, args.max_actions)
            else:
                host = f"[{args.host}]" if ":" in args.host else args.host
                print(f"Local play UI: http://{host}:{args.port}/", flush=True)
                # Expose only the UI protocol, not SDK metadata/scorecard/raw command APIs.
                app = Flask(__name__)
                app.add_url_rule("/api/healthcheck", view_func=lambda: "okay")
                web_routes(session)(sources["synthetic"][0], app)
                app.run(host=args.host, port=args.port, use_reloader=False, threaded=True)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"play_game: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
