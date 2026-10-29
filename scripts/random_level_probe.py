"""Explicit manual, CPU-only random diagnostic until the first completed level.

Independent SDK lifetimes and recordings; no model loading, training or changes
to the frozen evaluator. Never inspects game implementations.
"""

import argparse
from collections import Counter
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
import time

from arcengine import GameAction

from arc3.core.policy import Observation, RandomPolicy
from arc3.envs.sdk import SDK_VERSION, open_arcade


GAMES = ("ar25-0c556536", "g50t-5849a774")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_probe(wrapper, *, seed, max_actions, seconds, emit):
    observation = Observation.from_public(wrapper.observation_space)
    if observation.levels_completed != 0:
        raise ValueError("Probe must begin at the first level")
    policy = RandomPolicy(seed, observation.game_id)
    started = time.monotonic()
    result = dict(game=observation.game_id, initial_available_actions=list(observation.available_actions),
                  actions=0, returned_actions=0, resets=0, game_over_returns=0,
                  changed=0, unchanged=0, missing_frame=0, first_level_solved=False,
                  actions_to_first_level=None, stop_reason="action_budget")
    histogram = Counter()
    try:
        while result["actions"] < max_actions:
            if time.monotonic() - started >= seconds:
                result["stop_reason"] = "time_budget"
                break
            # Match the LoRA experiment's allow_undo=false without changing the
            # shared random policy. RESET remains available after GAME_OVER.
            permitted = replace(observation, available_actions=tuple(
                a for a in observation.available_actions if a != 7))
            action = policy.choose_action(permitted)
            if time.monotonic() - started >= seconds:
                result["stop_reason"] = "time_budget"
                break
            result["actions"] += 1
            emit(dict(event="request", action_number=result["actions"],
                      action_id=action.id, action_data=action.data))
            raw = wrapper.step(GameAction.from_id(action.id), data=action.data)
            current = Observation.from_public(raw)
            result["returned_actions"] += 1
            result["resets"] += int(action.id == 0)
            result["game_over_returns"] += int(current.state == "GAME_OVER")
            histogram[action.id] += 1
            changed = (observation.frame[-1] != current.frame[-1]
                       if observation.frame and current.frame else None)
            result["missing_frame" if changed is None else "changed" if changed else "unchanged"] += 1
            emit(dict(event="action", action_number=result["actions"],
                      action_id=action.id, action_data=action.data, changed=changed,
                      levels_completed=current.levels_completed, total_levels=current.win_levels,
                      state=current.state, elapsed_seconds=time.monotonic() - started))
            observation = current
            if observation.levels_completed > 0:
                result.update(first_level_solved=True, actions_to_first_level=result["actions"],
                              stop_reason="first_level_solved")
                break
            if observation.state == "WIN":
                raise RuntimeError("SDK WIN did not report a completed level")
    except (KeyboardInterrupt, InterruptedError):
        result["stop_reason"] = "cancelled"
    except Exception as exc:
        result.update(stop_reason="error", error_type=type(exc).__name__)
        logging.exception("Random probe failed; preserving confirmed results")
    elapsed = time.monotonic() - started
    result.update(levels_completed=observation.levels_completed, total_levels=observation.win_levels,
                  state=observation.state, elapsed_seconds=elapsed,
                  time_overrun_seconds=max(0., elapsed - seconds), action_histogram=dict(histogram),
                  censored=result["stop_reason"] in {"action_budget", "time_budget"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--environments-dir", required=True)
    parser.add_argument("--mode", choices=("normal", "offline"), default="normal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-actions", type=int, default=10000)
    parser.add_argument("--max-game-seconds", type=float, default=60.)
    args = parser.parse_args()
    if args.max_actions <= 0 or not math.isfinite(args.max_game_seconds) or args.max_game_seconds <= 0:
        parser.error("Action and time limits must be positive and finite")
    if args.output_dir.exists():
        parser.error("Output already exists; choose a new directory")
    args.output_dir.mkdir(parents=True)
    logging.basicConfig(filename=args.output_dir / "runner.log", level=logging.INFO, force=True)
    report = dict(format="arc3-random-first-level-v1", manual=True, seed=args.seed,
                  requested_games=list(GAMES), sdk=SDK_VERSION, policy="RandomPolicy; ACTION7 excluded",
                  max_actions=args.max_actions, max_game_seconds=args.max_game_seconds,
                  initial_make_reset_counted_separately=True, games=[], status="running",
                  started_at_utc=datetime.now(timezone.utc).isoformat(),
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  command=[sys.executable, "-m", "scripts.random_level_probe", *sys.argv[1:]])
    arcade = None
    started = time.monotonic()
    write_json(args.output_dir / "summary.json", report)
    try:
        arcade = open_arcade(args.mode, args.environments_dir,
                             recordings_dir=str(args.output_dir / "sdk-recordings"))
        for game in GAMES:
            setup_started = time.monotonic()
            wrapper = arcade.make(game, seed=args.seed, save_recording=True, include_frame_data=True)
            if wrapper is None:
                raise RuntimeError("SDK could not create the requested game")
            setup_seconds = time.monotonic() - setup_started
            with (args.output_dir / (game + ".jsonl")).open("x") as stream:
                def emit(event):
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
                    stream.flush()
                result = run_probe(wrapper, seed=args.seed, max_actions=args.max_actions,
                                   seconds=args.max_game_seconds, emit=emit)
            result.update(split="evaluation" if game == GAMES[0] else "training",
                          initialization_resets=1, setup_seconds=setup_seconds)
            report["games"].append(result)
            write_json(args.output_dir / "summary.json", report)
            print(json.dumps(result, allow_nan=False), flush=True)
            if result["stop_reason"] == "cancelled":
                break
        report["status"] = ("finished" if len(report["games"]) == len(GAMES) and all(
            g["stop_reason"] not in {"error", "cancelled"} for g in report["games"]) else "incomplete")
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        logging.exception("Random probe setup failed")
        raise
    finally:
        if arcade is not None:
            try:
                arcade.close_scorecard()
            except Exception as exc:
                report["scorecard_close_error_type"] = type(exc).__name__
        report["wall_seconds"] = time.monotonic() - started
        write_json(args.output_dir / "summary.json", report)
        fields = ("game", "split", "first_level_solved", "actions_to_first_level", "actions",
                  "returned_actions", "resets", "changed", "unchanged", "elapsed_seconds", "stop_reason")
        with (args.output_dir / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report["games"])
    if report["status"] != "finished":
        raise SystemExit("Probe incomplete; see retained partial results")


if __name__ == "__main__":
    main()
