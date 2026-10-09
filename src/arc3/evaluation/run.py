"""Report baseline metrics on the configured held-out games."""

import argparse
import json
import math
from pathlib import Path
from uuid import uuid4

from arc3.core.policy import RandomPolicy
from arc3.core.runtime import run_game
from arc3.envs.sdk import SDKEnvironment, SDK_VERSION, open_arcade
from arc3.envs.splits import load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", default="results/evaluation.json")
    parser.add_argument("--mode", choices=["offline", "normal"], default="offline")
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--trace-dir", help="Opt in to one .arc3 v1 file per evaluation attempt")
    parser.add_argument("--timeout-seconds", type=float, help="Per-game elapsed allowance, checked between actions")
    args = parser.parse_args()
    split = load_split(args.split)
    if args.max_actions <= 0:
        parser.error("--max-actions must be positive")
    if args.timeout_seconds is not None and (not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0):
        parser.error("--timeout-seconds must be finite and positive")
    output = Path(args.output)
    if output.suffix != ".json":
        parser.error("--output must end in .json")
    if output.exists():
        parser.error("--output already exists; choose a new evaluation result path")
    run_id = None
    if args.trace_dir is not None:
        from .recorder import EvaluationRecorder

        Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
        run_id = uuid4().hex
    arcade = open_arcade(args.mode, args.environments_dir)
    try:
        summaries = []
        for game_id in split.evaluation:
            env = SDKEnvironment(arcade, game_id, args.seed)
            recorder = None
            if run_id is not None:
                recorder = EvaluationRecorder(
                    args.trace_dir, run_id=run_id, seed=args.seed, sdk_version=SDK_VERSION,
                )
            summaries.append(run_game(
                env, RandomPolicy(args.seed, env.observation.game_id), args.max_actions,
                attempt_observer=recorder, timeout_seconds=args.timeout_seconds,
            ))
    finally:
        arcade.close_scorecard()
    metrics = {"lifecycle": "evaluation", "sdk_version": SDK_VERSION,
               "seed": args.seed, "max_actions": args.max_actions, "games": summaries}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
