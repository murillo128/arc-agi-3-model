"""Report baseline metrics on the configured held-out games."""

import argparse
import json
from pathlib import Path

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
    args = parser.parse_args()
    split = load_split(args.split)
    if args.max_actions <= 0:
        parser.error("--max-actions must be positive")
    output = Path(args.output)
    if output.suffix != ".json":
        parser.error("--output must end in .json")
    if output.exists():
        parser.error("--output already exists; choose a new evaluation result path")
    arcade = open_arcade(args.mode, args.environments_dir)
    try:
        summaries = []
        for game_id in split.evaluation:
            env = SDKEnvironment(arcade, game_id, args.seed)
            summaries.append(run_game(
                env, RandomPolicy(args.seed, env.observation.game_id), args.max_actions,
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
