"""Collect only configured training games, without reserved-level frames."""

import argparse
import json
from pathlib import Path

from arc3.core.policy import RandomPolicy
from arc3.core.runtime import run_game, transition_record
from arc3.envs.sdk import SDKEnvironment, SDK_VERSION, open_arcade
from arc3.envs.splits import load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", default="recordings/train.jsonl")
    parser.add_argument("--mode", choices=["offline", "normal"], default="offline")
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=80)
    args = parser.parse_args()
    split = load_split(args.split)
    if args.max_actions <= 0:
        parser.error("--max-actions must be positive")
    output = Path(args.output)
    if output.suffix != ".jsonl":
        parser.error("--output must end in .jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    # Refuse to overwrite or append to an unverified previous recording.
    with output.open("x", encoding="utf-8") as stream:
        arcade = open_arcade(args.mode, args.environments_dir)
        try:
            def record(transition):
                stream.write(json.dumps(transition_record(transition, args.seed)) + "\n")

            for game_id, level_cap in split.training.items():
                env = SDKEnvironment(arcade, game_id, args.seed)
                summaries.append(run_game(
                    env, RandomPolicy(args.seed, env.observation.game_id), args.max_actions,
                    on_transition=record, training_level_cap=level_cap,
                ))
        finally:
            arcade.close_scorecard()
    print(json.dumps({"lifecycle": "collection", "sdk_version": SDK_VERSION,
                      "seed": args.seed, "max_actions": args.max_actions,
                      "training_level_caps": split.training, "games": summaries}))


if __name__ == "__main__":
    main()
