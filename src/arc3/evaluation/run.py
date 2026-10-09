"""Evaluate a baseline on held-out local games without writing training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arc3.core.policy import RandomPolicy
from arc3.core.runner import run_game
from arc3.core.splits import games_for_split
from arc3.envs.arc3 import make_arcade, reset_action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON output; existing file is overwritten")
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    arcade = make_arcade(offline=args.offline)
    results = [
        run_game(arcade, game_id, RandomPolicy(game_id, args.seed),
                 max_actions=args.max_actions, reset_action=reset_action()).to_dict()
        for game_id in games_for_split(args.split, "evaluation")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seed": args.seed, "games": results}, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation saved to {args.output}")


if __name__ == "__main__":
    main()
