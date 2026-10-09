"""Capture real transitions for future world-model pretraining; no ML training yet."""

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
    parser.add_argument("--split", type=Path, required=True, help="JSON file with training_games and evaluation_games")
    parser.add_argument("--output", type=Path, required=True, help="JSONL output; existing file is overwritten")
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true", help="Require game sources already cached locally")
    args = parser.parse_args()
    games = games_for_split(args.split, "training")
    caps = json.loads(args.split.read_text(encoding="utf-8")).get("training_level_caps", {})
    if any(type(caps.get(game_id)) is not int or caps[game_id] < 1 for game_id in games):
        parser.error("Each training game needs a positive training_level_caps entry to protect the last level")
    arcade = make_arcade(offline=args.offline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for game_id in games:
            result = run_game(
                arcade, game_id, RandomPolicy(game_id, args.seed),
                max_actions=args.max_actions, reset_action=reset_action(),
                stop_at_completed_levels=caps[game_id],
                on_transition=lambda transition: destination.write(json.dumps(transition) + "\n"),
            )
            print(json.dumps(result.to_dict(), sort_keys=True))
    print(f"Real transitions saved to {args.output}; this command does not train a model")


if __name__ == "__main__":
    main()
