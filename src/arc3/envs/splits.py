"""Reject game overlap, including aliases for versions of the same game."""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GameSplit:
    training: dict[str, int]
    evaluation: tuple[str, ...]


def load_split(path: str | Path) -> GameSplit:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Split configuration must be an object")
    training = data.get("training_games")
    evaluation = data.get("evaluation_games")
    caps = data.get("training_level_caps")
    if not isinstance(training, list) or not isinstance(evaluation, list) or not isinstance(caps, dict):
        raise ValueError("Split needs training_games, evaluation_games and training_level_caps")
    if not training or not evaluation:
        raise ValueError("Both game splits must be nonempty")
    seen = set()
    for game_id in [*training, *evaluation]:
        if not isinstance(game_id, str) or not re.fullmatch(r"[a-z0-9]{4}(?:-[a-z0-9]+)?", game_id):
            raise ValueError(f"Invalid game ID: {game_id!r}")
        base_id = game_id.split("-", 1)[0]
        if base_id in seen:
            raise ValueError(f"Game split overlap or duplicate: {base_id}")
        seen.add(base_id)
    if any(type(caps.get(game_id)) is not int or caps[game_id] <= 0 for game_id in training):
        raise ValueError("Every training game needs a positive integer level cap")
    return GameSplit({game_id: caps[game_id] for game_id in training}, tuple(evaluation))
