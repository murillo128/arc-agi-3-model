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
    training = data.get("training")
    evaluation = data.get("evaluation")
    if not isinstance(training, dict) or not isinstance(evaluation, list):
        raise ValueError("Split needs a training game-to-level-cap object and evaluation list")
    if not training or not evaluation:
        raise ValueError("Both game splits must be nonempty")
    if any(type(cap) is not int or cap <= 0 for cap in training.values()):
        raise ValueError("Every training game needs a positive integer level cap")
    seen = set()
    for game_id in [*training, *evaluation]:
        if not isinstance(game_id, str) or not re.fullmatch(r"[a-z0-9]{4}(?:-[a-z0-9]+)?", game_id):
            raise ValueError(f"Invalid game ID: {game_id!r}")
        base_id = game_id.split("-", 1)[0]
        if base_id in seen:
            raise ValueError(f"Game split overlap or duplicate: {base_id}")
        seen.add(base_id)
    return GameSplit(training, tuple(evaluation))
