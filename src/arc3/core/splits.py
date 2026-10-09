"""Prevent accidental overlap between local training and evaluation games."""

from __future__ import annotations

import json
from pathlib import Path


def games_for_split(path: Path, split: str) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    train = set(data["training_games"])
    evaluate = set(data["evaluation_games"])
    if not train.isdisjoint(evaluate):
        raise ValueError("Training and evaluation games must not overlap")
    if split not in ("training", "evaluation"):
        raise ValueError(f"Invalid split: {split}")
    games = data[f"{split}_games"]
    if not games or any(not isinstance(item, str) or not item for item in games):
        raise ValueError(f"No valid {split} game IDs in {path}")
    return games
