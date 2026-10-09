"""Dependency-free public observation snapshot and seeded random policy."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Observation:
    game_id: str
    frame: tuple[tuple[tuple[int, ...], ...], ...]
    state: str
    levels_completed: int
    win_levels: int
    available_actions: tuple[int, ...]

    @classmethod
    def from_public(cls, raw: Any) -> Observation:
        """Copy only SDK FrameData/FrameDataRaw public fields, including pixels."""
        if raw is None:
            raise RuntimeError("SDK returned no public observation")
        return cls(
            game_id=raw.game_id,
            frame=tuple(
                tuple(tuple(int(pixel) for pixel in row) for row in layer)
                for layer in raw.frame
            ),
            state=raw.state.name,
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            available_actions=tuple(int(action) for action in raw.available_actions),
        )


@dataclass(frozen=True)
class Action:
    id: int
    data: dict[str, int]


class RandomPolicy:
    """A reproducible baseline with an independent RNG per game; never trains."""

    def __init__(self, seed: int, game_id: str) -> None:
        self.rng = random.Random(f"{seed}:{game_id}")

    def choose_action(self, observation: Observation) -> Action:
        if observation.state in {"NOT_PLAYED", "GAME_OVER"}:
            return Action(0, {})  # SDK RESET, also a real action.
        candidates = [a for a in observation.available_actions if a != 0]
        if not candidates:
            raise RuntimeError("No available actions in an active game")
        action_id = self.rng.choice(candidates)
        data = {}
        if action_id == 6:  # SDK ACTION6 uses public-frame pixel coordinates.
            if not observation.frame or not observation.frame[-1]:
                raise RuntimeError("A coordinate action requires a public frame")
            height = len(observation.frame[-1])
            width = len(observation.frame[-1][0])
            data = {"x": self.rng.randrange(width), "y": self.rng.randrange(height)}
        return Action(action_id, data)
