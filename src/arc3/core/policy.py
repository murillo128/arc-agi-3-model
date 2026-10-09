"""A small, deterministic baseline policy with no SDK dependency."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Sequence


@dataclass(frozen=True)
class Decision:
    action: Any
    data: dict[str, int] | None = None


def state_name(observation: Any) -> str:
    if observation is None:
        return "NOT_PLAYED"
    state = getattr(observation, "state", None)
    if state is None:
        return "NOT_PLAYED"
    return str(getattr(state, "name", state)).split(".")[-1].upper()


class RandomPolicy:
    """Per-game seeded baseline; later agents can implement the same interface."""

    def __init__(self, game_id: str, seed: int = 0) -> None:
        game_seed = int.from_bytes(hashlib.blake2b(game_id.encode(), digest_size=8).digest(), "big")
        self._rng = random.Random(seed ^ game_seed)

    def choose(self, observation: Any, actions: Sequence[Any], reset_action: Any) -> Decision:
        if state_name(observation) in ("NOT_PLAYED", "GAME_OVER"):
            return Decision(reset_action)
        candidates = [a for a in actions if getattr(a, "name", None) != "RESET"]
        if not candidates:
            raise ValueError("No non-reset action is available")
        action = self._rng.choice(candidates)
        complex_action = getattr(action, "is_complex", lambda: False)
        if complex_action():
            return Decision(action, {"x": self._rng.randrange(64), "y": self._rng.randrange(64)})
        return Decision(action)
