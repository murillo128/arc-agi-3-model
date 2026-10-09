"""Shared run loop for local collection and evaluation, with an injectable engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from .policy import RandomPolicy, state_name


@dataclass(frozen=True)
class GameResult:
    game_id: str
    actions: int
    levels_completed: int
    final_state: str

    @property
    def won(self) -> bool:
        return self.final_state == "WIN"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "won": self.won}


def _as_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return _as_json(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_as_json(item) for item in value]
    raise TypeError(f"Unsupported public frame value: {type(value).__name__}")


def snapshot(frame: Any) -> dict[str, Any] | None:
    """Record only publicly observed fields; never serialize game internals."""
    if frame is None:
        return None
    return {
        "frame": _as_json(getattr(frame, "frame", None)),
        "state": state_name(frame),
        "levels_completed": int(getattr(frame, "levels_completed", 0)),
    }


def run_game(
    arcade: Any,
    game_id: str,
    policy: RandomPolicy,
    *,
    max_actions: int,
    reset_action: Any,
    on_transition: Callable[[dict[str, Any]], None] | None = None,
    stop_at_completed_levels: int | None = None,
) -> GameResult:
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    if stop_at_completed_levels is not None and stop_at_completed_levels < 1:
        raise ValueError("stop_at_completed_levels must be positive")
    env = arcade.make(game_id)
    if env is None:
        raise RuntimeError(f"Could not create ARC-AGI-3 environment: {game_id}")
    frame = getattr(env, "observation_space", None)
    count = 0
    while (
        count < max_actions
        and state_name(frame) != "WIN"
        and (stop_at_completed_levels is None or getattr(frame, "levels_completed", 0) < stop_at_completed_levels)
    ):
        decision = policy.choose(frame, list(env.action_space), reset_action)
        previous = snapshot(frame) if on_transition else None
        next_frame = env.step(decision.action, data=decision.data or {})
        if next_frame is None:
            raise RuntimeError(f"Environment returned no observation after action {count}")
        # At a held-out boundary, the post-action frame can already show the
        # next (reserved) level. Drop that crossing transition entirely.
        crosses_boundary = (
            stop_at_completed_levels is not None
            and int(getattr(next_frame, "levels_completed", 0)) >= stop_at_completed_levels
        )
        if on_transition and not crosses_boundary:
            on_transition({
                "game_id": game_id,
                "step": count,
                "observation": previous,
                "action": decision.action.name,
                "action_data": decision.data or {},
                "next_observation": snapshot(next_frame),
            })
        frame = next_frame
        count += 1
    return GameResult(
        game_id=game_id,
        actions=count,
        levels_completed=int(getattr(frame, "levels_completed", 0)),
        final_state=state_name(frame),
    )
