"""Bounded real-action loop, independent of the SDK, CLI and Kaggle."""

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Protocol

from .policy import Action, Observation, RandomPolicy


class Environment(Protocol):
    @property
    def observation(self) -> Observation: ...

    def step(self, action: Action) -> Observation: ...


@dataclass(frozen=True)
class Transition:
    observation: Observation
    action: Action
    next_observation: Observation
    source: str = "real"


def run_game(
    env: Environment,
    policy: RandomPolicy,
    max_actions: int,
    *,
    on_transition: Callable[[Transition], None] | None = None,
    training_level_cap: int | None = None,
) -> dict:
    """Count every step; discard a boundary-crossing transition before recording.

    Caps count allowed levels from level 1. When collecting, the final level is
    always reserved, even if the configured cap is larger than the public count.
    The crossing action occurs in reality, but its new frame is never persisted.
    """
    if max_actions <= 0:
        raise ValueError("max_actions must be positive")
    if training_level_cap is not None and training_level_cap <= 0:
        raise ValueError("training_level_cap must be positive")
    started = perf_counter()
    observation = env.observation
    actions = recorded = 0
    stop_reason = "action_budget"

    def boundary(frame: Observation) -> bool:
        if training_level_cap is None:
            return False
        if frame.win_levels < 2:
            raise ValueError("Collection needs at least two known public levels")
        return frame.levels_completed >= min(training_level_cap, frame.win_levels - 1)

    while True:
        if boundary(observation):
            stop_reason = "training_level_boundary"
            break
        if observation.state == "WIN":
            stop_reason = "win"
            break
        if actions >= max_actions:
            break
        action = policy.choose_action(observation)
        next_observation = env.step(action)
        actions += 1
        if not boundary(next_observation) and on_transition is not None:
            on_transition(Transition(observation, action, next_observation))
            recorded += 1
        observation = next_observation

    return {
        "game_id": observation.game_id,
        "actions": actions,
        "recorded_transitions": recorded,
        "levels_completed": observation.levels_completed,
        "win_levels": observation.win_levels,
        "won": observation.state == "WIN",
        "state": observation.state,
        "stop_reason": stop_reason,
        "actions_per_completed_level": (
            actions / observation.levels_completed if observation.levels_completed else None
        ),
        "wall_seconds": perf_counter() - started,
    }


def transition_record(transition: Transition, seed: int) -> dict:
    return {"seed": seed, **asdict(transition)}
