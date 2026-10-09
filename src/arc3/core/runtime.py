"""Bounded real-action loop, independent of the SDK, CLI and Kaggle."""

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Protocol

from .policy import Action, Observation, Policy


class Environment(Protocol):
    @property
    def observation(self) -> Observation: ...

    def step(self, action: Action) -> Observation: ...


class AttemptObserver(Protocol):
    """Evaluation-only lifecycle seam; storage stays outside the action loop."""

    def start(self, observation: Observation) -> None: ...
    def before_action(self, observation: Observation, action: Action, policy: Policy) -> None: ...
    def after_action(self, action: Action, observation: Observation) -> None: ...
    def finish(self, reason: str, *, detail: str | None = None) -> None: ...


@dataclass(frozen=True)
class Transition:
    observation: Observation
    action: Action
    next_observation: Observation
    source: str = "real"


def run_game(
    env: Environment,
    policy: Policy,
    max_actions: int,
    *,
    on_transition: Callable[[Transition], None] | None = None,
    training_level_cap: int | None = None,
    attempt_observer: AttemptObserver | None = None,
    timeout_seconds: float | None = None,
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
    if attempt_observer is not None and (training_level_cap is not None or on_transition is not None):
        raise ValueError("Attempt tracing is evaluation-only, separate from collection")
    if timeout_seconds is not None:
        import math

        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
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

    try:
        if attempt_observer is not None:
            attempt_observer.start(observation)
        while True:
            if boundary(observation):
                stop_reason = "training_level_boundary"
                break
            if observation.state == "WIN":
                stop_reason = "win"
                break
            if actions >= max_actions:
                break
            if timeout_seconds is not None and perf_counter() - started >= timeout_seconds:
                stop_reason = "timeout"
                break
            action = policy.choose_action(observation)
            if attempt_observer is not None:
                attempt_observer.before_action(observation, action, policy)
            next_observation = env.step(action)
            actions += 1
            if attempt_observer is not None:
                attempt_observer.after_action(action, next_observation)
            if not boundary(next_observation) and on_transition is not None:
                on_transition(Transition(observation, action, next_observation))
                recorded += 1
            observation = next_observation
    except BaseException as exc:
        if attempt_observer is not None:
            reason = ("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit))
                      else "timeout" if isinstance(exc, TimeoutError) else "error")
            try:
                attempt_observer.finish(
                    reason, detail=f"{type(exc).__name__}; no unconfirmed action/result counted",
                )
            except Exception as recording_error:
                exc.add_note(f"Trace finalization also failed: {type(recording_error).__name__}")
        raise
    if attempt_observer is not None:
        attempt_observer.finish(stop_reason)

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
