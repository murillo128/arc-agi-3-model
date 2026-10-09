"""Deterministic recorder acceptance fixtures from original public SDK doubles.

Pixels and responses are literal test-owned examples, never actual game output.
Only identity and clocks are fixed; the real runtime, policy, adapter and recorder
produce the files. See README.md for the independent semantic expectations.
"""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

from arcengine import FrameData, GameState

from arc3.core.policy import Action, RandomPolicy
from arc3.core.runtime import run_game
from arc3.envs.sdk import SDKEnvironment, SDK_VERSION
from arc3.evaluation.recorder import EvaluationRecorder

GAME = "synthetic-recorder-v1"
SESSION = "00000000000000000000000000000001"


def frame(pixels, *, state=GameState.NOT_FINISHED, levels=0, actions=(1,)):
    return FrameData(game_id=GAME, frame=pixels, state=state,
                     levels_completed=levels, win_levels=3, available_actions=list(actions))


class PublicResponses:
    """SDK transport double; no environment implementation or game rules."""

    def __init__(self, responses):
        self.responses = responses

    def make(self, game_id, *, seed, save_recording):
        assert (game_id, seed, save_recording) == (GAME, 42, False)
        following = iter(self.responses[1:])
        return SimpleNamespace(observation_space=self.responses[0],
                               step=lambda action, *, data: next(following))


class PredictionExample:
    def choose_action(self, observation):
        return Action(6, {"x": 1, "y": 0})

    def trace_diagnostics(self, observation, action):
        # Explicit synthetic prediction, captured before the actual response.
        return {"prediction": {"target": "post_action_last_frame", "frames": {
            "dtype": "u8", "shape": [2, 2], "data": bytes([0, 1, 2, 3]),
        }}, "notes": ["Synthetic prediction; not SDK output or a trained model."]}


def record(directory, responses, policy, budget, times):
    with patch("arc3.evaluation.recorder.uuid4", return_value=UUID(hex=SESSION)), \
         patch("arc3.evaluation.recorder.datetime", wraps=datetime) as clock, \
         patch("arc3.evaluation.recorder.perf_counter", side_effect=times):
        clock.now.return_value = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
        recorder = EvaluationRecorder(directory, run_id="synthetic-acceptance-v1",
                                      seed=42, sdk_version=SDK_VERSION)
        return run_game(SDKEnvironment(PublicResponses(responses), GAME, 42), policy,
                        budget, attempt_observer=recorder)


def generate(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Fixture output directory must be empty")
    # Bootstrap RESET, click with two returned frames and a level change,
    # GAME_OVER, terminal RESET, next attempt's ordinary action. Five calls,
    # three recorded ordinary actions; both RESETs are outside active attempts.
    baseline = directory / "baseline"
    baseline.mkdir(exist_ok=True)
    record(baseline, [
        frame([], state=GameState.NOT_PLAYED, actions=(0,)),
        frame([[[0, 1], [2, 3]]], actions=(6,)),
        frame([[[4, 5], [6, 7]], [[8, 9], [10, 11]]], levels=1),
        frame([[[3, 2], [1, 0]]], state=GameState.GAME_OVER, levels=1, actions=(0,)),
        frame([[[9, 9], [9, 9]]]),
        frame([[[0, 0], [0, 0]]]),
    ], RandomPolicy(42, GAME), 5, [0.0, 0.25, 1.0, 1.25])
    prediction = directory / "prediction"
    prediction.mkdir(exist_ok=True)
    record(prediction, [frame([[[0, 1], [2, 3]]], actions=(6,)),
                        frame([[[3, 2], [1, 0]]])], PredictionExample(), 1, [0.0, 0.25])
    outputs = []
    for name, source in [
        ("recorder-baseline-0", baseline / f"{SESSION}-000000.arc3"),
        ("recorder-baseline-1", baseline / f"{SESSION}-000001.arc3"),
        ("recorder-prediction", prediction / f"{SESSION}-000000.arc3"),
    ]:
        target = directory / f"{name}.arc3"
        target.write_bytes(source.read_bytes())
        outputs.append(target)
    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Use a new directory; intermediate recorder files are retained")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    for path in generate(args.output):
        print(path)
