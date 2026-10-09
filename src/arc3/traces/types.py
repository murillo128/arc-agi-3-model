"""Plain Python wire values; tensor data is already little-endian, row-major BIN.

TypedDicts describe known fields. Unknown optional string keys are preserved by
the codec. No SDK, array library, model or device objects cross this boundary.
"""

from typing import Literal, NotRequired, TypedDict

type Value = None | bool | int | float | str | bytes | list[Value] | dict[str, Value]
type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
type Number = int | float


class Tensor(TypedDict):
    dtype: Literal["u8", "f16", "f32"]
    shape: list[int]
    data: bytes


class Observation(TypedDict):
    frames: Tensor
    state: str
    levels_completed: int
    win_levels: int
    available_actions: list[int]


class Action(TypedDict):
    id: int
    data: dict[str, int]


class Metadata(TypedDict):
    game_id: str
    run_id: str
    session_id: str
    attempt_index: int
    seed: int
    sdk_version: str
    source_split: Literal["train", "evaluation", "unspecified"]
    started_at: str
    model_id: NotRequired[str]
    checkpoint_id: NotRequired[str]
    config: NotRequired[JSONValue]


class Candidate(TypedDict):
    action: Action
    score: Number


class Decision(TypedDict):
    score_type: Literal["logit", "probability", "value"]
    candidates: list[Candidate]
    selected_value: NotRequired[Number]
    action_entropy: NotRequired[Number]


class Prediction(TypedDict):
    target: Literal["post_action_last_frame", "post_action_sequence"]
    frames: NotRequired[Tensor]
    latent: NotRequired[Tensor]
    uncertainty: NotRequired[Number]
    errors: NotRequired[dict[str, Number]]


class Learning(TypedDict, total=False):
    updates: int
    loss: Number
    replay_size: int


class Step(TypedDict):
    index: int
    action: Action
    observation: Observation
    decision: NotRequired[Decision]
    prediction: NotRequired[Prediction]
    learning: NotRequired[Learning]
    timing: NotRequired[dict[str, Number]]
    notes: NotRequired[list[str]]


class Termination(TypedDict):
    reason: Literal["win", "game_over", "reset", "action_budget", "timeout",
                    "error", "interrupted", "training_level_boundary"]
    final_state: str
    detail: NotRequired[str]
    reset_action: NotRequired[Action]


class Summary(TypedDict):
    real_actions: int
    levels_completed: int
    win_levels: int
    wall_seconds: Number


class Attempt(TypedDict):
    metadata: Metadata
    initial_observation: Observation
    steps: list[Step]
    termination: Termination
    summary: Summary
