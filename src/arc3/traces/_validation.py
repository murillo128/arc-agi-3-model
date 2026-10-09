"""Semantic validation owned by the frozen docs/arc3-format-v1.md contract."""

from datetime import date
import math
import re
import struct

from ._errors import Arc3Error

_EXACT_INT = 9_007_199_254_740_991
_TIMESTAMP = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})"
    r"(?:\.[0-9]{1,9})?(?:Z|([+-])([0-9]{2}):([0-9]{2}))"
)


def fail(path: str, message: str) -> None:
    raise Arc3Error(f"{path}: {message}")


def integer(value: object, path: str, minimum: int = -_EXACT_INT,
            maximum: int = _EXACT_INT) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        fail(path, f"expected integer in [{minimum}, {maximum}]")


def number(value: object, path: str, minimum: float | None = None) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        fail(path, "expected finite number")
    if type(value) is int:
        integer(value, path)
    if minimum is not None and value < minimum:
        fail(path, f"expected number >= {minimum}")


def text(value: object, path: str, *, identifier: bool = False) -> None:
    if type(value) is not str or (identifier and not value):
        fail(path, "expected nonempty STR" if identifier else "expected STR")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise Arc3Error(f"{path}: invalid UTF-8 text") from exc


def mapping(value: object, path: str, required: tuple[str, ...] = ()) -> None:
    if type(value) is not dict:
        fail(path, "expected map")
    for key in required:
        if key not in value or value[key] is None:
            fail(f"{path}.{key}", "missing required field")


def array(value: object, path: str) -> None:
    if type(value) is not list:
        fail(path, "expected array")


def enum(value: object, path: str, choices: tuple[str, ...]) -> None:
    if type(value) is not str or value not in choices:
        fail(path, f"expected one of {choices}")


def profile(value: object, path: str, depth: int = 0, *, config: bool = False) -> None:
    """Also validate unknown extensions and encoder input before serialization."""
    kind = type(value)
    if kind in (dict, list):
        if depth >= 64:
            fail(path, "nesting exceeds 64 containers")
        if kind is dict:
            for key, child in value.items():
                text(key, path + " key")
                profile(child, f"{path}.{key}", depth + 1, config=config)
        else:
            for i, child in enumerate(value):
                profile(child, f"{path}[{i}]", depth + 1, config=config)
    elif kind is str:
        text(value, path)
    elif kind is int:
        integer(value, path)
    elif kind is float:
        number(value, path)
    elif kind is bytes and not config:
        pass
    elif value is None or kind is bool:
        pass
    else:
        fail(path, "value outside JSON config profile" if config else "value outside MessagePack profile")


def tensor(value: object, path: str, cap: int, *, rank: int | None = None,
           observed: bool = False) -> None:
    mapping(value, path, ("dtype", "shape", "data"))
    enum(value["dtype"], path + ".dtype", ("u8",) if observed else ("u8", "f16", "f32"))
    shape, data = value["shape"], value["data"]
    array(shape, path + ".shape")
    if not 1 <= len(shape) <= 8 or (rank is not None and len(shape) != rank):
        fail(path + ".shape", "invalid tensor rank")
    if type(data) is not bytes:
        fail(path + ".data", "expected BIN")
    for dim in shape:
        integer(dim, path + ".shape dimension", 0, cap)
    if observed and shape == [0, 0, 0]:
        if data:
            fail(path + ".data", "empty observation requires empty BIN")
        return
    size = {"u8": 1, "f16": 2, "f32": 4}[value["dtype"]]
    for dim in shape:
        if dim == 0 or dim > cap // size:
            fail(path + ".shape", "nonpositive dimension or tensor byte product exceeds cap")
        size *= dim
    if size != len(data):
        fail(path + ".data", "BIN length differs from dtype/shape byte count")
    if value["dtype"] != "u8":
        fmt = "<e" if value["dtype"] == "f16" else "<f"
        if any(not math.isfinite(item[0]) for item in struct.iter_unpack(fmt, data)):
            fail(path + ".data", "nonfinite tensor element")


def observation(value: object, path: str, cap: int) -> None:
    mapping(value, path, ("frames", "state", "levels_completed", "win_levels", "available_actions"))
    tensor(value["frames"], path + ".frames", cap, rank=3, observed=True)
    text(value["state"], path + ".state", identifier=True)
    for key in ("levels_completed", "win_levels"):
        integer(value[key], path + "." + key, 0)
    array(value["available_actions"], path + ".available_actions")
    for action in value["available_actions"]:
        integer(action, path + ".available_actions entry", 0, 7)


def action(value: object, path: str, available: list[int] | None = None) -> None:
    mapping(value, path, ("id", "data"))
    integer(value["id"], path + ".id", 1, 7)
    mapping(value["data"], path + ".data")
    for key, param in value["data"].items():
        integer(param, path + ".data." + key)
    if value["id"] == 6:
        for key in ("x", "y"):
            if key not in value["data"]:
                fail(path + ".data." + key, "click requires coordinate")
            integer(value["data"][key], path + ".data." + key, 0, 63)
    if available is not None and value["id"] not in available:
        fail(path + ".id", "action unavailable in pre-action observation")


def diagnostics(step: dict, path: str, previous: dict, cap: int) -> None:
    if "decision" in step:
        value = step["decision"]
        dest = path + ".decision"
        mapping(value, dest, ("score_type", "candidates"))
        enum(value["score_type"], dest + ".score_type", ("logit", "probability", "value"))
        array(value["candidates"], dest + ".candidates")
        if not value["candidates"]:
            fail(dest + ".candidates", "expected at least one candidate")
        for i, candidate in enumerate(value["candidates"]):
            location = f"{dest}.candidates[{i}]"
            mapping(candidate, location, ("action", "score"))
            action(candidate["action"], location + ".action", previous["available_actions"])
            number(candidate["score"], location + ".score")
            if value["score_type"] == "probability" and not 0 <= candidate["score"] <= 1:
                fail(location + ".score", "probability outside [0, 1]")
        if "selected_value" in value:
            number(value["selected_value"], dest + ".selected_value")
        if "action_entropy" in value:
            number(value["action_entropy"], dest + ".action_entropy", 0)
    if "prediction" in step:
        value = step["prediction"]
        dest = path + ".prediction"
        mapping(value, dest, ("target",))
        enum(value["target"], dest + ".target", ("post_action_last_frame", "post_action_sequence"))
        if not any(key in value for key in ("frames", "latent", "uncertainty")):
            fail(dest, "prediction requires frames, latent or uncertainty")
        if "frames" in value:
            rank = 2 if value["target"] == "post_action_last_frame" else 3
            tensor(value["frames"], dest + ".frames", cap, rank=rank)
        if "latent" in value:
            tensor(value["latent"], dest + ".latent", cap)
        if "uncertainty" in value:
            number(value["uncertainty"], dest + ".uncertainty", 0)
        if "errors" in value:
            mapping(value["errors"], dest + ".errors")
            for key in ("mae", "mse"):
                if key not in value["errors"]:
                    continue
                number(value["errors"][key], dest + ".errors." + key, 0)
                actual = step["observation"]["frames"]["shape"]
                target = actual[1:] if value["target"] == "post_action_last_frame" else actual
                if not actual[0] or "frames" not in value or value["frames"]["shape"] != target:
                    fail(dest + ".errors." + key, "pixel errors require comparable predicted/observed shapes")
    if "learning" in step:
        value = step["learning"]
        dest = path + ".learning"
        mapping(value, dest)
        for key in ("updates", "replay_size"):
            if key in value:
                integer(value[key], dest + "." + key, 0)
        if "loss" in value:
            number(value["loss"], dest + ".loss")
    if "timing" in step:
        mapping(step["timing"], path + ".timing")
        for key, value in step["timing"].items():
            number(value, path + ".timing." + key, 0)
    if "notes" in step:
        array(step["notes"], path + ".notes")
        for note in step["notes"]:
            text(note, path + ".notes entry")


def validate_attempt(attempt: object, cap: int) -> None:
    profile(attempt, "attempt")
    mapping(attempt, "attempt", ("metadata", "initial_observation", "steps", "termination", "summary"))
    metadata = attempt["metadata"]
    mapping(metadata, "metadata", ("game_id", "run_id", "session_id", "attempt_index", "seed",
                                   "sdk_version", "source_split", "started_at"))
    for key in ("game_id", "run_id", "session_id", "sdk_version", "model_id", "checkpoint_id"):
        if key in metadata:
            text(metadata[key], "metadata." + key, identifier=True)
    integer(metadata["attempt_index"], "metadata.attempt_index", 0)
    integer(metadata["seed"], "metadata.seed")
    enum(metadata["source_split"], "metadata.source_split", ("train", "evaluation", "unspecified"))
    text(metadata["started_at"], "metadata.started_at")
    match = _TIMESTAMP.fullmatch(metadata["started_at"])
    if match is None:
        fail("metadata.started_at", "invalid timestamp profile")
    year, month, day, hour, minute, second = (int(match[i]) for i in range(1, 7))
    try:
        date(year, month, day)
    except ValueError as exc:
        raise Arc3Error("metadata.started_at: invalid calendar date") from exc
    if hour > 23 or minute > 59 or second > 59 or (
        match[7] is not None and (int(match[8]) > 23 or int(match[9]) > 59)
    ):
        fail("metadata.started_at", "invalid clock time or UTC offset")
    if "config" in metadata:
        profile(metadata["config"], "metadata.config", config=True)

    previous = attempt["initial_observation"]
    observation(previous, "initial_observation", cap)
    array(attempt["steps"], "steps")
    for i, step in enumerate(attempt["steps"]):
        path = f"steps[{i}]"
        mapping(step, path, ("index", "action", "observation"))
        integer(step["index"], path + ".index", 0)
        if step["index"] != i:
            fail(path + ".index", "indices must be contiguous from zero")
        if previous["state"] in ("WIN", "GAME_OVER", "NOT_PLAYED"):
            fail(path, "ordinary step after terminal or initialization state")
        action(step["action"], path + ".action", previous["available_actions"])
        observation(step["observation"], path + ".observation", cap)
        diagnostics(step, path, previous, cap)
        previous = step["observation"]

    termination, summary = attempt["termination"], attempt["summary"]
    mapping(termination, "termination", ("reason", "final_state"))
    reason = termination["reason"]
    enum(reason, "termination.reason", ("win", "game_over", "reset", "action_budget", "timeout",
                                       "error", "interrupted", "training_level_boundary"))
    text(termination["final_state"], "termination.final_state", identifier=True)
    if "detail" in termination:
        text(termination["detail"], "termination.detail")
    reset = int("reset_action" in termination)
    if reset:
        if reason != "reset":
            fail("termination.reset_action", "only permitted for reset termination")
        value = termination["reset_action"]
        mapping(value, "termination.reset_action", ("id", "data"))
        integer(value["id"], "termination.reset_action.id", 0, 0)
        mapping(value["data"], "termination.reset_action.data")
        if value["data"]:
            fail("termination.reset_action.data", "RESET data must be empty")
    mapping(summary, "summary", ("real_actions", "levels_completed", "win_levels", "wall_seconds"))
    for key in ("real_actions", "levels_completed", "win_levels"):
        integer(summary[key], "summary." + key, 0)
    number(summary["wall_seconds"], "summary.wall_seconds", 0)
    count = len(attempt["steps"]) + reset
    extra = int(reason in ("training_level_boundary", "error", "timeout", "interrupted"))
    if not count <= summary["real_actions"] <= count + extra:
        fail("summary.real_actions", "inconsistent executed-action count")
    if reason != "training_level_boundary":
        if termination["final_state"] != previous["state"]:
            fail("termination.final_state", "differs from final persisted observation")
        for key in ("levels_completed", "win_levels"):
            if summary[key] != previous[key]:
                fail("summary." + key, "differs from final persisted observation")
    terminal_reason = {"WIN": "win", "GAME_OVER": "game_over"}
    if previous["state"] in terminal_reason and reason != terminal_reason[previous["state"]]:
        fail("termination.reason", "known persisted terminal state requires corresponding reason")
    if reason in terminal_reason.values():
        expected = "WIN" if reason == "win" else "GAME_OVER"
        if termination["final_state"] != expected or previous["state"] != expected:
            fail("termination", "terminal reason requires matching persisted terminal state")
