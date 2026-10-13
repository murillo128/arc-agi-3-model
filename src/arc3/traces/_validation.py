"""Frozen v1.0 semantics plus minor-gated docs/arc3-format-v1.1.md additions."""

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


def state_tensor(value: object, path: str, cap: int) -> None:
    tensor(value, path, cap, rank=1)
    if value["dtype"] != "f32":
        fail(path + ".dtype", "state requires f32")


def representation(value: object, path: str, cap: int) -> None:
    mapping(value, path, ("z", "h"))
    for key in ("z", "h"):
        state_tensor(value[key], path + "." + key, cap)


def extension_diagnostics(step: dict, path: str, cap: int, observations: list[dict]) -> None:
    """Known 1.1 fields; minor 0 continues to treat these names as unknown."""
    if "representation" in step:
        representation(step["representation"], path + ".representation", cap)
        if not step["observation"]["frames"]["shape"][0]:
            fail(path + ".representation", "real representation requires a visual result")
    if "policy_mode" in step:
        enum(step["policy_mode"], path + ".policy_mode", ("random", "learned"))
    prediction = step.get("prediction", {})
    dest = path + ".prediction"
    if "dynamics" in prediction:
        value = prediction["dynamics"]
        mapping(value, dest + ".dynamics", ("z", "h", "delta_z", "h_next"))
        if prediction["target"] != "post_action_last_frame" or "latent" not in prediction:
            fail(dest + ".dynamics", "requires latent and last-frame target")
        for key in ("z", "h", "delta_z", "h_next"):
            state_tensor(value[key], dest + ".dynamics." + key, cap)
        state_tensor(prediction["latent"], dest + ".latent", cap)
        if value["z"]["shape"] != value["delta_z"]["shape"] or value["z"]["shape"] != prediction["latent"]["shape"]:
            fail(dest + ".dynamics", "latent dimensions must match")
        if value["h"]["shape"] != value["h_next"]["shape"]:
            fail(dest + ".dynamics", "memory dimensions must match")
        # Numerical residual disagreement is a viewer diagnostic, not malformed.
    for key, source in (("latent_mse", prediction.get("latent")),
                        ("no_change_mse", prediction.get("dynamics", {}).get("z"))):
        errors = prediction.get("errors", {})
        if key in errors:
            number(errors[key], dest + ".errors." + key, 0)
            actual = step.get("representation", {}).get("z")
            if prediction["target"] != "post_action_last_frame" or source is None or actual is None or source["shape"] != actual["shape"]:
                fail(dest + ".errors." + key, "latent error requires matched real/predicted tensors and last-frame target")
    if "visual_delta" in prediction:
        value = prediction["visual_delta"]
        location = dest + ".visual_delta"
        mapping(value, location, ("reference_position", "reference_frame", "delta"))
        if prediction["target"] != "post_action_last_frame" or "frames" not in prediction:
            fail(location, "requires absolute frames and last-frame target")
        integer(value["reference_position"], location + ".reference_position", 0, len(observations) - 1)
        for key, maximum in (("reference_frame", 15), ("delta", 16)):
            tensor(value[key], location + "." + key, cap, rank=2)
            if value[key]["dtype"] != "u8" or any(v > maximum for v in value[key]["data"]):
                fail(location + "." + key, "invalid u8 palette/KEEP code")
        frames, reference, delta = prediction["frames"], value["reference_frame"], value["delta"]
        if frames["dtype"] != "u8" or any(v > 15 for v in frames["data"]):
            fail(dest + ".frames", "visual decoder requires u8 colors 0..15")
        if frames["shape"] != reference["shape"] or delta["shape"] != reference["shape"]:
            fail(location, "internal geometry must match")
        observed = observations[value["reference_position"]]["frames"]
        size = len(reference["data"])
        if not observed["shape"][0] or observed["shape"][1:] != reference["shape"] or observed["data"][-size:] != reference["data"]:
            fail(location, "reference must equal a nonempty causal observation's last frame")
        if any((r if d == 16 else d) != p for r, d, p in zip(reference["data"], delta["data"], frames["data"])):
            fail(location, "KEEP/replacement delta differs from absolute predicted board")
    learning = step.get("learning", {})
    if "modules" in learning:
        modules = learning["modules"]
        location = path + ".learning.modules"
        mapping(modules, location)
        if not modules:
            fail(location, "expected nonempty module map")
        for key in ("world_model", "decoder", "actor", "critic"):
            if key not in modules:
                continue
            value, module_path = modules[key], location + "." + key
            mapping(value, module_path)
            if not any(k in value for k in ("updates", "loss", "losses", "reset_reason")):
                fail(module_path, "module requires actual reported data")
            if "updates" in value:
                integer(value["updates"], module_path + ".updates", 0)
            if "loss" in value:
                number(value["loss"], module_path + ".loss")
            if "losses" in value:
                mapping(value["losses"], module_path + ".losses")
                if not value["losses"]:
                    fail(module_path + ".losses", "expected nonempty loss map")
                for name, loss in value["losses"].items():
                    text(name, module_path + ".losses key", identifier=True)
                    number(loss, module_path + ".losses." + name)
            if "reset_reason" in value:
                if key not in ("actor", "critic"):
                    fail(module_path + ".reset_reason", "reset events are actor/critic only")
                text(value["reset_reason"], module_path + ".reset_reason", identifier=True)


def diagnostics(step: dict, path: str, previous: dict, cap: int, *, minor: int = 0,
                observations: list[dict] | None = None) -> None:
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
    if minor >= 1:
        extension_diagnostics(step, path, cap, observations if observations is not None else [previous])


def validate_attempt(attempt: object, cap: int, *, minor: int = 0) -> None:
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
    if minor >= 1 and "initial_representation" in attempt:
        representation(attempt["initial_representation"], "initial_representation", cap)
    observations = [previous]
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
        diagnostics(step, path, previous, cap, minor=minor, observations=observations)
        previous = step["observation"]
        observations.append(previous)

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
