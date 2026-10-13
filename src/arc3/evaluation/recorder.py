"""Public evaluation attempts, finalized locally with the frozen .arc3 codec."""

from copy import deepcopy
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import struct
import tempfile
from time import perf_counter
from uuid import uuid4

from arc3.core.policy import Action, Observation, Policy
from arc3.traces import write_attempt
from arc3.traces._errors import MAX_UNCOMPRESSED_BYTES
from arc3.traces._validation import diagnostics as validate_diagnostics, profile, representation
from arc3.traces.types import Attempt, Observation as WireObservation


def observation_snapshot(observation: Observation) -> WireObservation:
    """Preserve every layer and reject shapes v1 cannot represent losslessly."""
    layers = observation.frame
    if layers:
        height = len(layers[0])
        width = len(layers[0][0]) if height else 0
        if not height or not width or any(
            len(layer) != height or any(len(row) != width for row in layer)
            for layer in layers
        ):
            raise ValueError(".arc3 requires nonempty rectangular SDK frames of equal size")
        shape = [len(layers), height, width]
        data = bytes(pixel for layer in layers for row in layer for pixel in row)
    else:
        shape, data = [0, 0, 0], b""
    return {
        "frames": {"dtype": "u8", "shape": shape, "data": data},
        "state": observation.state,
        "levels_completed": observation.levels_completed,
        "win_levels": observation.win_levels,
        "available_actions": list(observation.available_actions),
    }


def _prediction_errors(step: dict) -> None:
    """Merge owned metrics from frozen snapshots, never replay-training losses."""
    prediction = step.get("prediction", {})
    if prediction.get("target") == "post_action_last_frame" and "representation" in step:
        real = step.get("representation", {}).get("z")
        for key, source in (("latent_mse", prediction.get("latent")),
                            ("no_change_mse", prediction.get("dynamics", {}).get("z"))):
            if real is not None and source is not None and source["shape"] == real["shape"]:
                def values(tensor):
                    if tensor["dtype"] == "u8":
                        return tensor["data"]
                    return [item[0] for item in struct.iter_unpack(
                        "<f" if tensor["dtype"] == "f32" else "<e", tensor["data"])]
                actual_values, predicted_values = values(real), values(source)
                if actual_values and len(actual_values) == len(predicted_values):
                    mse = math.fsum((a - b) ** 2 for a, b in zip(actual_values, predicted_values)) / len(actual_values)
                    prediction.setdefault("errors", {})[key] = mse
    predicted = prediction.get("frames")
    actual = step["observation"]["frames"]
    if predicted is None or actual["shape"][0] == 0:
        return
    if prediction["target"] == "post_action_last_frame":
        shape = actual["shape"][1:]
        count = shape[0] * shape[1]
        target = actual["data"][-count:]
    else:
        shape, target = actual["shape"], actual["data"]
    if predicted["shape"] != shape:
        return
    if predicted["dtype"] == "u8":
        values = predicted["data"]
    else:
        fmt = {"f16": "<e", "f32": "<f"}[predicted["dtype"]]
        values = [value[0] for value in struct.iter_unpack(fmt, predicted["data"])]
    if len(values) != len(target):
        return  # Codec still rejects an invalid tensor, without invented errors.
    differences = [predicted_pixel - real_pixel for predicted_pixel, real_pixel in zip(values, target)]
    prediction.setdefault("errors", {}).update({
        "mae": sum(abs(value) for value in differences) / len(target),
        "mse": sum(value * value for value in differences) / len(target),
    })


class EvaluationRecorder:
    """One session per game; no training imports, updates or SDK recording.

    Unconfirmed failing SDK calls are not counted. Abrupt process death cannot
    finalize the in-memory attempt; published .arc3 files are always complete.
    """

    def __init__(self, directory: str | Path, *, run_id: str, seed: int,
                 sdk_version: str, format_minor: int = 0) -> None:
        if type(format_minor) is not int or format_minor not in (0, 1):
            raise ValueError("format_minor must be 0 or 1")
        self.format_minor = format_minor
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.session_id = uuid4().hex
        self.seed = seed
        self.sdk_version = sdk_version
        self._game_id: str | None = None
        self._attempt_index = 0
        self._attempt: Attempt | None = None
        self._started = 0.0
        self._real_actions = 0
        self._pending: dict = {}
        self._policy: Policy | None = None
        self._pre_action: Observation | None = None

    def start(self, observation: Observation, policy: Policy | None = None) -> None:
        self._pending = {}
        self._pre_action = None
        self._policy = policy
        if self._game_id is not None and observation.game_id != self._game_id:
            raise ValueError("A trace session cannot change game_id")
        self._game_id = observation.game_id
        if observation.state == "NOT_PLAYED":
            return  # Bootstrap RESET is session accounting, not an empty attempt.
        self._started = perf_counter()
        self._real_actions = 0
        self._attempt = {
            "metadata": {
                "game_id": observation.game_id, "run_id": self.run_id,
                "session_id": self.session_id, "attempt_index": self._attempt_index,
                "seed": self.seed, "sdk_version": self.sdk_version,
                "source_split": "evaluation",
                "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
            "initial_observation": observation_snapshot(observation),
            "steps": [],
        }
        hook = getattr(policy, "trace_initial_diagnostics", None)
        if hook is not None:
            initial = deepcopy(hook(observation))
            profile(initial, "initial diagnostics")
            if not isinstance(initial, dict) or initial.keys() - {"initial_representation"}:
                raise ValueError("Initial diagnostics permit only initial_representation")
            if self.format_minor >= 1 and "initial_representation" in initial:
                representation(initial["initial_representation"], "initial_representation", MAX_UNCOMPRESSED_BYTES)
            self._attempt.update(initial)
        if observation.state in {"WIN", "GAME_OVER"}:
            self.finish("win" if observation.state == "WIN" else "game_over")

    def before_action(self, observation: Observation, action: Action, policy: Policy) -> None:
        self._policy = policy
        self._pre_action = observation
        self._pending = {"action": {"id": action.id, "data": dict(action.data)}}
        if action.id == 0:
            return
        hook = getattr(policy, "trace_diagnostics", None)
        if hook is not None:
            diagnostics = deepcopy(hook(observation, Action(action.id, dict(action.data))))
            if not isinstance(diagnostics, dict) or diagnostics.keys() - {
                "decision", "prediction", "learning", "timing", "notes", "policy_mode",
            }:
                raise ValueError("Diagnostics must contain only optional .arc3 step groups")
            profile(diagnostics, "pre-action diagnostics")
            errors = diagnostics.get("prediction", {}).get("errors", {})
            owned_errors = ("mae", "mse", "latent_mse", "no_change_mse") if self.format_minor >= 1 else ("mae", "mse")
            if any(key in errors for key in owned_errors):
                raise ValueError("Prediction errors must be computed after the real action")
            validate_diagnostics({**diagnostics, "observation": observation_snapshot(observation)},
                                 "pending", observation_snapshot(observation), MAX_UNCOMPRESSED_BYTES,
                                 minor=self.format_minor, observations=self._observations())
            self._pending.update(diagnostics)

    def _observations(self) -> list[WireObservation]:
        return [self._attempt["initial_observation"],
                *(step["observation"] for step in self._attempt["steps"])]

    def after_action(self, action: Action, observation: Observation) -> None:
        if action.id == 0:
            if self._attempt is not None:
                self._real_actions += 1
                self._attempt["termination"] = {
                    "reason": "reset", "final_state": self._last_observation()["state"],
                    "reset_action": self._pending["action"],
                }
                self.finish("reset")
            self.start(observation, self._policy)
            return
        if self._attempt is None:
            raise RuntimeError("An ordinary action requires an active attempt")
        self._real_actions += 1
        if observation.game_id != self._game_id:
            raise ValueError("A trace session cannot change game_id")
        step = {"index": len(self._attempt["steps"]), **self._pending,
                "observation": observation_snapshot(observation)}
        self._attempt["steps"].append(step)
        self._pending = {}
        pre_action, self._pre_action = self._pre_action, None
        hook = getattr(self._policy, "trace_result_diagnostics", None)
        try:
            if hook is not None:
                result = deepcopy(hook(pre_action, Action(step["action"]["id"], dict(step["action"]["data"])), observation))
                profile(result, "result diagnostics")
                if not isinstance(result, dict) or result.keys() - {"representation", "learning", "timing", "notes"}:
                    raise ValueError("Result diagnostics permit only representation, learning, timing and notes")
                if self.format_minor >= 1 and not observation.frame:
                    result.pop("representation", None)
                # Stage and validate before attachment so a hook failure cannot
                # corrupt the already confirmed action/result or frozen prediction.
                candidate = {**step, **result}
                for key in ("learning", "timing"):
                    if key in step and key in result and isinstance(result[key], dict):
                        candidate[key] = {**step[key], **result[key]}
                        if key == "learning" and self.format_minor >= 1 and "modules" in step[key] and isinstance(result[key].get("modules"), dict):
                            candidate[key]["modules"] = {**step[key]["modules"], **result[key]["modules"]}
                if "notes" in step and "notes" in result and isinstance(result["notes"], list):
                    candidate["notes"] = step["notes"] + result["notes"]
                validate_diagnostics(candidate, "result", self._observations()[-2], MAX_UNCOMPRESSED_BYTES,
                                     minor=self.format_minor, observations=self._observations()[:-1])
                step.update(candidate)
        finally:
            # Even a failing result hook retains and scores the confirmed pixels.
            scoring = step if self.format_minor >= 1 else {k: v for k, v in step.items() if k != "representation"}
            _prediction_errors(scoring)
        if observation.state in {"WIN", "GAME_OVER"}:
            self.finish("win" if observation.state == "WIN" else "game_over")

    def _last_observation(self) -> WireObservation:
        return (self._attempt["steps"][-1]["observation"] if self._attempt["steps"]
                else self._attempt["initial_observation"])

    def finish(self, reason: str, *, detail: str | None = None) -> None:
        self._pending = {}
        self._pre_action = None
        if self._attempt is None:
            return
        last = self._last_observation()
        # Known terminal evidence always wins over a later budget/error/reset.
        reason = {"WIN": "win", "GAME_OVER": "game_over"}.get(last["state"], reason)
        if "termination" not in self._attempt:
            self._attempt["termination"] = {"reason": reason, "final_state": last["state"]}
        if detail is not None:
            self._attempt["termination"]["detail"] = detail
        self._attempt["summary"] = {
            "real_actions": self._real_actions, "levels_completed": last["levels_completed"],
            "win_levels": last["win_levels"], "wall_seconds": perf_counter() - self._started,
        }
        attempt, self._attempt = self._attempt, None
        # Encode/fsync using the shared codec, then publish with a no-clobber
        # hard link. Even concurrent writers or repeated IDs cannot overwrite.
        fd, temporary = tempfile.mkstemp(prefix=".arc3-", suffix=".tmp", dir=self.directory)
        os.close(fd)
        try:
            write_attempt(temporary, attempt, format_minor=self.format_minor)
            basename = f"{self.session_id}-{self._attempt_index:06d}"
            collision = 0
            while True:
                suffix = f"-{collision}" if collision else ""
                target = self.directory / f"{basename}{suffix}.arc3"
                try:
                    os.link(temporary, target)
                    break
                except FileExistsError:
                    collision += 1
        finally:
            Path(temporary).unlink(missing_ok=True)
        self._attempt_index += 1
