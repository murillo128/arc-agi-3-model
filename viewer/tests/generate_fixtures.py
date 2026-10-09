"""Small synthetic conformance vectors; no games, backend, or SDK execution.

Valid samples use the production Python encoder. Invalid samples bypass it with
msgpack/pyzstd and literal mutations from frozen format section 8. Each vector's
accept/reject expectation is also checked by the Python decoder before emission.
Generated binary/JSON artifacts stay ignored; expected pixels are spec literals.
"""

from copy import deepcopy
import json
from pathlib import Path
import struct
import sys

import msgpack
import pyzstd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from arc3.traces import Arc3Error, decode_attempt, encode_attempt  # noqa: E402

OUTPUT = ROOT / "viewer" / ".fixtures"
OUTPUT.mkdir(exist_ok=True)
golden = bytes.fromhex((ROOT / "tests/fixtures/arc3-v1-baseline.hex").read_text())
baseline = decode_attempt(golden)
vectors = []


def json_value(value):
    if isinstance(value, bytes):
        return {"$bin_hex": value.hex()}
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    return value


def add(name, data, valid=False, *, cap=None, error=""):
    try:
        decoded = decode_attempt(data, **({"max_uncompressed_bytes": cap} if cap else {}))
    except Arc3Error:
        if valid:
            raise
    else:
        if not valid:
            raise AssertionError(f"Python accepted invalid vector {name}")
    vector = {"name": name, "hex": data.hex(), "valid": valid}
    if valid:
        vector["expected"] = json_value(decoded)
    if cap:
        vector["cap"] = cap
    if error:
        vector["error"] = error
    vectors.append(vector)


def raw_wire(raw, *, size=None, checksum=True, content_size=True, minor=0):
    frame = pyzstd.compress(raw, {
        pyzstd.CParameter.checksumFlag: int(checksum),
        pyzstd.CParameter.contentSizeFlag: int(content_size),
    })
    return b"ARC3" + struct.pack("<HHQ", 1, minor, len(raw) if size is None else size) + frame


def wire(value, **kwargs):
    return raw_wire(msgpack.packb(value, use_bin_type=True), **kwargs)


def change(name, path, value, error):
    payload = deepcopy(baseline)
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    add(name, wire(payload), error=error)


add("independent-golden", golden, True)
add("python-one-frame", encode_attempt(baseline), True)
multi = deepcopy(baseline)
multi["steps"][0]["action"] = {"id": 6, "data": {"x": 63, "y": 0, "extra": -2}}
multi["steps"][0]["observation"]["frames"] = {"dtype": "u8", "shape": [3, 2, 2], "data": bytes.fromhex("0302010003020100ff100700")}
multi["metadata"]["run_id"] = "<img src=x onerror=alert(1)>"
multi["metadata"]["config"] = {"nested": [None, True, "π", 0.0]}
multi["steps"][0]["notes"] = ["π", "<script>throw 'unsafe'</script>"]
multi["steps"][0]["prediction"] = {
    "target": "post_action_last_frame",
    "frames": {"dtype": "f32", "shape": [2, 2], "data": bytes.fromhex("000000000000803f0000004000004040")},
    "latent": {"dtype": "f16", "shape": [2], "data": bytes.fromhex("003c00c0")},
    "uncertainty": 0.5, "errors": {"mae": 2.0, "mse": 5.0},
}
multi["steps"][0]["decision"] = {"score_type": "probability", "candidates": [
    {"action": {"id": 1, "data": {}}, "score": 0.5},
    {"action": {"id": 6, "data": {"x": 1, "y": 0}}, "score": 0.25},
], "selected_value": -0.25, "action_entropy": 1.0397207708399179}
multi["steps"][0]["learning"] = {"updates": 1, "loss": 0.125, "replay_size": 8}
multi["steps"][0]["timing"] = {"decision_ms": 2.5, "other_ms": 0.0}
add("python-multi-frame-diagnostics", encode_attempt(multi), True)
empty = deepcopy(baseline)
empty["steps"] = []
empty["initial_observation"]["frames"] = {"dtype": "u8", "shape": [0, 0, 0], "data": b""}
empty["initial_observation"]["state"] = "FUTURE_STATE"
empty["termination"] = {"reason": "reset", "final_state": "FUTURE_STATE", "reset_action": {"id": 0, "data": {}}}
add("python-empty-reset", encode_attempt(empty), True)
future = deepcopy(baseline)
future["__proto__"] = {"safe": 1}
future["extension"] = {"raw": b"\x01", "float": 1e100, "nil": None}
add("forward-minor-unknown-keys", wire(future, minor=65535), True)
add("content-size-omitted", wire(baseline, content_size=False), True)
size = struct.unpack_from("<Q", golden, 8)[0]
add("lower-cap-equality", golden, True, cap=size)
add("lower-cap-refusal", golden, cap=size - 1, error="reader cap")
for reason, state in (("win", "WIN"), ("game_over", "GAME_OVER"), ("timeout", "NOT_FINISHED"), ("interrupted", "NOT_FINISHED"), ("error", "NOT_FINISHED"), ("training_level_boundary", "WIN")):
    payload = deepcopy(baseline)
    payload["termination"] = {"reason": reason, "final_state": state}
    if reason == "training_level_boundary":
        payload["summary"]["real_actions"] = 2
        payload["summary"]["levels_completed"] = 2
    else:
        payload["steps"][0]["observation"]["state"] = state
    add(f"python-termination-{reason}", encode_attempt(payload), True)

add("truncated-header", golden[:15], error="header")
add("wrong-magic", b"ARX3" + golden[4:], error="magic")
add("unsupported-major", golden[:4] + b"\x02\x00" + golden[6:], error="major")
for n in (0, 536870913, 18446744073709551615):
    add(f"header-size-{n}", golden[:8] + struct.pack("<Q", n) + golden[16:], error="length")
for n in (size - 1, size + 1):
    add(f"declared-size-{n}", raw_wire(msgpack.packb(baseline, use_bin_type=True), size=n, content_size=False), error="length|decoding")
add("missing-checksum", wire(baseline, checksum=False), error="checksum")
add("corrupt-checksum", golden[:-1] + bytes([golden[-1] ^ 1]), error="checksum")
for suffix in (b"\0", golden[16:], bytes.fromhex("502a4d1800000000")):
    add(f"suffix-{suffix[:4].hex()}", golden + suffix, error="trailing")
add("truncated-checksum", golden[:-1], error="incomplete")
add("truncated-block", golden[:30], error="incomplete")
add("skippable-frame", golden[:16] + bytes.fromhex("502a4d1800000000"), error="ordinary")
dict_frame = bytearray(golden)
dict_frame[20] |= 1
dict_frame[21:21] = b"\x01"  # single-segment frame: dictionary ID follows descriptor
add("dictionary-id", bytes(dict_frame), error="dictionaries")
window_frame = bytearray(wire(baseline, content_size=False))
window_frame[21] = 0xa0  # window exponent 20 -> 1 GiB
add("window-over-cap", bytes(window_frame), error="window")
reserved = bytearray(golden)
reserved[20] |= 8
add("reserved-frame-header", bytes(reserved), error="reserved")
raw = msgpack.packb(baseline, use_bin_type=True)
add("array-root", raw_wire(b"\x90"), error="root")
add("msgpack-suffix", raw_wire(raw + b"\xc0"), error="trailing")
# B has five top-level entries in a fixmap. Add a sixth duplicate key.
add("duplicate-key", raw_wire(b"\x86" + raw[1:] + msgpack.packb("summary") + b"\x80"), error="duplicate")
payload = deepcopy(baseline)
del payload["termination"]
add("missing-termination", wire(payload), error="missing")
add("invalid-utf8", raw_wire(raw.replace(b"fixture-v1", b"\xffixture-v1")), error="UTF-8")
for v in (False, 0.0):
    change(f"index-type-{type(v).__name__}", ["steps", 0, "index"], v, "integer")
change("unsafe-integer", ["metadata", "seed"], 9007199254740992, "unsafe integer")
for shape in ([-1, 2, 2], [1, 2], [0, 536870913, 0], [0, 2, 2], [2, 536870912, 536870912]):
    change(f"invalid-shape-{shape}", ["steps", 0, "observation", "frames", "shape"], shape, "shape")
change("tensor-byte-length", ["steps", 0, "observation", "frames", "data"], b"\x03\x02\x01", "BIN length")
change("tensor-str", ["steps", 0, "observation", "frames", "data"], "03020100", "BIN")
change("observed-float-dtype", ["steps", 0, "observation", "frames", "dtype"], "f32", "dtype")
for shape, dtype in (([1] * 9, "u8"), ([536870912], "f32")):
    change(f"latent-shape-{dtype}", ["steps", 0, "prediction"], {"target": "post_action_last_frame", "latent": {"dtype": dtype, "shape": shape, "data": b""}}, "shape")
change("index-gap", ["steps", 0, "index"], 1, "contiguous")
change("regular-reset", ["steps", 0, "action", "id"], 0, "integer")
change("unavailable-action", ["steps", 0, "action", "id"], 7, "unavailable")
change("click-coordinate", ["steps", 0, "action"], {"id": 6, "data": {"x": 64, "y": 0}}, "integer")
change("click-missing-y", ["steps", 0, "action"], {"id": 6, "data": {"x": 1}}, "integer")
change("null-prediction", ["steps", 0, "prediction"], None, "map")
change("empty-prediction", ["steps", 0, "prediction"], {"target": "post_action_last_frame"}, "requires")
change("bad-probability", ["steps", 0, "decision"], {"score_type": "probability", "candidates": [{"action": {"id": 1, "data": {}}, "score": 1.01}]}, "probability")
change("negative-timing", ["steps", 0, "timing"], {"future_ms": -1}, "number")
change("nonfinite-f16", ["steps", 0, "prediction"], {"target": "post_action_last_frame", "latent": {"dtype": "f16", "shape": [1], "data": bytes.fromhex("007c")}}, "nonfinite")
change("nonfinite-f32", ["steps", 0, "prediction"], {"target": "post_action_last_frame", "latent": {"dtype": "f32", "shape": [1], "data": bytes.fromhex("0000c07f")}}, "nonfinite")
change("mismatched-error", ["steps", 0, "prediction"], {"target": "post_action_last_frame", "frames": {"dtype": "u8", "shape": [1, 1], "data": b"\0"}, "errors": {"mae": 0}}, "comparable")
change("unknown-reason", ["termination", "reason"], "done", "expected")
change("misplaced-reset", ["termination", "reset_action"], {"id": 0, "data": {}}, "reset")
for n in (0, 3):
    change(f"real-action-count-{n}", ["summary", "real_actions"], n, "count")
change("final-state-mismatch", ["termination", "final_state"], "WIN", "differs")
change("summary-counter-mismatch", ["summary", "levels_completed"], 1, "differs")
change("invalid-split", ["metadata", "source_split"], "held_out", "expected")
for timestamp in ("2026-10-09T12:00:00", "2025-02-29T00:00:00Z", "0000-01-01T00:00:00Z", "2026-01-01T00:00:00+24:00"):
    change(f"timestamp-{timestamp}", ["metadata", "started_at"], timestamp, "timestamp|calendar")
change("config-bin", ["metadata", "config"], {"tensor": b"\0"}, "config")
change("known-optional-null", ["metadata", "model_id"], None, "STR")
change("terminal-continuation", ["initial_observation", "state"], "GAME_OVER", "terminal")
add("huge-array-count", raw_wire(b"\x81\xa1x\xdd\xff\xff\xff\xff"), error="count")
add("huge-map-count", raw_wire(b"\xdf\xff\xff\xff\xff"), error="count")
add("huge-bin-length", raw_wire(b"\x81\xa1x\xc6\xff\xff\xff\xff"), error="length")
add("excess-depth", raw_wire(b"\x81\xa1x" + b"\x91" * 64 + b"\xc0"), error="nesting")
add("extension-tag", raw_wire(b"\x81\xa1x\xd4\x01\x00"), error="extension")
add("non-str-key", raw_wire(b"\x81\x01\xc0"), error="keys")
change("nonfinite-number", ["summary", "wall_seconds"], float("inf"), "nonfinite")

# Replay interaction fixtures are original synthetic public observations, not game assets.
zero = deepcopy(baseline)
zero["steps"] = []
zero["summary"]["real_actions"] = 0
zero["summary"]["wall_seconds"] = 0.0
empty_result = deepcopy(baseline)
empty_result["steps"][0]["observation"]["frames"] = {"dtype": "u8", "shape": [0, 0, 0], "data": b""}
unexecuted_reset = deepcopy(zero)
unexecuted_reset["termination"] = {"reason": "reset", "final_state": "NOT_FINISHED"}
reset = deepcopy(multi)
reset["termination"] = {"reason": "reset", "final_state": "NOT_FINISHED", "reset_action": {"id": 0, "data": {}}}
reset["summary"]["real_actions"] = 2

def scene(height, width, phase):
    # Test-owned colored grid with a border and changing tiles; no privileged rules.
    return bytes(5 if x in (0, width - 1) or y in (0, height - 1) else
                 9 if (x + phase) % 4 == 0 else 11 if (x, y) == (phase + 2, 3) else
                 15 if y % 3 == 0 else 1 for y in range(height) for x in range(width))

replay = deepcopy(baseline)
replay["initial_observation"]["frames"] = {"dtype": "u8", "shape": [1, 8, 12], "data": scene(8, 12, 0)}
replay["steps"] = []
for index, (action, level, state, count, height, width) in enumerate([
    ({"id": 1, "data": {}}, 0, "NOT_FINISHED", 2, 8, 12),
    ({"id": 6, "data": {"x": 9, "y": 4}}, 1, "NOT_FINISHED", 3, 6, 10),
    ({"id": 7, "data": {}}, 2, "WIN", 1, 6, 10),
]):
    observation = deepcopy(baseline["initial_observation"])
    observation.update({"levels_completed": level, "state": state, "available_actions": [] if state == "WIN" else [1, 6, 7]})
    observation["frames"] = {"dtype": "u8", "shape": [count, height, width], "data": b"".join(scene(height, width, index + frame + 1) for frame in range(count))}
    replay["steps"].append({"index": index, "action": action, "observation": observation})
replay["termination"] = {"reason": "win", "final_state": "WIN"}
replay["summary"].update({"real_actions": 3, "levels_completed": 2, "wall_seconds": 1.5})
for name, payload in (("empty-result", empty_result), ("zero-step", zero), ("unexecuted-reset", unexecuted_reset), ("closing-reset", reset), ("replay-levels", replay)):
    encoded = encode_attempt(payload)
    decode_attempt(encoded)
    (OUTPUT / f"{name}.arc3").write_bytes(encoded)

(OUTPUT / "vectors.json").write_text(json.dumps(vectors, ensure_ascii=False), encoding="utf-8")
for name, data in (("one-frame", encode_attempt(baseline)), ("multi-frame", encode_attempt(multi)), ("empty", encode_attempt(empty))):
    (OUTPUT / f"{name}.arc3").write_bytes(data)
print(f"Generated {len(vectors)} Python-checked vectors and 8 synthetic .arc3 files in {OUTPUT}")
