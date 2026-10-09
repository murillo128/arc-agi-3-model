"""Offline codec conformance; literal expectations from format v1 section 8.

Negative payloads use the installed wire libraries directly, bypassing the
production encoder so its validation cannot hide a reader defect.
"""

from copy import deepcopy
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

import msgpack
import pyzstd

from arc3.traces import (
    Arc3Error, Arc3ResourceLimitError, MAX_UNCOMPRESSED_BYTES,
    decode_attempt, encode_attempt, read_attempt, write_attempt,
)
from arc3.traces.__main__ import main


def baseline():
    """Spec baseline B; all bytes and values are synthetic, independently fixed."""
    return {
        "metadata": {
            "game_id": "fixture-v1", "run_id": "run-1", "session_id": "session-1",
            "attempt_index": 0, "seed": 42, "sdk_version": "0.9.9",
            "source_split": "unspecified", "started_at": "2026-10-09T12:00:00Z",
        },
        "initial_observation": {
            "frames": {"dtype": "u8", "shape": [1, 2, 2], "data": bytes.fromhex("00010203")},
            "state": "NOT_FINISHED", "levels_completed": 0, "win_levels": 2,
            "available_actions": [1, 6],
        },
        "steps": [{
            "index": 0, "action": {"id": 1, "data": {}},
            "observation": {
                "frames": {"dtype": "u8", "shape": [1, 2, 2], "data": bytes.fromhex("03020100")},
                "state": "NOT_FINISHED", "levels_completed": 0, "win_levels": 2,
                "available_actions": [1, 6],
            },
        }],
        "termination": {"reason": "action_budget", "final_state": "NOT_FINISHED"},
        "summary": {"real_actions": 1, "levels_completed": 0, "win_levels": 2, "wall_seconds": 0.25},
    }


def prediction():
    return {
        "target": "post_action_last_frame",
        "frames": {"dtype": "f32", "shape": [2, 2],
                   "data": bytes.fromhex("000000000000803f0000004000004040")},
        "latent": {"dtype": "f16", "shape": [2], "data": bytes.fromhex("003c00c0")},
        "uncertainty": 0.5, "errors": {"mae": 2.0, "mse": 5.0},
    }


def decision():
    return {
        "score_type": "probability",
        "candidates": [
            {"action": {"id": 1, "data": {}}, "score": 0.5},
            {"action": {"id": 6, "data": {"x": 1, "y": 0}}, "score": 0.25},
        ],
        "selected_value": -0.25, "action_entropy": 1.0397207708399179,
    }


def raw_wire(raw, *, minor=0, size=None, checksum=True, content_size=True):
    """Independent envelope construction with third-party serialization/compression."""
    frame = pyzstd.compress(raw, {
        pyzstd.CParameter.checksumFlag: int(checksum),
        pyzstd.CParameter.contentSizeFlag: int(content_size),
    })
    return b"ARC3" + struct.pack("<HHQ", 1, minor, len(raw) if size is None else size) + frame


def wire(value, **kwargs):
    return raw_wire(msgpack.packb(value, use_bin_type=True), **kwargs)


def change(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


class Arc3CodecTests(unittest.TestCase):
    def test_independent_golden_baseline_and_wire_primitives(self):
        path = Path(__file__).parent / "fixtures" / "arc3-v1-baseline.hex"
        golden = bytes.fromhex(path.read_text())
        decoded = decode_attempt(golden)
        self.assertEqual(decoded, baseline())
        self.assertEqual(list(decoded["initial_observation"]["frames"]["data"]), [0, 1, 2, 3])
        self.assertEqual(list(decoded["steps"][0]["observation"]["frames"]["data"]), [3, 2, 1, 0])
        self.assertEqual(msgpack.unpackb(bytes.fromhex("c40403020100")), b"\x03\x02\x01\x00")
        self.assertEqual(struct.unpack("<2e", msgpack.unpackb(bytes.fromhex("c404003c00c0"))), (1.0, -2.0))
        encoded = encode_attempt(decoded)
        n = struct.unpack_from("<Q", encoded, 8)[0]
        self.assertEqual(encoded[:8], b"ARC3\x01\x00\x00\x00")
        self.assertEqual(len(pyzstd.decompress(encoded[16:])), n)
        self.assertTrue(encoded[20] & 4)
        self.assertIn(bytes.fromhex("c40403020100"), pyzstd.decompress(encoded[16:]))
        self.assertEqual(decode_attempt(encoded), baseline())

    def test_positive_spec_vectors_v01_through_v16(self):
        for vector in range(1, 17):
            value = baseline()
            step = value["steps"][0]
            obs = step["observation"]
            summary = value["summary"]
            if vector == 2:
                obs["frames"] = {"dtype": "u8", "shape": [2, 2, 2], "data": bytes.fromhex("0405060708090a0b")}
            elif vector == 3:
                step["action"] = {"id": 6, "data": {"x": 1, "y": 0}}
            elif vector == 4:
                obs["levels_completed"] = summary["levels_completed"] = 1
            elif vector == 5:
                obs.update(state="WIN", levels_completed=2, available_actions=[])
                value["termination"] = {"reason": "win", "final_state": "WIN"}
                summary["levels_completed"] = 2
            elif vector == 6:
                obs.update(state="GAME_OVER", available_actions=[0])
                value["termination"] = {"reason": "game_over", "final_state": "GAME_OVER"}
            elif vector == 7:
                value["termination"] = {"reason": "reset", "final_state": "NOT_FINISHED",
                                        "reset_action": {"id": 0, "data": {}}}
                summary["real_actions"] = 2
            elif vector == 8:
                value["metadata"]["attempt_index"] = 1
                value["initial_observation"]["frames"]["data"] = bytes.fromhex("09090909")
                value["steps"] = []
                summary.update(real_actions=0, wall_seconds=0.0)
            elif vector == 9:
                value["steps"] = []
                value["termination"] = {"reason": "training_level_boundary", "final_state": "NOT_FINISHED"}
                value["metadata"]["source_split"] = "train"
                summary["levels_completed"] = 1
            elif vector == 10:
                obs["frames"] = {"dtype": "u8", "shape": [0, 0, 0], "data": b""}
            elif vector == 11:
                step["prediction"] = prediction()
                value["metadata"]["config"] = {"uncertainty_estimator": "synthetic fixture", "uncertainty_units": "unitless"}
            elif vector == 12:
                step.update(decision=decision(), learning={"updates": 1, "loss": 0.125, "replay_size": 8},
                            timing={"decision_ms": 2.5, "environment_ms": 10.0, "learning_ms": 1.0},
                            notes=["Synthetic diagnostic example"])
                value["metadata"]["config"] = {"loss_objective": "synthetic fixture", "loss_aggregation": "one update"}
            elif vector == 13:
                step["prediction"] = {"target": "post_action_sequence",
                                      "frames": {"dtype": "u8", "shape": [1, 1, 1], "data": b"\x07"}}
            elif vector == 14:
                value["public_note"] = "fixture"
            elif vector == 15:
                value["termination"] = {"reason": "error", "final_state": "NOT_FINISHED",
                                        "detail": "Action executed; response unavailable"}
                summary["real_actions"] = 2
            elif vector == 16:
                value["termination"]["reason"] = "reset"
            with self.subTest(vector=vector):
                decoded = decode_attempt(wire(value, minor=int(vector == 14)))
                self.assertEqual(decoded, value)
                self.assertEqual(decode_attempt(encode_attempt(value)), value)
                if vector == 2:
                    self.assertEqual(list(decoded["steps"][0]["observation"]["frames"]["data"]),
                                     [4, 5, 6, 7, 8, 9, 10, 11])
                    self.assertEqual(decoded["summary"]["real_actions"], 1)
                if vector == 7:
                    self.assertEqual(len(decoded["steps"]), 1)
                    self.assertNotIn("observation", decoded["termination"])
                if vector == 11:
                    pred = decoded["steps"][0]["prediction"]
                    self.assertEqual(struct.unpack("<4f", pred["frames"]["data"]), (0.0, 1.0, 2.0, 3.0))
                    self.assertEqual(struct.unpack("<2e", pred["latent"]["data"]), (1.0, -2.0))
                    self.assertEqual(pred["errors"], {"mae": 2.0, "mse": 5.0})

    def test_unknown_fields_states_utf8_and_non64_pixel_dimensions(self):
        value = baseline()
        value["public_note"] = "π"
        value["extension"] = {"nil": None, "flag": True, "bin": b"abc", "list": [-9_007_199_254_740_991]}
        value["metadata"].update(seed=0, model_id="model", checkpoint_id="checkpoint", config=None)
        value["initial_observation"]["state"] = "FUTURE_PUBLIC_STATE"
        obs = value["steps"][0]["observation"]
        obs["state"] = value["termination"]["final_state"] = "OTHER_PUBLIC_STATE"
        obs["frames"] = {"dtype": "u8", "shape": [1, 1, 65], "data": bytes(range(65))}
        value["steps"][0]["action"] = {"id": 6, "data": {"x": 63, "y": 63, "extra": -1}}
        decoded = decode_attempt(wire(value, minor=65535))
        self.assertEqual(decoded, value)
        self.assertEqual(decode_attempt(encode_attempt(value)), value)

    def test_multiple_steps_use_previous_availability_and_preserve_animation_repeats(self):
        value = baseline()
        value["steps"][0]["observation"]["available_actions"] = [7]
        second = deepcopy(value["steps"][0])
        second.update(index=1, action={"id": 7, "data": {}})
        second["observation"]["frames"] = {"dtype": "u8", "shape": [2, 1, 1], "data": b"\x04\x04"}
        value["steps"].append(second)
        value["summary"]["real_actions"] = 2
        self.assertEqual(decode_attempt(encode_attempt(value)), value)
        second["action"]["id"] = 1
        with self.assertRaisesRegex(Arc3Error, "pre-action"):
            decode_attempt(wire(value))

    def test_empty_step_attempts_terminal_reasons_and_exceptional_counts(self):
        for state, reason in (("NOT_PLAYED", "action_budget"), ("WIN", "win"),
                              ("GAME_OVER", "game_over"), ("NOT_FINISHED", "reset")):
            value = baseline()
            value["steps"] = []
            value["initial_observation"]["state"] = state
            value["termination"] = {"reason": reason, "final_state": state}
            value["summary"]["real_actions"] = 0
            self.assertEqual(decode_attempt(encode_attempt(value)), value)
        for reason in ("training_level_boundary", "error", "timeout", "interrupted"):
            for count in (1, 2):
                value = baseline()
                value["termination"]["reason"] = reason
                value["summary"]["real_actions"] = count
                self.assertEqual(decode_attempt(wire(value)), value)

    def test_header_checksum_truncation_and_suffix_vectors_e01_e08(self):
        valid = wire(baseline())
        corrupt = bytearray(valid)
        corrupt[-1] ^= 1
        cases = [
            ("short header", valid[:15], "header"),
            ("magic", b"ARX3" + valid[4:], "magic"),
            ("major", valid[:4] + b"\x02\x00" + valid[6:], "major"),
            ("absent checksum", wire(baseline(), checksum=False), "checksum"),
            ("bad checksum", bytes(corrupt), "checksum"),
            ("checksum truncated", valid[:-1], "incomplete"),
            ("data block truncated", valid[:-12], "incomplete"),
            ("byte suffix", valid + b"\x00", "trailing"),
            ("second empty frame", valid + pyzstd.compress(b""), "trailing"),
            ("skippable suffix", valid + bytes.fromhex("502a4d1800000000"), "trailing"),
            ("skippable first frame", valid[:16] + bytes.fromhex("502a4d1800000000"), "ordinary"),
            ("dictionary", valid[:16] + bytes.fromhex("28b52ffd050001"), "dictionaries"),
            ("1GiB window", valid[:16] + bytes.fromhex("28b52ffd04a0"), "window"),
            ("reserved frame bit", valid[:20] + bytes([valid[20] | 0x08]) + valid[21:], "header"),
        ]
        for length in (0, MAX_UNCOMPRESSED_BYTES + 1, 2**64 - 1):
            cases.append((f"length {length}", valid[:8] + struct.pack("<Q", length) + valid[16:], "length"))
        for delta in (-1, 1):
            n = struct.unpack_from("<Q", valid, 8)[0]
            cases.append((f"size {delta}", valid[:8] + struct.pack("<Q", n + delta) + valid[16:], "size"))
        for name, data, error in cases:
            with self.subTest(case=name), self.assertRaisesRegex(Arc3Error, error):
                decode_attempt(data)

    def test_omitted_content_size_hard_ceiling_and_checksum_completion(self):
        raw = msgpack.packb(baseline(), use_bin_type=True)
        self.assertEqual(decode_attempt(raw_wire(raw, content_size=False)), baseline())
        for size in (len(raw) - 1, len(raw) + 1):
            with self.subTest(size=size), self.assertRaisesRegex(Arc3Error, "length"):
                decode_attempt(raw_wire(raw, size=size, content_size=False))
        # High compression cannot evade a tiny envelope cap even with no FCS.
        bomb = raw_wire(b"x" * 2_000_000, size=1, content_size=False)
        with self.assertRaisesRegex(Arc3Error, "output exceeds"):
            decode_attempt(bomb)

        # Legal block layouts must finish checksum verification at exactly N,
        # even when the last block produces no output. Assemble raw blocks
        # independently; the checksum depends only on the unchanged raw bytes.
        checksum = pyzstd.compress(raw, {pyzstd.CParameter.checksumFlag: 1})[-4:]
        frame_header = bytes.fromhex("28b52ffda4") + len(raw).to_bytes(4, "little")
        blocks = (len(raw) << 3).to_bytes(3, "little") + raw + bytes.fromhex("000000010000")
        envelope = b"ARC3" + struct.pack("<HHQ", 1, 0, len(raw))
        self.assertEqual(decode_attempt(envelope + frame_header + blocks + checksum), baseline())

    def test_lower_cap_equality_and_cap_configuration(self):
        valid = wire(baseline())  # Single segment: window == declared N.
        size = struct.unpack_from("<Q", valid, 8)[0]
        self.assertEqual(decode_attempt(valid, max_uncompressed_bytes=size), baseline())
        with self.assertRaises(Arc3ResourceLimitError):
            decode_attempt(valid, max_uncompressed_bytes=size - 1)
        for cap in (0, -1, True, 1.0, MAX_UNCOMPRESSED_BYTES + 1):
            with self.subTest(cap=cap), self.assertRaises(Arc3Error):
                decode_attempt(valid, max_uncompressed_bytes=cap)
        # N equality at the v1 cap is allowed; the intentionally absent frame is
        # rejected later. No 512 MiB fixture/allocation is needed.
        with self.assertRaisesRegex(Arc3Error, "ordinary"):
            decode_attempt(b"ARC3" + struct.pack("<HHQ", 1, 0, MAX_UNCOMPRESSED_BYTES))
        with self.assertRaisesRegex(Arc3ResourceLimitError, "window"):
            decode_attempt(wire(baseline(), content_size=False), max_uncompressed_bytes=size)

    def test_messagepack_profile_and_allocation_vectors_e09_e10_e18(self):
        raw = msgpack.packb(baseline(), use_bin_type=True)
        # B's top-level map is fixmap(5); add a sixth literal key/value without
        # a Python dict collapsing the duplicate before decoder validation.
        duplicate = b"\x86" + raw[1:] + msgpack.packb("summary") + msgpack.packb(baseline()["summary"])
        missing = baseline()
        del missing["termination"]
        cases = [
            ("array root", msgpack.packb([]), "root"),
            ("second value", raw + b"\xc0", "trailing"),
            ("duplicate root key", duplicate, "duplicate"),
            ("duplicate nested key", b"\x81\xa1x\x82\xa1y\x01\xa1y\x02", "duplicate"),
            ("non STR key", b"\x81\x01\x00", "STR"),
            ("array key", b"\x81\x90\x00", "STR"),
            ("missing required field", msgpack.packb(missing, use_bin_type=True), "termination"),
            ("invalid UTF8", b"\x81\xa1x\xa1\xff", "MessagePack"),
            ("huge array count", b"\x81\xa1x\xdd\xff\xff\xff\xff", "count"),
            ("huge map count", b"\x81\xa1x\xdf\xff\xff\xff\xff", "count"),
            ("huge BIN length", b"\x81\xa1x\xc6\xff\xff\xff\xff", "length"),
            ("huge STR length", b"\x81\xa1x\xdb\xff\xff\xff\xff", "length"),
            ("truncated uint64", b"\x81\xa1x\xcf\x01", "truncated"),
            ("container depth65", b"\x81\xa1x" + b"\x91" * 64 + b"\xc0", "nesting"),
            ("timestamp EXT", b"\x81\xa1x\xd6\xff\x00\x00\x00\x00", "extension"),
            ("reserved tag", b"\x81\xa1x\xc1", "invalid tag"),
        ]
        for name, payload, error in cases:
            with self.subTest(case=name), self.assertRaisesRegex(Arc3Error, error):
                decode_attempt(raw_wire(payload))
        extension = None
        for _ in range(63):
            extension = [extension]
        value = baseline()
        value["extension"] = extension
        self.assertEqual(decode_attempt(wire(value)), value)  # Depth 64 boundary.
        value["extension"] = [extension]
        with self.assertRaisesRegex(Arc3Error, "nesting"):
            encode_attempt(value)

    def test_semantic_rejection_vectors_e11_e17_e19_e20(self):
        frame = ("steps", 0, "observation", "frames")
        step = ("steps", 0)
        cases = [
            (step + ("index",), False), (step + ("index",), 0.0),
            (("metadata", "seed"), 9_007_199_254_740_992),
            (frame + ("shape",), [-1, 2, 2]), (frame + ("shape",), [1, 2]),
            (frame + ("data",), b"\x03\x02\x01"), (frame + ("data",), "03020100"),
            (frame + ("dtype",), "f32"), (frame + ("shape",), [0, 536_870_913, 0]),
            (frame + ("shape",), [0, 2, 2]),
            (step + ("prediction",), {"target": "post_action_last_frame",
                                      "latent": {"dtype": "u8", "shape": [1] * 9, "data": b"\x00"}}),
            (step + ("prediction",), {"target": "post_action_last_frame",
                                      "latent": {"dtype": "f32", "shape": [536_870_912], "data": b""}}),
            (step + ("index",), 1), (step + ("action", "id"), 0), (step + ("action", "id"), 7),
            (step + ("action",), {"id": 6, "data": {"x": 64, "y": 0}}),
            (step + ("prediction",), None),
            (step + ("timing",), {"decision_ms": -1}),
            (step + ("prediction",), {"target": "post_action_last_frame",
                                      "latent": {"dtype": "f16", "shape": [1], "data": bytes.fromhex("007c")}}),
            (("termination", "reason"), "done"),
            (("termination", "reset_action"), {"id": 0, "data": {}}),
            (("summary", "real_actions"), 0), (("summary", "real_actions"), 3),
            (("termination", "final_state"), "WIN"),
            (("metadata", "source_split"), "held_out"),
            (("metadata", "started_at"), "2026-10-09T12:00:00"),
            (("metadata", "config"), {"nested": [b"BIN is not JSON"]}),
            (("initial_observation", "state"), "GAME_OVER"),
            (step + ("prediction",), {"target": "post_action_sequence",
                                      "frames": {"dtype": "u8", "shape": [1, 1, 1], "data": b"\x07"},
                                      "errors": {"mae": 0}}),
        ]
        invalid_decision = decision()
        invalid_decision["candidates"][0]["score"] = 1.01
        cases.append((step + ("decision",), invalid_decision))
        for path, replacement in cases:
            value = baseline()
            change(value, path, replacement)
            with self.subTest(path=path, value=replacement):
                with self.assertRaises(Arc3Error):
                    decode_attempt(wire(value))
                with self.assertRaises(Arc3Error):
                    encode_attempt(value)

    def test_every_required_field_is_checked(self):
        original = baseline()
        locations = [(), ("metadata",), ("initial_observation",), ("initial_observation", "frames"),
                     ("steps", 0), ("steps", 0, "action"), ("steps", 0, "observation"),
                     ("termination",), ("summary",)]
        for location in locations:
            target = original
            for key in location:
                target = target[key]
            for field in target:
                for null in (False, True):
                    value = baseline()
                    parent = value
                    for key in location:
                        parent = parent[key]
                    if null:
                        parent[field] = None
                    else:
                        del parent[field]
                    with self.subTest(location=location, field=field, null=null), self.assertRaises(Arc3Error):
                        decode_attempt(wire(value))

        value = baseline()
        value["steps"][0].update(decision=decision(), prediction=prediction())
        for location, fields in (
            (("steps", 0, "decision"), ("score_type", "candidates")),
            (("steps", 0, "decision", "candidates", 0), ("action", "score")),
            (("steps", 0, "prediction"), ("target",)),
            (("steps", 0, "prediction", "latent"), ("dtype", "shape", "data")),
        ):
            for field in fields:
                malformed = deepcopy(value)
                parent = malformed
                for key in location:
                    parent = parent[key]
                del parent[field]
                with self.subTest(location=location, field=field), self.assertRaises(Arc3Error):
                    decode_attempt(wire(malformed))

    def test_additional_optional_numeric_timestamp_and_termination_guards(self):
        cases = [
            (("metadata", "model_id"), None), (("metadata", "checkpoint_id"), ""),
            (("metadata", "started_at"), "2026-02-29T12:00:00Z"),
            (("metadata", "started_at"), "0000-01-01T00:00:00Z"),
            (("metadata", "started_at"), "2026-10-09T24:00:00Z"),
            (("metadata", "started_at"), "2026-10-09T12:00:60Z"),
            (("metadata", "started_at"), "2026-10-09T12:00:00+24:00"),
            (("metadata", "started_at"), "2026-10-09T12:00:00.1234567890Z"),
            (("steps", 0, "action", "data"), {"extra": True}),
            (("steps", 0, "action"), {"id": 6, "data": {"x": 0}}),
            (("steps", 0, "decision"), {"score_type": "value", "candidates": []}),
            (("steps", 0, "prediction"), {"target": "post_action_last_frame"}),
            (("steps", 0, "prediction"), {"target": "future", "uncertainty": 0}),
            (("steps", 0, "prediction"), {"target": "post_action_last_frame", "uncertainty": None}),
            (("steps", 0, "prediction"), {"target": "post_action_last_frame", "uncertainty": 0, "errors": {"mse": 1}}),
            (("steps", 0, "learning"), {"updates": 1.0}),
            (("steps", 0, "learning"), {"replay_size": -1}),
            (("steps", 0, "learning"), {"loss": float("nan")}),
            (("steps", 0, "learning"), None),
            (("steps", 0, "timing"), None),
            (("steps", 0, "timing"), {"future_timing": -1}),
            (("steps", 0, "notes"), [None]),
            (("summary", "levels_completed"), 1), (("summary", "win_levels"), 3),
            (("summary", "wall_seconds"), -0.1), (("summary", "wall_seconds"), float("inf")),
            (("termination", "reason"), "win"), (("termination", "detail"), None),
            (("extension",), float("nan")), (("extension",), 2**63),
        ]
        for path, replacement in cases:
            value = baseline()
            change(value, path, replacement)
            with self.subTest(path=path, replacement=replacement), self.assertRaises(Arc3Error):
                decode_attempt(wire(value))
        for started_at in ("0001-01-01T00:00:00Z", "2000-02-29T23:59:59.123456789+23:59",
                           "9999-12-31T23:59:59-00:00"):
            value = baseline()
            value["metadata"]["started_at"] = started_at
            self.assertEqual(decode_attempt(encode_attempt(value)), value)
        for bad_reset in ({"id": False, "data": {}}, {"id": 0, "data": {"x": 0}}):
            value = baseline()
            value["termination"].update(reason="reset", reset_action=bad_reset)
            value["summary"]["real_actions"] = 2
            with self.assertRaises(Arc3Error):
                decode_attempt(wire(value))
        value = baseline()
        value["steps"][0]["observation"]["state"] = "WIN"
        value["termination"]["final_state"] = "WIN"
        with self.assertRaisesRegex(Arc3Error, "corresponding reason"):
            decode_attempt(wire(value))

        for state in ("WIN", "GAME_OVER", "NOT_PLAYED"):
            value = baseline()
            value["steps"][0]["observation"]["state"] = state
            following = deepcopy(value["steps"][0])
            following["index"] = 1
            value["steps"].append(following)
            value["summary"]["real_actions"] = 2
            with self.subTest(state=state), self.assertRaisesRegex(Arc3Error, "ordinary step after"):
                decode_attempt(wire(value))

        value = baseline()
        value["public_note"] = "\ud800"
        with self.assertRaisesRegex(Arc3Error, "UTF-8"):
            encode_attempt(value)

    def test_float_tensor_finiteness_and_unaligned_little_endian_bytes(self):
        for dtype, hex_bytes in (("f16", "007c"), ("f16", "007e"),
                                 ("f32", "0000807f"), ("f32", "0000c07f")):
            value = baseline()
            value["steps"][0]["prediction"] = {
                "target": "post_action_last_frame", "latent": {
                    "dtype": dtype, "shape": [1], "data": bytes.fromhex(hex_bytes),
                },
            }
            with self.subTest(dtype=dtype, data=hex_bytes), self.assertRaisesRegex(Arc3Error, "nonfinite"):
                decode_attempt(wire(value))
        value["steps"][0]["prediction"]["latent"]["data"] = bytes.fromhex("0000803f")
        value["public_note"] = "π"
        self.assertEqual(struct.unpack("<f", decode_attempt(wire(value))["steps"][0]["prediction"]["latent"]["data"]), (1.0,))

    def test_atomic_storage_read_validation_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "attempt.arc3"
            write_attempt(target, baseline())
            before = target.read_bytes()
            self.assertEqual(read_attempt(target), baseline())
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(main([str(target)]), 0)
            self.assertEqual(json.loads(stdout.getvalue()), {
                "valid": True, "game_id": "fixture-v1", "attempt_index": 0, "steps": 1, "termination": "action_budget",
            })
            invalid = baseline()
            invalid["summary"]["real_actions"] = 99
            with self.assertRaises(Arc3Error):
                write_attempt(target, invalid)
            self.assertEqual(target.read_bytes(), before)
            for stage in ("fsync", "replace"):
                for exists in (True, False):
                    path = target if exists else Path(directory) / "failed.arc3"
                    with self.subTest(stage=stage, exists=exists):
                        with patch(f"arc3.traces.codec.os.{stage}", side_effect=OSError("injected failure")):
                            with self.assertRaisesRegex(OSError, "injected failure"):
                                write_attempt(path, baseline())
                        if exists:
                            self.assertEqual(path.read_bytes(), before)
                        else:
                            self.assertFalse(path.exists())
                        self.assertEqual(list(Path(directory).iterdir()), [target])
            replacement = baseline()
            replacement["metadata"]["attempt_index"] = 1
            write_attempt(target, replacement)
            self.assertEqual(read_attempt(target), replacement)
            target.write_bytes(before[:-1])
            with self.assertRaises(Arc3Error):
                read_attempt(target)
            with redirect_stderr(stderr):
                self.assertEqual(main([str(target)]), 1)
            self.assertIn("Invalid .arc3", stderr.getvalue())
            target.write_bytes(before)
            with self.assertRaises(Arc3ResourceLimitError):
                read_attempt(target, max_uncompressed_bytes=1)
            # Convenience file reader's additional, documented input-memory cap.
            target.write_bytes(before + b"\x00" * 1_049_000)
            with self.assertRaisesRegex(Arc3ResourceLimitError, "compressed file"):
                read_attempt(target, max_uncompressed_bytes=600)


if __name__ == "__main__":
    unittest.main()
