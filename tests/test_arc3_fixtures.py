"""Independent semantic expectations for the stored recorder golden samples."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from arcengine import GameState
from fixtures.generate_arc3 import GAME, PublicResponses, frame, generate
from fixtures.capture_public_sdk import capture
from arc3.traces import read_attempt

FIXTURES = Path(__file__).parent / "fixtures"


class FixtureTests(unittest.TestCase):
    def test_recorder_outputs_are_deterministic_and_match_golden_semantics(self):
        hashes = json.loads((FIXTURES / "arc3-sha256.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            first = generate(Path(directory) / "first")
            second = generate(Path(directory) / "second")
            for path, repeated in zip(first, second):
                golden = FIXTURES / path.name
                self.assertEqual(path.read_bytes(), repeated.read_bytes())
                # Compression/map ordering is not canonical across dependency
                # versions; frozen bytes keep their hash, new output keeps semantics.
                self.assertEqual(read_attempt(path), read_attempt(golden))
                self.assertEqual(hashlib.sha256(golden.read_bytes()).hexdigest(), hashes[path.name])
                self.assertLess(path.stat().st_size, 1024)

    def test_terminal_reset_starts_a_new_attempt_without_duplicate_actions(self):
        first = read_attempt(FIXTURES / "recorder-baseline-0.arc3")
        next_attempt = read_attempt(FIXTURES / "recorder-baseline-1.arc3")
        self.assertEqual(first["initial_observation"]["frames"]["data"], bytes([0, 1, 2, 3]))
        self.assertEqual(first["steps"][0]["action"], {"id": 6, "data": {"x": 1, "y": 0}})
        self.assertEqual(first["steps"][0]["observation"]["frames"], {
            "dtype": "u8", "shape": [2, 2, 2], "data": bytes(range(4, 12)),
        })
        self.assertEqual([s["observation"]["levels_completed"] for s in first["steps"]], [1, 1])
        self.assertEqual(first["termination"], {"reason": "game_over", "final_state": "GAME_OVER"})
        self.assertEqual([first["summary"]["real_actions"], next_attempt["summary"]["real_actions"]], [2, 1])
        self.assertEqual(next_attempt["metadata"]["attempt_index"], 1)
        self.assertEqual(first["metadata"]["session_id"], next_attempt["metadata"]["session_id"])
        self.assertEqual(next_attempt["initial_observation"]["frames"]["data"], bytes([9] * 4))
        self.assertEqual(next_attempt["initial_observation"]["levels_completed"], 0)
        self.assertEqual(next_attempt["termination"]["reason"], "action_budget")
        for attempt in (first, next_attempt):
            self.assertNotIn("model_id", attempt["metadata"])
            for step in attempt["steps"]:
                self.assertTrue({"prediction", "decision", "learning", "timing", "notes"}.isdisjoint(step))

    def test_prediction_is_synthetic_and_real_pixels_stay_separate(self):
        attempt = read_attempt(FIXTURES / "recorder-prediction.arc3")
        step = attempt["steps"][0]
        self.assertEqual(step["prediction"]["frames"]["data"], bytes([0, 1, 2, 3]))
        self.assertEqual(step["observation"]["frames"]["data"], bytes([3, 2, 1, 0]))
        self.assertEqual(step["prediction"]["errors"], {"mae": 2.0, "mse": 5.0})
        self.assertEqual(attempt["summary"]["real_actions"], 1)
        self.assertIn("Synthetic prediction", step["notes"][0])

    def test_public_extraction_synthesizes_pixels_and_excludes_boundary_action(self):
        arcade = PublicResponses([
            frame([[[15] * 3] * 3], actions=(1, 2, 3)),
            frame([[[14] * 3] * 3], actions=(1, 2, 3)),
            frame([[[13] * 3] * 3], levels=1),
        ])
        report, attempt = capture(arcade, GAME, 42, 1)
        self.assertEqual(report["initial_observation"]["source_shape"], [1, 3, 3])
        self.assertEqual(len(report["steps"]), 1)
        self.assertEqual(attempt["initial_observation"]["frames"]["data"], bytes([0, 1, 2, 3]))
        self.assertEqual(attempt["steps"][0]["observation"]["frames"]["data"], bytes([1, 2, 3, 4]))
        self.assertEqual([s["action"]["id"] for s in attempt["steps"]], [1])
        self.assertEqual(attempt["termination"]["reason"], "training_level_boundary")
        self.assertEqual(attempt["summary"]["real_actions"], 2)
        self.assertEqual(attempt["summary"]["levels_completed"], 1)
        self.assertNotIn("prediction", attempt["steps"][0])

    def test_public_extraction_bootstrap_and_reserved_initialization(self):
        # Bootstrap is session usage; its returned frame becomes the initial
        # observation, and the subsequent three actions each have one step.
        arcade = PublicResponses([frame([], state=GameState.NOT_PLAYED),
                                  *[frame([[[15]]], actions=(1, 2, 3)) for _ in range(4)]])
        report, attempt = capture(arcade, GAME, 42, 1)
        self.assertEqual(report["session_actions"], 4)
        self.assertEqual(attempt["summary"]["real_actions"], 3)
        self.assertEqual([s["action"]["id"] for s in attempt["steps"]], [1, 2, 3])
        self.assertEqual(attempt["initial_observation"]["frames"]["data"], bytes([0, 1, 2, 3]))
        with self.assertRaisesRegex(ValueError, "reserved/unknown level"):
            capture(PublicResponses([frame([[[15]]], levels=1)]), GAME, 42, 1)
        with self.assertRaisesRegex(ValueError, "No permitted initial observation"):
            capture(PublicResponses([frame([], state=GameState.NOT_PLAYED),
                                     frame([[[15]]], levels=1)]), GAME, 42, 1)
