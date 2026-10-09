"""Recorder acceptance through scripted public SDK frames, not game rules."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from arcengine import FrameData, GameAction, GameState

from arc3.core.policy import Action, RandomPolicy
from arc3.core.runtime import run_game
from arc3.envs.sdk import SDKEnvironment, SDK_VERSION
from arc3.evaluation import run as evaluation
from arc3.evaluation.recorder import EvaluationRecorder
from arc3.traces import read_attempt
from test_smoke import ScriptedArcade


def frame(*, state=GameState.NOT_FINISHED, pixels=None, levels=0, actions=(1, 6), game="fixture-v1"):
    return FrameData(
        game_id=game, frame=pixels if pixels is not None else [[[0, 1], [2, 3]]],
        state=state, levels_completed=levels, win_levels=3, available_actions=list(actions),
    )


class ScriptedPolicy:
    def __init__(self, *actions):
        self.actions = iter(actions)

    def choose_action(self, observation):
        return next(self.actions)


class RecorderTests(unittest.TestCase):
    def run_recorded(self, directory, frames, policy, budget, **kwargs):
        arcade = ScriptedArcade({"fixture": frames})
        env = SDKEnvironment(arcade, "fixture", 42)
        recorder = EvaluationRecorder(directory, run_id="run-1", seed=42, sdk_version=SDK_VERSION)
        metrics = run_game(env, policy, budget, attempt_observer=recorder, **kwargs)
        attempts = sorted((read_attempt(path) for path in Path(directory).glob("*.arc3")),
                          key=lambda attempt: attempt["metadata"]["attempt_index"])
        return metrics, attempts, arcade

    def test_startup_reset_multiframe_click_and_level_change(self):
        # Expected bytes and counts come directly from these public responses.
        initial = frame(pixels=[[[1, 2, 3]], [[4, 5, 6]]], actions=(6, 1))
        following = frame(pixels=[[[7, 8, 9]], [[7, 8, 9]], [[10, 11, 12]]], levels=1, actions=(1,))
        with tempfile.TemporaryDirectory() as directory:
            metrics, attempts, arcade = self.run_recorded(
                directory, [frame(state=GameState.NOT_PLAYED, pixels=[]), initial,
                            following, frame(state=GameState.WIN, levels=3)],
                ScriptedPolicy(Action(0, {}), Action(6, {"x": 63, "y": 0}), Action(1, {})), 3,
            )
            self.assertEqual(len(attempts), 1)
            attempt = attempts[0]
            self.assertEqual(attempt["initial_observation"]["frames"],
                             {"dtype": "u8", "shape": [2, 1, 3], "data": bytes([1, 2, 3, 4, 5, 6])})
            self.assertEqual(attempt["initial_observation"]["available_actions"], [6, 1])
            self.assertEqual(attempt["steps"][0]["observation"]["frames"],
                             {"dtype": "u8", "shape": [3, 1, 3], "data": bytes([7, 8, 9, 7, 8, 9, 10, 11, 12])})
            self.assertEqual(attempt["steps"][0]["action"], {"id": 6, "data": {"x": 63, "y": 0}})
            self.assertEqual(attempt["steps"][0]["observation"]["levels_completed"], 1)
            self.assertEqual([step["index"] for step in attempt["steps"]], [0, 1])
            self.assertEqual(attempt["termination"], {"reason": "win", "final_state": "WIN"})
            self.assertEqual((attempt["summary"]["real_actions"], metrics["actions"]), (2, 3))
            self.assertEqual([action for action, _ in arcade.actions], [GameAction.RESET, GameAction.ACTION6, GameAction.ACTION1])
            self.assertEqual(attempt["metadata"]["game_id"], "fixture-v1")
            self.assertEqual(attempt["metadata"]["source_split"], "evaluation")
            self.assertEqual((attempt["metadata"]["seed"], attempt["metadata"]["sdk_version"]), (42, "0.9.9"))
            self.assertTrue(attempt["metadata"]["started_at"].endswith("Z"))
            self.assertEqual(attempt["metadata"]["attempt_index"], 0)
            self.assertGreaterEqual(attempt["summary"]["wall_seconds"], 0)

    def test_game_over_is_final_before_baseline_reset(self):
        for budget, reasons in [(1, ["game_over"]), (3, ["game_over", "action_budget"])]:
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as directory:
                metrics, attempts, arcade = self.run_recorded(
                    directory, [frame(actions=(1,)), frame(state=GameState.GAME_OVER),
                                frame(levels=1, actions=(1,)), frame(levels=1)],
                    RandomPolicy(42, "fixture-v1"), budget,
                )
                self.assertEqual([attempt["termination"]["reason"] for attempt in attempts], reasons)
                self.assertEqual(attempts[0]["termination"]["final_state"], "GAME_OVER")
                self.assertNotIn("reset_action", attempts[0]["termination"])
                self.assertEqual(attempts[0]["summary"]["real_actions"], 1)
                self.assertFalse(metrics["won"])
                self.assertEqual(metrics["stop_reason"], "action_budget")
                self.assertEqual(len(arcade.actions), budget)
                if budget == 3:
                    self.assertEqual(attempts[1]["initial_observation"]["levels_completed"], 1)
                    self.assertEqual(attempts[1]["summary"]["real_actions"], 1)
                    self.assertEqual([attempt["metadata"]["attempt_index"] for attempt in attempts], [0, 1])
                    for key in ("game_id", "run_id", "session_id"):
                        self.assertEqual(attempts[0]["metadata"][key], attempts[1]["metadata"][key])

    def test_explicit_reset_closes_old_attempt_and_starts_post_reset_file(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, attempts, _ = self.run_recorded(
                directory, [frame(), frame(levels=1), frame(pixels=[[[9]]], levels=1), frame(levels=2)],
                ScriptedPolicy(Action(1, {}), Action(0, {}), Action(1, {})), 3,
            )
            first, second = attempts
            self.assertEqual(first["termination"], {
                "reason": "reset", "final_state": "NOT_FINISHED", "reset_action": {"id": 0, "data": {}},
            })
            self.assertEqual([step["action"]["id"] for step in first["steps"]], [1])
            self.assertEqual(first["summary"]["real_actions"], 2)
            self.assertEqual(first["summary"]["levels_completed"], 1)
            self.assertEqual(second["initial_observation"]["frames"]["data"], b"\x09")
            self.assertEqual(second["initial_observation"]["levels_completed"], 1)
            self.assertEqual(second["metadata"]["attempt_index"], 1)
            self.assertEqual(second["summary"]["real_actions"], 1)
            self.assertEqual(metrics["actions"], 3)

    def test_reset_at_budget_still_starts_zero_step_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            _, attempts, _ = self.run_recorded(directory, [frame(), frame()], ScriptedPolicy(Action(0, {})), 1)
            self.assertEqual([attempt["termination"]["reason"] for attempt in attempts], ["reset", "action_budget"])
            self.assertEqual([attempt["summary"]["real_actions"] for attempt in attempts], [1, 0])

    def test_failure_timeout_and_interruption_keep_only_verified_steps(self):
        for error, reason in [(RuntimeError("private detail"), "error"), (TimeoutError("private detail"), "timeout"),
                              (KeyboardInterrupt(), "interrupted")]:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                arcade = ScriptedArcade({"fixture": [frame(), frame(levels=1)]})
                env = SDKEnvironment(arcade, "fixture", 42)
                original_step = env.step

                def step(action):
                    if len(arcade.actions) == 1:
                        raise error
                    return original_step(action)

                env.step = step
                recorder = EvaluationRecorder(directory, run_id="run-1", seed=42, sdk_version=SDK_VERSION)
                with self.assertRaises(type(error)):
                    run_game(env, ScriptedPolicy(Action(1, {}), Action(1, {})), 3, attempt_observer=recorder)
                attempt = read_attempt(next(Path(directory).glob("*.arc3")))
                self.assertEqual(len(attempt["steps"]), 1)
                self.assertEqual(attempt["summary"]["real_actions"], 1)
                self.assertEqual(attempt["termination"]["reason"], reason)
                self.assertEqual(attempt["termination"]["final_state"], "NOT_FINISHED")
                self.assertNotIn("private detail", attempt["termination"]["detail"])
                self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_missing_sdk_response_does_not_invent_execution_or_next_attempt(self):
        for action in (Action(1, {}), Action(0, {})):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, "no public observation"):
                    self.run_recorded(directory, [frame(), None], ScriptedPolicy(action), 1)
                paths = list(Path(directory).glob("*.arc3"))
                self.assertEqual(len(paths), 1)
                attempt = read_attempt(paths[0])
                self.assertEqual(attempt["steps"], [])
                self.assertEqual(attempt["summary"]["real_actions"], 0)
                self.assertEqual(attempt["termination"]["reason"], "error")
                self.assertNotIn("reset_action", attempt["termination"])

    def test_bootstrap_failure_has_no_empty_initialization_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "no public observation"):
                self.run_recorded(directory, [frame(state=GameState.NOT_PLAYED, pixels=[]), None],
                                  ScriptedPolicy(Action(0, {})), 1)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_unrepresentable_response_retains_previous_frames_and_counts_confirmed_action(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "rectangular SDK frames"):
                self.run_recorded(directory, [frame(), frame(pixels=[[[1, 2], [3]]])],
                                  ScriptedPolicy(Action(1, {})), 1)
            attempt = read_attempt(next(Path(directory).glob("*.arc3")))
            self.assertEqual(attempt["steps"], [])
            self.assertEqual(attempt["initial_observation"]["frames"]["data"], bytes([0, 1, 2, 3]))
            self.assertEqual(attempt["summary"]["real_actions"], 1)
            self.assertEqual(attempt["termination"]["reason"], "error")

    def test_cli_failure_closes_scorecard_and_leaves_trace_without_success_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.json"
            split.write_text(json.dumps({"training_games": ["ls20"], "evaluation_games": ["ft09"],
                                         "training_level_caps": {"ls20": 1}}))
            output, traces = root / "metrics.json", root / "traces"
            arcade = ScriptedArcade({"ft09": [frame(actions=(1,)), None]})
            arguments = ["evaluate", "--split", str(split), "--output", str(output), "--trace-dir", str(traces)]
            with patch.object(sys, "argv", arguments), patch.object(evaluation, "open_arcade", return_value=arcade), self.assertRaisesRegex(RuntimeError, "no public observation"):
                evaluation.main()
            self.assertTrue(arcade.closed)
            self.assertFalse(output.exists())
            attempt = read_attempt(next(traces.glob("*.arc3")))
            self.assertEqual(attempt["termination"]["reason"], "error")

    def test_elapsed_timeout_and_empty_public_sequence(self):
        with tempfile.TemporaryDirectory() as directory, patch("arc3.core.runtime.perf_counter", side_effect=[0, 2, 2]):
            metrics, attempts, arcade = self.run_recorded(
                directory, [frame(pixels=[])], ScriptedPolicy(), 3, timeout_seconds=1,
            )
            self.assertEqual(metrics["stop_reason"], "timeout")
            self.assertEqual(arcade.actions, [])
            self.assertEqual(attempts[0]["initial_observation"]["frames"], {"dtype": "u8", "shape": [0, 0, 0], "data": b""})
            self.assertEqual(attempts[0]["termination"]["reason"], "timeout")
            self.assertEqual(attempts[0]["summary"]["real_actions"], 0)

    def test_diagnostics_are_copied_before_execution_and_errors_use_real_pixels(self):
        cases = [
            ("u8", bytes([1, 1, 1, 1]), "post_action_last_frame", [2, 2], {"mae": 2.5, "mse": 9.0}),
            ("f16", struct.pack("<4e", 1, 1, 1, 1), "post_action_last_frame", [2, 2], {"mae": 2.5, "mse": 9.0}),
            ("f32", struct.pack("<4f", 1, 1, 1, 1), "post_action_last_frame", [2, 2], {"mae": 2.5, "mse": 9.0}),
            ("u8", bytes([1] * 8), "post_action_sequence", [2, 2, 2], {"mae": 5.25, "mse": 36.5}),
        ]
        for dtype, data, target, shape, errors in cases:
            with self.subTest(dtype=dtype, target=target), tempfile.TemporaryDirectory() as directory:
                arcade = ScriptedArcade({"fixture": [frame(actions=(1,)),
                    frame(pixels=[[[9, 9], [9, 9]], [[0, 2], [4, 6]]])]})
                env = SDKEnvironment(arcade, "fixture", 42)
                diagnostics = {
                    "prediction": {"target": target, "frames": {"dtype": dtype, "shape": shape, "data": data}},
                    "decision": {"score_type": "value", "candidates": [{"action": {"id": 1, "data": {}}, "score": 0.5}]},
                }
                test = self

                class Policy:
                    def choose_action(self, observation):
                        return Action(1, {})

                    def trace_diagnostics(self, observation, action):
                        test.assertEqual(len(arcade.actions), 0)
                        test.assertEqual(observation.frame, (((0, 1), (2, 3)),))
                        return diagnostics

                original_step = env.step

                def step(action):
                    diagnostics["prediction"]["frames"]["data"] = bytes([99] * 4)
                    diagnostics["decision"]["candidates"][0]["score"] = 999
                    return original_step(action)

                env.step = step
                recorder = EvaluationRecorder(directory, run_id="run-1", seed=42, sdk_version=SDK_VERSION)
                run_game(env, Policy(), 1, attempt_observer=recorder)
                recorded = read_attempt(next(Path(directory).glob("*.arc3")))["steps"][0]
                self.assertEqual(recorded["prediction"]["frames"]["data"], data)
                self.assertEqual(recorded["decision"]["candidates"][0]["score"], 0.5)
                self.assertEqual(recorded["prediction"]["errors"], errors)

    def test_prediction_mismatch_has_no_pixel_errors(self):
        class Policy:
            def choose_action(self, observation):
                return Action(1, {})

            def trace_diagnostics(self, observation, action):
                return {"prediction": {"target": "post_action_sequence", "frames": {"dtype": "u8", "shape": [1, 1, 1], "data": b"\x05"}}}

        with tempfile.TemporaryDirectory() as directory:
            _, attempts, _ = self.run_recorded(directory, [frame(), frame()], Policy(), 1)
            self.assertNotIn("errors", attempts[0]["steps"][0]["prediction"])

    def test_atomic_publication_never_overwrites_and_cleans_failed_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EvaluationRecorder(directory, run_id="run-1", seed=42, sdk_version=SDK_VERSION)
            target = Path(directory) / f"{recorder.session_id}-000000.arc3"
            target.write_bytes(b"keep-existing-file")
            env = SDKEnvironment(ScriptedArcade({"fixture": [frame(), frame()]}), "fixture", 42)
            run_game(env, ScriptedPolicy(Action(1, {})), 1, attempt_observer=recorder)
            self.assertEqual(target.read_bytes(), b"keep-existing-file")
            new_file = Path(directory) / f"{recorder.session_id}-000000-1.arc3"
            self.assertEqual(read_attempt(new_file)["termination"]["reason"], "action_budget")
            self.assertEqual(set(Path(directory).iterdir()), {target, new_file})
        with tempfile.TemporaryDirectory() as directory, patch("arc3.evaluation.recorder.os.link", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.run_recorded(directory, [frame(), frame()], ScriptedPolicy(Action(1, {})), 1)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cli_opt_in_preserves_metrics_actions_and_training_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.json"
            split.write_text(json.dumps({"training_games": ["ls20"], "evaluation_games": ["ft09", "vc33"], "training_level_caps": {"ls20": 1}}))
            training = root / "train.jsonl"
            training.write_bytes(b"existing-training-data\n")
            results, executed = [], []
            for tracing in (False, True):
                output = root / f"metrics-{tracing}.json"
                arguments = ["evaluate", "--split", str(split), "--output", str(output), "--max-actions", "1", "--seed", "42"]
                if tracing:
                    arguments += ["--trace-dir", str(root / "traces")]
                arcade = ScriptedArcade({"ft09": [frame(actions=(1,)), frame()],
                                         "vc33": [frame(game="other-v2", actions=(1,)), frame(game="other-v2")]})
                stdout = io.StringIO()
                with patch.object(sys, "argv", arguments), patch.object(evaluation, "open_arcade", return_value=arcade), patch("arc3.core.runtime.perf_counter", return_value=10), redirect_stdout(stdout):
                    evaluation.main()
                results.append(json.loads(output.read_text()))
                self.assertEqual(json.loads(stdout.getvalue()), results[-1])
                self.assertEqual(arcade.makes, [("ft09", 42, False), ("vc33", 42, False)])
                self.assertTrue(arcade.closed)
                executed.append(arcade.actions)
                if not tracing:
                    self.assertFalse((root / "traces").exists())
            expected_game = {"game_id": "fixture-v1", "actions": 1, "recorded_transitions": 0,
                             "levels_completed": 0, "win_levels": 3, "won": False, "state": "NOT_FINISHED",
                             "stop_reason": "action_budget", "actions_per_completed_level": None, "wall_seconds": 0}
            self.assertEqual(results[0], {"lifecycle": "evaluation", "sdk_version": "0.9.9", "seed": 42,
                                         "max_actions": 1, "games": [expected_game, {**expected_game, "game_id": "other-v2"}]})
            self.assertEqual(results[0], results[1])
            self.assertEqual(executed[0], executed[1])
            self.assertEqual(training.read_bytes(), b"existing-training-data\n")
            attempts = [read_attempt(path) for path in (root / "traces").glob("*.arc3")]
            self.assertEqual(len(attempts), 2)
            self.assertEqual(len({attempt["metadata"]["run_id"] for attempt in attempts}), 1)
            self.assertEqual(len({attempt["metadata"]["session_id"] for attempt in attempts}), 2)
            for attempt in attempts:
                self.assertEqual(set(attempt["steps"][0]), {"index", "action", "observation"})
                self.assertNotIn("model_id", attempt["metadata"])
                self.assertNotIn("checkpoint_id", attempt["metadata"])
                self.assertEqual(attempt["termination"]["reason"], "action_budget")

    def test_disabled_trace_does_not_load_codec_or_call_diagnostics(self):
        class Policy:
            def choose_action(self, observation):
                return Action(1, {})

            def trace_diagnostics(self, observation, action):
                raise AssertionError("Disabled tracing must not collect diagnostics")

        env = SDKEnvironment(ScriptedArcade({"fixture": [frame(), frame()]}), "fixture", 42)
        self.assertEqual(run_game(env, Policy(), 1)["actions"], 1)
        result = subprocess.run([sys.executable, "-c", "import sys; import arc3.evaluation.run; assert 'arc3.traces' not in sys.modules; assert 'arc3.evaluation.recorder' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_trace_cannot_share_training_collection_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EvaluationRecorder(directory, run_id="run-1", seed=42, sdk_version=SDK_VERSION)
            env = SDKEnvironment(ScriptedArcade({"fixture": [frame()]}), "fixture", 42)
            for options in ({"training_level_cap": 1}, {"on_transition": lambda transition: None}):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, "evaluation-only"):
                    run_game(env, RandomPolicy(42, "fixture-v1"), 1, attempt_observer=recorder, **options)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
