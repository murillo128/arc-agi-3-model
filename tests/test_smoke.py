"""Small CPU smoke suite: actual SDK types with scripted public responses.

The transport double has no game rules; these tests do not claim live-game or
Kaggle gateway success. They require the installed package dependencies.
"""

import importlib.util
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from arc_agi import Arcade
from arcengine import FrameData, GameAction, GameState

from arc3.core.policy import Observation, RandomPolicy
from arc3.core.runtime import run_game
from arc3.envs.sdk import SDKEnvironment
from arc3.envs.splits import load_split
from arc3.evaluation import run as evaluation
from arc3.kaggle import build
from arc3.training import collect


def public_frame(*, levels=0, total=3, state=GameState.NOT_FINISHED, pixel=1, game="ls20"):
    return FrameData(game_id=game, frame=[[[pixel, pixel], [pixel, pixel]]],
                     state=state, levels_completed=levels, win_levels=total,
                     available_actions=[6])


class ScriptedArcade:
    def __init__(self, frames):
        self.frames = frames
        self.makes = []
        self.actions = []
        self.closed = False

    def make(self, game_id, *, seed, save_recording):
        self.makes.append((game_id, seed, save_recording))
        initial, *following = self.frames[game_id]
        responses = iter(following)

        def step(action, *, data):
            self.actions.append((action, data))
            return next(responses)

        return SimpleNamespace(observation_space=initial, step=step)

    def close_scorecard(self):
        self.closed = True


class SmokeTests(unittest.TestCase):
    def test_pinned_sdk_and_real_transition_budget(self):
        self.assertEqual(version("arc-agi"), "0.9.9")
        self.assertTrue({"game_id", "seed", "save_recording"} <= set(inspect.signature(Arcade.make).parameters))
        arcade = ScriptedArcade({"ls20": [public_frame(state=GameState.NOT_PLAYED),
                                           public_frame(state=GameState.GAME_OVER), public_frame()]})
        transitions = []
        result = run_game(SDKEnvironment(arcade, "ls20", 7), RandomPolicy(7, "ls20"),
                          2, on_transition=transitions.append)
        self.assertEqual(arcade.makes, [("ls20", 7, False)])
        self.assertEqual([action for action, _ in arcade.actions], [GameAction.RESET, GameAction.RESET])
        self.assertEqual((result["actions"], result["stop_reason"]), (2, "action_budget"))
        self.assertEqual([t.source for t in transitions], ["real", "real"])
        self.assertEqual(transitions[0].observation.state, "NOT_PLAYED")
        self.assertEqual(transitions[0].next_observation.state, "GAME_OVER")

    def test_win_and_missing_observation_stop_without_fabricating_transitions(self):
        for response, expected_count in [(public_frame(state=GameState.WIN, levels=3), 1), (None, 0)]:
            with self.subTest(response=response):
                arcade = ScriptedArcade({"ls20": [public_frame(), response]})
                transitions = []
                env = SDKEnvironment(arcade, "ls20", 0)
                if response is None:
                    with self.assertRaisesRegex(RuntimeError, "no public observation"):
                        run_game(env, RandomPolicy(0, "ls20"), 80, on_transition=transitions.append)
                else:
                    result = run_game(env, RandomPolicy(0, "ls20"), 80, on_transition=transitions.append)
                    self.assertEqual((result["won"], result["actions"], result["stop_reason"]), (True, 1, "win"))
                self.assertEqual(len(transitions), expected_count)
                self.assertEqual(len(arcade.actions), 1)

    def test_split_rejects_overlap_version_aliases_and_missing_caps(self):
        cases = [
            (["ls20"], ["ft09"], {"ls20": 1}, True),
            (["ls20"], ["ls20"], {"ls20": 1}, False),
            (["ls20"], ["ls20-9607627b"], {"ls20": 1}, False),
            (["ls20-9607627b"], ["ls20"], {"ls20-9607627b": 1}, False),
            (["ls20-abc"], ["ls20-def"], {"ls20-abc": 1}, False),
            (["ls20"], ["ft09"], {"ls20": 0}, False),
            (["ls20"], ["ft09"], {"ls20": True}, False),
            (["ls20"], ["ft09"], {}, False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            for training, evaluation, caps, valid in cases:
                config = {"training_games": training, "evaluation_games": evaluation,
                          "training_level_caps": caps}
                with self.subTest(config=config):
                    path.write_text(json.dumps(config))
                    if valid:
                        self.assertEqual(load_split(path).training, {"ls20": 1})
                    else:
                        with self.assertRaises(ValueError):
                            load_split(path)

    def test_reserved_level_boundary_discards_crossing_frame(self):
        # Cap 1 stops at level 2; an overlarge cap still reserves the last level.
        for cap, start in [(1, 0), (99, 1)]:
            with self.subTest(cap=cap):
                arcade = ScriptedArcade({"ls20": [public_frame(levels=start),
                    public_frame(levels=start, pixel=2), public_frame(levels=start + 1, pixel=9)]})
                transitions = []
                result = run_game(SDKEnvironment(arcade, "ls20", 0), RandomPolicy(0, "ls20"),
                                  80, on_transition=transitions.append, training_level_cap=cap)
                self.assertEqual((result["actions"], len(transitions)), (2, 1))
                self.assertEqual(transitions[0].next_observation.frame, (((2, 2), (2, 2)),))
                self.assertEqual(result["stop_reason"], "training_level_boundary")
        for initial in [public_frame(total=1), public_frame(total=0)]:
            arcade = ScriptedArcade({"ls20": [initial]})
            with self.assertRaisesRegex(ValueError, "two known public levels"):
                run_game(SDKEnvironment(arcade, "ls20", 0), RandomPolicy(0, "ls20"),
                         80, training_level_cap=1)
            self.assertEqual(arcade.actions, [])

    def test_lifecycles_keep_training_records_and_evaluation_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.json"
            split.write_text(json.dumps({"training_games": ["ls20"], "evaluation_games": ["ft09"],
                                         "training_level_caps": {"ls20": 1}}))
            recording = root / "train.jsonl"
            train_arcade = ScriptedArcade({"ls20": [public_frame(), public_frame(pixel=2),
                                                    public_frame(levels=1, pixel=9)]})
            arguments = ["collect", "--split", str(split), "--output", str(recording)]
            with patch.object(sys, "argv", arguments), patch.object(collect, "open_arcade", return_value=train_arcade), redirect_stdout(io.StringIO()):
                collect.main()
            records = [json.loads(line) for line in recording.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["next_observation"]["frame"], [[[2, 2], [2, 2]]])
            self.assertEqual((records[0]["source"], records[0]["seed"]), ("real", 0))
            self.assertEqual(train_arcade.makes, [("ls20", 0, False)])
            self.assertTrue(train_arcade.closed)
            contents = recording.read_bytes()
            metrics = root / "metrics.json"
            eval_arcade = ScriptedArcade({"ft09": [public_frame(game="ft09"),
                public_frame(game="ft09", state=GameState.WIN, levels=3)]})
            arguments = ["evaluate", "--split", str(split), "--output", str(metrics)]
            with patch.object(sys, "argv", arguments), patch.object(evaluation, "open_arcade", return_value=eval_arcade), redirect_stdout(io.StringIO()):
                evaluation.main()
            report = json.loads(metrics.read_text())
            self.assertEqual(eval_arcade.makes, [("ft09", 0, False)])
            self.assertTrue(eval_arcade.closed)
            self.assertEqual((report["games"][0]["won"], report["games"][0]["recorded_transitions"]), (True, 0))
            self.assertEqual(recording.read_bytes(), contents)
            self.assertEqual(set(root.iterdir()), {split, recording, metrics})

    def test_notebook_embeds_executable_shared_policy_without_local_dependencies(self):
        notebook = build.build_notebook(seed=23, max_actions=4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile(cell["source"], "generated-cell", "exec")
        scope = {}
        exec(notebook["cells"][2]["source"], scope)
        root = scope["source_root"]
        try:
            spec = importlib.util.spec_from_file_location("embedded_policy", root / "arc3/core/policy.py")
            embedded = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {spec.name: embedded}):
                spec.loader.exec_module(embedded)
            observation = Observation.from_public(public_frame())
            local, packaged = RandomPolicy(23, "ls20"), embedded.RandomPolicy(23, "ls20")
            for _ in range(4):
                action = packaged.choose_action(observation)
                expected = local.choose_action(observation)
                self.assertEqual((action.id, action.data), (expected.id, expected.data))
                self.assertEqual(action.id, 6)
                self.assertTrue(0 <= action.data["x"] < 2 and 0 <= action.data["y"] < 2)
            compile(scope["sources"]["my_agent.py"], "embedded-agent", "exec")
            # Isolate the adapter boundary; the real framework/gateway is external.
            class AgentInterface:
                def __init__(self, game_id, arc_env):
                    self.game_id, self.arc_env, self.action_counter = game_id, arc_env, 0

                def _convert_raw_frame_data(self, raw):
                    return raw

                def do_action_request(self, action):
                    # Public transport behavior of the pinned upstream Agent.
                    return self.arc_env.step(action, data=action.action_data.model_dump())

            agent_module = ModuleType("agents.agent")
            agent_module.Agent = AgentInterface
            adapter = {}
            with patch.dict(sys.modules, {"agents": ModuleType("agents"), "agents.agent": agent_module}):
                exec(scope["sources"]["my_agent.py"], adapter)
            sent = []
            transport = SimpleNamespace(step=lambda action, **kwargs: sent.append((action, kwargs)))
            first = adapter["MyAgent"]("ls20", transport)
            second = adapter["MyAgent"]("ft09", transport)
            first_action = first.choose_action([], public_frame())
            # Simulate another thread mutating the SDK enum between choose/send.
            second.choose_action([], public_frame(game="ft09"))
            with patch.object(GameAction.ACTION6, "action_data", GameAction.ACTION6.action_type(x=63, y=63)):
                first.do_action_request(first_action)
            self.assertEqual(sent[0][0], GameAction.ACTION6)
            self.assertTrue(all(0 <= sent[0][1]["data"][axis] < 2 for axis in ("x", "y")))
            first.action_counter = 4
            self.assertTrue(first.is_done([], public_frame()))
            self.assertFalse(notebook["metadata"]["kaggle"]["isInternetEnabled"])
            self.assertFalse(notebook["metadata"]["kaggle"]["isGpuEnabled"])
            self.assertIn("--no-index", notebook["cells"][1]["source"])
            self.assertIn("KAGGLE_IS_COMPETITION_RERUN", notebook["cells"][3]["source"])
        finally:
            shutil.rmtree(root)

    def test_kernel_metadata_requires_explicit_username(self):
        with tempfile.TemporaryDirectory() as directory:
            notebook = Path(directory) / "submission.ipynb"
            arguments = ["build", "--output", str(notebook)]
            with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
                build.main()
            metadata = notebook.with_name("kernel-metadata.json")
            self.assertTrue(notebook.exists())
            self.assertFalse(metadata.exists())
            with patch.object(sys, "argv", [*arguments, "--username", "example-user"]), redirect_stdout(io.StringIO()):
                build.main()
            settings = json.loads(metadata.read_text())
            self.assertEqual(settings["id"], "example-user/arc3-random-baseline")
            self.assertEqual(settings["competition_sources"], ["arc-prize-2026-arc-agi-3"])
            self.assertFalse(settings["enable_gpu"] or settings["enable_internet"])

    def test_notebook_cli_from_installed_package_layout(self):
        # Exercise relocation and the default output without build tools/network.
        # A separate built-wheel check verifies actual distribution contents.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            installation = root / "site-packages"
            shutil.copytree(Path(files("arc3")), installation / "arc3",
                            ignore=shutil.ignore_patterns("__pycache__"))
            working = root / "working"
            working.mkdir()
            process = subprocess.run(
                [sys.executable, "-m", "arc3.kaggle.build", "--accelerator", "cpu"],
                cwd=working, env={**os.environ, "PYTHONPATH": str(installation)},
                capture_output=True, text=True, check=True,
            )
            output = working / "artifacts/submission.ipynb"
            self.assertIn("artifacts/submission.ipynb", process.stdout)
            self.assertEqual(json.loads(output.read_text())["nbformat"], 4)
            self.assertFalse(output.with_name("kernel-metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
