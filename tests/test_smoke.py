"""Three cheap, independent checks. No SDK downloads, GPU, or training."""

import json
from enum import Enum, auto
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from arc3.core.policy import RandomPolicy
from arc3.core.runner import run_game
from arc3.core.splits import games_for_split
from arc3.kaggle.build import build_notebook


class Action(Enum):
    RESET = auto()
    ACTION1 = auto()

    def is_complex(self):
        return False


class Engine:
    def __init__(self):
        self.calls = 0
        self.action_space = [Action.ACTION1]
        self.observation_space = SimpleNamespace(state="NOT_PLAYED", levels_completed=0, frame=[[[0]]])

    def step(self, action, *, data):
        self.calls += 1
        if self.calls == 1:
            assert action is Action.RESET
            return SimpleNamespace(state="IN_PROGRESS", levels_completed=0, frame=[[[1]]])
        assert action is Action.ACTION1
        return SimpleNamespace(state="WIN", levels_completed=1, frame=[[[2]]])


class Arcade:
    def __init__(self):
        self.makes = 0
        self.env = Engine()

    def make(self, game_id):
        self.makes += 1
        return self.env


class MinimalChecks(unittest.TestCase):
    def test_game_is_created_once_and_transitions_are_real(self):
        arcade = Arcade()
        transitions = []
        result = run_game(arcade, "demo", RandomPolicy("demo", seed=5),
                          max_actions=5, reset_action=Action.RESET, on_transition=transitions.append)
        self.assertEqual(arcade.makes, 1)
        self.assertEqual((result.actions, result.levels_completed, result.won), (2, 1, True))
        self.assertEqual([t["action"] for t in transitions], ["RESET", "ACTION1"])
        self.assertEqual(transitions[1]["observation"]["frame"], [[[1]]])
        self.assertEqual(transitions[1]["next_observation"]["frame"], [[[2]]])

    def test_training_does_not_record_reserved_next_level(self):
        arcade = Arcade()
        transitions = []
        result = run_game(
            arcade, "demo", RandomPolicy("demo"), max_actions=5,
            reset_action=Action.RESET, stop_at_completed_levels=1,
            on_transition=transitions.append,
        )
        self.assertEqual(result.levels_completed, 1)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["next_observation"]["frame"], [[[1]]])

    def test_splits_must_be_disjoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "splits.json"
            path.write_text(json.dumps({"training_games": ["a"], "evaluation_games": ["a"]}))
            with self.assertRaisesRegex(ValueError, "overlap"):
                games_for_split(path, "training")
            path.write_text(json.dumps({"training_games": ["a"], "evaluation_games": ["b"]}))
            self.assertEqual(games_for_split(path, "evaluation"), ["b"])

    def test_kaggle_notebook_contains_shared_policy(self):
        notebook = build_notebook(accelerator="t4")
        self.assertEqual(notebook["nbformat"], 4)
        source = "\n".join(str(cell["source"]) for cell in notebook["cells"])
        self.assertIn("arc3/core/policy.py", source)
        self.assertIn("KAGGLE_IS_COMPETITION_RERUN", source)
        self.assertIn("submission.parquet", source)


if __name__ == "__main__":
    unittest.main()
