"""Probe stopping/counting contracts; scripted public SDK frames, no game rules."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from arcengine import FrameData, GameAction, GameState

from scripts.random_level_probe import run_probe


def frame(*, levels=0, state=GameState.NOT_FINISHED, pixel=1, actions=(1, 6, 7)):
    return FrameData(game_id="probe", frame=[[[pixel, pixel], [pixel, pixel]]],
                     levels_completed=levels, win_levels=3, state=state,
                     available_actions=list(actions))


class RandomLevelProbeTests(unittest.TestCase):
    def test_first_level_stops_immediately_and_counts_noop_and_reset(self):
        responses = iter([frame(), frame(state=GameState.GAME_OVER), frame(), frame(levels=1, pixel=2)])
        actions, events = [], []
        def step(action, *, data):
            actions.append(action)
            return next(responses)
        result = run_probe(SimpleNamespace(observation_space=frame(), step=step),
                           seed=42, max_actions=10, seconds=60, emit=events.append)
        self.assertEqual(result["actions_to_first_level"], 4)
        self.assertEqual((result["resets"], result["game_over_returns"]), (1, 1))
        self.assertEqual((result["changed"], result["unchanged"]), (1, 3))
        self.assertEqual(actions[2], GameAction.RESET)
        self.assertEqual(result["stop_reason"], "first_level_solved")
        self.assertEqual(len(events), 8)

    def test_action_cap_reproducibility_click_coordinates_and_no_undo(self):
        runs = []
        for _ in range(2):
            actions = []
            def step(action, *, data):
                actions.append((action.value, data))
                return frame()
            result = run_probe(SimpleNamespace(observation_space=frame(), step=step),
                               seed=42, max_actions=40, seconds=60, emit=lambda event: None)
            self.assertEqual((result["actions"], result["unchanged"]), (40, 40))
            self.assertEqual(result["stop_reason"], "action_budget")
            self.assertIsNone(result["actions_to_first_level"])
            self.assertTrue(result["censored"])
            self.assertEqual({action for action, data in actions}, {1, 6})
            for action, data in actions:
                if action == 6:
                    self.assertTrue(0 <= data["x"] < 2 and 0 <= data["y"] < 2)
            runs.append(actions)
        self.assertEqual(runs[0], runs[1])

    def test_late_sdk_return_is_counted_before_time_budget_or_success(self):
        for solved in (False, True):
            with self.subTest(solved=solved):
                clock = [0.]
                def step(action, *, data):
                    clock[0] = 2.
                    return frame(levels=int(solved))
                with patch("scripts.random_level_probe.time.monotonic", side_effect=lambda: clock[0]):
                    result = run_probe(SimpleNamespace(observation_space=frame(), step=step),
                                       seed=42, max_actions=10, seconds=1., emit=lambda event: None)
                self.assertEqual((result["actions"], result["returned_actions"]), (1, 1))
                self.assertEqual(result["time_overrun_seconds"], 1.)
                self.assertEqual(result["stop_reason"], "first_level_solved" if solved else "time_budget")

    def test_missing_sdk_result_retains_attempt_without_claiming_success(self):
        with self.assertLogs(level="ERROR"):
            result = run_probe(SimpleNamespace(observation_space=frame(), step=lambda *a, **k: None),
                               seed=42, max_actions=10, seconds=60, emit=lambda event: None)
        self.assertEqual((result["actions"], result["returned_actions"]), (1, 0))
        self.assertFalse(result["first_level_solved"])
        self.assertEqual(result["stop_reason"], "error")


if __name__ == "__main__":
    unittest.main()
