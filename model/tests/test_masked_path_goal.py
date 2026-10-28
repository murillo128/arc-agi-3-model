"""Masked goal inputs retain only the current real context outside the target."""
from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model import agent_checkpoint
from model.agent import Agent, AgentConfig
from model.path_planning import PathConfig, PathPlanner, PathValue
from model.planning import fingerprint
from model.tests.test_agent import make_agent, obs


class GoalObservationTests(unittest.TestCase):
    def test_literal_composition_is_an_independent_uint8_copy(self):
        from model.goal_observation import compose_goal_observation
        goal = np.array([[1, 2], [3, 4]], dtype=np.int64)
        mask = np.array([[True, False], [False, True]])
        current = np.array([[5, 6], [7, 8]], dtype=np.uint8)
        originals = [value.copy() for value in (goal, mask, current)]
        for value in (goal, mask, current):
            value.flags.writeable = False
        result = compose_goal_observation(goal, mask, current)
        np.testing.assert_array_equal(result, [[1, 6], [7, 4]])
        self.assertEqual(result.dtype, np.uint8)
        result[:] = 0
        for value, original in zip((goal, mask, current), originals):
            np.testing.assert_array_equal(value, original)
            self.assertFalse(np.shares_memory(result, value))

    def test_geometry_mismatches_are_rejected_before_broadcasting(self):
        from model.goal_observation import compose_goal_observation
        for goal, mask, current in (
                ([[1, 2]], [[True]], [[3, 4]]),
                ([[1, 2]], [[True, False]], [[3], [4]]),
                ([1, 2], [True, False], [3, 4])):
            with self.subTest(goal=goal, mask=mask, current=current):
                with self.assertRaisesRegex(ValueError, 'geometry'):
                    compose_goal_observation(goal, mask, current)


class MaskedPathGoalTests(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent(frozen=False)
        engine = self.agent.engine
        self.value = PathValue(engine.model.config.latent, engine.model.config.memory, width=4)
        # The normal initial head is constant. Nonzero outputs let this regression
        # also compare real scores from the spatial target-sensitive readout.
        with torch.no_grad():
            self.value.success.weight.normal_(std=.1)
            self.value.readout[-1].weight.normal_(std=.1)
        self.planner = PathPlanner(engine, self.value, PathConfig(horizon=3, beam=2, clicks=2))
        self.observation = obs(2, actions=(1, 2))
        self.state = engine.initial(self.observation)
        self.mask = np.zeros((4, 4), dtype=bool)
        self.mask[:2, :2] = True
        self.goal = np.zeros((4, 4), dtype=np.uint8)
        self.goal[self.mask] = 5

    def test_excluded_goal_colors_leave_encoded_target_scores_and_action_identical(self):
        engine = self.agent.engine
        goals = [self.goal.copy(), self.goal.copy()]
        goals[1][~self.mask] = 9
        decisions = []
        weights = fingerprint(engine.model)
        for goal in goals:
            calls = []
            def record(module, inputs, outputs):
                calls.append((inputs[2].clone(), tuple(value.clone() for value in outputs)))
            handle = self.value.register_forward_hook(record)
            try:
                with patch.object(engine, 'encode', wraps=engine.encode) as encode:
                    self.planner.set_goal(goal, self.mask)
                    rng, diagnostics = random.Random(91), {}
                    action = self.planner.choose(self.state, self.observation, rng, diagnostics=diagnostics)
                    encoded_frames = [call.args[0][0].final.copy() for call in encode.call_args_list]
            finally:
                handle.remove()
            decisions.append((action, diagnostics, rng.getstate(), calls, encoded_frames))
            np.testing.assert_array_equal(self.planner.goal['frame'], goal)
            np.testing.assert_array_equal(self.planner.goal['pixel_mask'], self.mask)
        first, second = decisions
        torch.testing.assert_close(first[3][0][0], second[3][0][0], rtol=0, atol=0)
        self.assertEqual(first[:3], second[:3])
        expected = np.full((4, 4), 2, dtype=np.uint8)
        expected[:2, :2] = 5
        for decision in decisions:
            self.assertEqual(len(decision[4]), 1, 'One goal encoding per real decision')
            np.testing.assert_array_equal(decision[4][0], expected)
            self.assertEqual(len(decision[3]), 3)
            for target, _ in decision[3]:
                torch.testing.assert_close(target, decision[3][0][0], rtol=0, atol=0)
        for (_, expected_scores), (_, actual_scores) in zip(first[3], second[3]):
            for expected_score, actual_score in zip(expected_scores, actual_scores):
                torch.testing.assert_close(actual_score, expected_score, rtol=0, atol=0)
        self.assertEqual(fingerprint(engine.model), weights)
        self.assertTrue(all(p.grad is None for p in engine.model.parameters()))

    def test_partial_goal_waits_for_real_context_and_refreshes_next_decision(self):
        engine = self.agent.engine
        with patch.object(engine, 'encode', wraps=engine.encode) as encode, \
                patch.object(engine, 'decode', side_effect=AssertionError('No imagined decoding')):
            self.planner.set_goal(self.goal, self.mask)
            self.assertEqual(encode.call_count, 0)
            self.assertIsNone(self.planner.goal['z'])
            self.planner.choose(self.state, self.observation, random.Random(91))
            first_target = self.planner.goal['z'].clone()
            changed = np.full((4, 4), 7, dtype=np.uint8)
            changed[self.mask] = 3
            next_observation = replace(self.observation, frames=changed[None])
            self.planner.choose(self.state, next_observation, random.Random(91))
            self.assertEqual(encode.call_count, 2)
            second_input = encode.call_args.args[0][0].final
            expected = np.full((4, 4), 7, dtype=np.uint8)
            expected[:2, :2] = 5
            np.testing.assert_array_equal(second_input, expected)
            self.assertFalse(torch.equal(first_target, self.planner.goal['z']))
        np.testing.assert_array_equal(self.planner.goal['frame'], self.goal)

    def test_full_goal_keeps_eager_encoding_without_per_decision_work(self):
        engine = self.agent.engine
        with patch.object(engine, 'encode', wraps=engine.encode) as encode:
            self.planner.set_goal(self.goal, np.ones_like(self.mask))
            target = self.planner.goal['z'].clone()
            self.assertEqual(encode.call_count, 1)
            for value in (2, 7):
                self.planner.choose(self.state, obs(value, actions=(1, 2)), random.Random(91))
            self.assertEqual(encode.call_count, 1)
            torch.testing.assert_close(self.planner.goal['z'], target, rtol=0, atol=0)

    def test_partial_raw_goal_and_resumed_decision_round_trip(self):
        agent = Agent(self.agent.engine, self.agent.heads,
                      AgentConfig(mode='path_mpc', path=self.planner.config),
                      frozen=False, path_evaluator=self.value)
        agent.observe(self.observation)
        agent.set_goal(self.goal, self.mask)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'agent.pt'
            agent_checkpoint.save(path, agent, include_runtime=True)
            restored = agent_checkpoint.load(path, resume_runtime=True)
            np.testing.assert_array_equal(restored.planner.goal['frame'], self.goal)
            np.testing.assert_array_equal(restored.planner.goal['pixel_mask'], self.mask)
            self.assertEqual(restored.act(), agent.act())
            torch.testing.assert_close(restored.planner.goal['z'], agent.planner.goal['z'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
