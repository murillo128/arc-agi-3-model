from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.agent import Agent
from model.config import NetworkConfig, TrainConfig
from model.data import Observation
from model.diagnostics import action_probe, main
from model.learning import Engine
from model.networks import WorldModel
from model.planning import OutcomeValue


def observation(board):
    return Observation(board[None].copy(), (1, 2), 'NOT_FINISHED', 0, 1)


class BranchFactory:
    def __init__(self, equivalent=False, unstable=False):
        self.equivalent, self.unstable, self.resets = equivalent, unstable, 0

    def make(self, seed, level):
        self.resets += 1
        board = np.zeros((2, 2), dtype=np.uint8)
        if self.unstable:
            board[1, 1] = self.resets % 2

        def step(action):
            board[0, 0 if self.equivalent else action.id - 1] ^= 1
            return observation(board)

        return SimpleNamespace(observation=observation(board), step=step)


class PerfectEngine:
    def initial(self, obs):
        return SimpleNamespace(z=True, board=obs.final)

    def observe(self, state, action, obs):
        return self.initial(obs)

    def prediction(self, state, action, reference):
        board = state.board.copy()
        board[0, action.id - 1] ^= 1
        return board, None, None


class DiagnosticTests(unittest.TestCase):
    def test_correct_action_beats_wrong_action_and_no_change_on_real_branches(self):
        progress = []
        report = action_probe(PerfectEngine(), BranchFactory(), 1, [1], anchors=2,
                              progress=lambda **p: progress.append(p))
        self.assertEqual(report['anchors'], 2)
        self.assertEqual(report['predictor']['samples'], 4)
        self.assertEqual(report['predictor']['delta_error'], 0.)
        self.assertEqual(report['no_change']['delta_error'], 1.)
        self.assertEqual(report['wrong_action']['delta_error'], 1.)
        self.assertEqual(report['wrong_action']['invented'], 4)
        self.assertEqual(report['wrong_action']['missed'], 4)
        self.assertEqual(report['distinct_pairs'], 4)
        self.assertEqual(report['real_separation_pixels'], 8)
        self.assertEqual(report['predicted_separation_pixels'], 8)
        self.assertEqual(len(progress), 4)

    def test_equivalent_real_actions_do_not_claim_action_discrimination(self):
        report = action_probe(PerfectEngine(), BranchFactory(equivalent=True), 1, [1], anchors=1)
        self.assertEqual(report['distinct_pairs'], 0)
        self.assertIsNone(report['distinct_correct']['delta_error'])

    def test_nondeterministic_reset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not reproducible'):
            action_probe(PerfectEngine(), BranchFactory(unstable=True), 1, [1], anchors=1)

    def test_runner_evaluates_three_policies_with_same_small_frozen_agent(self):
        torch.set_num_threads(2)
        engine = Engine(WorldModel(NetworkConfig.small()), TrainConfig())
        agent = Agent(engine, OutcomeValue(16, 32, width=8))
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
                patch('model.diagnostics.agent_checkpoint.load', return_value=agent), \
                patch('model.diagnostics.SYNTHETIC_LEVELS', {'sc01-v1': 1}):
            main(['--agents', 'unused.pt', '--labels', 'small', '--output', tmp,
                  '--device', 'cpu', '--episodes', '1', '--steps', '2', '--probe-seeds', '1', '--anchors', '1'])
            report = json.loads((Path(tmp) / 'report.json').read_text())
        self.assertEqual(report['status'], 'complete')
        result = report['results'][0]
        self.assertTrue(result['weights_unchanged'])
        self.assertEqual(set(result['policies']), {'random', 'q', 'value_mpc'})
        for policy in result['policies'].values():
            self.assertTrue(policy['complete'])
            self.assertEqual(policy['summary']['episodes'], 1)
            self.assertLessEqual(policy['summary']['actions'], 2)
        self.assertEqual(result['probes'][0]['anchors'], 1)
