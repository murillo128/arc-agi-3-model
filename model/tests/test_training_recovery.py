"""Recovery and fixed-data diagnostics through real small neural modules."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from model import agent_checkpoint, campaign, learning_probe
from model.agent import Agent
from model.config import TrainConfig
from model.planning import OutcomeValue, fingerprint
from model.tests.test_campaign_world import engine, SyntheticFixture


class TrainingRecoveryTests(unittest.TestCase):
    def test_failed_evaluation_resumes_saved_agent_without_retraining(self):
        e = engine()
        heads = OutcomeValue(16, 32, width=8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            world, initial, output = root/'world.pt', root/'heads.pt', root/'output'
            torch.save({'training': asdict(TrainConfig())}, world)
            torch.save({}, initial)
            arguments = ['--checkpoint', str(world), '--heads', str(initial), '--output', str(output),
                         '--pilot', '--device', 'cpu', '--actions', '4', '--seeds', '0',
                         '--dev-episodes', '1', '--variants', 'world_reference']
            with patch('model.campaign.SYNTHETIC_LEVELS', {'fixture': 1}), \
                 patch('model.campaign.Factory', return_value=SyntheticFixture(steps=2)), \
                 patch('model.campaign.load_source', return_value=e), \
                 patch('model.campaign.load_heads', return_value=heads), \
                 patch('model.campaign.train_world', wraps=campaign.train_world) as train, \
                 patch('model.campaign.evaluate_cell_world', side_effect=RuntimeError('evaluation failed')), \
                 redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                    campaign.main(arguments)
                self.assertEqual(train.call_count, 1)
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report['cells'], [])
            self.assertEqual(report['pending_evaluation']['variant'], 'world_reference')
            saved = agent_checkpoint.load(output/'pending-agent.pt')
            expected = fingerprint(saved.engine.model)
            self.assertEqual(expected, fingerprint(e.model))
            with patch('model.campaign.SYNTHETIC_LEVELS', {'fixture': 1}), \
                 patch('model.campaign.Factory', return_value=SyntheticFixture()), \
                 patch('model.campaign.train_world', side_effect=AssertionError('must not retrain')), \
                 patch('model.campaign.evaluate_cell_world', return_value=({}, {'world_fingerprint': expected})), \
                 patch('model.campaign.evaluate_behavior', return_value={}), \
                 redirect_stdout(io.StringIO()):
                campaign.main(arguments)
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report['status'], 'pilot_complete')
            self.assertEqual(len(report['cells']), 1)
            self.assertNotIn('pending_evaluation', report)
            restored = agent_checkpoint.load(output/'candidate.pt')
            self.assertEqual(fingerprint(restored.engine.model), expected)

    def test_learning_probe_uses_equal_initial_weights_and_keeps_representation_frozen(self):
        e = engine()
        agent = Agent(e, OutcomeValue(16, 32, width=8))
        factory = SyntheticFixture(steps=3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root/'agent.pt', root/'probe'
            agent_checkpoint.save(source, agent)
            with patch('model.learning_probe.SYNTHETIC_LEVELS', {'fixture': 1}), \
                 patch('model.learning_probe.Factory', return_value=factory), redirect_stdout(io.StringIO()):
                learning_probe.main(['--agent', str(source), '--output', str(output), '--device', 'cpu',
                                     '--episodes', '2', '--steps', '3', '--updates', '2', '--evaluate-every', '2'])
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(set(seed for seed, _ in factory.calls)), 4)
            self.assertEqual(report['results'][0]['initial_world'], report['results'][1]['initial_world'])
            for row in report['results']:
                self.assertTrue(row['frozen_representation_verified'])
                self.assertTrue(row['dynamics_changed'])
                self.assertTrue(row['saved_before_evaluation'])
                self.assertEqual([p['updates'] for p in row['points']], [0, 2])
                self.assertEqual(row['points'][0]['fit']['rollouts']['2']['samples'], 4)
                loaded = agent_checkpoint.load(output/f'{row["variant"]}.pt')
                self.assertEqual(fingerprint(loaded.engine.model.encoder), fingerprint(e.model.encoder))
                self.assertEqual(fingerprint(loaded.engine.model.decoder), fingerprint(e.model.decoder))


if __name__ == '__main__':
    unittest.main()
