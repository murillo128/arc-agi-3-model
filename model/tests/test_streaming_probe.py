"""End-to-end small CPU proof for paired exposure, selection and calibration."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model import agent_checkpoint, streaming_probe
from model.agent import Agent
from model.planning import OutcomeValue, fingerprint
from model.tests.test_campaign_world import engine, SyntheticFixture


class StreamingProbeTests(unittest.TestCase):
    def test_paired_real_data_best_weights_and_frozen_calibration(self):
        e = engine()
        agent = Agent(e, OutcomeValue(16, 32, width=8))
        factory = SyntheticFixture(steps=3, native_progress=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / 'agent.pt', root / 'probe'
            agent_checkpoint.save(source, agent)
            with patch('model.streaming_probe.SYNTHETIC_LEVELS', {'fixture': 2}), \
                 patch('model.streaming_probe.Factory', return_value=factory), redirect_stdout(io.StringIO()):
                streaming_probe.main(['--agent', str(source), '--output', str(output), '--device', 'cpu',
                    '--actions', '12', '--steps', '3', '--world-steps', '3', '--world-episodes', '1',
                    '--head-epochs', '1', '--behavior-episodes', '1', '--train-every', '2', '--evaluate-every', '6'])
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['status'], 'complete')
            a, b = report['results']
            self.assertEqual(a['data_digest'], b['data_digest'])
            self.assertEqual(a['initial_world'], b['initial_world'])
            self.assertEqual(a['updates'], b['updates'])
            self.assertEqual(report['coverage']['actions'], 12)
            self.assertEqual([r['actions'] for r in report['coverage']['per_pair']], [6, 6])
            # Four episodes were collected once, not recollected for recipe two.
            self.assertEqual(sum(seed % 2 == 0 for seed, _ in factory.calls), 4)
            for row in (a, b):
                self.assertTrue(row['dynamics_changed'])
                self.assertTrue(row['frozen_representation_verified'])
                self.assertEqual(row['best']['selection_error'], min(p['selection_error'] for p in row['points']))
                self.assertEqual(row['points'][-1]['actions'], 12)
                # The progress transition enters a new native level: it is
                # retained for value labels but excluded from world targets.
                self.assertEqual(row['points'][0]['development']['rollouts']['1']['samples'], 4)
            self.assertEqual(report['calibration']['records'], 12)
            self.assertEqual(report['calibration']['successes'], 4)
            self.assertEqual(report['calibration']['positive_records'], 12)
            self.assertTrue(report['calibration']['world_unchanged'])
            self.assertTrue(report['evaluation_weights_unchanged'])
            self.assertEqual(set(report['policies']), {'random', 'q', 'value_mpc'})
            for policy in report['policies'].values():
                self.assertTrue(policy['complete'])
                self.assertEqual(policy['summary']['wins'], 2)
            loaded = agent_checkpoint.load(output / 'calibrated-agent.pt')
            self.assertEqual(fingerprint(loaded.engine.model.encoder), fingerprint(e.model.encoder))
            self.assertEqual(fingerprint(loaded.engine.model.decoder), fingerprint(e.model.decoder))
            self.assertNotEqual(fingerprint(loaded.heads), fingerprint(agent.heads))

    def test_quota_cut_is_censored_and_selection_weights_pairs_equally(self):
        factory = SyntheticFixture(steps=20)
        rows = list(streaming_probe.fresh_episodes({'a': factory}, [('a', 1), ('a', 2)], 9, 4, 0))
        self.assertEqual([reason for _, _, reason in rows], ['budget', 'budget', 'interrupted'])
        self.assertEqual(sum(len(ep.actions) for _, ep, _ in rows), 9)
        def horizon(samples, error):
            return {'samples': samples, 'readouts': {'contextual': {'prediction_delta_error': error}}}
        # Sample counts must not allow an easy level to dominate selection.
        data = {'per_pair': [{'rollouts': {'1': horizon(100, .1), '2': horizon(90, .3)}},
                             {'rollouts': {'1': horizon(2, .7), '2': horizon(1, .9)}}]}
        self.assertAlmostEqual(streaming_probe.selection_error(data), .5)


if __name__ == '__main__':
    unittest.main()
