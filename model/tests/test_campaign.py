from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from model.campaign import CONTROLS, WORLD_VARIANTS, compare, imagination_gate, main, quotas, world_config
from model.config import TrainConfig
from model.planning import OutcomeValue, PlanConfig, fingerprint


class CampaignProtocol(unittest.TestCase):
    def test_budget_is_total_across_ten_pairs_not_per_pair(self):
        pairs = [(game, level) for game in ('a', 'b') for level in range(1, 6)]
        assigned = quotas(16384, pairs)
        self.assertEqual(sum(assigned.values()), 16384)
        self.assertEqual(sorted(assigned.values()), [1638]*6 + [1639]*4)

    def test_ablations_change_only_their_declared_loss_or_readout(self):
        base = TrainConfig(auto_representation=True, freeze_representation=True)
        reference = world_config(base, 'rollout2', 16384)
        self.assertEqual(world_config(base, 'sigreg_only', 16384).rollout_weight, .5)
        self.assertEqual(world_config(base, 'sigreg_only', 16384).decoder_readout, reference.decoder_readout)
        local = world_config(base, 'local_decoder', 16384)
        self.assertEqual(local.variance_weight, reference.variance_weight)
        self.assertEqual(local.decoder_readout, 'local')
        staged = world_config(base, 'staged_freeze', 16384)
        self.assertEqual(staged.representation_joint_steps, 8192)
        self.assertFalse(staged.auto_representation)
        self.assertFalse(staged.freeze_representation)
        sigreg = world_config(base, 'sigreg_only', 16384)
        differences = [key for key in asdict(staged) if getattr(staged, key) != getattr(sigreg, key)]
        self.assertEqual(differences, ['variance_weight'])
        self.assertEqual(sigreg.variance_weight, 0.)
        self.assertEqual(CONTROLS['sigreg_only'], 'staged_freeze')
        self.assertLess(WORLD_VARIANTS.index('staged_freeze'), WORLD_VARIANTS.index('sigreg_only'))

    @staticmethod
    def reports(variant, success, seconds=10.):
        return [{'variant': variant, 'training': {'seconds': seconds}, 'behavior': {
            'actions': 320, 'decision_ms': 2., 'results': [
                {'training_seed': seed, 'game': 'game', 'level': 1, 'seed': i, 'success': success}
                for i in range(32)]}} for seed in range(3)]

    def test_comparison_rejects_unpaired_evidence_and_accepts_actual_simplification(self):
        reference = self.reports('staged_freeze', True)
        same = self.reports('sigreg_only', True)
        self.assertTrue(compare(reference, same)['promote'])
        # Merely choosing local readout has not removed saved contextual weights.
        self.assertFalse(compare(reference, self.reports('local_decoder', True))['promote'])
        worse = self.reports('sigreg_only', False, seconds=1.)
        self.assertFalse(compare(reference, worse)['promote'])
        changed = deepcopy(same)
        changed[0]['behavior']['results'][0]['seed'] = 999
        with self.assertRaisesRegex(ValueError, 'identical'):
            compare(reference, changed)


class ImaginationAdmission(unittest.TestCase):
    def setUp(self):
        self.engine = SimpleNamespace(model=torch.nn.Linear(1, 1), device='cpu', config=TrainConfig())
        self.pairs = [('synthetic', 1), ('synthetic', 2)]

    def measured(self):
        return {'per_pair': [dict(game=game, level=level, rollouts={'2': {
            'samples': 5, 'latent_mse': .1, 'identity_latent_mse': .2,
            'shuffled_action_latent_mse': .15, 'action_sensitivity_mse': .03,
            'readouts': {name: {'prediction_delta_error': .2, 'reconstruction_delta_error': .01}
                         for name in ('contextual', 'local')}}}) for game, level in self.pairs]}

    def reference(self, measured):
        return {'variant': 'mc', 'seed': 1, 'world_fingerprint': fingerprint(self.engine.model), 'world': measured}

    def test_gate_reuses_only_exact_frozen_world_diagnostics_and_records_seeds(self):
        reference = self.reference(self.measured())
        with patch('model.campaign.evaluate_world', return_value=self.measured()) as evaluate:
            result = imagination_gate(self.engine, {}, self.pairs, 1, reference)
            self.assertTrue(result['passed'])
            self.assertEqual(result['evidence_source'], 'mc_development')
            evaluate.assert_not_called()
            reference['world_fingerprint'] = 'another-world'
            result = imagination_gate(self.engine, {}, self.pairs, 1, reference)
            evaluate.assert_called_once_with(self.engine, {}, self.pairs, 1, episodes=2, diagnostics=True)
            self.assertEqual(result['evidence_source'], 'fresh_source_world_diagnostic')
        self.assertTrue(all(len(row['evaluation_seeds']) == 2 for row in result['per_pair']))
        self.assertTrue(all(seed % 2 == 1 for row in result['per_pair'] for seed in row['evaluation_seeds']))

    def test_bad_or_missing_pair_cannot_be_hidden_by_good_average_and_nan_is_json_safe(self):
        for kind in ('worse', 'nan', 'missing'):
            with self.subTest(kind=kind):
                measured = self.measured()
                bad = measured['per_pair'][1]['rollouts']['2']
                if kind == 'worse':
                    bad['latent_mse'] = .21  # Macro .155 would beat identity .2.
                elif kind == 'nan':
                    bad['latent_mse'] = float('nan')
                else:
                    bad['samples'] = 0
                result = imagination_gate(self.engine, {}, self.pairs, 1, self.reference(measured))
                self.assertFalse(result['passed'])
                self.assertTrue(result['per_pair'][0]['passed'])
                self.assertFalse(result['per_pair'][1]['passed'])
                json.dumps(result, allow_nan=False)

    def test_failed_gate_skips_training_and_preserves_imitation_candidate(self):
        self.pairs = [('synthetic', 1)]
        measured = self.measured()
        measured['per_pair'][0]['rollouts']['2']['latent_mse'] = 1.
        heads = OutcomeValue(16, 32, width=8)
        trained = []
        planner = SimpleNamespace(config=PlanConfig(clicks=16))
        def train(engine, heads, config, factories, pairs, **kwargs):
            trained.append(kwargs['variant'])
            return planner, None, {'actions': kwargs['total_actions'], 'seconds': 1.}
        def save(path, agent):
            Path(path).write_text(trained[-1])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            world, initial, output = root/'world.pt', root/'heads.pt', root/'campaign'
            torch.save({'training': asdict(TrainConfig())}, world)
            torch.save({}, initial)
            with patch('model.campaign.SYNTHETIC_LEVELS', {'synthetic': 1}), \
                 patch('model.campaign.Factory'), \
                 patch('model.campaign.load_source', return_value=self.engine), \
                 patch('model.campaign.load_heads', return_value=heads), \
                 patch('model.campaign.evaluate_world', return_value=measured), \
                 patch('model.campaign.evaluate_behavior', return_value={}), \
                 patch('model.campaign.Agent'), \
                 patch('model.campaign.agent_checkpoint.save', side_effect=save), \
                 patch('model.campaign_behavior.train_behavior', side_effect=train), \
                 redirect_stdout(io.StringIO()):
                arguments = ['--checkpoint', str(world), '--heads', str(initial), '--output', str(output),
                    '--pilot', '--device', 'cpu', '--actions', '10', '--seeds', '1', '--dev-episodes', '2',
                    '--variants', 'mc', 'actor_imitation', 'actor_imagined']
                main(arguments)
                main(arguments)  # A resumed finished campaign must not retrain deferred cells.
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(trained, ['mc', 'actor_imitation'])
            self.assertEqual((output/'candidate.pt').read_text(), 'actor_imitation')
            self.assertEqual(len(report['cells']), 2)
            self.assertEqual(report['deferred_cells'][0]['training'], {'actions': 0, 'updates': 0})
            self.assertEqual(report['deferred_variants']['actor_imagined']['fallback'], 'actor_imitation')
            self.assertEqual(report['status'], 'pilot_complete')


if __name__ == '__main__':
    unittest.main()
