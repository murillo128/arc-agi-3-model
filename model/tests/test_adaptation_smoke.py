"""Smoke evidence distinguishes simulated inputs from public SDK experience."""
from dataclasses import asdict, replace
from contextlib import redirect_stderr
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from model.adaptation import AdaptationConfig
from model.adaptation_checkpoint import export
from model.adaptation_heads import VisualGoal
from model.adaptation_smoke import _tensor_digest, _training_level_one, local_smoke, self_check
from model.config import NetworkConfig, TrainConfig
from model.networks import WorldModel
from model.path_planning import PathValue


def portable_fixture(root):
    # Exercise the production feature combination with small CPU dimensions.
    config = replace(NetworkConfig.small(), patch=2, latent=16, pixel_latent=True,
                     local_dynamics=True, rotation_adapter_width=16,
                     learnable_click_sigma=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        world = WorldModel(config)
        goal = VisualGoal(config.latent, config.memory, config.patch, 8)
        path = PathValue(config.latent, config.memory, 8)
    source, output = root / 'training-fixture.pt', root / 'base.pt'
    torch.save(dict(kind='game_sequences', version=1, network=asdict(config),
                    training=asdict(TrainConfig()), weights=world.state_dict(),
                    planning=dict(version=1, config=dict(width=8), goal=goal.state_dict(),
                                  path=path.state_dict()), progress=dict(batches=0)), source)
    export(source, output)
    return output


class AdaptationSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_digest_detects_exact_single_bit_changes_and_covers_shape(self):
        tensor = torch.tensor([1., 2.])
        before, size = _tensor_digest([('tensor', tensor)])
        self.assertEqual(size, 8)
        self.assertEqual(_tensor_digest([('tensor', tensor.clone())])[0], before)
        tensor.view(torch.int32)[0] ^= 1
        self.assertNotEqual(_tensor_digest([('tensor', tensor)])[0], before)
        self.assertNotEqual(_tensor_digest([('tensor', tensor.reshape(1, 2))])[0],
                            _tensor_digest([('tensor', tensor)])[0])

    def test_level_gate_excludes_all_crossings_and_unknown_reserves(self):
        for completed, wins, expected in ((0, 5, True), (1, 5, False), (4, 5, False),
                                          (0, 0, False), (0, 1, False), (0, 2, True)):
            with self.subTest(completed=completed, wins=wins):
                self.assertEqual(_training_level_one(SimpleNamespace(
                    levels_completed=completed, win_levels=wins)), expected)

    def test_fixture_self_check_runs_real_optimizers_without_claiming_sdk_data(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = portable_fixture(Path(directory))
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            progress = io.StringIO()
            with redirect_stderr(progress):
                report = self_check(checkpoint, config=dict(per_observation_seconds=30.,
                                      total_seconds=120., reconstruction_threshold=0.))
            events = [json.loads(line) for line in progress.getvalue().splitlines()]
            self.assertEqual([event['stage'] for event in events],
                             ['loading', 'loaded', 'checking', 'learning', 'verifying', 'complete'])
            self.assertEqual([event['elapsed_seconds'] for event in events],
                             sorted(event['elapsed_seconds'] for event in events))
            self.assertEqual(report['runtime']['device'], 'cpu')
            self.assertEqual(report['runtime']['precision'], 'fp32')
            self.assertFalse(report['runtime']['native_bf16'])
            self.assertEqual(report['runtime']['checkpoint_loading'], 'staged_sequential_eager')
            self.assertEqual(report['runtime']['parameter_dtype'], 'torch.float32')
            timings = report['timings']
            self.assertTrue(all(math.isfinite(v) and v >= 0 for v in timings.values()))
            self.assertGreater(timings['load_seconds'], 0)
            self.assertGreaterEqual(timings['startup_seconds'], timings['load_seconds'])
            self.assertGreaterEqual(report['elapsed_seconds'], sum(timings[key] for key in
                ('startup_seconds', 'interaction_seconds', 'verification_seconds')))
            self.assertTrue(report['passed'], report)
            self.assertEqual(report['kind'], 'offline-fixture-self-check')
            self.assertEqual(report['real_sdk_transitions'], 0)
            self.assertEqual(report['confirmed_observations'], 3)
            self.assertTrue(all(report['isolation'].values()))
            self.assertEqual(report['base_sha256_before'], report['base_sha256_after'])
            self.assertEqual(digest, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            game = report['games'][0]
            self.assertEqual(game['simulated_confirmations'], 2)
            self.assertGreater(report['base_bytes'], game['adapter_bytes'])
            for group in ('representation', 'predictor', 'path', 'goal'):
                self.assertGreater(game['diagnostics']['updates'][group], 0, group)
                self.assertLessEqual(game['diagnostics']['losses'][group]['post'],
                                     game['diagnostics']['losses'][group]['pre'], group)

    def test_public_sdk_smoke_uses_level_one_without_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = portable_fixture(Path(directory))
            config = AdaptationConfig(stopping='steps', representation_steps=1, per_observation_seconds=30.,
                                      total_seconds=120., reconstruction_threshold=0.,
                                      horizon=1, beam=1, clicks=2)
            # The smoke must never import the trainer's privileged collection API.
            import sys
            with patch.dict(sys.modules, {'model.environment': None,
                                          'model.sequence_planning': None}):
                report = local_smoke(checkpoint, actions=1, config=config,
                                     environments_dir=Path(__file__).parents[2] / 'games')
            self.assertTrue(report['passed'], report)
            self.assertEqual(report['kind'], 'public-sdk-level1-smoke')
            self.assertEqual(report['real_sdk_transitions'], 2)
            self.assertEqual(report['runtime']['precision'], 'fp32')
            self.assertGreater(report['timings']['load_seconds'], 0)
            self.assertGreater(report['timings']['interaction_seconds'], 0)
            self.assertEqual([game['game'] for game in report['games']], ['sc01-v1', 'si01-v1'])
            self.assertEqual([game['executed_sdk_actions'] for game in report['games']], [1, 1])
            self.assertTrue(report['base_immutable'])
            self.assertTrue(all(report['isolation'].values()))
            # The fixed default seed's first sc01 level-1 SDK action changes
            # the returned board. That real pair supplies predictor supervision
            # and an already reached hindsight goal, so all four phases must
            # perform useful bounded work. No oracle labels or seed/LR search
            # are involved; both fixture weights and policy seed are fixed.
            observed = report['games'][0]['diagnostics']
            self.assertEqual(observed['confirmed_transitions'], 1)
            self.assertEqual(observed['predictive_pairs'], 1)
            self.assertEqual(observed['boundary_skips'], 0)
            self.assertEqual(observed['numerical_failures'], 0)
            for group in ('representation', 'predictor', 'path', 'goal'):
                self.assertGreater(observed['updates'][group], 0, group)
                losses = observed['losses'][group]
                self.assertTrue(math.isfinite(losses['pre']) and math.isfinite(losses['post']), group)
                self.assertLess(losses['post'], losses['pre'], group)


if __name__ == '__main__':
    unittest.main()
