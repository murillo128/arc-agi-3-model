from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from model.config import NetworkConfig
from model.data import Observation
from model.networks import WorldModel
from model.reconstruction_audit import (FrozenReconstructor, ReconstructionMetrics,
                                        UniqueReservoir, collect_target, weights_digest)
from model.sequence_exploration import SequenceExplorationConfig


def observation(color=0, level=1, state='NOT_FINISHED', frames=None):
    return Observation(np.full((1, 4, 8), color, dtype=np.uint8) if frames is None else frames,
                       (1,), state, level-1, 3)


class ReconstructionAuditTests(unittest.TestCase):
    def test_pixel_color_and_changed_errors_have_independent_denominators(self):
        actual = np.array([[0, 0, 0, 0], [0, 0, 1, 2]], dtype=np.uint8)
        predicted = actual.copy()
        predicted[0, 0] = 1  # one of six background pixels
        predicted[1, 2] = 0  # all of color 1; color 2 remains perfect
        previous = actual.copy()
        previous[1, 2:] = 0
        metrics = ReconstructionMetrics()
        metrics.add(actual, predicted, previous)
        result = metrics.report()
        self.assertEqual(result['pixel_error'], 2/8)
        self.assertAlmostEqual(result['balanced_color_error'], (1/6+1+0)/3)
        self.assertEqual(result['non_dominant_error'], .5)
        self.assertEqual(result['changed_error'], .5)
        self.assertEqual(result['perfect_fraction'], 0)
        self.assertEqual(result['colors'][0], dict(color=0, wrong=1, pixels=6))

    def test_missing_masks_are_not_zero_error_and_perfect_frames_are_counted(self):
        metrics = ReconstructionMetrics()
        self.assertIsNone(metrics.report()['pixel_error'])
        actual = np.zeros((2, 2), dtype=np.uint8)
        metrics.add(actual, actual, actual)
        row = metrics.report()
        self.assertEqual(row['pixel_error'], 0)
        self.assertEqual(row['balanced_color_error'], 0)
        self.assertEqual(row['perfect_fraction'], 1)
        self.assertIsNone(row['non_dominant_error'])
        self.assertIsNone(row['changed_error'])
        with self.assertRaises(ValueError):
            metrics.add(actual, actual[:, :1])

    def test_reservoir_is_bounded_and_deduplicates_settled_frames_not_levels(self):
        pool = UniqueReservoir(3, 42)
        for i in range(100):
            pool.add(observation(), None, seed=0, action_index=i)
        self.assertEqual(len(pool.seen), 1)
        pool.add(observation(level=2), None, seed=0, action_index=100)
        self.assertEqual(len(pool.seen), 2)
        for color in range(1, 16):
            pool.add(observation(color), None, seed=0, action_index=color)
        self.assertEqual(len(pool.samples), 3)
        self.assertEqual(len(pool.seen), 17)
        pool.add(observation(2, state='GAME_OVER'), None, seed=0, action_index=101)
        self.assertEqual(len(pool.seen), 17)

    def test_final_level_requires_real_entry_and_discards_old_level_animation(self):
        class Env:
            observation = observation()
            def __init__(self):
                self.calls = []
                animation = np.stack([np.zeros((4, 8), dtype=np.uint8), np.full((4, 8), 2, dtype=np.uint8)])
                self.results = iter([observation(1, level=2), observation(level=3, frames=animation),
                                     observation(3, level=4, state='WIN')])
            def step(self, action):
                self.calls.append(action.id)
                return next(self.results)
        env = Env()
        factory = SimpleNamespace(game_id='fixture', make=lambda seed: env)
        row = collect_target(factory, 3, [7], 3, 64, 10, None, SequenceExplorationConfig())
        self.assertEqual(env.calls, [1, 1, 1])
        self.assertEqual(row['reached_levels'], [1, 2, 3])
        self.assertEqual(row['status'], 'sampled')
        self.assertEqual(len(row['samples']), 1)
        self.assertEqual(row['samples'][0]['frames'].shape, (1, 4, 8))
        self.assertTrue(np.all(row['samples'][0]['frames'] == 2))
        self.assertIsNone(row['samples'][0]['previous'])

    def test_unreachable_level_is_missing_and_resets_use_public_actions(self):
        class Env:
            observation = observation()
            def __init__(self): self.calls = []
            def step(self, action):
                self.calls.append(action.id)
                return observation() if action.id == 0 else observation(state='GAME_OVER')
        env = Env()
        row = collect_target(SimpleNamespace(game_id='fixture', make=lambda seed: env),
                             3, [9], 4, 64, 10, None, SequenceExplorationConfig())
        self.assertEqual(env.calls, [1, 0, 1, 0])
        self.assertEqual(row['status'], 'unreached')
        self.assertEqual(row['samples'], [])
        self.assertEqual(row['actions'], 4)

    def test_cpu_reconstruction_preserves_weights_and_real_animation_context(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        config = NetworkConfig.small()
        reference = WorldModel(config).eval()
        weights = {k: v.clone() for k, v in reference.state_dict().items()
                   if k.startswith(('encoder.', 'decoder.'))}
        digest = weights_digest(weights)
        model = FrozenReconstructor(config, weights)
        obs = observation(frames=np.stack([np.zeros((4, 8), dtype=np.uint8),
                                           np.ones((4, 8), dtype=np.uint8)]))
        captured = []
        handle = model.encoder.temporal.register_forward_hook(lambda module, args, result: captured.append(result))
        prediction = model.reconstruct(obs)
        handle.remove()
        self.assertEqual(len(captured), 2)
        self.assertEqual(prediction.shape, (4, 8))
        with torch.inference_mode():
            z = reference.encode([obs], reference.encoder.spatial, chunk=1)[0]
            expected = reference.decoder(z[None], 4, 8).argmax(-1)[0].numpy()
        np.testing.assert_array_equal(prediction, expected)
        self.assertEqual(weights_digest(model.state_dict()), digest)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        self.assertFalse(hasattr(model, 'dynamics'))


if __name__ == '__main__':
    unittest.main()
