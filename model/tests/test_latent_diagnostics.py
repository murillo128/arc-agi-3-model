"""Numerical oracles for read-only latent diagnostics, independent of a network."""
import unittest

import numpy as np
import torch

from model.learning import State
from model.latent_diagnostics import LatentProbe, summarize_latents


def state(z, geometry=(1, 1)):
    return State(torch.tensor(z, dtype=torch.float32), torch.zeros(1, 1), geometry)


def probe(samples, geometry=(1, 1), patch=1):
    result = LatentProbe(patch)
    for z in samples:
        result.sample(state(z, geometry))
    return result


class LatentDiagnosticsTests(unittest.TestCase):
    def test_positional_codes_and_different_geometries_cannot_fake_diversity(self):
        # Each geometry is constant across observations, despite different codes
        # at each spatial position and very different offsets between geometries.
        a = probe([[[7., 8.], [9., 10.]]] * 4, (1, 2))
        b = probe([[[100., 200.]]] * 4)
        result = summarize_latents([a, b])
        self.assertEqual(result['std_mean'], 0.)
        self.assertEqual(result['within_episode_std'], 0.)
        self.assertEqual(result['inactive_fraction'], 1.)
        self.assertEqual(result['effective_rank'], 0.)
        self.assertEqual(result['geometry_count'], 2)

    def test_isotropic_and_redundant_dimensions_have_different_ranks(self):
        corners = probe([[[-1., -1.]], [[-1., 1.]], [[1., -1.]], [[1., 1.]]])
        rng = torch.get_rng_state().clone()
        result = summarize_latents([corners])
        self.assertEqual(result['std_mean'], 1.)
        self.assertEqual(result['inactive_fraction'], 0.)
        self.assertAlmostEqual(result['effective_rank'], 2., places=10)
        self.assertEqual(result['max_rank'], 2)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        redundant = probe([[[-1., -1.]], [[1., 1.]], [[-1., -1.]], [[1., 1.]]])
        result = summarize_latents([redundant])
        self.assertEqual(result['std_mean'], 1.)
        self.assertAlmostEqual(result['effective_rank'], 1., places=10)

    def test_between_episode_variation_does_not_hide_stationary_episodes(self):
        a = probe([[[-1., 2.]]] * 3)
        b = probe([[[1., 2.]]] * 3)
        result = summarize_latents([a, b])
        self.assertEqual(result['std_mean'], .5)
        self.assertEqual(result['within_episode_std'], 0.)
        self.assertEqual(result['inactive_fraction'], .5)
        self.assertEqual(result['sample_count'], 6)
        self.assertEqual(result['episode_count'], 2)

    def test_missing_evidence_and_sampling_limit(self):
        for samples in ([], [probe([[[3., 4.]]])]):
            with self.subTest(samples=len(samples)):
                result = summarize_latents(samples)
                for key in ('std_mean', 'within_episode_std', 'inactive_fraction',
                            'effective_rank', 'max_rank', 'changed_rms', 'unchanged_rms'):
                    self.assertIsNone(result[key], key)
        bounded = probe([[[0.]]] * 8 + [[[1000.]]])
        result = summarize_latents([bounded])
        self.assertEqual(result['sample_count'], 8)
        self.assertEqual(result['std_mean'], 0.)
        # CPU snapshots own their storage and retain no autograd graph.
        original = state([[3.]])
        original.z.requires_grad_(True)
        saved = LatentProbe(1)
        saved.sample(original)
        with torch.no_grad():
            original.z.add_(10)
        self.assertEqual(saved.samples[0][1].item(), 3.)
        self.assertFalse(saved.samples[0][1].requires_grad)
        # The expensive rank calculation stays bounded even with more episodes;
        # coordinate diversity still uses every retained real observation.
        many = [probe([[[float(i), float(i % 2)]]] * 8) for i in range(9)]
        result = summarize_latents(many)
        self.assertEqual(result['sample_count'], 72)
        self.assertEqual(result['rank_sample_count'], 64)
        self.assertEqual(result['max_rank'], 2)

    def test_changed_and_static_patches_use_real_pixels_including_padded_edges(self):
        before = np.zeros((3, 3), dtype=np.uint8)
        after = before.copy()
        after[2, 2] = 1
        previous = state([[0., 0.]] * 4, (3, 3))
        current = state([[0., 0.], [0., 0.], [0., 0.], [3., 4.]], (3, 3))
        measured = LatentProbe(2)
        measured.transition(previous, current, before, after)
        result = summarize_latents([measured])
        self.assertAlmostEqual(result['changed_rms'], (25 / 2) ** .5)
        self.assertEqual(result['unchanged_rms'], 0.)
        self.assertEqual((result['changed_regions'], result['unchanged_regions']), (1, 3))
        self.assertEqual(result['transitions'], 1)
        stationary = LatentProbe(2)
        stationary.transition(previous, current, before, before)
        result = summarize_latents([stationary])
        self.assertIsNone(result['changed_rms'])
        self.assertAlmostEqual(result['unchanged_rms'], (25 / 8) ** .5)
        # A real change with zero latent response is measured zero, not missing.
        insensitive = LatentProbe(2)
        insensitive.transition(previous, previous, before, after)
        self.assertEqual(summarize_latents([insensitive])['changed_rms'], 0.)

    def test_incompatible_transitions_are_excluded_from_sensitivity(self):
        measured = LatentProbe(1)
        measured.transition(state([[1.]]), state([[2.], [3.]], (1, 2)),
                            np.zeros((1, 1)), np.zeros((1, 2)))
        result = summarize_latents([measured])
        self.assertEqual(result['skipped_transitions'], 1)
        self.assertEqual(result['transitions'], 0)
        self.assertIsNone(result['changed_rms'])
        self.assertIsNone(result['unchanged_rms'])


if __name__ == '__main__':
    unittest.main()
