"""Numerical, SDK and persistence boundaries; no browser or trained artifacts."""
from contextlib import nullcontext, redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import random
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch

from model import checkpoint
from model.config import NetworkConfig, TrainConfig
from model.data import Action, Episode, Observation, Replay, frame_features
from model.environment import Factory
from model.learning import Engine, Learner, delta_log_probabilities, visual_delta_loss, soft_delta_error
from model.metrics import Metrics, VisualQuality, delta_counts
from model.networks import Block, SpatialDynamics, WorldModel, sigreg, variance_floor
from model.runner import main, train, validation

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(2)


def observation(value, shape=(5, 7)):
    frames = np.zeros((2, *shape), dtype=np.uint8)
    frames[0, 1, 1] = value
    frames[1, 1, 2] = value
    return Observation(frames, (1, 2, 3, 4), 'NOT_FINISHED', 0, 1)


def episode(length=4):
    return Episode([observation(i % 15 + 1) for i in range(length + 1)], [Action(4)] * length)


class NumericalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = WorldModel(NetworkConfig.small())
        self.config = replace(TrainConfig(), min_observations=4, loss_steps=4, frame_chunk=2)
        self.engine = Engine(self.model, self.config)

    def test_scaled_transformer_residual_preserves_identity_and_input_gradient(self):
        block = Block(4, 1, scale=.1)
        # Replace nonlinear branches with independent constant-output modules.
        # The exact oracle is x + .1*2 + .1*3, with an identity input Jacobian.
        class Constant(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value
            def forward(self, x, mask=None):
                return torch.full_like(x, self.value)
        block.attention = Constant(2.)
        block.down = Constant(3.)
        block.up = torch.nn.Identity()
        x = torch.randn(1, 2, 4, requires_grad=True)
        result = block(x, torch.ones(1, 2, dtype=torch.bool))
        torch.testing.assert_close(result, x + .5)
        result.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_batched_animation_recurrence_matches_independent_observations(self):
        independent = WorldModel(NetworkConfig.small())
        independent.load_state_dict(self.model.state_dict())
        obs = [observation(1), observation(3), observation(4, (9, 11)),
               replace(observation(5), frames=observation(5).frames[:1])]
        batched = self.model.encode(obs, self.model.encoder.spatial, chunk=2)
        reference = [independent.encoder.aggregate(independent.encoder.spatial(
            frame_features(o, 'cpu', 4))) for o in obs]
        for actual, expected in zip(batched, reference):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        for states in (batched, reference):
            sum((z * torch.arange(z.numel()).reshape_as(z)).mean() for z in states).backward()
        for actual, expected in zip(self.model.encoder.parameters(), independent.encoder.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-5, rtol=1e-4)

    def test_pixel_latent_readout_preserves_native_pixel_coordinates(self):
        config = replace(NetworkConfig.small(), latent=64, patch=2, pixel_latent=True)
        model = WorldModel(config)
        class ReadColors(torch.nn.Module):
            def forward(self, pixels):
                return pixels[..., :16]
        model.decoder.color = ReadColors()
        # Independent space-to-depth oracle on a padded rectangular canvas.
        board = torch.arange(24).reshape(4, 6) % 16
        z = torch.nn.functional.one_hot(board, 16).float().reshape(2, 2, 3, 2, 16)
        z = z.permute(0, 2, 1, 3, 4).reshape(1, 6, 64)
        actual = model.decoder(z, 3, 5, local=True).argmax(-1)[0]
        torch.testing.assert_close(actual, board[:3, :5])

    def test_pixel_latent_joint_learning_and_checkpoint_restore(self):
        config = replace(NetworkConfig.small(), latent=64, pixel_latent=True,
                         broadcast_conditioning=True, local_dynamics=True, residual_scale=.1, position_scale=.1)
        model = WorldModel(config)
        learner = Learner(Engine(model, self.config))
        result = learner.update([episode(2), episode(3)])
        result = learner.update([episode(2), episode(3)])  # unlock zero-initialized local head
        self.assertEqual(result['samples'], 5)
        for name in ('encoder.pixel_embedding.weight', 'dynamics.action_token.weight',
                     'decoder.context.weight', 'decoder.color.weight', 'dynamics.local.input.weight',
                     'dynamics.local.condition.weight', 'dynamics.local.output.weight'):
            self.assertGreater(float(dict(model.named_parameters())[name].grad.abs().sum()), 0.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            checkpoint.save(path, learner, {}, {}, random.Random(1))
            saved = checkpoint.load(path)
            restored = WorldModel(NetworkConfig(**saved['network']))
            restored.load_state_dict(saved['weights'])
            self.assertEqual(restored.config, config)

    def test_local_dynamics_moves_latents_across_patch_boundaries(self):
        config = replace(NetworkConfig.small(), latent=64, pixel_latent=True, local_dynamics=True)
        local = SpatialDynamics(config)
        channels = 4
        class NoCondition(torch.nn.Module):
            def forward(self, x):
                return x.new_zeros(x.shape[0], channels)
        local.input = torch.nn.Identity()
        local.condition = NoCondition()
        local.blocks = torch.nn.ModuleList()
        local.output = torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        with torch.no_grad():
            local.output.weight.zero_()
            for channel in range(channels):
                local.output.weight[channel, channel, 1, 0] = 1
        board = torch.arange(8 * 8 * channels).reshape(8, 8, channels).float()
        def pack(value):
            return torch.stack([value[r:r+4, c:c+4].reshape(-1)
                                for r in (0,4) for c in (0,4)])[None]
        expected = torch.zeros_like(board)
        expected[:, 1:] = board[:, :-1]
        actual = local(pack(board), torch.zeros(1,config.memory),
                       torch.zeros(1,config.action_embedding), 5, 7)
        torch.testing.assert_close(actual, pack(expected), rtol=0, atol=0)

    def test_first_transition_trains_without_waiting_for_regularization(self):
        engine = Engine(self.model, replace(self.config, min_observations=64, loss_steps=1))
        result = Learner(engine).update([episode(1)])
        self.assertIsNotNone(result)
        self.assertEqual(result['samples'], 1)
        self.assertEqual(result['sigreg'], 0.)
        self.assertEqual(result['variance_floor'], 0.)
        for component in (self.model.encoder, self.model.dynamics, self.model.decoder):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in component.parameters()
                                   if p.grad is not None), 0.)

    def test_prepared_inputs_reuse_storage_without_changing_gradients(self):
        ep = episode()
        ep.prepare('cpu', 4)
        original = ep.prepared[0][1]
        ep.prepare('cpu', 4)
        self.assertIs(ep.prepared[0][1], original)
        self.assertFalse(any(t.requires_grad for item in ep.prepared for t in item if t is not None))
        torch.testing.assert_close(ep.prepared[1][0], torch.tensor(ep.observations[1].frames, dtype=torch.long))
        self.assertEqual(float(ep.prepared[1][2][4]), 1.)
        self.assertEqual(float(original[0, 16 + 17, 1, 1]), 1.)
        other = WorldModel(NetworkConfig.small())
        other.load_state_dict(self.model.state_dict())
        for model, data in ((self.model, ep), (other, episode())):
            torch.manual_seed(91)
            Learner(Engine(model, self.config)).update([data])
        for (name, a), b in zip(self.model.named_parameters(), other.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
        before = len(ep.prepared)
        ep.append(Action(2), observation(9))
        with self.assertRaises(MemoryError):
            ep.prepare('cpu', 4, byte_limit=ep.device_bytes)
        self.assertEqual(len(ep.prepared), before)

    def test_optimizer_precision_change_requires_explicit_reset(self):
        learner = Learner(self.engine)
        learner.update([episode()])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            rng = random.Random(1)
            checkpoint.save(path, learner, {}, {}, rng)
            payload = checkpoint.load(path)
            self.assertEqual(payload['runtime'], {'fp8': False, 'optimizer_bits': 32})
            payload['runtime']['optimizer_bits'] = 8
            restored = Learner(Engine(WorldModel(NetworkConfig.small()), self.config))
            with self.assertRaisesRegex(ValueError, 'precision changed'):
                checkpoint.restore(payload, restored, rng)
            checkpoint.restore(payload, restored, rng, reset_optimizer=True)
            self.assertTrue(all(not opt.state for opt in restored.optimizers))
            for a, b in zip(learner.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_delta_denominator_rejects_background_shortcut(self):
        before = np.array([[0, 1], [2, 3]])
        actual = np.array([[0, 4], [5, 3]])
        predicted = np.array([[0, 4], [2, 0]])
        self.assertEqual(delta_counts(before, actual, predicted), (2, 3, 2))
        self.assertEqual(delta_counts(before, before, before), (0, 0, 0))
        metrics = Metrics()
        metrics.add(before, actual, predicted, np.array([1.]), np.array([3.]), np.array([1.]))
        self.assertAlmostEqual(metrics.report()['delta_error'], 2 / 3)
        self.assertEqual(metrics.report()['latent_delta_mse'], 1.)
        self.assertFalse(metrics.passes(self.config))

    def test_animation_channels_and_native_padding(self):
        obs = observation(7)
        features = frame_features(obs, 'cpu', 4)
        self.assertEqual(tuple(features.shape), (2, 35, 8, 8))
        self.assertEqual(features[0, 16 + 17, 1, 1], 1.)
        self.assertEqual(features[1, 16 + 0, 1, 1], 1.)  # cleared pixel
        self.assertEqual(features[1, 16 + 7, 1, 2], 1.)  # new color
        self.assertEqual(features[1, 16 + 16, 0, 0], 1.)  # KEEP
        self.assertEqual(features[:, :, 5:, :].sum(), 0.)

    def test_delta_error_decomposition_counts_distinct_failures(self):
        before = np.zeros((1, 5), dtype=np.uint8)
        actual = np.array([[0, 1, 2, 3, 0]])
        predicted = np.array([[0, 0, 4, 3, 5]])
        metrics = Metrics()
        for _ in range(2):
            metrics.add(before, actual, predicted, np.zeros(1), np.zeros(1), np.zeros(1))
        result = metrics.report()
        for key in ('missed_changes', 'extra_changes', 'wrong_replacement_colors', 'correct_changes'):
            self.assertEqual(result[key], 2)
        self.assertEqual(result['delta_error'], .75)
        self.assertEqual(metrics.wrong, 6)
        self.assertEqual(metrics.union, 8)

    def test_gate_requires_coverage_and_noncollapsed_latents(self):
        metrics = Metrics()
        before, after = np.zeros((2, 2)), np.ones((2, 2))
        for _ in range(128):
            metrics.add(before, after, after, np.array([-1.]), np.array([1.]), np.array([2.]))
        self.assertFalse(metrics.passes(self.config))  # perfect prediction is insufficient
        metrics.observe_latent([-1.])
        metrics.observe_latent([1.])
        self.assertTrue(metrics.passes(self.config))
        metrics.skipped = 1
        self.assertFalse(metrics.passes(self.config))
        metrics.skipped = 0
        metrics.changed_transitions = 0
        self.assertFalse(metrics.passes(self.config))

    def test_residual_identity_action_conditioning_and_readout(self):
        z = torch.randn(2, 4, 16, requires_grad=True)
        h = torch.zeros(2, 32)
        with torch.no_grad():
            action_states, _, _ = self.model.dynamics(z[:1].expand(2, -1, -1), h,
                self.engine.actions([Action(1), Action(4)]))
        self.assertFalse(torch.allclose(action_states[0], action_states[1], atol=1e-8, rtol=0))
        with torch.no_grad():
            self.model.dynamics.delta.weight.zero_()
            self.model.dynamics.delta.bias.zero_()
        z_hat, _, delta = self.model.dynamics(z, h, self.engine.actions([Action(1), Action(4)]))
        torch.testing.assert_close(z_hat, z, rtol=0, atol=0)
        self.assertEqual(delta.count_nonzero(), 0)
        logits = self.model.decoder(z_hat, 5, 7)
        self.assertEqual(tuple(logits.shape), (2, 5, 7, 16))
        logits.square().mean().backward()
        self.assertGreater(float(z.grad.abs().sum()), 0.)
        self.assertGreater(float(self.model.dynamics.delta.weight.grad.abs().sum()), 0.)
        self.assertTrue(any(p.grad is not None and p.grad.count_nonzero() for p in self.model.decoder.parameters()))

    def test_update_changes_all_three_networks_and_checkpoint_resumes(self):
        learner = Learner(self.engine)
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        result = learner.update([episode()])
        self.assertTrue(np.isfinite(list(result.values())).all())
        for prefix in ('encoder.', 'dynamics.', 'decoder.'):
            self.assertTrue(any(not torch.equal(p, before[n]) for n, p in self.model.named_parameters() if n.startswith(prefix)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            rng = random.Random(91)
            checkpoint.save(path, learner, {'actions': 4}, {'game': 'sc01-v1'}, rng)
            payload = checkpoint.load(path)
            restored = Learner(Engine(WorldModel(NetworkConfig.small()), self.config))
            checkpoint.restore(payload, restored, rng)
            for a, b in zip(self.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

            self.assertEqual(payload['progress']['actions'], 4)
            self.assertEqual(rng.random(), random.Random(91).random())
            self.assertNotIn('replay', payload)
            self.assertEqual(len(restored.optimizers[0].state), len(learner.optimizers[0].state))
            learner.update([episode()])
            checkpoint.restore(payload, restored, rng)  # restore random projections too
            restored.update([episode()])
            for a, b in zip(self.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_soft_delta_union_error_and_false_change_gradients(self):
        before = torch.tensor([[[0, 1, 2]]])
        actual = torch.tensor([[[0, 3, 4]]])
        probability = torch.empty(1, 1, 3, 16)
        for i, (color, correct) in enumerate(((0, .8), (3, .6), (4, .7))):
            probability[0, 0, i].fill_((1 - correct) / 15)
            probability[0, 0, i, color] = correct
        logits = probability.log().requires_grad_(True)
        loss = soft_delta_error(logits, before, actual)
        # One static cell contributes .2 false changes; two real changed
        # cells contribute .4 + .3 wrong colors. Expected union = 2 + .2.
        self.assertAlmostEqual(float(loss.detach()[0]), .9 / 2.2, places=6)
        loss.sum().backward()
        for i, color in enumerate((0, 3, 4)):
            self.assertLess(float(logits.grad[0, 0, i, color]), 0.)
        self.assertAlmostEqual(float(soft_delta_error(logits.detach(), actual, actual)[0]), .3, places=6)
        # Hard limit: one false change, one missed change, one right change.
        hard = torch.full_like(logits, -100.)
        for i, color in enumerate((5, 1, 4)):
            hard[0, 0, i, color] = 100.
        self.assertAlmostEqual(float(soft_delta_error(hard, before, actual)[0]), 2 / 3, places=6)

    def test_local_readout_preserves_patch_locality(self):
        z = torch.randn(1, 4, 16, requires_grad=True)
        decoded = self.model.decoder(z, 5, 7, local=True)
        perturbed = z.detach().clone()
        perturbed[:, 1] += torch.randn(16)
        after = self.model.decoder(perturbed, 5, 7, local=True)
        # Row-major patch 1 covers rows 0..3, columns 4..6 after native crop.
        torch.testing.assert_close(decoded[:, :, :4], after[:, :, :4], rtol=0, atol=0)
        torch.testing.assert_close(decoded[:, 4:], after[:, 4:], rtol=0, atol=0)
        self.assertGreater(float((decoded[:, :4, 4:] - after[:, :4, 4:]).detach().abs().sum()), 0.)
        decoded[:, :4, 4:].square().sum().backward()
        self.assertGreater(float(z.grad[:, 1].abs().sum()), 0.)
        self.assertEqual(float(z.grad[:, [0, 2, 3]].abs().sum()), 0.)

    def test_visual_targets_train_encoder_and_predictor(self):
        # Isolate supervision from real pixels: neither latent alignment nor
        # distribution regularization may account for these gradients.
        with patch('model.learning.F.mse_loss', side_effect=lambda x, y: x.sum() * 0), \
                patch('model.learning.sigreg', side_effect=lambda z, slices: z.sum() * 0), \
                patch('model.learning.variance_floor', side_effect=lambda z: z.sum() * 0):
            Learner(self.engine).update([episode()])
        for component in (self.model.encoder, self.model.dynamics, self.model.decoder):
            magnitude = sum(float(p.grad.abs().sum()) for p in component.parameters() if p.grad is not None)
            self.assertGreater(magnitude, 1e-8)

    def test_delta_probability_labels_and_balanced_nll(self):
        # Each pixel has independently specified probabilities: color 0=.25,
        # color 1=.50, remaining colors share .25. Three KEEP, one replacement.
        probabilities = torch.tensor([.25, .5] + [.25 / 14] * 14)
        logits = probabilities.log().expand(1, 2, 2, 16).clone().requires_grad_(True)
        before = torch.tensor([[[0, 1], [0, 1]]])
        actual = torch.tensor([[[0, 0], [0, 1]]])
        delta = delta_log_probabilities(logits, before).exp()
        torch.testing.assert_close(delta.sum(-1), torch.ones(1, 2, 2))
        torch.testing.assert_close(delta[..., 16], torch.tensor([[[.25, .5], [.25, .5]]]))
        self.assertEqual(float(delta.gather(-1, before[..., None]).sum()), 0.)
        expected = .5 * (-np.log(.25) + (-2 * np.log(.25) - np.log(.5)) / 3)
        loss = visual_delta_loss(logits, before, actual)
        self.assertAlmostEqual(float(loss.detach()[0]), expected, places=6)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        # No-change and all-change boundaries use ordinary mean NLL.
        for target in (before, 1 - before):
            expected = torch.nn.functional.cross_entropy(logits.permute(0, 3, 1, 2), target)
            torch.testing.assert_close(visual_delta_loss(logits, before, target)[0], expected)

    def test_visual_nll_weights_each_pixel_equally_within_changed_static_groups(self):
        before = torch.tensor([[[0, 0, 0, 1, 2]]])
        actual = torch.tensor([[[0, 0, 0, 1, 3]]])
        probabilities = torch.tensor([.25, .5, .025, .125] + [.1 / 12] * 12)
        logits = probabilities.log().expand(1, 1, 5, 16).clone().requires_grad_(True)
        # One changed pixel gets half the group mass; each of four static pixels
        # gets one eighth, regardless of whether its real color is rare.
        expected = .5 * (-np.log(.125) + (-3 * np.log(.25) - np.log(.5)) / 4)
        self.assertAlmostEqual(float(visual_delta_loss(logits, before, actual).detach()[0]), expected, places=6)
        uniform = torch.zeros(1, 1, 5, 16, requires_grad=True)
        visual_delta_loss(uniform, before, actual).sum().backward()
        for index, color in enumerate((0, 0, 0, 1)):
            self.assertAlmostEqual(float(uniform.grad[0, 0, index, color]), -15 / 128, places=6)
        self.assertAlmostEqual(float(uniform.grad[0, 0, 4, 3]), -15 / 32, places=6)
        # A single nonempty group uses the ordinary mean over all five pixels.
        for reference in (actual, torch.full_like(actual, 15)):
            expected = (-3 * np.log(.25) - np.log(.5) - np.log(.125)) / 5
            self.assertAlmostEqual(float(visual_delta_loss(logits, reference, actual).detach()[0]), expected, places=6)

    def test_visual_quality_exposes_static_rare_color_failure(self):
        quality = VisualQuality()
        before = np.array([[0, 0, 0, 1, 2]])
        actual = np.array([[0, 0, 0, 1, 3]])
        quality.add(before, actual, np.array([[0, 0, 0, 0, 3]]))
        result = quality.report()
        self.assertAlmostEqual(result['color_error'], 1 / 3)
        self.assertEqual(result['static_color_error'], .5)
        self.assertEqual(result['changed_color_error'], 0.)
        self.assertEqual(result['by_color']['1'], {'pixels': 1, 'error': 1.})

    def test_false_change_cost_preserves_missed_change_gradients(self):
        before = torch.zeros(1, 1, 4, dtype=torch.long)
        actual = torch.tensor([[[0, 0, 0, 1]]])
        probabilities = torch.tensor([.5, .25] + [.25 / 14] * 14)
        for target, expected, weights in (
            (actual, .5 * (-2 * np.log(.5) - np.log(.25)), (1 / 3, 1 / 3, 1 / 3, .5)),
            (before, -2 * np.log(.5), (.5,) * 4),
            (torch.ones_like(before), -np.log(.25), (.25,) * 4),
        ):
            with self.subTest(target=target.tolist()):
                logits = probabilities.log().expand(1, 1, 4, 16).clone().requires_grad_(True)
                loss = visual_delta_loss(logits, before, target, false_change_weight=2.)
                self.assertAlmostEqual(float(loss.detach()[0]), expected, places=6)
                loss.sum().backward()
                # Independent CE derivative: coefficient * (p - one_hot(label)).
                for index, (color, weight) in enumerate(zip(target.flatten().tolist(), weights)):
                    expected_gradient = weight * probabilities.clone()
                    expected_gradient[color] -= weight
                    torch.testing.assert_close(logits.grad[0, 0, index], expected_gradient)
        for weight in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                TrainConfig(false_change_weight=weight)

    def test_false_change_weight_only_changes_predictive_objective(self):
        candidate = WorldModel(NetworkConfig.small())
        candidate.load_state_dict(self.model.state_dict())
        weighted = Engine(candidate, replace(self.config, false_change_weight=2.))
        baseline = Learner(self.engine).update([episode()])
        result = Learner(weighted).update([episode()])
        self.assertGreater(result['decoder_ce'], baseline['decoder_ce'])
        for key in ('reconstruction_ce', 'local_reconstruction_ce', 'decoder_overlap',
                    'reconstruction_overlap', 'local_reconstruction_overlap', 'latent_mse'):
            self.assertAlmostEqual(result[key], baseline[key], places=6)

    def test_sigreg_integrates_both_sides_of_even_statistic(self):
        t = np.linspace(-3, 3, 33)
        phi = np.exp(-t ** 2 / 2)
        # For four zero vectors the empirical characteristic function is 1.
        expected = 4 * np.trapezoid((1 - phi) ** 2 * phi, t)
        self.assertAlmostEqual(float(sigreg(torch.zeros(4, 16), 32)), expected, places=6)

    def test_position_codes_cannot_fake_observation_diversity(self):
        positional = torch.arange(64).reshape(1, 4, 16).float().expand(8, -1, -1)
        self.assertAlmostEqual(float(variance_floor(positional)), .9999, places=6)
        metrics = Metrics()
        for _ in range(8):
            metrics.observe_latent(positional[0].numpy(), (8, 8))
            metrics.observe_latent(np.ones((6, 16)), (8, 12))
        self.assertEqual(metrics.report()['latent_std'], 0.)
        # A small real observation difference must receive a restoring gradient.
        z = torch.tensor([-0.001, 0.001]).reshape(2, 1, 1).requires_grad_(True)
        variance_floor(z).backward()
        self.assertGreater(float(z.grad[0]), 0.)
        self.assertLess(float(z.grad[1]), 0.)

    def test_joint_backward_is_independent_of_decoder_chunking(self):
        gradients = []
        for chunk, recompute in ((1, True), (4, True), (4, False)):
            model = WorldModel(NetworkConfig.small())
            model.load_state_dict(self.model.state_dict())
            engine = Engine(model, replace(self.config, frame_chunk=chunk))
            # Hold encoder batching fixed: this tests the decoder VJP bridge,
            # not FP32 batch-GEMM rounding amplified by the variance hinge.
            engine.encode = lambda observations, training=False, prepared=None: model.encode(
                observations, engine.spatial, chunk=4, checkpoint_blocks=training, prepared=prepared)
            learner = Learner(engine)
            torch.manual_seed(51)
            def direct(function, *args, **kwargs):
                kwargs.pop('use_reentrant')
                return function(*args, **kwargs)
            context = nullcontext() if recompute else patch('model.learning.activation_checkpoint', side_effect=direct)
            with context:
                learner.update([episode()])
            gradients.append({n: p.grad.clone() for n, p in model.named_parameters()})
        for other in gradients[1:]:
            for name in gradients[0]:
                torch.testing.assert_close(gradients[0][name], other[name], rtol=2e-4, atol=1e-6,
                                           msg=lambda message: name + ': ' + message)

    def test_short_episodes_can_train_without_crossing_boundaries(self):
        replay = Replay(4)
        a, b = episode(1), episode(1)
        replay.add(a)
        replay.add(b)
        replay.add(a)
        sampled = replay.sample(random.Random(4), 1, 4)
        self.assertEqual(len(sampled), 2)
        self.assertTrue(all(len(e.observations) == 2 and len(e.actions) == 1 for e in sampled))
        self.assertIsNotNone(Learner(self.engine).update(sampled))

    def test_replay_bounds_device_inputs_and_keeps_causal_episode_prefixes(self):
        older, live = episode(4), episode(2)
        older.prepare('cpu', 4)
        live.prepare('cpu', 4)
        replay = Replay(4, byte_limit=older.device_bytes + live.device_bytes)
        replay.add(older)
        replay.add(live)
        # A sampled history starts at its own reset and never contains a later
        # observation than its last action. The live terminal pair is retained.
        live.observations[-1] = replace(live.observations[-1], state='GAME_OVER')
        samples = replay.sample(random.Random(5), 2, 2, latest=live, random_prefix=True)
        self.assertTrue(samples[0].observations[-1].terminal)
        self.assertEqual(samples[0].actions, live.actions)
        self.assertIs(samples[1].observations[0], older.observations[0])
        self.assertEqual(len(samples[1].observations), len(samples[1].actions) + 1)
        self.assertEqual(len(samples[1].prepared), len(samples[1].observations))
        self.assertIs(samples[1].prepared[0][1], older.prepared[0][1])
        # Reserve the next input before allocation by evicting an old episode.
        del samples
        live.append(Action(1), observation(7))
        replay.prepare(live, 'cpu', 4, torch.float32)
        self.assertEqual(len(replay.episodes), 1)
        self.assertIs(replay.episodes[0], live)
        self.assertLessEqual(sum(e.device_bytes for e in replay.episodes), replay.byte_limit)

    def test_replay_prepared_batch_matches_independent_reencoding(self):
        a, b = episode(2), episode(3)
        for ep in (a, b):
            ep.prepare('cpu', 4)
        other = WorldModel(NetworkConfig.small())
        other.load_state_dict(self.model.state_dict())
        results = []
        for model, batch in ((self.model, [a, b]), (other, [episode(2), episode(3)])):
            torch.manual_seed(89)
            results.append(Learner(Engine(model, self.config)).update(batch))
        self.assertEqual(results[0]['samples'], 5)
        self.assertAlmostEqual(results[0]['loss'], results[1]['loss'], places=5)
        for parameter, reference in zip(self.model.parameters(), other.parameters()):
            torch.testing.assert_close(parameter, reference)

    def test_real_memory_rebuild_matches_streaming_inference(self):
        ep = episode()
        state = self.engine.initial(ep.observations[0])
        for action, obs in zip(ep.actions, ep.observations[1:]):
            state = self.engine.observe(state, action, obs)
        rebuilt = self.engine.rebuild(ep)
        torch.testing.assert_close(state.z, rebuilt.z, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(state.h, rebuilt.h, atol=1e-6, rtol=1e-5)

    def test_empty_response_is_not_a_fabricated_visual_target(self):
        empty = Observation(np.empty((0, 0, 0), dtype=np.uint8), (1,), 'NOT_FINISHED', 0, 1)
        initial = self.engine.initial(empty)
        self.assertIsNone(initial.z)
        self.assertIsNone(self.engine.prediction(initial, Action(1), None))
        current = self.engine.initial(observation(1))
        after = self.engine.observe(current, Action(1), empty)
        torch.testing.assert_close(after.z, current.z, rtol=0, atol=0)
        self.assertFalse(torch.equal(after.h, current.h))
        ep = Episode([observation(1), empty, observation(2), observation(3)], [Action(1)] * 3)
        learner = Learner(Engine(self.model, replace(self.config, min_observations=2)))
        result = learner.update([ep])
        self.assertEqual(result['samples'], 1)  # only the final pair has two real boards


class SDKTests(unittest.TestCase):
    def test_statistics_start_after_first_completed_action_on_resume(self):
        import json
        clock = [0.]
        class Environment:
            observation = observation(1)
            def step(self, action):
                return observation(2)
        class Environments:
            def make(self, seed, level):
                return Environment()
        engine = Engine(WorldModel(NetworkConfig.small()),
                        replace(TrainConfig(), max_steps=9805, max_episode_steps=2))
        learner = Learner(engine)
        update = learner.update
        def timed_update(episodes):
            result = update(episodes)
            # Independent clock: warmup is 1100 s, later actions take 2 s.
            clock[0] = 1100. if clock[0] == 0 else clock[0] + 2
            return result
        learner.update = timed_update
        progress = dict(actions=9802, updates=1220, episodes=443, level_index=0,
                        level_actions=9802, passes=0, best_error=1.)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            with patch('model.runner.time.monotonic', side_effect=lambda: clock[0]):
                train(engine, learner, Environments(), {'levels': [1]}, progress,
                      random.Random(1), Path(directory))
            saved = checkpoint.load(Path(directory) / 'latest.pt')
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        points = [r.get('statistics', {}) for r in rows if r['event'] == 'progress']
        self.assertEqual([p.get('actions') for p in points], [0, 1, 2])
        self.assertEqual([p['actions_per_second'] for p in points], [None, .5, .5])
        self.assertEqual([p['elapsed_seconds'] for p in points], [0, 2, 4])
        self.assertEqual([p['updates'] for p in points], [0, 1, 2])
        self.assertEqual([p['episodes'] for p in points], [0, 0, 1])
        self.assertEqual(points[-1]['level_actions'], 2)
        self.assertEqual(rows[-1]['statistics'], points[-1])
        self.assertEqual(saved['progress']['actions'], 9805)
        self.assertEqual(saved['progress']['updates'], 1223)

    def test_every_action_trains_terminal_before_releasing_episode(self):
        class Environment:
            def __init__(self, color):
                self.color, self.steps = color, 0
                self.observation = observation(color)
            def step(self, action):
                self.steps += 1
                return replace(observation(self.color + self.steps),
                               state='GAME_OVER' if self.steps == 2 else 'NOT_FINISHED')
        class Environments:
            def __init__(self):
                self.count = 0
            def make(self, seed, level):
                self.count += 1
                return Environment(self.count)
        config = replace(TrainConfig(), max_steps=5, min_observations=64)
        engine = Engine(WorldModel(NetworkConfig.small()), config)
        learner, seen, refs = Learner(engine), [], []
        update = learner.update
        def record(episodes):
            self.assertEqual(len(episodes), 1)
            ep = episodes[0]
            seen.append((len(ep.actions), ep.observations[-1].terminal,
                         int(ep.observations[0].frames[0, 1, 1])))
            refs.append(weakref.ref(ep))
            return update(episodes)
        learner.update = record
        progress = dict(actions=0, updates=0, episodes=0, level_index=0, level_actions=0, passes=0, best_error=1.)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            train(engine, learner, Environments(), {'levels': [1]}, progress,
                  random.Random(1), Path(directory))
        self.assertEqual(seen, [(1, False, 1), (2, True, 1), (1, False, 2), (2, True, 2), (1, False, 3)])
        self.assertEqual(progress['updates'], 5)
        self.assertTrue(all(ref() is None for ref in refs))

    def test_unlimited_practice_stops_on_sustained_collapse_or_level_pass(self):
        # Only validation verdicts are controlled; the CLI, SDK interaction and
        # checkpoint/final stop path are real. This does not assert convergence.
        for passes, spreads, reason, actions in (
            (False, [.2, .001, .001, .001], 'encoder_collapsed', 4),
            (True, [.2, .2, .2], 'level_converged', 3),
        ):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                reports = [{'1': {'passes': passes, 'delta_error': .001, 'latent_std': std}} for std in spreads]
                output = io.StringIO()
                with redirect_stdout(output), patch('model.runner.validation', side_effect=reports):
                    main(['train', '--profile', 'small', '--device', 'cpu', '--no-compile',
                          '--levels', '1', '2', '--max-steps', '0', '--max-level-steps', '0',
                          '--stop-after-level', '--validate-every', '1', '--output', directory])
                import json
                stopped = json.loads(output.getvalue().splitlines()[-1])
                self.assertEqual(stopped['reason'], reason)
                self.assertEqual(stopped['actions'], actions)
                saved = checkpoint.load(Path(directory) / 'latest.pt')
                self.assertEqual(saved['progress']['actions'], actions)

    def test_isolated_level_fresh_seed_and_native_terminal_frame(self):
        # Independent environment-side reference is used only in this test.
        sys.path.insert(0, str(ROOT / 'tests'))
        from square_cross_reference import reference_path
        factory = Factory('sc01-v1', ROOT / 'games')
        env = factory.make(42, 4)
        self.assertEqual(env.wrapper._game.level_index, 3)
        self.assertEqual(env.observation.win_levels, 1)
        same = factory.make(42, 4)
        different = factory.make(44, 4)
        np.testing.assert_array_equal(env.observation.frames, same.observation.frames)
        self.assertFalse(np.array_equal(env.observation.frames, different.observation.frames))
        path = reference_path(env.wrapper._game.scene)
        for action in path:
            actual = env.step(Action(action))
        self.assertEqual(actual.state, 'WIN')
        self.assertEqual(actual.levels_completed, 1)
        self.assertEqual(env.wrapper._game.level_index, 3)
        self.assertEqual(len(actual.frames), 1)  # no next level scene appended
        self.assertEqual(actual.action_input.id, path[-1])

    def test_other_sdk_game_native_actions_and_frozen_validation(self):
        factory = Factory('si01-v1', ROOT / 'games')
        env = factory.make(7)
        actual = env.step(Action(1))
        self.assertEqual(actual.action_input, Action(1))
        clickable = factory.make(7, 2)
        self.assertIn(6, clickable.observation.available_actions)
        actual = clickable.step(Action(6, {'x': 20, 'y': 20}))
        self.assertEqual(actual.action_input, Action(6, {'x': 20, 'y': 20}))
        model = WorldModel(NetworkConfig.small())
        config = replace(TrainConfig(), validation_steps=2, validation_episodes=1, min_validation_transitions=2)
        engine = Engine(model, config)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        # Even a passing predictor/latent gate must not hide a broken decoder.
        with redirect_stdout(io.StringIO()), patch('model.runner.Metrics.passes', return_value=True):
            result = validation(engine, factory, [None], 0)
        self.assertEqual(result['None']['transitions'], 2)
        self.assertFalse(result['None']['passes'])
        self.assertGreater(result['None']['decoder_delta_error'], config.delta_error)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_cli_resume_advances_counters_with_empty_replay(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            common = ['--device', 'cpu', '--no-compile', '--output', directory]
            main(['train', '--profile', 'small', '--max-steps', '4',
                  '--min-observations', '4', *common])
            path = Path(directory) / 'latest.pt'
            payload = checkpoint.load(path)
            payload['progress'].update(best_error=.25, passes=2, collapse_streak=1)
            torch.save(payload, path)
            main(['train', '--checkpoint', str(path), '--max-steps', '8', *common])
            saved = checkpoint.load(path)
            self.assertEqual(saved['progress']['actions'], 8)
            # One update per action, including after resume with an empty episode.
            self.assertEqual(saved['progress']['updates'], 8)
            self.assertEqual(saved['progress']['episodes'], 2)
            self.assertEqual(saved['progress']['best_error'], .25)
            self.assertEqual(saved['progress']['passes'], 2)
            self.assertEqual(saved['progress']['collapse_streak'], 1)
            self.assertIn('empty device input cache', output.getvalue())


if __name__ == '__main__':
    unittest.main()
