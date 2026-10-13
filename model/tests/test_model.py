"""Numerical, SDK and persistence boundaries; no browser or trained artifacts."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model import checkpoint
from model.config import NetworkConfig, TrainConfig
from model.data import Action, Episode, Observation, Replay, frame_features
from model.environment import Factory
from model.learning import Engine, Learner, delta_log_probabilities, visual_delta_loss
from model.metrics import Metrics, delta_counts
from model.networks import WorldModel, sigreg, variance_floor
from model.runner import main, validation

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
        for chunk in (1, 4):
            model = WorldModel(NetworkConfig.small())
            model.load_state_dict(self.model.state_dict())
            engine = Engine(model, replace(self.config, frame_chunk=chunk))
            # Hold encoder batching fixed: this tests the decoder VJP bridge,
            # not FP32 batch-GEMM rounding amplified by the variance hinge.
            engine.encode = lambda observations, training=False: model.encode(
                observations, engine.spatial, chunk=4, checkpoint_blocks=training)
            learner = Learner(engine)
            torch.manual_seed(51)
            learner.update([episode()])
            gradients.append({n: p.grad.clone() for n, p in model.named_parameters()})
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=2e-4, atol=1e-6,
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
        config = replace(TrainConfig(), validation_steps=2, validation_episodes=1)
        engine = Engine(model, config)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with redirect_stdout(io.StringIO()):
            result = validation(engine, factory, [None], 0)
        self.assertEqual(result['None']['transitions'], 2)
        self.assertFalse(result['None']['passes'])
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_cli_resume_advances_counters_with_empty_replay(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            common = ['--device', 'cpu', '--no-compile', '--output', directory]
            main(['train', '--profile', 'small', '--max-steps', '4', '--train-every', '4',
                  '--min-observations', '4', *common])
            path = Path(directory) / 'latest.pt'
            main(['train', '--checkpoint', str(path), '--max-steps', '8', *common])
            saved = checkpoint.load(path)
            self.assertEqual(saved['progress']['actions'], 8)
            self.assertEqual(saved['progress']['updates'], 2)
            self.assertEqual(saved['progress']['episodes'], 2)
            self.assertIn('empty RAM replay', output.getvalue())


if __name__ == '__main__':
    unittest.main()
