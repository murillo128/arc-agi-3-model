"""Independent two-step and gradient contracts, using small CPU modules."""
from dataclasses import replace
import unittest
import random
from unittest.mock import patch

import numpy as np
import torch

from model.config import NetworkConfig, TrainConfig
from model.data import Action, Episode, Observation, Replay
from model.learning import Engine, Learner, evaluate_rollouts
from model.networks import WorldModel
from torch.utils.checkpoint import checkpoint


torch.set_num_threads(2)


def observation(value, shape=(4, 4), **changes):
    return replace(Observation(np.full((1, *shape), value, dtype=np.uint8),
                               (1, 2), 'NOT_FINISHED', 0, 2), **changes)


def episode():
    return Episode([observation(1), observation(3), observation(9)], [Action(1), Action(2)])


class RolloutTests(unittest.TestCase):
    def test_recomputed_history_preserves_joint_frozen_and_burn_in_gradients(self):
        data = Episode([observation(i % 12 + 1) for i in range(9)], [Action(1)] * 8)
        for frozen in (False, True):
            for burn_in in (0, 2):
                with self.subTest(frozen=frozen, burn_in=burn_in):
                    config = TrainConfig(freeze_representation=frozen, burn_in_steps=burn_in,
                                         rollout_weight=.5, loss_steps=2)
                    models = [WorldModel(NetworkConfig.small()) for _ in range(2)]
                    models[1].load_state_dict(models[0].state_dict())
                    results = []
                    for model, recompute in zip(models, (False, True)):
                        engine = Engine(model, config)
                        def checkpoint_or_reference(function, *args, **kwargs):
                            if function == engine.advance and not recompute:
                                return function(*args)
                            return checkpoint(function, *args, **kwargs)
                        torch.manual_seed(91)
                        with patch('model.learning.activation_checkpoint', side_effect=checkpoint_or_reference):
                            results.append(Learner(engine).update([data]))
                    self.assertAlmostEqual(results[0]['loss'], results[1]['loss'], places=5)
                    for (name, expected), actual in zip(models[0].named_parameters(), models[1].parameters()):
                        self.assertEqual(expected.grad is None, actual.grad is None, name)
                        if expected.grad is not None:
                            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-4, msg=name)

    def test_endpoint_loss_uses_predicted_state_and_memory_with_unbroken_gradient(self):
        model = WorldModel(NetworkConfig.small())
        config = TrainConfig(rollout_weight=.5, loss_steps=1, freeze_representation=True,
                             sigreg_weight=0, variance_weight=0)
        engine = Engine(model, config)
        # Independent scalar recurrence z' = w*z + h + action1,
        # h' = h + w*z. At w=2, z0=1: z1=3, h1=2, z2=8.
        # Real z1=4, z2=9 (first prediction is deliberately inaccurate).
        # Rollout derivative is 2*(8-9)*(2*w+2)=-12.
        # One-step endpoint uses real z1 and has derivative 2*(10-9)*5=10.
        with torch.no_grad():
            model.dynamics.delta.bias[0] = 2.
        weight = model.dynamics.delta.bias
        captured = []
        weight.register_hook(lambda grad: captured.append(float(grad[0])))

        def encode(observations, **kwargs):
            return [torch.full((1, 16), float(obs.final[0, 0])) for obs in observations]

        def advance(z, h, action):
            return h + weight[0] * z.mean((1, 2))[:, None], action

        def predict(z, h, action, *geometry):
            following = weight[0] * z + h[:, :1, None] + action[:, 1, None, None]
            return following, advance(z, h, action)[0], following - z

        engine.encode, engine.advance, engine.predict = encode, advance, predict
        data = Episode([observation(1), observation(4), observation(9)], [Action(1), Action(2)])
        zero_visual = lambda logits, *args: logits.sum((1, 2, 3)) * 0
        with patch('model.learning.visual_delta_loss', side_effect=zero_visual), \
                patch('model.learning.soft_delta_error', side_effect=zero_visual):
            result = Learner(engine).update([data])
        self.assertEqual(result['samples'], 1)
        self.assertEqual(result['rollout_samples'], 1)
        self.assertAlmostEqual(result['rollout_mse'], 1.)
        self.assertAlmostEqual(result['loss'], 1.5)
        self.assertAlmostEqual(sum(captured), 4., places=5)

    def test_missing_frames_and_geometry_changes_cannot_be_rollout_targets(self):
        config = TrainConfig(rollout_weight=.5, loss_steps=3, sigreg_weight=0, variance_weight=0)
        engine = Engine(WorldModel(NetworkConfig.small()), config)
        learner = Learner(engine)
        for middle in (replace(observation(3), frames=np.empty((0, 0, 0), dtype=np.uint8)),
                       observation(3, shape=(8, 4))):
            data = Episode([observation(1), middle, observation(4), observation(5)],
                           [Action(1), Action(2), Action(1)])
            result = learner.update([data])
            self.assertEqual(result['samples'], 1)
            self.assertEqual(result['rollout_samples'], 0)

    def test_terminal_endpoint_is_valid_but_episode_cannot_continue_after_it(self):
        learner = Learner(Engine(WorldModel(NetworkConfig.small()), TrainConfig(rollout_weight=.5)))
        data = episode()
        data.observations[-1] = replace(data.observations[-1], state='WIN', levels_completed=1)
        self.assertEqual(learner.update([data])['rollout_samples'], 1)
        data.append(Action(1), observation(4, levels_completed=1))
        with self.assertRaisesRegex(ValueError, 'terminal'):
            learner.update([data])
        data = episode()
        data.observations[-1] = replace(data.observations[-1], levels_completed=1)
        with self.assertRaisesRegex(ValueError, 'level boundary'):
            learner.update([data])

    def test_fragmented_backward_matches_ordinary_autograd(self):
        for frozen in (False, True):
            for readout in ('contextual', 'local'):
                with self.subTest(frozen=frozen, readout=readout):
                    config = TrainConfig(rollout_weight=.5, loss_steps=2, frame_chunk=1,
                                         freeze_representation=frozen, decoder_readout=readout)
                    models = [WorldModel(NetworkConfig.small()) for _ in range(2)]
                    models[1].load_state_dict(models[0].state_dict())
                    results = []
                    for model, mode in zip(models, ('fragmented', 'ordinary')):
                        torch.manual_seed(91)
                        learner = Learner(Engine(model, replace(config, backward_mode=mode)))
                        results.append(learner.update([episode()]))
                    self.assertAlmostEqual(results[0]['loss'], results[1]['loss'], places=5)
                    if readout == 'local' and not frozen:
                        self.assertEqual(results[0]['local_reconstruction_ce'], 0.)
                    for (name, actual), expected in zip(models[0].named_parameters(), models[1].parameters()):
                        self.assertEqual(actual.grad is None, expected.grad is None, name)
                        if actual.grad is not None:
                            torch.testing.assert_close(actual.grad, expected.grad,
                                                       atol=2e-6, rtol=2e-4, msg=name)

    def test_legacy_configuration_has_no_rollout_and_keeps_contextual_readout(self):
        config = TrainConfig()
        self.assertEqual(config.rollout_weight, 0)
        self.assertEqual(config.decoder_readout, 'contextual')
        result = Learner(Engine(WorldModel(NetworkConfig.small()), config)).update([episode()])
        self.assertEqual(result['rollout_samples'], 0)
        self.assertEqual(result['rollout_mse'], 0.)

    def test_frozen_diagnostics_compare_matching_anchors_and_preserve_weights(self):
        model = WorldModel(NetworkConfig.small())
        engine = Engine(model, TrainConfig())
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        model.train()
        data = episode()
        result = evaluate_rollouts(engine, data)
        self.assertTrue(model.training)
        self.assertEqual([result[str(h)]['samples'] for h in (1, 2, 4)], [2, 1, 0])
        with torch.no_grad():
            start, end = engine.encode([data.observations[0], data.observations[-1]])
            expected = float((start - end).square().mean())
        self.assertAlmostEqual(result['2']['identity_latent_mse'], expected, places=6)
        self.assertIsNone(result['4']['latent_mse'])
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
            self.assertIsNone(parameter.grad)
        self.assertEqual(set(result['2']['readouts']), {'contextual', 'local'})

    def test_frozen_latent_cache_reuses_only_encoder_outputs_and_invalidates(self):
        model = WorldModel(NetworkConfig.small())
        engine = Engine(model, TrainConfig(freeze_representation=True, cache_frozen_latents=True))
        learner = Learner(engine)
        data = episode()
        replay = Replay(2)
        replay.add(data)
        sampled = replay.sample(random.Random(0), 1, 2)
        learner.update(sampled)
        self.assertEqual(len(data.latent_cache.values), 3)
        cached = data.latent_cache.values[0]
        self.assertFalse(cached.requires_grad)
        learner.update(replay.sample(random.Random(0), 1, 2))
        self.assertIs(cached, data.latent_cache.values[0])
        # Checkpoint copies/in-place weight edits are detected by the encoder
        # parameter versions; changing dynamics did not invalidate z above.
        with torch.no_grad():
            next(model.encoder.parameters()).add_(.01)
        engine.encode_frozen_episode(data)
        self.assertIsNot(cached, data.latent_cache.values[0])
        with torch.no_grad():
            direct = engine.encode(data.observations)
        for actual, expected in zip(data.latent_cache.values, direct):
            torch.testing.assert_close(actual, expected)
        learner.set_representation_frozen(False)
        self.assertEqual(data.latent_cache.values, [])
        learner.update([data])
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() for p in model.encoder.parameters()))

    def test_cache_byte_budget_yields_to_real_inputs(self):
        data = Episode([observation(1)], [])
        data.prepare('cpu', 4)
        initial_bytes = data.device_bytes
        data.append(Action(1), observation(2))
        data.prepare('cpu', 4)
        full_bytes = data.device_bytes
        replay = Replay(1, full_bytes)
        replay.add(data)
        engine = Engine(WorldModel(NetworkConfig.small()),
                        TrainConfig(freeze_representation=True, cache_frozen_latents=True))
        engine.encode_frozen_episode(data)
        self.assertEqual(data.latent_cache.values, [])  # no room beyond real inputs
        data = Episode([observation(1)], [])
        replay = Replay(1, full_bytes)
        replay.prepare(data, 'cpu', 4, torch.float32)
        engine.encode_frozen_episode(data)
        self.assertGreater(data.device_bytes, initial_bytes)
        data.append(Action(1), observation(2))
        replay.prepare(data, 'cpu', 4, torch.float32)
        self.assertEqual(data.device_bytes, full_bytes)
        self.assertEqual(data.latent_cache.values, [])

    def test_burn_in_preserves_endpoint_and_rollout_counts(self):
        data = Episode([observation(i % 12 + 1) for i in range(9)], [Action(1)] * 8)
        config = TrainConfig(burn_in_steps=2, rollout_weight=.5, loss_steps=1)
        model = WorldModel(NetworkConfig.small())
        engine = Engine(model, config)
        encoded = []
        original = engine.encode
        def traced(observations, **kwargs):
            encoded.extend(float(obs.final[0, 0]) for obs in observations)
            return original(observations, **kwargs)
        engine.encode = traced
        result = Learner(engine).update([data])
        self.assertEqual(result['samples'], 1)
        self.assertEqual(result['rollout_samples'], 1)
        # Last two actions need z6; warmup is z4,z5, not the entire z0 prefix.
        self.assertTrue(encoded)
        self.assertGreaterEqual(min(encoded), 5.)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() for p in model.dynamics.parameters()))


if __name__ == '__main__':
    unittest.main()
