"""Gradient ownership and numerical objectives of the incremental learners."""
from contextlib import nullcontext
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from model.config import NetworkConfig, TrainConfig
from model.data import Action, Episode, Observation
from model.learning import Engine, State, visual_delta_loss
from model.networks import WorldModel
from model.online_learning import OnlinePredictor, OnlineRepresentation
from model.path_planning import real_states
from model.planning import fingerprint


torch.set_num_threads(2)


class OnlineLearningTests(unittest.TestCase):
    def test_delayed_admission_keeps_weights_and_optimizer_fresh_then_learns(self):
        torch.manual_seed(67)
        config = replace(NetworkConfig.small(), latent=64, pixel_latent=True,
                         broadcast_conditioning=True, local_dynamics=True,
                         residual_scale=.1, position_scale=.1)
        engine = Engine(WorldModel(config), replace(TrainConfig(), variance_weight=.1))
        representation = OnlineRepresentation(engine)
        predictor = OnlinePredictor(engine, threshold=None)
        observations = []
        for column in (1, 2, 3):
            frame = np.zeros((1, 8, 8), dtype=np.uint8)
            frame[0, 3, column] = 9
            observations.append(Observation(frame, (1, 2, 3, 4), 'NOT_FINISHED', 0, 1))
        episode = Episode(observations, [Action(4), Action(4)])
        initial = {name: fingerprint(getattr(engine.model, name))
                   for name in ('encoder', 'decoder', 'dynamics')}
        gradients = {name: [] for name in ('action_token.weight', 'local.input.weight',
                     'local.condition.weight', 'local.output.weight', 'delta.weight')}
        handles = [dict(engine.model.dynamics.named_parameters())[name].register_hook(
            lambda grad, name=name: gradients[name].append(float(grad.norm()))) for name in gradients]
        # Inference during warmup must not create updates, decay or Adam moments.
        for _ in range(8):
            state = engine.initial(observations[0])
            engine.prediction(state, Action(4), observations[0].final)
            representation.update(observations[:2])
            self.assertEqual(fingerprint(engine.model.dynamics), initial['dynamics'])
            self.assertFalse(predictor.optimizer.state)
            self.assertEqual(predictor.updates, 0)
            self.assertTrue(all(p.grad is None for p in engine.model.dynamics.parameters()))
            self.assertTrue(all(not values for values in gradients.values()))
        for name in ('encoder', 'decoder'):
            self.assertNotEqual(fingerprint(getattr(engine.model, name)), initial[name])
        fixed = {name: fingerprint(getattr(engine.model, name)) for name in ('encoder', 'decoder')}
        try:
            # First update opens the zero-initialized local output; the second
            # must also propagate into its input and action/history conditioning.
            for step in (1, 2):
                states = real_states(engine, episode)
                result = predictor.update(states[0], Action(4), observations[0].final,
                                          observations[1], states[1].z, 0.)
                self.assertTrue(math.isfinite(result['total']))
                self.assertEqual(predictor.updates, step)
                self.assertTrue(all(float(s['step']) == step for s in predictor.optimizer.state.values()))
                for name in fixed:
                    self.assertEqual(fingerprint(getattr(engine.model, name)), fixed[name])
        finally:
            for handle in handles:
                handle.remove()
        self.assertNotEqual(fingerprint(engine.model.dynamics), initial['dynamics'])
        for name, values in gradients.items():
            self.assertTrue(any(x > 0 and math.isfinite(x) for x in values), name)

    def test_balanced_delta_nll_keeps_changed_gradients_when_false_change_weight_doubles(self):
        before = torch.tensor([[[1, 2, 3, 4]]])
        actual = torch.tensor([[[1, 0, 4, 4]]])  # Includes a replacement by palette index zero.
        probability = torch.empty(1, 1, 4, 16)
        for i, (color, correct) in enumerate(((1, .8), (0, .2), (4, .4), (4, .6))):
            probability[0, 0, i].fill_((1 - correct) / 15)
            probability[0, 0, i, color] = correct
        logits = probability.log().requires_grad_()
        weighted = visual_delta_loss(logits, before, actual, 2.).sum()
        expected = .5 * ((-math.log(.2) - math.log(.4)) / 2
                         + 2 * (-math.log(.8) - math.log(.6)) / 2)
        self.assertAlmostEqual(float(weighted.detach()), expected, places=6)
        grad = torch.autograd.grad(weighted, logits)[0]
        unweighted = torch.autograd.grad(visual_delta_loss(logits, before, actual).sum(), logits)[0]
        torch.testing.assert_close(grad[:, :, 1:3], unweighted[:, :, 1:3])
        torch.testing.assert_close(grad[:, :, [0, 3]], 2 * unweighted[:, :, [0, 3]])
        for i, color in enumerate((1, 0, 4, 4)):
            self.assertLess(float(grad[0, 0, i, color]), 0.)
        # Extra unchanged background must not dilute the changed-pixel NLL.
        indices = [1, 2] + [0, 3] * 100
        padded = visual_delta_loss(logits[:, :, indices], before[:, :, indices], actual[:, :, indices], 2.)
        self.assertAlmostEqual(float(padded.detach()[0]), expected, places=6)

    def test_static_and_all_changed_targets_have_correct_finite_nll_gradients(self):
        for before, actual, expected in ((3, 3, 2 * math.log(16)), (3, 0, math.log(16))):
            with self.subTest(before=before, actual=actual):
                logits = torch.zeros(1, 1, 1, 16, requires_grad=True)
                loss = visual_delta_loss(logits, torch.tensor([[[before]]]), torch.tensor([[[actual]]]), 2.)
                self.assertAlmostEqual(float(loss.detach()[0]), expected, places=6)
                loss.sum().backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertLess(float(logits.grad[0, 0, 0, actual]), 0.)

    def test_online_rollout_has_full_predicted_state_and_memory_gradient(self):
        class Dynamics(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.))

            def forward(self, z, h, action, height, width):
                delta = h[:, :, None] + self.weight
                return z + delta, h + self.weight, delta

        class UniformDecoder(nn.Module):
            def forward(self, z, height, width, **kwargs):
                return z.sum() * 0 + torch.zeros(1, height, width, 16)

        model = nn.Module()
        model.encoder, model.dynamics, model.decoder = nn.Linear(1, 1), Dynamics(), UniformDecoder()
        engine = SimpleNamespace(model=model, device=torch.device('cpu'), optimizer_bits=32,
            config=TrainConfig(), autocast=nullcontext, predict=model.dynamics, decode=model.decoder,
            actions=lambda actions: torch.tensor([a.features() for a in actions]))
        online = OnlinePredictor(engine, threshold=None, horizon=3)
        # True dynamics w=2: (z,h)=(1,0) -> (3,2) -> (7,4) -> (13,6).
        # Current w=1: last-step error is -1; imagined endpoint is 1+6*w=7,
        # error -6. dLoss/dw = -2 + 2*(-6)*6 = -74, with no target gradient.
        state = State(torch.tensor([[7.]], requires_grad=True), torch.tensor([[4.]], requires_grad=True), (1, 1))
        anchor = State(torch.tensor([[1.]]), torch.tensor([[0.]]), (1, 1))
        target = torch.tensor([[13.]], requires_grad=True)
        after = Observation(np.ones((1, 1, 1), dtype=np.uint8), (1,), 'NOT_FINISHED', 0, 1)
        captured = []
        model.dynamics.weight.register_hook(lambda grad: captured.append(float(grad)))
        result = online.update(state, Action(1), np.zeros((1, 1), dtype=np.uint8), after, target, 1.,
            rollout=(anchor, [Action(1)] * 3, np.zeros((1, 1), dtype=np.uint8)))
        self.assertAlmostEqual(result['latent'], 1.)
        self.assertAlmostEqual(result['total'], 37 + 2 * math.log(16) + 2 * 15 / 16, places=5)
        self.assertEqual(captured, [-74.])
        self.assertIsNone(target.grad)
        self.assertIsNone(state.z.grad)
        self.assertIsNone(state.h.grad)


if __name__ == '__main__':
    unittest.main()
