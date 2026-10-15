"""Gradient ownership and numerical objectives of the incremental learners."""
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from model.config import NetworkConfig, TrainConfig
from model.data import Action, Episode, Observation
from model.learning import Engine, State, change_union_delta_loss, visual_delta_loss
from model.networks import WorldModel
from model.online_learning import OnlinePredictor, OnlineRepresentation
from model.path_planning import real_states
from model.planning import fingerprint


torch.set_num_threads(2)


def union_example():
    # Correct change, missed change, invented change, wrong color, correct keep.
    before = torch.tensor([[[0, 1, 2, 3, 4]]])
    actual = torch.tensor([[[5, 6, 2, 7, 4]]])
    probability = torch.empty(1, 1, 5, 16)
    for i, (target, winner, correct) in enumerate(((5, 5, .8), (6, 1, .1),
                                                  (2, 0, .2), (7, 8, .3), (4, 4, .7))):
        probability[0, 0, i].fill_((1 - correct) / 15 if target == winner else (1 - correct - .6) / 14)
        probability[0, 0, i, winner] = correct if target == winner else .6
        probability[0, 0, i, target] = correct
    return before, actual, probability


class OnlineLearningTests(unittest.TestCase):
    def test_frozen_encoder_keeps_exact_latents_while_decoder_and_predictor_learn(self):
        torch.manual_seed(19)
        engine = Engine(WorldModel(NetworkConfig.small()),
                        replace(TrainConfig(), variance_weight=.1, sigreg_weight=.1))
        online = OnlineRepresentation(engine, freeze_encoder=True)
        predictor = OnlinePredictor(engine, threshold=None)
        observations = [Observation(np.full((1, 8, 8), color, dtype=np.uint8),
                        (1, 2, 3, 4), 'NOT_FINISHED', 0, 1) for color in (2, 3)]
        episode = Episode(observations, [Action(4)])
        initial = {name: fingerprint(getattr(engine.model, name))
                   for name in ('encoder', 'decoder', 'dynamics')}
        with torch.no_grad():
            latent = torch.stack(engine.encode(observations))
        encoder_parameters = {id(p) for p in engine.model.encoder.parameters()}
        self.assertFalse(encoder_parameters & {id(p) for p in online.parameters})
        for _ in range(2):
            result = online.update(observations)
            self.assertTrue(math.isfinite(result['total']))
            self.assertNotIn('sigreg', result['terms'])
            self.assertNotIn('variance_floor', result['terms'])
            states = real_states(engine, episode)
            result = predictor.update(states[0], Action(4), observations[0].final,
                                      observations[1], states[1].z, 1.)
            self.assertTrue(math.isfinite(result['total']))
            self.assertEqual(fingerprint(engine.model.encoder), initial['encoder'])
            self.assertTrue(all(p.grad is None for p in engine.model.encoder.parameters()))
            with torch.no_grad():
                self.assertTrue(torch.equal(torch.stack(engine.encode(observations)), latent))
        for name in ('decoder', 'dynamics'):
            self.assertNotEqual(fingerprint(getattr(engine.model, name)), initial[name])

    def test_union_nll_counts_missed_invented_and_wrong_colors_without_background_dilution(self):
        before, actual, probability = union_example()
        logits = probability.log().requires_grad_()
        loss = change_union_delta_loss(logits, before, actual).sum()
        expected = -(math.log(.8) + math.log(.1) + math.log(.2) + math.log(.3)) / 4
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)
        gradient = torch.autograd.grad(loss, logits)[0]
        for i, correct in enumerate((.8, .1, .2, .3)):
            self.assertAlmostEqual(float(gradient[0, 0, i, actual[0, 0, i]]), (correct - 1) / 4, places=6)
        self.assertEqual(int(torch.count_nonzero(gradient[:, :, 4])), 0)
        indices = [0, 1, 2, 3] + [4] * 100
        padded = logits.detach()[:, :, indices].clone().requires_grad_()
        padded_loss = change_union_delta_loss(padded, before[:, :, indices], actual[:, :, indices]).sum()
        self.assertAlmostEqual(float(padded_loss.detach()), expected, places=6)
        padded_loss.backward()
        torch.testing.assert_close(padded.grad[:, :, :4], gradient[:, :, :4])
        self.assertEqual(int(torch.count_nonzero(padded.grad[:, :, 4:])), 0)
        for weight in (1., 2., 100.):
            torch.testing.assert_close(visual_delta_loss(logits, before, actual, weight, mode='union').sum(), loss)

    def test_union_nll_empty_and_mixed_batches_are_finite_and_normalized_per_image(self):
        before, actual, probability = union_example()
        empty_logits = torch.full_like(probability, -2.).scatter(-1, before[..., None], 2.)
        logits = torch.cat((probability.log(), empty_logits)).requires_grad_()
        loss = change_union_delta_loss(logits, before.expand(2, -1, -1), torch.cat((actual, before)))
        self.assertEqual(loss.shape, (2,))
        self.assertAlmostEqual(float(loss.detach()[0]), -(math.log(.8) + math.log(.1) + math.log(.2) + math.log(.3)) / 4, places=6)
        self.assertEqual(float(loss.detach()[1]), 0.)
        loss.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(int(torch.count_nonzero(logits.grad[1])), 0)
        empty_logits.requires_grad_()
        change_union_delta_loss(empty_logits, before, before).sum().backward()
        self.assertEqual(int(torch.count_nonzero(empty_logits.grad)), 0)

    def test_online_union_nll_reaches_dynamics_at_one_and_three_steps_with_decoder_frozen(self):
        before, actual, probability = union_example()

        class Dynamics(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.))

            def forward(self, z, h, action, height, width):
                return z + self.weight, h, torch.ones_like(z) * self.weight

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(probability.log())

            def forward(self, z, height, width, **kwargs):
                return self.logits + z.sum() * torch.nn.functional.one_hot(actual, 16)

        model = nn.Module()
        model.encoder, model.dynamics, model.decoder = nn.Linear(1, 1), Dynamics(), Decoder()
        engine = SimpleNamespace(model=model, device=torch.device('cpu'), optimizer_bits=32,
            config=TrainConfig(prediction_delta_loss='union', false_change_weight=2., delta_overlap_weight=0.),
            autocast=nullcontext, predict=model.dynamics, decode=model.decoder,
            actions=lambda actions: torch.tensor([a.features() for a in actions]))
        online = OnlinePredictor(engine, threshold=None)
        state = State(torch.zeros(1, 1), torch.zeros(1, 1), (1, 5))
        after = Observation(actual.numpy().astype(np.uint8), (1,), 'NOT_FINISHED', 0, 1)
        captured = []
        model.dynamics.weight.register_hook(lambda grad: captured.append(float(grad)))
        decoder_before = fingerprint(model.decoder)
        result = online.update(state, Action(1), before[0].numpy(), after, state.z, 1.,
            rollout=(state, [Action(1)] * 3, before[0].numpy()))
        expected = -(math.log(.8) + math.log(.1) + math.log(.2) + math.log(.3)) / 4
        self.assertAlmostEqual(result['visual'], expected, places=6)
        self.assertAlmostEqual(result['rollout'], expected, places=6)
        self.assertAlmostEqual(result['total'], 2 * expected, places=6)
        # One-step dz/dw=1; three-step dz/dw=3. Latent MSE has zero gradient here.
        self.assertAlmostEqual(captured[0], (.8 + .1 + .2 + .3 - 4), places=6)
        self.assertGreater(float(model.dynamics.weight.detach()), 0.)
        self.assertEqual(fingerprint(model.decoder), decoder_before)
        self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))

    def test_local_reconstruction_has_its_weighted_loss_and_encoder_gradient(self):
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.context = nn.Parameter(torch.tensor(0.))
                self.local = nn.Parameter(torch.tensor(math.log(5.)))

            def forward(self, z, height, width, *, local=False, **kwargs):
                # Context: uniform 16 colors. Local: P(color 0)=5/(5+15)=1/4.
                score = z[:, 0, 0] * (self.local if local else self.context)
                return score[:, None, None, None] * torch.nn.functional.one_hot(torch.tensor(0), 16)

        for weight, readout in ((.5, 'contextual'), (0., 'contextual'), (.5, 'local')):
            with self.subTest(weight=weight, readout=readout):
                model = nn.Module()
                model.encoder, model.decoder, model.dynamics = nn.Linear(1, 1, bias=False), Decoder(), nn.Linear(1, 1)
                model.config = NetworkConfig.small()
                with torch.no_grad():
                    model.encoder.weight.fill_(1.)
                config = replace(TrainConfig(), sigreg_weight=0., variance_weight=0.,
                                 local_reconstruction_weight=weight, decoder_readout=readout,
                                 prediction_delta_loss='union')
                engine = SimpleNamespace(model=model, config=config, device=torch.device('cpu'), optimizer_bits=32,
                    autocast=nullcontext, invalidate_latent_cache=lambda: None,
                    encode=lambda observations, **kwargs: [model.encoder.weight for _ in observations],
                    decode=partial(model.decoder.forward, local=readout == 'local'))
                online = OnlineRepresentation(engine)
                observation = Observation(np.zeros((1, 1, 1), dtype=np.uint8), (1,), 'NOT_FINISHED', 0, 1)
                gradients = []
                model.encoder.weight.register_hook(lambda grad: gradients.append(float(grad)))
                rng = torch.get_rng_state().clone()
                result = online.update([observation, observation])
                context_loss, local_loss = 2 * math.log(16) + 15 / 16, 2 * math.log(4) + 3 / 4
                expected = local_loss if readout == 'local' else context_loss + weight * local_loss
                self.assertAlmostEqual(result['total'], expected, places=5)
                # d/ds [2*(-log p0)+(1-p0)] = -27/16 at p0=1/4.
                factor = 1. if readout == 'local' else weight
                self.assertAlmostEqual(gradients[0], -factor * 27 / 16 * math.log(5), places=5)
                self.assertEqual('local_reconstruction' in result['terms'], bool(weight and readout != 'local'))
                self.assertNotIn('sigreg', result['terms'])
                self.assertNotIn('variance_floor', result['terms'])
                self.assertAlmostEqual(result['total'], sum(result['terms'].values()), places=5)
                torch.testing.assert_close(torch.get_rng_state(), rng)

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
        self.assertAlmostEqual(result['total'], sum(result['terms'].values()), places=5)
        self.assertIsNone(target.grad)
        self.assertIsNone(state.z.grad)
        self.assertIsNone(state.h.grad)


if __name__ == '__main__':
    unittest.main()
