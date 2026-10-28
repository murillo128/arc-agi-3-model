"""Low-rank routing contracts; runtime lifecycle is tested separately."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import unittest

import numpy as np
import torch
from torch import nn

from model.adaptation_heads import VisualGoal
from model.config import NetworkConfig, TrainConfig
from model.data import Action, Observation
from model.learning import Engine
from model.lora import AdapterRouter
from model.networks import WorldModel
from model.path_planning import PathValue


class LoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_effect_all_five_groups_and_spatial_exclusions(self):
        torch.manual_seed(7)
        config = replace(NetworkConfig.small(), patch=2, latent=16, pixel_latent=True,
                         local_dynamics=True, rotation_adapter_width=4)
        world = WorldModel(config).eval()
        goal = VisualGoal(config.latent, config.memory, config.patch, 8)
        path = PathValue(config.latent, config.memory, 8)
        engine = Engine(world, TrainConfig())
        observation = Observation(np.arange(16, dtype=np.uint8).reshape(1, 4, 4), (1,), 'NOT_FINISHED', 0, 1)
        def outputs():
            z = engine.encode([observation])[0][None]
            h = torch.zeros(1, config.memory)
            predicted, next_h, delta = engine.predict(z, h, engine.actions([Action(1)]), 4, 4)
            return (z, predicted, next_h, delta, engine.decode(z, 4, 4),
                    *goal(z, h, 4, 4), *path(z, h, z, torch.ones(1, 4, 1), 2, 2, torch.ones(1)))
        before = [value.detach().clone() for value in outputs()]
        weights = {name: value.clone() for name, value in world.state_dict().items()}
        router = AdapterRouter(dict(encoder=world.encoder, decoder=world.decoder,
                                    predictor=world.dynamics, goal=goal, path=path))
        self.assertEqual({target.group for target in router.targets}, {'encoder', 'decoder', 'predictor', 'goal', 'path'})
        self.assertTrue(router.report['excluded'], 'The rotation adapter contains excluded depthwise layers')
        bank = router.new_bank()
        with router.activate(bank, groups=tuple(router.modules)):
            actual = outputs()
            for expected, value in zip(before, actual):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            loss = sum(value.square().mean() for value in actual)
            loss.backward()
            for group in router.modules:
                gradients = [p.grad for p in bank.parameters((group,)) if p.grad is not None]
                self.assertTrue(gradients, group)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients), group)
                self.assertTrue(any(torch.count_nonzero(g) for g in gradients), group)
                bank.optimizers[group].step()
        for name, weight in world.state_dict().items():
            self.assertTrue(torch.equal(weight, weights[name]))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in world.parameters()))
        self.assertLess(bank.bytes, sum(p.numel() * p.element_size() for p in world.parameters()))

    def test_updates_reduce_loss_and_select_only_requested_group(self):
        torch.manual_seed(2)
        modules = {'encoder': nn.Linear(3, 2), 'decoder': nn.Conv2d(2, 2, 3, padding=1)}
        router = AdapterRouter(modules)
        bank = router.new_bank(rank=2, learning_rate=.05)
        original_decoder = [p.clone() for p in bank.parameters(('decoder',))]
        x, target = torch.tensor([[1., 2., -1.], [-1., 0., 2.]]), torch.tensor([[2., 0.], [0., 2.]])
        with router.activate(bank, groups=('encoder',)):
            initial = float((modules['encoder'](x) - target).square().mean().detach())
            for _ in range(20):
                optimizer = bank.optimizers['encoder']
                optimizer.zero_grad(set_to_none=True)
                loss = (modules['encoder'](x) - target).square().mean()
                loss.backward()
                optimizer.step()
            final = float((modules['encoder'](x) - target).square().mean().detach())
        self.assertLess(final, initial)
        for before, after in zip(original_decoder, bank.parameters(('decoder',))):
            self.assertTrue(torch.equal(before, after))
        self.assertFalse(bank.optimizers['decoder'].state)

    def test_concurrent_banks_equal_isolated_and_restore_after_exception(self):
        torch.manual_seed(1)
        module = nn.Linear(3, 2)
        router = AdapterRouter({'encoder': module})
        x = torch.tensor([[1., 2., 3.]])
        def train(bank, target):
            for _ in range(4):
                with router.activate(bank, groups=('encoder',)):
                    bank.optimizers['encoder'].zero_grad(set_to_none=True)
                    (module(x) - target).square().mean().backward()
                    bank.optimizers['encoder'].step()
            with router.activate(bank), torch.no_grad():
                return module(x).clone()
        first, second = [router.new_bank(seed=s, learning_rate=.01) for s in (10, 20)]
        isolated = [train(router.new_bank(seed=s, learning_rate=.01), target)
                    for s, target in ((10, 1.), (20, -1.))]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(train, first, 1.), pool.submit(train, second, -1.)]
            actual = [future.result() for future in futures]
        for expected, value in zip(isolated, actual):
            torch.testing.assert_close(expected, value, rtol=0, atol=0)
        first_state = first.optimizers['encoder'].state[first.parameters()[0]]
        second_state = second.optimizers['encoder'].state[second.parameters()[0]]
        self.assertIsNot(first_state['exp_avg'], second_state['exp_avg'])
        with router.activate(first, groups=('encoder',)):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with router.activate(second):
                    raise RuntimeError('injected')
            self.assertIs(router.active, first)
            self.assertTrue(all(p.requires_grad for p in first.parameters()))
            parameter = first.parameters()[0]
            parameter.grad = torch.ones_like(parameter)
            gradient = parameter.grad
            with router.activate(first, groups=()):
                self.assertFalse(any(p.requires_grad for p in first.parameters()))
            self.assertIs(parameter.grad, gradient)
            self.assertTrue(all(p.requires_grad for p in first.parameters()))
        self.assertIsNone(router.active)
        for bank in (first, second):
            self.assertTrue(all(p.device.type == 'cpu' and p.grad is None for p in bank.parameters()))

    def test_conv_adapter_matches_explicit_low_rank_kernel(self):
        torch.manual_seed(4)
        module = nn.Conv2d(2, 3, (3, 2), stride=2, padding=1, dilation=2)
        router = AdapterRouter({'predictor': module})
        bank = router.new_bank(rank=2, alpha=3)
        a, b = bank.tensors['predictor']
        with torch.no_grad():
            b.fill_(.2)
        x = torch.randn(1, 2, 9, 8)
        expected = torch.nn.functional.conv2d(x, module.weight + (b[:, :, 0, 0] @ a.flatten(1)).reshape_as(module.weight) * 1.5,
                                              module.bias, stride=2, padding=1, dilation=2)
        with router.activate(bank):
            torch.testing.assert_close(module(x), expected, rtol=1e-5, atol=1e-6)

    def test_rejects_invalid_bank_and_groups(self):
        router = AdapterRouter({'encoder': nn.Linear(2, 2)})
        for kwargs in ({'rank': 0}, {'rank': True}, {'alpha': float('nan')}, {'learning_rate': 0}, {'seed': 'x'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                router.new_bank(**kwargs)
        other = AdapterRouter({'encoder': nn.Linear(2, 2)}).new_bank()
        with self.assertRaises(ValueError), router.activate(other):
            pass
        with self.assertRaises(ValueError), router.activate(router.new_bank(), groups=('missing',)):
            pass


if __name__ == '__main__':
    unittest.main()
