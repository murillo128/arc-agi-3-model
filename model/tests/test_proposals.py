import random
import math
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.config import NetworkConfig, TrainConfig
from model.data import Action, Observation
from model.learning import Engine
from model.networks import WorldModel
from model.planning import OutcomeValue, fingerprint
from model.proposals import ProposalPolicy, TeacherRecord, imitation_loss, imagined_loss


class ProposalContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.engine = Engine(WorldModel(NetworkConfig.small()), TrainConfig())
        observation = Observation(np.zeros((1, 8, 8), dtype=np.uint8), (1, 2, 6), 'NOT_FINISHED', 0, 5)
        self.state = self.engine.initial(observation)
        config = self.engine.model.config
        self.policy = ProposalPolicy(config.latent, config.memory, width=8, patch=config.patch)
        self.heads = OutcomeValue(config.latent, config.memory, width=8)
        self.row = TeacherRecord(self.state.z.detach(), self.state.h[0].detach(), (8, 8),
                                 (1, 2, 6), 2, Action(2))

    def test_only_teacher_labels_are_imitated_and_learning_increases_teacher_probability(self):
        before = float(imitation_loss(self.policy, [self.row], 'cpu').detach())
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=.01)
        optimizer.zero_grad()
        imitation_loss(self.policy, [self.row], 'cpu').backward()
        optimizer.step()
        after = float(imitation_loss(self.policy, [self.row], 'cpu').detach())
        self.assertLess(after, before)
        self.row.source = 'random_exploration'
        with self.assertRaisesRegex(ValueError, 'MPC decisions'):
            imitation_loss(self.policy, [self.row], 'cpu')

    def test_click_coordinate_domain_and_rng_are_independent_of_frame_geometry(self):
        args = (self.state.z, self.state.h, (8, 8), (6,))
        first = self.policy.propose(*args, random.Random(4), count=64)
        second = self.policy.propose(*args, random.Random(4), count=64)
        self.assertEqual(first, second)
        self.assertTrue(all(a.id == 6 and 0 <= a.data['x'] < 64 and 0 <= a.data['y'] < 64 for a in first))
        self.assertTrue(any(a.data['x'] >= 8 for a in first))

    def test_imagination_updates_actor_critic_but_never_world_or_outcome_heads(self):
        self.row.z, self.row.h = self.row.z.half(), self.row.h.half()
        before_world, before_heads = fingerprint(self.engine.model), fingerprint(self.heads)
        loss, metrics = imagined_loss(self.engine, self.heads, self.policy, [self.row], horizon=2)
        loss.backward()
        self.assertEqual(metrics['imagined_horizon'], 2)
        self.assertGreater(float(self.policy.actor.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(self.policy.critic.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in self.engine.model.parameters()))
        self.assertTrue(all(p.grad is None for p in self.heads.parameters()))
        self.assertEqual(fingerprint(self.engine.model), before_world)
        self.assertEqual(fingerprint(self.heads), before_heads)

    def test_route_second_action_is_conditioned_on_its_first_successor(self):
        seen = []
        def propose(z, h, *args, **kwargs):
            seen.append((z.clone(), h.clone()))
            return [Action(1)]
        def predict(z, h, actions, *geometry):
            return z + 3, h + 7, torch.full_like(z, 3)
        before = self.state.z.clone()
        with patch.object(self.policy, 'propose', side_effect=propose), patch.object(self.engine, 'predict', side_effect=predict):
            route = self.policy.propose_route(self.engine, self.state, (1, 2), random.Random(4), 8, remaining_budget=2)
        self.assertEqual(route, [Action(1), Action(1)])
        torch.testing.assert_close(seen[1][0], seen[0][0] + 3)
        torch.testing.assert_close(seen[1][1], seen[0][1] + 7)
        torch.testing.assert_close(self.state.z, before)

    def test_budget_one_limits_imagination_even_with_long_requested_horizon(self):
        self.row.remaining_budget = 1
        loss, metrics = imagined_loss(self.engine, self.heads, self.policy, [self.row], horizon=8)
        self.assertEqual(metrics['imagined_horizon'], 1)
        self.assertTrue(torch.isfinite(loss))

    def test_slow_critic_supplies_bootstrap_and_last_action_progress_is_not_erased(self):
        self.row.available_actions = (1,)
        self.row.remaining_budget = 2
        with torch.no_grad():
            self.policy.critic.weight.zero_()
            self.policy.critic.bias.zero_()
            self.heads.action[-1].weight.zero_()
            self.heads.action[-1].bias.copy_(torch.tensor([100., -100., -100., 0.]))
        slow = deepcopy(self.policy).eval().requires_grad_(False)
        with torch.no_grad():
            slow.critic.bias.fill_(math.log(.8 / .2))
        loss, _ = imagined_loss(self.engine, self.heads, self.policy, [self.row],
                                horizon=1, discount=.5, lambda_=1., target_policy=slow)
        loss.backward()
        # Continue, gamma=.5, slow V=.8 => target=.4. Online V=.5,
        # critic loss weight=.5 => bias gradient .5*(.5-.4)=.05.
        self.assertAlmostEqual(self.policy.critic.bias.grad.item(), .05, places=6)
        self.assertTrue(all(p.grad is None for p in slow.parameters()))
        self.policy.zero_grad(set_to_none=True)
        self.row.remaining_budget = 1
        with torch.no_grad():
            self.heads.action[-1].bias.copy_(torch.tensor([-100., 100., -100., 0.]))
        loss, _ = imagined_loss(self.engine, self.heads, self.policy, [self.row],
                                horizon=1, discount=.5, lambda_=1., target_policy=slow)
        loss.backward()
        # A certain progress on the last action has target 1, not zero.
        self.assertAlmostEqual(self.policy.critic.bias.grad.item(), -.25, places=6)


if __name__ == '__main__':
    unittest.main()
