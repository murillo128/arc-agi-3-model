"""Bounded experiment accounting and provenance, using real small CPU modules."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from model.campaign_behavior import TeacherPlanner, train_behavior
from model.behavior_experiment import collect
from model.config import NetworkConfig, TrainConfig
from model.data import Action, Observation
from model.learning import Engine
from model.networks import WorldModel
from model.planning import GoalValue, PlanConfig, fingerprint


class PublicFactory:
    def make(self, seed, level):
        initial = Observation(np.full((1, 8, 8), level, np.uint8), (1, 2), 'NOT_FINISHED', 0, 5)
        position = 0
        def step(action):
            nonlocal position
            position += 1
            return replace(initial, frames=np.full((1, 8, 8), (level + position) % 16, np.uint8),
                           levels_completed=int(position >= 2), action_input=action)
        return SimpleNamespace(observation=initial, step=step)


class CampaignBehaviorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.engine = Engine(WorldModel(NetworkConfig.small()), TrainConfig())
        self.heads = GoalValue(16, 32, width=8)
        self.config = PlanConfig(horizon=1, beam=2)

    def test_exact_balanced_actions_and_final_callback_use_same_frozen_world(self):
        events, callbacks = [], []
        source = fingerprint(self.engine.model)
        with redirect_stdout(io.StringIO()):
            planner, policy, report = train_behavior(self.engine, self.heads, self.config,
                {'a': PublicFactory(), 'b': PublicFactory()}, [('a', 1), ('a', 2), ('b', 1)],
                total_actions=10, seed=0, emit=lambda event, **data: events.append((event, data)),
                check_callback=lambda actions, p, a, r: callbacks.append((actions, p, a)))
        self.assertEqual(report['actions'], 10)
        self.assertEqual([p['actions'] for p in report['per_pair']], [4, 3, 3])
        self.assertEqual(callbacks, [(10, planner, None)])
        self.assertIsNone(policy)
        self.assertEqual(fingerprint(self.engine.model), source)
        self.assertTrue(report['world_unchanged'])
        self.assertEqual(planner.config.clicks, 16)
        self.assertEqual(events[-1][0], 'campaign_progress')
        self.assertTrue(all(e[1]['actions'] <= 10 for e in events))

    def test_actor_stage_preserves_world_and_evaluator_and_has_imitation_first(self):
        source, utility = fingerprint(self.engine.model), fingerprint(self.heads)
        events = []
        with redirect_stdout(io.StringIO()):
            planner, policy, report = train_behavior(self.engine, self.heads, self.config,
                {'a': PublicFactory()}, [('a', 1)], total_actions=8, seed=1, variant='actor_imagined',
                emit=lambda event, **data: events.append(data))
        self.assertIs(planner.proposal_policy, policy)
        self.assertEqual(report['actions'], 8)
        self.assertGreater(report['teacher_records'], 0)
        self.assertEqual([c['actions'] for c in report['checks']], [4, 8])
        self.assertTrue(any('imitation_loss' in (e.get('loss') or {}) and e['actions'] == 4 for e in events))
        self.assertTrue(any(e.get('imagined_horizon') == 2 and e['actions'] == 8 for e in events))
        self.assertEqual(fingerprint(self.engine.model), source)
        self.assertEqual(fingerprint(self.heads), utility)
        self.assertTrue(all(p.grad is None for p in self.engine.model.parameters()))

    def test_random_exploration_never_enters_teacher_store(self):
        store = []
        class Teacher:
            def reset(self): pass
            def after_real_action(self, *args): pass
            def choose(self, state, observation, rng, mode, diagnostics, remaining_budget=None):
                return Action(2)
        teacher = TeacherPlanner(Teacher(), store)
        with redirect_stdout(io.StringIO()):
            collect(self.engine, PublicFactory(), 1, [101], 2, .95, [], policy=teacher,
                    mode='value_mpc', epsilon=1)
        self.assertEqual(store, [])
        initial = PublicFactory().make(101, 1).observation
        state = self.engine.initial(initial)
        selected = teacher.choose(state, initial, random.Random(1), remaining_budget=2)
        self.assertEqual(selected, Action(2))
        self.assertEqual(len(store), 1)
        self.assertEqual(store[0].source, 'mpc_teacher')
        self.assertEqual(store[0].remaining_budget, 2)
        torch.testing.assert_close(store[0].z.float(), state.z.float(), rtol=.001, atol=.001)


if __name__ == '__main__':
    unittest.main()
