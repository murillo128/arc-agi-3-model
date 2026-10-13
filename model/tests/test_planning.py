"""Public outcome, frozen-weight and independent planning-oracle contracts."""
from contextlib import nullcontext, redirect_stderr
from dataclasses import replace
from pathlib import Path
import random
import io
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from model.behavior_experiment import collect, episode_seeds, evaluate, fit, load_heads, main, save_heads
from model.config import NetworkConfig, TrainConfig
from model.data import Observation
from model.environment import Factory
from model.learning import Engine, State
from model.networks import WorldModel
from model.planning import GoalValue, PlanConfig, Planner, candidates, fingerprint, freeze, outcome, returns
from model.behavior_trace import record_episode
from arc3.traces import encode_attempt, decode_attempt


def obs(state='NOT_FINISHED', completed=0):
    return Observation(np.zeros((1, 8, 8), dtype=np.uint8), (1, 2, 6), state, completed, 5)


class Contracts(unittest.TestCase):
    def test_public_games_fail_closed_before_loading_any_checkpoint(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(['--checkpoint', '/not-read.pt', '--output', '/not-created', '--game', 'ls20-9607627b'])
        self.assertEqual(error.exception.code, 2)

    def test_real_labels_follow_actions_and_empty_frames_preserve_history(self):
        start = replace(obs(), frames=np.full((1, 8, 8), 5, np.uint8), available_actions=(1,))
        empty = replace(start, frames=np.empty((0, 0, 0), np.uint8))
        later = replace(start, frames=np.full((1, 8, 8), 8, np.uint8))
        class FakeFactory:
            def make(self, seed, level):
                responses = iter([empty, later, replace(later, state='WIN')])
                return SimpleNamespace(observation=start, step=lambda action: next(responses))
        class FakeEngine:
            device = 'cpu'
            model = SimpleNamespace(config=SimpleNamespace(memory=1))
            def autocast(self):
                return nullcontext()
            def encode(self, observations):
                return [None if o.final is None else torch.tensor([[float(o.final[0, 0])]]) for o in observations]
            def actions(self, actions):
                return torch.tensor([[a.id] for a in actions])
            def advance(self, z, h, action):
                return h + z[:, 0] + action, None
        records = []
        collect(FakeEngine(), FakeFactory(), 1, [2], 3, .5, records)
        self.assertEqual([r[0].item() for r in records], [5., 5., 8.])
        self.assertEqual([r[1].item() for r in records], [0., 6., 12.])
        self.assertEqual([r[3] for r in records], [0, 0, 1])
        self.assertEqual([r[4] for r in records], [.25, .5, 1.])

    def test_public_outcomes_and_finite_returns(self):
        start = obs()
        self.assertEqual(outcome(start, obs(completed=1)), 1)
        self.assertEqual(outcome(start, obs('WIN')), 1)
        self.assertEqual(outcome(start, obs('GAME_OVER')), 2)
        self.assertEqual(outcome(start, start), 0)  # includes budget expiration
        self.assertEqual(returns([0, 0, 1], .5), [.25, .5, 1.])
        self.assertEqual(returns([0, 2], .5), [0., 0.])
        self.assertEqual(returns([0, 0], .5), [0., 0.])
        self.assertEqual(returns([1, 0, 1], .5), [1., .5, 1.])

    def test_candidates_only_legal_and_bounded_clicks(self):
        result = candidates(obs(), random.Random(1), clicks=3)
        self.assertEqual([x.id for x in result], [1, 2, 6, 6, 6])
        self.assertEqual(len({tuple(x.data.items()) for x in result if x.id == 6}), 3)
        self.assertTrue(all(0 <= v < 64 for x in result for v in x.data.values()))
        self.assertEqual(result, candidates(obs(), random.Random(1), clicks=3))
        with self.assertRaises(ValueError):
            candidates(replace(obs(), available_actions=(0,)), random.Random())
        for setting in ({'horizon': 0}, {'discount': 1.1}, {'beam': -1}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                PlanConfig(**setting)

    def test_partitions_are_disjoint(self):
        sets = [set(episode_seeds(split, 20)) for split in ('train', 'dev', 'final')]
        self.assertEqual(len(set.union(*sets)), 60)
        with self.assertRaises(ValueError):
            episode_seeds('train', 1, 1_000_000)

    def test_head_learning_leaves_world_byte_identical_and_rejects_mismatch(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)
        net = WorldModel(NetworkConfig.small())
        engine = Engine(net, TrainConfig())
        freeze(engine)
        original = fingerprint(net)
        heads = GoalValue(16, 32, width=8)
        original_head = heads.value.weight.detach().clone()
        records = [(torch.randn(4, 16), torch.randn(32), torch.eye(12)[1], 1, 1., (8, 8)),
                   (torch.randn(4, 16), torch.randn(32), torch.eye(12)[2], 2, 0., (8, 8))]
        fit(engine, heads, records, 2, 2, .001, 1)
        self.assertFalse(torch.equal(original_head, heads.value.weight))
        self.assertEqual(fingerprint(net), original)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in net.parameters()))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'heads.pt'
            save_heads(path, heads, original, PlanConfig(), {})
            with self.assertRaisesRegex(ValueError, 'different world-model'):
                load_heads(path, heads, 'wrong')
            copy = GoalValue(16, 32, width=8)
            load_heads(path, copy, original)
            self.assertTrue(torch.equal(copy.value.weight, heads.value.weight))

    def test_real_sdk_cpu_paired_evaluation_has_no_updates(self):
        torch.set_num_threads(2)
        net = WorldModel(NetworkConfig.small())
        engine = Engine(net, TrainConfig())
        heads = GoalValue(16, 32, width=8)
        planner = Planner(engine, heads, PlanConfig(horizon=1))
        factory = Factory('sc01-v1', Path(__file__).resolve().parents[2] / 'games')
        before = fingerprint(net)
        hbefore = {k: v.clone() for k, v in heads.state_dict().items()}
        a = evaluate(engine, planner, factory, 2, [1231], 2, 'mpc')
        b = evaluate(engine, planner, factory, 2, [1231], 2, 'mpc')
        for key in ('wins', 'actions'):
            self.assertEqual(a[key], b[key])
        self.assertEqual(a['results'][0]['seed'], 1231)
        self.assertEqual(before, fingerprint(net))
        self.assertTrue(all(torch.equal(hbefore[k], v) for k, v in heads.state_dict().items()))
        self.assertTrue(a['results'][0]['truncated'] or a['results'][0]['game_over'] or a['wins'])
        attempt = record_episode(engine, planner, factory, 2, 91991, 2, 'mpc', before)
        decoded = decode_attempt(encode_attempt(attempt, format_minor=1))
        self.assertEqual(decoded['summary']['real_actions'], len(attempt['steps']))
        for step in decoded['steps']:
            self.assertEqual(step['planning']['steps'][0]['action'], step['action'])
            self.assertEqual(step['planning']['horizon'], 1)
            self.assertNotIn('learning', step)
        self.assertEqual(before, fingerprint(net))
        self.assertTrue(all(torch.equal(hbefore[k], v) for k, v in heads.state_dict().items()))

    def test_latent_planning_oracle_and_independent_memory(self):
        # Two real choices: action 1 moves latent x left, action 2 right.
        # The independent terminal value oracle prefers x=+1. No images.
        class Readout(nn.Module):
            def forward(self, z, h, action, rows, columns):
                event_q = torch.zeros((len(z), 4))
                event_q[:, 0] = 30.  # deterministic continuation
                return z[:, 0, 0], event_q
        class FakeEngine:
            model = nn.Linear(1, 1)
            model.config = SimpleNamespace(patch=1)
            def autocast(self):
                return nullcontext()
            def actions(self, actions):
                return torch.tensor([a.features() for a in actions])
            def predict(self, z, h, action, height, width):
                d = (action[:, 2] - action[:, 1])[:, None, None]
                return z + d, h + 1, d
            def decode(self, *args):
                raise AssertionError('Planner must not decode pixels')
        state = State(torch.zeros(1, 1), torch.zeros(1, 1), (1, 1))
        planner = Planner(FakeEngine(), Readout(), PlanConfig(horizon=2, beam=2))
        chosen = planner.choose(state, replace(obs(), available_actions=(1, 2)), random.Random(5))
        self.assertEqual(chosen.id, 2)
        diagnostics = {}
        traced = planner.choose(state, replace(obs(), available_actions=(1, 2)), random.Random(5), diagnostics=diagnostics)
        self.assertEqual(traced, chosen)
        plan = diagnostics['planning']
        self.assertEqual([s['action']['id'] for s in plan['steps']], [2, 2])
        self.assertEqual(plan['horizon'], 2)
        self.assertAlmostEqual(plan['state_value'], .5)
        self.assertAlmostEqual(plan['steps'][0]['successor_value'], .7310586, places=6)
        self.assertAlmostEqual(plan['selected_score'], .95 ** 2 * .8807971, places=6)
        q_trace = {}
        planner.choose(state, replace(obs(), available_actions=(1, 2)), random.Random(5), 'q', q_trace)
        self.assertEqual(q_trace['planning']['horizon'], 1)
        self.assertNotIn('successor_value', q_trace['planning']['steps'][0])
        self.assertEqual(state.z.item(), 0.)
        self.assertEqual(state.h.item(), 0.)


if __name__ == '__main__':
    unittest.main()
