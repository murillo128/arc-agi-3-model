"""All-game replay, weighted updates and bounded accelerator state contracts."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.adaptation import AdaptationConfig, AdaptivePolicy, Record
from model.adaptation_convergence import fixed_hindsight, recency_weights, weighted_mean
from model.adaptation_smoke import base_digest
from model.tests.test_adaptation import runtime, observed, no_learning


class HistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_keeps_earliest_frames_actions_and_goal_outcomes_across_levels(self):
        policy = AdaptivePolicy(runtime(), no_learning())
        initial = observed()
        current = initial
        for i in range(70):
            action = policy.choose_action(current)
            policy.goal = dict(frame=observed(4).final, mask=np.ones((8, 8), bool),
                start=i, budget=1, created_action=i)
            current = observed((i + 1) % 3 + 1, level=(i + 1) // 20)
            self.assertTrue(policy.confirm_result(current, action))
        self.assertIsNone(policy.config.replay_size)
        self.assertEqual([r.number for r in policy.history], list(range(71)))
        np.testing.assert_array_equal(policy.history[0].observation.frames, initial.frames)
        self.assertTrue(all(r.action.id == 4 for r in list(policy.history)[1:]))
        self.assertEqual(len(policy._examples('representation')), 71)
        pairs = policy._examples('predictor')
        self.assertEqual(len(pairs), 67)
        self.assertEqual((pairs[0][0].number, pairs[0][1].number), (0, 1))
        self.assertEqual((pairs[-1][0].number, pairs[-1][1].number), (69, 70))
        self.assertEqual(len(policy.path_labels), 67)
        self.assertEqual(policy.path_labels[0]['start'], 0)
        self.assertEqual(policy.path_labels[0]['end'], 1)
        self.assertEqual(policy.path_labels[-1]['end'], 70)
        for row in policy._examples('goal'):
            self.assertEqual(row['start'] // 20, row['end'] // 20)
        self.assertGreater(min(policy._weights('predictor', pairs)), .25)
        self.assertEqual(policy._weights('predictor', pairs)[-1], 1.)

    def test_weights_have_hand_calculated_decay_floor_and_outcome_age(self):
        config = AdaptationConfig()
        self.assertEqual(recency_weights([128, 96, 64, 0], 128, config),
                         [1., .625, .4375, .296875])
        self.assertEqual(recency_weights([0], 100000, config), [.25])
        self.assertEqual(recency_weights([0, 80], 100, replace(config, history_weight_floor=1)), [1., 1.])
        policy = AdaptivePolicy(runtime(), no_learning())
        policy.choose_action(observed())
        policy.confirm_result(observed(2))
        policy.choose_action(observed(2))
        policy.confirm_result(observed(3))
        weights = policy._weights('path', [dict(start=0, end=2), dict(start=0, end=1)])
        self.assertEqual(weights[0], 1.)
        self.assertLess(weights[1], 1.)
        for kwargs in ({'history_half_life': 0}, {'history_half_life': float('inf')},
                       {'history_weight_floor': 0}, {'history_weight_floor': 1.1},
                       {'history_weight_floor': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AdaptationConfig(**kwargs)

    def test_accumulated_weighted_gradient_is_independent_of_minibatch_partition(self):
        gradients = []
        for batch_size in (1, 2, 4):
            policy = AdaptivePolicy(runtime(), no_learning(learning_rate=.01), seed=3)
            policy.choose_action(observed())
            policy.confirm_result(observed(2))
            policy.choose_action(observed(2))
            policy.confirm_result(observed(3))
            policy.config = replace(policy.config, stopping='convergence', representation_steps=1,
                minibatch=batch_size, history_half_life=1.)
            samples = [r.observation for r in policy.history]
            slopes = {id(obs): v for obs, v in zip(samples, [1., 1., -1.5])}
            parameter = policy.bank.parameters(('encoder',))[0]
            with torch.no_grad():
                parameter.zero_()
            optimizer = policy.bank.optimizers['encoder']
            original = optimizer.step
            before = base_digest(policy.runtime)
            finished = [False]

            def step():
                gradients.append(float(parameter.grad.flatten()[0]))
                original()
                finished[0] = True

            def objective(name, *, samples=None, states=None, measured=None, weights=None):
                def calculate():
                    if measured is not None:
                        measured.extend(dict(reconstruction_error=0. if finished[0] else 1.) for _ in samples)
                    losses = [10. + parameter.flatten()[0] * slopes[id(obs)] for obs in samples]
                    return weighted_mean(losses, weights), torch.tensor(0. if finished[0] else 1.)
                return calculate

            with patch.object(policy, '_objective', side_effect=objective), patch.object(optimizer, 'step', side_effect=step):
                policy._adapt()
            fit = policy.diagnostics['convergence']['representation']
            self.assertEqual(fit['steps'], 1)
            self.assertEqual(fit['samples'], 3)
            self.assertEqual(fit['reason'], 'threshold')
            self.assertAlmostEqual(fit['weights']['total'], 2.0625)
            self.assertEqual(base_digest(policy.runtime), before)
        # Weights are 7/16, 10/16 and 16/16. Weighted gradient is
        # (7 + 10 - 24) / 33 = -7/33, opposite the unweighted +1/6.
        for gradient in gradients:
            self.assertAlmostEqual(gradient, -7 / 33, places=6)

    def test_real_model_update_matches_across_minibatch_partitions(self):
        outcomes = []
        for batch_size in (1, 4):
            policy = AdaptivePolicy(runtime(), no_learning(learning_rate=.01), seed=3)
            policy.choose_action(observed())
            policy.confirm_result(observed(2))
            policy.choose_action(observed(2))
            policy.confirm_result(observed(3))
            policy.config = replace(policy.config, stopping='convergence', representation_steps=1,
                                    minibatch=batch_size, history_half_life=1.)
            policy._adapt()
            self.assertEqual(policy.diagnostics['convergence']['representation']['reason'], 'threshold')
            outcomes.append([p.detach().clone() for p in policy.bank.parameters(('encoder', 'decoder'))])
        for a, b in zip(*outcomes):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_cpu_history_states_and_selected_current_state_match_and_refresh(self):
        policy = AdaptivePolicy(runtime(), no_learning())
        for i in range(40):
            current = observed(i % 3 + 1)
            policy.choose_action(current)
            policy.confirm_result(observed((i + 1) % 3 + 1))
        with policy.runtime.router.activate(policy.bank):
            all_states = policy._states(cpu=True)
            selected = policy._states(numbers={40})
            self.assertEqual(set(all_states), set(range(41)))
            self.assertEqual(set(selected), {40})
            self.assertTrue(all(s.z.device.type == s.h.device.type == 'cpu' for s in all_states.values()))
            torch.testing.assert_close(all_states[40].z, selected[40].z.cpu())
            torch.testing.assert_close(all_states[40].h, selected[40].h.cpu())
            with torch.no_grad():
                next(pair[1] for name, pair in policy.bank.tensors.items() if name.startswith('encoder.')).fill_(.1)
            refreshed = policy._states(cpu=True)
            self.assertFalse(torch.equal(all_states[0].z, refreshed[0].z))
            self.assertFalse(torch.equal(all_states[40].h, refreshed[40].h))
            self.assertEqual(torch.count_nonzero(refreshed[0].h).item(), 0)

    def test_linear_hindsight_matches_independent_all_pairs_reference(self):
        rng = np.random.default_rng(73)
        rows = []
        for i, value in enumerate(rng.integers(1, 4, size=100)):
            rows.append(Record(i, observed(int(value), level=i // 20), boundary=i > 0 and i % 20 == 0))
        expected = []
        for start, before in enumerate(rows):
            candidates = []
            for end in range(start + 1, len(rows)):
                if rows[end].boundary:
                    break
                if not np.array_equal(before.observation.final, rows[end].observation.final):
                    candidates.append(end)
            if candidates:
                expected.append((start, candidates[-1]))
        actual = fixed_hindsight(rows)
        self.assertEqual([(r['start'], r['end']) for r in actual], expected)
        self.assertTrue(all(r['progress'] == (r['end'] in (19, 39, 59, 79)) for r in actual))
        self.assertTrue(all(r['distance'] == r['end'] - r['start'] for r in actual))
        # Also account for missing frames as boundaries, not a source of targets.
        missing = replace(observed(), frames=np.empty((0, 0, 0), np.uint8))
        interrupted = [Record(0, observed()), Record(1, observed(2)),
                       Record(2, missing, boundary=True), Record(3, observed(3), boundary=True),
                       Record(4, observed())]
        self.assertEqual([(r['start'], r['end']) for r in fixed_hindsight(interrupted)], [(0, 1), (3, 4)])
        won = fixed_hindsight([Record(0, observed()), Record(1, observed(2, state='WIN'))])
        self.assertTrue(won[0]['progress'])


if __name__ == '__main__':
    unittest.main()
