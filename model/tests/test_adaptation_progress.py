"""Native success labels and genuine multi-step per-game adaptation."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from model.adaptation import AdaptationConfig, AdaptivePolicy, Record
from model.adaptation_convergence import stopping_reason, summarize
from model.adaptation_smoke import base_digest
from model.data import Action, Observation
from model.learning import State
from model.tests.test_adaptation import no_learning, observed, runtime


class ProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_changed_visited_states_do_not_supervise_goal_but_remain_path_evidence(self):
        policy = AdaptivePolicy(runtime(), no_learning())
        policy.choose_action(observed())
        policy.confirm_result(observed(2))
        self.assertEqual(policy._examples('goal'), [])
        self.assertEqual([(r['start'], r['end']) for r in policy._examples('path')], [(0, 1)])
        policy.choose_action(observed(2))
        policy.confirm_result(observed(4, level=1))
        goals = policy._examples('goal')
        self.assertEqual([(r['start'], r['end']) for r in goals], [(0, 1)])
        self.assertTrue(goals[0]['progress'])
        np.testing.assert_array_equal(goals[0]['frame'], observed(2).final)

    def test_new_pair_updates_even_when_initial_error_passes_but_not_twice_or_on_reset(self):
        config = AdaptationConfig(representation_steps=0, path_steps=0, goal_steps=0,
                                  delta_threshold=1., exploration=1.)
        policy = AdaptivePolicy(runtime(), config, seed=3)
        policy.choose_action(observed())
        policy.confirm_result(observed(2))
        self.assertEqual(policy.diagnostics['updates']['predictor'], 1)
        self.assertEqual(policy.diagnostics['convergence']['predictor']['steps'], 1)
        self.assertFalse(policy.confirm_result(observed(2)))
        policy.choose_action(observed(2))
        policy.confirm_result(observed(3, level=1))
        self.assertEqual(policy.diagnostics['updates']['predictor'], 1)
        policy.choose_action(observed(3, level=1))
        policy.confirm_result(observed(3, level=1))  # an actual no-op is new evidence
        self.assertEqual(policy.diagnostics['updates']['predictor'], 2)

    def test_goal_requires_native_success_and_never_targets_terminal_or_next_level(self):
        for outcome, level, expected in [('NOT_FINISHED', 0, []), ('GAME_OVER', 0, []),
                                         ('WIN', 0, [(0, 1)]), ('NOT_FINISHED', 1, [(0, 1)])]:
            with self.subTest(outcome=outcome, level=level):
                policy = AdaptivePolicy(runtime(), no_learning())
                policy.choose_action(observed())
                policy.confirm_result(observed(2))
                policy.choose_action(observed(2))
                policy.confirm_result(observed(5, state=outcome, level=level))
                goals = policy._examples('goal')
                self.assertEqual([(r['start'], r['end']) for r in goals], expected)
                for row in goals:
                    np.testing.assert_array_equal(row['frame'], observed(2).final)
                if outcome == 'GAME_OVER':
                    self.assertEqual(policy.choose_action(policy.observation).id, 0)
                    policy.confirm_result(observed(6, level=1))
                    self.assertEqual(policy._examples('goal'), [])

    def test_no_native_progress_leaves_goal_and_adam_untouched_in_both_modes(self):
        for mode in ('steps', 'convergence'):
            with self.subTest(mode=mode):
                config = replace(no_learning(), stopping=mode, goal_steps=1)
                policy = AdaptivePolicy(runtime(), config, seed=3)
                before = [p.detach().clone() for p in policy.bank.parameters(('goal',))]
                policy.choose_action(observed())
                policy.confirm_result(observed(2))
                self.assertEqual(policy.diagnostics['updates']['goal'], 0)
                self.assertFalse(policy.bank.optimizers['goal'].state)
                self.assertTrue(all(torch.equal(a, b) for a, b in
                                    zip(before, policy.bank.parameters(('goal',)))))
                if mode == 'convergence':
                    self.assertEqual(policy.diagnostics['convergence']['goal']['reason'], 'no_progress')

    def test_three_steps_reuse_predicted_latent_memory_and_real_action_order(self):
        policy = AdaptivePolicy(runtime(), no_learning())
        # Real observed states 0 -> 1 -> 3 -> 6 with actions +1, +2, +3.
        # One-step predictions are perfect. Imagined memory adds an error of
        # +1 then +2, giving 9 at the third step: one wrong pixel / one changed.
        for i, color in enumerate((0, 1, 3, 6)):
            obs = Observation(np.full((1, 1, 1), color, np.uint8), (1, 2, 3), 'NOT_FINISHED', 0, 1)
            policy.history.append(Record(i, obs, None if not i else Action(i)))
        states = {r.number: State(torch.tensor([[float(r.observation.final.item())]]),
                                 torch.zeros(1, 1), (1, 1)) for r in policy.history}
        scale = torch.tensor(1., requires_grad=True)

        def predict(z, h, actions, height, width):
            change = actions[:, :8].argmax(-1).float()[:, None, None] + h[:, :, None] * scale
            return z + change, h + 1, change

        def decode(z, height, width):
            return F.one_hot(z.long().reshape(-1, 1, 1), 16).float() * 10

        measured = []
        with patch.object(policy.runtime.engine, 'predict', side_effect=predict), \
             patch.object(policy.runtime.engine, 'decode', side_effect=decode):
            loss, _ = policy._objective('predictor', samples=policy._examples('predictor'),
                                       states=states, measured=measured)()
            loss.backward()
        # Endpoint latent error is (3*scale)^2; average of two horizons in
        # the final example, then three examples: derivative = 3 at scale=1.
        self.assertAlmostEqual(float(scale.grad), 3.)
        metrics = summarize('predictor', measured)
        self.assertEqual(metrics['delta_error'], 0.)
        self.assertEqual(metrics['rollout_samples'], 1)
        self.assertEqual(metrics['rollout_wrong'], 1)
        self.assertEqual(metrics['rollout_union'], 1)
        self.assertEqual(metrics['rollout_delta_error'], 1.)
        self.assertEqual(metrics['latest_rollout_delta_error'], 1.)
        self.assertIsNone(stopping_reason('predictor', metrics, AdaptationConfig()))

    def test_rollouts_require_three_real_same_scene_actions(self):
        missing = replace(observed(3), frames=np.empty((0, 0, 0), np.uint8))
        variants = [dict(boundary=True), dict(action=Action(0)),
                    dict(observation=observed(3, level=1)), dict(observation=missing),
                    dict(observation=observed(3, state='GAME_OVER')),
                    dict(observation=replace(observed(3), frames=np.zeros((1, 4, 6), np.uint8)))]
        for variant in variants:
            with self.subTest(variant=list(variant)):
                policy = AdaptivePolicy(runtime(), no_learning())
                rows = [Record(i, observed(i+1), None if i == 0 else Action(4)) for i in range(4)]
                rows[2] = replace(rows[2], **variant)
                policy.history.extend(rows)
                measured = []
                with policy.runtime.router.activate(policy.bank), torch.no_grad():
                    policy._objective('predictor', samples=policy._examples('predictor'),
                                      states=policy._states(cpu=True), measured=measured)()
                self.assertEqual(sum(r['rollout_samples'] for r in measured), 0)

    def test_real_predictor_trains_three_step_loss_without_changing_base(self):
        policy = AdaptivePolicy(runtime(), no_learning(), seed=3)
        for a, b in ((1, 2), (2, 3), (3, 4)):
            policy.choose_action(observed(a))
            policy.confirm_result(observed(b))
        before = [p.detach().clone() for p in policy.bank.parameters(('predictor',))]
        frozen = base_digest(policy.runtime)
        policy.config = replace(policy.config, predictor_steps=1)
        policy._adapt()
        self.assertEqual(policy.diagnostics['updates']['predictor'], 1)
        self.assertEqual(policy.diagnostics['numerical_failures'], 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, policy.bank.parameters(('predictor',)))))
        self.assertEqual(base_digest(policy.runtime), frozen)

    def test_thresholds_and_latest_rollout_guard_and_missing_evidence(self):
        c = AdaptationConfig()
        self.assertEqual(c.delta_threshold, .05)
        self.assertEqual(c.rollout_delta_threshold, .10)
        good = dict(delta_error=.05, latest_delta_error=.05, rollout_samples=2,
                    rollout_delta_error=.10, latest_rollout_delta_error=.10)
        self.assertEqual(stopping_reason('predictor', good, c), 'threshold')
        for key, value in [('delta_error', .051), ('latest_delta_error', .051),
                           ('rollout_delta_error', .101), ('latest_rollout_delta_error', .101)]:
            with self.subTest(key=key):
                self.assertIsNone(stopping_reason('predictor', good | {key: value}, c))
        missing = good | dict(rollout_samples=0, rollout_delta_error=None, latest_rollout_delta_error=None)
        self.assertEqual(stopping_reason('predictor', missing, c), 'insufficient_rollout_history')
        unchanged = missing | dict(rollout_samples=1, delta_error=None, latest_delta_error=None)
        self.assertEqual(stopping_reason('predictor', unchanged, c), 'no_change')
        for invalid in (-.1, 1.1, float('nan'), True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                AdaptationConfig(rollout_delta_threshold=invalid)

    def test_selective_goal_reset_preserves_other_adapters_adam_history_and_base(self):
        policy = AdaptivePolicy(runtime(), no_learning(), seed=8)
        pristine = [p.detach().clone() for p in policy.bank.parameters(('goal',))]
        policy.choose_action(observed())
        with self.assertRaisesRegex(ValueError, 'pending'):
            policy.reset_goal_adaptation()
        policy.confirm_result(observed(2))
        base = base_digest(policy.runtime)
        with policy.runtime.router.activate(policy.bank, groups=tuple(policy.bank.groups)):
            for group, opt in policy.bank.optimizers.items():
                for p in policy.bank.parameters((group,)):
                    p.grad = torch.ones_like(p)
                opt.step()
                opt.zero_grad(set_to_none=True)
        others = {g: ([p.detach().clone() for p in policy.bank.parameters((g,))],
                      deepcopy(opt.state_dict())) for g, opt in policy.bank.optimizers.items() if g != 'goal'}
        history = list(policy.history)
        counts = dict(policy.diagnostics['updates'])
        policy.diagnostics['losses']['goal'] = {'pre': .01, 'post': .001}
        policy.goal = dict(start=0)
        policy.reset_goal_adaptation()
        self.assertIsNone(policy.goal)
        self.assertNotIn('goal', policy.diagnostics['losses'])
        self.assertEqual(policy.diagnostics['goal_adapter_resets'], 1)
        self.assertEqual(policy.diagnostics['updates'], counts)
        self.assertTrue(all(a is b for a, b in zip(history, policy.history)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(pristine, policy.bank.parameters(('goal',)))))
        self.assertFalse(policy.bank.optimizers['goal'].state)
        self.assertEqual(base_digest(policy.runtime), base)
        for group, (parameters, opt) in others.items():
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(parameters, policy.bank.parameters((group,)))))
            actual = policy.bank.optimizers[group].state_dict()
            self.assertEqual(actual['param_groups'], opt['param_groups'])
            for key, values in opt['state'].items():
                for name, expected in values.items():
                    self.assertTrue(torch.equal(actual['state'][key][name], expected))


if __name__ == '__main__':
    unittest.main()
