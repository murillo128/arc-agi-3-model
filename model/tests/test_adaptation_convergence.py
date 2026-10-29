"""Convergence contracts: real CPU learning, fixed evidence and honest stops."""
from copy import deepcopy
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import torch

from model.adaptation import AdaptationConfig, AdaptivePolicy
from model.adaptation_convergence import delta_metrics, goal_metrics, path_metrics, summarize, stopping_reason
from model.adaptation_smoke import bank_digest, base_digest
from model.tests.test_adaptation import runtime, observed, no_learning


class ConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_has_no_action_time_or_update_ceiling(self):
        config = AdaptationConfig()
        self.assertEqual(config.stopping, 'convergence')
        self.assertIsNone(config.per_observation_seconds)
        self.assertIsNone(config.total_seconds)
        policy = AdaptivePolicy(runtime(), no_learning())
        for i in range(81):
            policy.choose_action(observed(i % 3 + 1))
            self.assertTrue(policy.confirm_result(observed((i + 1) % 3 + 1)))
        self.assertIsNone(policy.max_actions)
        self.assertEqual(policy.actions_taken, 81)
        self.assertIsNotNone(policy.choose_action(policy.observation))

    def test_real_lora_fits_all_heads_beyond_one_update_without_mutating_base(self):
        shared = runtime()
        config = AdaptationConfig(learning_rate=.01, exploration=1.)
        policy = AdaptivePolicy(shared, config, seed=3)
        inactive = AdaptivePolicy(shared, config, seed=4)
        frozen, other = base_digest(shared), bank_digest(inactive.bank)
        action = policy.choose_action(observed())
        self.assertEqual(policy.diagnostics['convergence']['predictor']['reason'], 'no_data')
        policy.confirm_result(observed(3), action)
        fits = policy.diagnostics['convergence']
        for name in ('representation', 'path'):
            self.assertEqual(fits[name]['reason'], 'threshold', (name, fits[name]))
            self.assertTrue(fits[name]['threshold_met'])
            self.assertGreaterEqual(fits[name]['elapsed_seconds'], 0.)
        self.assertEqual(fits['goal']['reason'], 'no_progress')
        self.assertEqual(fits['predictor']['reason'], 'insufficient_rollout_history')
        self.assertFalse(fits['predictor']['threshold_met'])
        self.assertEqual(fits['predictor']['final']['metrics']['rollout_samples'], 0)
        self.assertIsNone(fits['predictor']['final']['metrics']['rollout_delta_error'])
        for name in ('predictor', 'path'):
            self.assertGreater(fits[name]['steps'], 1)
            self.assertLess(fits[name]['final']['loss'], fits[name]['initial']['loss'])
        self.assertLessEqual(fits['predictor']['final']['metrics']['delta_error'], .05)
        self.assertLessEqual(fits['path']['final']['metrics']['cost_error'], .10)
        policy.choose_action(observed(3))
        policy.confirm_result(observed(4, level=1))
        goal = policy.diagnostics['convergence']['goal']
        self.assertEqual(goal['reason'], 'threshold')
        self.assertGreater(goal['steps'], 1)
        self.assertLess(goal['final']['loss'], goal['initial']['loss'])
        self.assertLessEqual(goal['final']['metrics']['mask_error'], .10)
        self.assertEqual(base_digest(shared), frozen)
        self.assertEqual(bank_digest(inactive.bank), other)
        self.assertTrue(all(p.grad is None for p in shared.engine.model.parameters()))
        self.assertIsNone(shared.router.active)

    def test_hand_counted_errors_do_not_hide_omissions_or_inventions(self):
        before = torch.tensor([[[0, 0, 0, 0, 0]]])
        actual = torch.tensor([[[1, 0, 2, 3, 0]]])
        predicted = torch.tensor([[[0, 4, 5, 3, 0]]])
        stats = delta_metrics(before, actual, predicted)
        self.assertEqual(stats, dict(missed=1, invented=1, wrong_color=1, wrong=3, union=4, changed=3))
        self.assertEqual(summarize('predictor', [stats])['delta_error'], .75)
        empty = delta_metrics(before, before, before)
        empty_metric = summarize('predictor', [empty])
        self.assertIsNone(empty_metric['delta_error'])
        self.assertEqual(stopping_reason('predictor', empty_metric, AdaptationConfig()), 'insufficient_rollout_history')
        # Historical success must not hide a completely wrong new transition.
        easy = dict(missed=0, invented=0, wrong_color=0, wrong=0, union=100, changed=100)
        metrics = summarize('predictor', [easy, stats])
        self.assertLess(metrics['delta_error'], .10)
        self.assertIsNone(stopping_reason('predictor', metrics, AdaptationConfig()))

    def test_goal_and_path_use_separate_fractional_errors_not_raw_loss(self):
        before = torch.zeros(1, 1, 4, dtype=torch.long)
        actual = torch.tensor([[[1, 1, 0, 0]]])
        mask = torch.tensor([[[True, True, False, False]]])
        logits = torch.zeros(1, 1, 4, 16)
        logits[0, 0, 0, 1] = 1
        logits[0, 0, 1, 0] = 1
        metrics = goal_metrics(logits, torch.tensor([[[1., -1., -1., -1.]]]), before, actual, mask)
        self.assertEqual(metrics, dict(color_error=.5, mask_error=.25))
        metrics = path_metrics(torch.tensor([math.log(9.)]), torch.tensor([math.log1p(3.)]), True, 4)
        self.assertAlmostEqual(metrics['reachability_error'], .1, places=6)
        self.assertAlmostEqual(metrics['cost_error'], .25, places=6)
        failed = path_metrics(torch.tensor([0.]), torch.tensor([1000.]), False, 4)
        self.assertEqual(failed, dict(reachability_error=.5, cost_error=None))

    def test_empty_latest_change_unions_cannot_borrow_pooled_convergence(self):
        cases = [((0, 1, 2, 2), 'latest_delta_error', 'no_latest_change', 2, 1),
                 ((0, 1, 2, 3, 1), 'latest_rollout_delta_error', 'no_latest_rollout_change', 4, 1)]
        config = AdaptationConfig()
        for colors, missing, reason, one_step_union, rollout_union in cases:
            with self.subTest(colors=colors):
                frames = [torch.tensor([[[color]]]) for color in colors]
                rows = []
                for i in range(1, len(frames)):
                    # Perfect predictions: every changed single pixel contributes
                    # one to the union, while equal start/end colors contribute zero.
                    stats = delta_metrics(frames[i-1], frames[i], frames[i])
                    stats.update(rollout_samples=int(i >= 3), rollout_wrong=0, rollout_union=0)
                    if i >= 3:
                        rollout = delta_metrics(frames[i-3], frames[i], frames[i])
                        stats.update(rollout_wrong=rollout['wrong'], rollout_union=rollout['union'])
                    rows.append(stats)
                metrics = summarize('predictor', rows)
                self.assertEqual(metrics['union'], one_step_union)
                self.assertEqual(metrics['rollout_union'], rollout_union)
                self.assertEqual(metrics['rollout_samples'], len(colors)-3)
                self.assertEqual(metrics['delta_error'], 0.)
                self.assertEqual(metrics['rollout_delta_error'], 0.)
                self.assertIsNone(metrics[missing])
                self.assertEqual(stopping_reason('predictor', metrics, config), reason)
                # Missing evidence must not stop fitting any available failing error.
                for key in ('delta_error', 'latest_delta_error',
                            'rollout_delta_error', 'latest_rollout_delta_error'):
                    if key != missing:
                        self.assertIsNone(stopping_reason('predictor', metrics | {key: 1.}, config))

    def test_unavailable_latest_guards_keep_required_update_and_allow_next_action(self):
        for missing, reason in [('latest_delta_error', 'no_latest_change'),
                                ('latest_rollout_delta_error', 'no_latest_rollout_change')]:
            with self.subTest(missing=missing):
                policy = AdaptivePolicy(runtime(), no_learning(learning_rate=1.), seed=3)
                for before, after in ((1, 2), (2, 3), (3, 4), (4, 2)):
                    policy.choose_action(observed(before))
                    policy.confirm_result(observed(after))
                policy.config = replace(policy.config, stopping='convergence', predictor_steps=1)
                parameter = policy.bank.parameters(('predictor',))[0]
                with torch.no_grad():
                    parameter.zero_()

                def objective(name, *, samples=None, states=None, measured=None, weights=None):
                    def calculate():
                        if measured is not None:
                            for _, after in samples:
                                latest = after.number == policy.history[-1].number
                                measured.append(dict(wrong=0, union=int(not (latest and missing == 'latest_delta_error')),
                                    rollout_samples=int(after.number >= 3), rollout_wrong=0,
                                    rollout_union=int(after.number >= 3 and not (latest and missing == 'latest_rollout_delta_error'))))
                        # Adam at LR=1 overshoots this optimum. The initial loss
                        # remains best, so accidental rollback would erase the update.
                        return (parameter.flatten()[0] - .1).square(), None
                    return calculate

                with patch.object(policy, '_objective', side_effect=objective):
                    policy._adapt()
                result = policy.diagnostics['convergence']['predictor']
                self.assertEqual(result['reason'], reason)
                self.assertFalse(result['threshold_met'])
                self.assertIsNone(result['final']['metrics'][missing])
                self.assertEqual(result['minimum_steps'], 1)
                self.assertEqual(result['steps'], 1)
                self.assertEqual(policy.diagnostics['updates']['predictor'], 1)
                self.assertGreater(result['final']['loss'], result['initial']['loss'])
                self.assertFalse(result['restored_best'])
                self.assertNotEqual(float(parameter.flatten()[0].detach()), 0.)
                state = policy.bank.optimizers['predictor'].state[parameter]
                self.assertEqual(float(state['step']), 1.)
                self.assertIsNotNone(policy.choose_action(policy.observation))

    def test_full_replay_is_deterministic_and_keeps_latest_eligible_samples(self):
        policy = AdaptivePolicy(runtime(), no_learning(replay_size=3), seed=8)
        for i in range(5):
            policy.choose_action(observed(i % 3 + 1))
            policy.confirm_result(observed((i + 1) % 3 + 1))
        rng = policy.rng.getstate()
        self.assertEqual(len(policy._examples('representation')), 4)
        pairs = policy._examples('predictor')
        self.assertEqual([(a.number, b.number) for a, b in pairs], [(2, 3), (3, 4), (4, 5)])
        self.assertEqual(policy._examples('goal'), [])  # no native success yet
        labels = policy._examples('path')
        self.assertEqual(len({r['start'] for r in labels}), len(labels))
        self.assertTrue(any(r['end'] == 5 for r in labels))
        self.assertEqual(policy.rng.getstate(), rng)
        self.assertEqual([(r['start'], r['end']) for r in labels],
                         [(r['start'], r['end']) for r in policy._examples('path')])

    def test_plateau_restores_best_parameters_and_adam_moments(self):
        policy = AdaptivePolicy(runtime(), AdaptationConfig(learning_rate=1., exploration=1.,
            plateau_patience=2, predictor_steps=0, goal_steps=0, path_steps=0))
        parameter = policy.bank.parameters(('encoder',))[0]
        with torch.no_grad():
            parameter.zero_()
        opt = policy.bank.optimizers['encoder']
        original = opt.step
        saved = []

        def step():
            original()
            saved.append((parameter.detach().clone(), deepcopy(opt.state_dict())))

        def objective(name, *, samples=None, states=None, measured=None, weights=None):
            def calculate():
                if measured is not None:
                    measured.append(dict(reconstruction_error=1.))
                # Independent scalar quadratic; Adam at LR=1 overshoots its
                # optimum after the second update. Error stays unmet on purpose.
                return (parameter.flatten()[0] - 2.).square(), torch.tensor(1.)
            return calculate

        with patch.object(policy, '_objective', side_effect=objective), patch.object(opt, 'step', side_effect=step):
            policy.choose_action(observed())
        result = policy.diagnostics['convergence']['representation']
        self.assertEqual(result['reason'], 'plateau')
        self.assertFalse(result['threshold_met'])
        self.assertTrue(result['restored_best'])
        self.assertEqual(result['steps'], 4)
        self.assertTrue(torch.equal(parameter, saved[1][0]))
        current = opt.state_dict()
        self.assertEqual(current['param_groups'], saved[1][1]['param_groups'])
        for key, state in current['state'].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, saved[1][1]['state'][key][name]))
        self.assertEqual(float(next(iter(current['state'].values()))['step']), 2.)

    def test_cancel_during_adaptation_preserves_admission_and_returns_no_new_action(self):
        policy = AdaptivePolicy(runtime(), AdaptationConfig(learning_rate=.01, exploration=1.), seed=3)
        policy.choose_action(observed())
        original = policy._update

        def stop_after_step(*args, **kwargs):
            done = original(*args, **kwargs)
            policy.request_stop()
            return done

        with patch.object(policy, '_update', side_effect=stop_after_step):
            self.assertTrue(policy.confirm_result(observed(3)))
        self.assertEqual(policy.actions_taken, 1)
        self.assertEqual(policy.diagnostics['confirmed_transitions'], 1)
        self.assertEqual(policy.diagnostics['convergence']['predictor']['reason'], 'cancelled')
        self.assertIsNone(policy.pending_action)
        with self.assertRaises(InterruptedError):
            policy.choose_action(observed(3))

    def test_convergence_numerical_failure_restores_bank(self):
        policy = AdaptivePolicy(runtime(), AdaptationConfig(learning_rate=.01), seed=3)
        before = bank_digest(policy.bank)
        opt = policy.bank.optimizers['encoder']
        original = opt.step

        def corrupt():
            original()
            with torch.no_grad():
                policy.bank.parameters(('encoder',))[0].fill_(float('nan'))

        with patch.object(opt, 'step', side_effect=corrupt):
            policy.choose_action(observed())
        self.assertEqual(policy.diagnostics['convergence']['representation']['reason'], 'numerical_failure')
        self.assertEqual(policy.diagnostics['numerical_failures'], 1)
        self.assertEqual(bank_digest(policy.bank), before)
        self.assertFalse(opt.state)

    def test_configuration_rejects_bad_thresholds_and_accepts_optional_limits(self):
        self.assertEqual(AdaptationConfig(stopping='steps', total_seconds=300).total_seconds, 300)
        for kwargs in ({'stopping': 'forever'}, {'delta_threshold': -1}, {'goal_mask_threshold': 2},
                       {'plateau_patience': 0}, {'plateau_min_relative_improvement': 0},
                       {'path_cost_threshold': float('nan')}, {'total_seconds': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AdaptationConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
