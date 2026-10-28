"""Public confirmation, isolated learning and causal label contracts."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.adaptation import AdaptationConfig, AdaptivePolicy, Runtime, color_macro
from model.adaptation_heads import VisualGoal
from model.adaptation_smoke import bank_digest, base_digest
from model.config import NetworkConfig, TrainConfig
from model.data import Action, Observation
from model.learning import Engine
from model.networks import WorldModel
from model.path_planning import PathValue


def runtime():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(145)
        c = replace(NetworkConfig.small(), patch=2, pixel_latent=True,
                    local_dynamics=True, rotation_adapter_width=4)
        return Runtime(SimpleNamespace(engine=Engine(WorldModel(c).eval(), TrainConfig()),
            goal=VisualGoal(c.latent, c.memory, c.patch, 8).eval(),
            path=PathValue(c.latent, c.memory, 8).eval(), metadata={}))


def observed(column=1, *, level=0, state='NOT_FINISHED', actions=(4,)):
    frame = np.zeros((1, 8, 8), dtype=np.uint8)
    frame[0, 2:4, column:column+2] = 9
    return Observation(frame, actions, state, level, 5)


def config(**kwargs):
    return replace(AdaptationConfig(), **(dict(stopping='steps', exploration=1., representation_steps=1,
                   per_observation_seconds=30., total_seconds=300.) | kwargs))


def no_learning(**kwargs):
    return replace(AdaptationConfig(), stopping='steps', exploration=1., representation_steps=0,
                   predictor_steps=0, path_steps=0, goal_steps=0, **kwargs)


class AdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_adaptive_precision_uses_native_support_on_the_selected_device(self):
        # Simulate an engine whose training policy accepts BF16 emulation.
        # No CUDA allocation: only the capability-selection boundary is doubled.
        for native in (False, True):
            with self.subTest(native=native):
                bundle = runtime()
                bundle.engine.device = torch.device('cuda:1')
                bundle.engine.amp = True
                selected = []

                @contextmanager
                def device_scope(device):
                    selected.append(device)
                    try:
                        yield
                    finally:
                        selected.pop()

                def supported(including_emulation=True):
                    return including_emulation or (selected == [torch.device('cuda:1')] and native)

                with patch('torch.cuda.device', side_effect=device_scope), \
                     patch('torch.cuda.is_bf16_supported', side_effect=supported):
                    adapted = Runtime(bundle)
                self.assertEqual(adapted.engine.amp, native)
                self.assertEqual(selected, [])
                with patch('torch.autocast') as autocast:
                    with adapted.engine.autocast():
                        pass
                if native:
                    autocast.assert_called_once_with('cuda', dtype=torch.bfloat16)
                else:
                    autocast.assert_not_called()

    def test_adaptive_cpu_precision_does_not_probe_cuda(self):
        with patch('torch.cuda.is_bf16_supported', side_effect=AssertionError('CUDA probe on CPU')):
            self.assertFalse(runtime().engine.amp)

    def test_initial_only_representation_then_all_groups_learn_from_real_confirmation(self):
        shared = runtime()
        policy = AdaptivePolicy(shared, config(learning_rate=.001), seed=3)
        before = base_digest(shared)
        pristine = {k: [p.clone() for p in policy.bank.parameters((k,))] for k in policy.bank.groups}
        action = policy.choose_action(observed())
        self.assertEqual(policy.diagnostics['updates'], dict(representation=1, predictor=0, path=0, goal=0))
        pending_updates = dict(policy.diagnostics['updates'])
        self.assertEqual(policy.choose_action(observed()), action)
        self.assertEqual(policy.diagnostics['updates'], pending_updates)
        self.assertEqual(policy.diagnostics['confirmed_transitions'], 0)
        policy.confirm_result(observed(3), action)
        self.assertEqual(policy.diagnostics['predictive_pairs'], 1)
        self.assertTrue(all(v > 0 for v in policy.diagnostics['updates'].values()))
        for group in pristine:
            self.assertTrue(any(not torch.equal(a, b) for a, b in
                zip(pristine[group], policy.bank.parameters((group,)))), group)
        for metric in policy.diagnostics['losses'].values():
            self.assertTrue(np.isfinite([metric['pre'], metric['post']]).all())
            self.assertLess(metric['post'], metric['pre'])
        self.assertEqual(base_digest(shared), before)
        self.assertTrue(all(p.grad is None for p in shared.engine.model.parameters()))

    def test_pending_cancel_duplicate_terminal_and_exact_action_budget(self):
        policy = AdaptivePolicy(runtime(), no_learning(), max_actions=2)
        action = policy.choose_action(observed())
        with self.assertRaisesRegex(ValueError, 'explicit SDK'):
            policy.choose_action(observed(2))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            policy.confirm_result(observed(2), Action(1))
        policy.cancel_pending()
        self.assertFalse(policy.confirm_result(observed(2)))
        self.assertEqual(len(policy.history), 1)
        action = policy.choose_action(observed())
        policy.confirm_result(observed(2), action)
        action = policy.choose_action(observed(2))
        policy.confirm_result(observed(3, state='WIN'), action)
        self.assertFalse(policy.confirm_result(observed(3, state='WIN'), action))
        self.assertEqual(policy.diagnostics['confirmed_transitions'], 2)
        self.assertEqual(policy.diagnostics['native_outcomes']['WIN'], 1)
        self.assertEqual(policy.history[-1].observation.state, 'WIN')
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            policy.choose_action(observed(3, state='WIN'))

    def test_boundaries_keep_adapters_but_exclude_pairs_and_clear_goals(self):
        policy = AdaptivePolicy(runtime(), config(), seed=9)
        policy.choose_action(observed())
        bank = policy.bank
        policy.confirm_result(observed(2, level=1))
        self.assertIs(policy.bank, bank)
        self.assertEqual(policy.diagnostics['predictive_pairs'], 0)
        self.assertEqual(policy._hindsight(), [])
        self.assertIsNone(policy.goal)
        # Boundary animations are reduced to their final frame for memory, but
        # the next public choose still accepts the complete returned observation.
        multi = observed(3, level=2)
        multi = replace(multi, frames=np.concatenate((observed().frames, multi.frames)))
        policy.choose_action(observed(2, level=1))
        policy.confirm_result(multi)
        policy.choose_action(multi)
        self.assertEqual(len(policy.history[-1].observation.frames), 1)
        policy.cancel_pending()
        self.assertEqual(policy.diagnostics['boundary_skips'], 2)

    def test_replay_truncation_rebuilds_zero_carry_and_refreshes_encoder(self):
        policy = AdaptivePolicy(runtime(), no_learning(replay_size=2), seed=1)
        for a, b in ((1, 2), (2, 3), (3, 4)):
            policy.choose_action(observed(a))
            policy.confirm_result(observed(b))
        self.assertEqual(len(policy.history), 3)
        with policy.runtime.router.activate(policy.bank):
            old = policy._states()
            self.assertEqual(torch.count_nonzero(old[1].h).item(), 0)
            pair = next(pair for name, pair in policy.bank.tensors.items() if name.startswith('encoder.'))
            with torch.no_grad():
                pair[1].fill_(.1)
            new = policy._states()
            self.assertFalse(torch.equal(old[3].z, new[3].z))
            self.assertFalse(torch.equal(old[3].h, new[3].h))
            self.assertEqual(torch.count_nonzero(new[1].h).item(), 0)

    def test_hindsight_masks_real_noops_and_pending_or_censored_goals(self):
        policy = AdaptivePolicy(runtime(), no_learning())
        policy.choose_action(observed())
        self.assertEqual(policy._hindsight(), [])
        policy.goal = dict(frame=observed(4).final, mask=np.ones((8, 8), bool),
                           start=0, budget=2, created_action=0)
        policy.confirm_result(observed())
        self.assertEqual(policy.diagnostics['predictive_pairs'], 1)  # real no-op
        self.assertEqual(policy._hindsight(), [])
        self.assertEqual(len(policy.path_labels), 0)  # pending is not failure
        policy.choose_action(observed())
        policy.confirm_result(observed(2))
        self.assertEqual(policy.path_labels[0]['success'], False)
        self.assertEqual(policy.path_labels[0]['distance'], 2)
        labels = policy._hindsight()
        self.assertTrue(labels)
        for label in labels:
            np.testing.assert_array_equal(label['mask'], observed().final != observed(2).final)
            np.testing.assert_array_equal(label['frame'], observed(2).final)
            self.assertTrue(label['success'])
        policy.choose_action(observed(2))
        policy.goal = dict(frame=observed(4).final, mask=np.ones((8, 8), bool),
                           start=2, budget=2, created_action=2)
        policy.confirm_result(observed(3, level=1))
        self.assertEqual(len(policy.path_labels), 1)  # boundary adds no failure
        self.assertEqual(policy.diagnostics['goal_outcomes']['censored'], 1)

    def test_empty_geometry_reset_and_no_undo_have_explicit_semantics(self):
        policy = AdaptivePolicy(runtime(), no_learning())
        self.assertEqual(policy.choose_action(observed(actions=(4, 7))).id, 4)
        policy.confirm_result(replace(observed(2), frames=np.zeros((0, 0, 0), np.uint8)))
        self.assertEqual(policy.diagnostics['boundary_skips'], 1)
        policy.choose_action(policy.observation)
        policy.confirm_result(observed(2, state='GAME_OVER'))
        self.assertEqual(policy.choose_action(policy.observation).id, 0)
        policy.confirm_result(observed(3))
        self.assertEqual(policy.diagnostics['predictive_pairs'], 0)
        self.assertEqual(policy.history[-1].action.id, 0)
        with policy.runtime.router.activate(policy.bank):
            state = policy._states()[policy.history[-1].number]
            self.assertTrue(torch.isfinite(state.h).all())

    def test_budget_stops_learning_but_legal_inference_remains(self):
        policy = AdaptivePolicy(runtime(), config(total_seconds=.00001))
        with patch('model.adaptation.time.monotonic', side_effect=[0., 1., 2.]):
            action = policy.choose_action(observed())
        self.assertEqual(action.id, 4)
        self.assertEqual(sum(policy.diagnostics['updates'].values()), 0)
        self.assertEqual(policy.diagnostics['budget_stops'], 1)

    def test_elapsed_limit_is_checked_between_steps_and_total_budget_persists(self):
        policy = AdaptivePolicy(runtime(), config(per_observation_seconds=10., total_seconds=10.))
        with patch('model.adaptation.time.monotonic', side_effect=[0., 0., 11., 12.]):
            policy.choose_action(observed())
        self.assertEqual(policy.diagnostics['updates']['representation'], 1)
        self.assertEqual(policy.diagnostics['updates']['predictor'], 0)
        self.assertEqual(policy.diagnostics['adaptation_seconds'], 12.)
        policy.confirm_result(observed(2))
        self.assertEqual(sum(policy.diagnostics['updates'].values()), 1)
        self.assertEqual(policy.choose_action(observed(2)).id, 4)

    def test_geometry_change_and_observed_progress_do_not_supply_cross_scene_targets(self):
        policy = AdaptivePolicy(runtime(), no_learning())
        policy.choose_action(observed())
        policy.confirm_result(observed(2))
        self.assertFalse(any(row['progress'] for row in policy._hindsight()))
        policy.choose_action(observed(2))
        new_scene = replace(observed(3, level=1), frames=np.zeros((1, 4, 6), np.uint8))
        policy.confirm_result(new_scene)
        self.assertEqual(policy.diagnostics['predictive_pairs'], 1)
        labels = policy._hindsight()
        self.assertTrue(all(row['progress'] for row in labels))
        self.assertTrue(all(row['frame'].shape == (8, 8) for row in labels))
        with policy.runtime.router.activate(policy.bank):
            self.assertEqual(policy._states()[2].geometry, (4, 6))

    def test_transactional_numerical_failure_does_not_change_any_bank_or_base(self):
        shared = runtime()
        active, other = AdaptivePolicy(shared, config()), AdaptivePolicy(shared, config(), seed=1)
        base, inactive = base_digest(shared), bank_digest(other.bank)
        # Adam failure happens after backward; rollback must also restore moments.
        optimizer = active.bank.optimizers['encoder']
        original = optimizer.step
        def corrupt():
            original()
            with torch.no_grad():
                active.bank.parameters(('encoder',))[0].fill_(float('nan'))
        bank_before = bank_digest(active.bank)
        with patch.object(optimizer, 'step', side_effect=corrupt):
            active.choose_action(observed())
        self.assertEqual(active.diagnostics['numerical_failures'], 1)
        self.assertEqual(bank_digest(active.bank), bank_before)
        self.assertFalse(optimizer.state)
        self.assertEqual(bank_digest(other.bank), inactive)
        self.assertEqual(base_digest(shared), base)
        self.assertIsNone(shared.router.active)

    def test_concurrent_game_histories_match_isolated_learning(self):
        shared = runtime()
        def run(policy):
            for left, right in ((1, 2), (2, 3)):
                action = policy.choose_action(observed(left))
                policy.confirm_result(observed(right), action)
            return bank_digest(policy.bank), policy.diagnostics['updates']
        policies = [AdaptivePolicy(shared, config(), seed=i) for i in (2, 3)]
        with ThreadPoolExecutor(max_workers=2) as executor:
            concurrent = list(executor.map(run, policies))
        isolated = [run(AdaptivePolicy(shared, config(), seed=i)) for i in (2, 3)]
        self.assertEqual(concurrent, isolated)
        self.assertIsNot(policies[0].history, policies[1].history)

    def test_color_macro_error_cannot_be_dominated_by_background(self):
        target = torch.zeros(1, 1, 100, dtype=torch.long)
        target[0, 0, 0] = 9
        logits = torch.zeros(1, 1, 100, 16)
        logits[..., 0] = 10
        _, error = color_macro(logits, target)
        self.assertEqual(float(error), .5)

    def test_config_rejects_invalid_or_nonfinite_values(self):
        for kwargs in ({'replay_size': 0}, {'minibatch': 5}, {'per_observation_seconds': float('nan')},
                       {'exploration': 0}, {'predictor_steps': -1}, {'allow_undo': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AdaptationConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
