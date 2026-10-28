"""Runner contracts on scripted public SDK responses, never gameplay scores."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from arcengine import FrameData, GameState

from model.data import Action
from model.online_lora_experiment import aggregate_counts, counts, run_game


def frame(value=0, state=GameState.NOT_FINISHED, levels=0):
    return FrameData(game_id='scripted', frame=[[[value]]], state=state,
                     levels_completed=levels, win_levels=2, available_actions=[4])


class Policy:
    def __init__(self, budget, events):
        self.max_actions, self.events, self.actions_taken = budget, events, 0
        self.pending = None
        self.diagnostics = dict(adaptation_seconds=0., updates={}, numerical_failures=0)

    def choose_action(self, observation):
        self.events.append(('choose', self.actions_taken))
        self.pending = Action(0 if observation.state == 'GAME_OVER' else 4)
        return self.pending

    def cancel_pending(self):
        self.pending = None
        self.events.append(('cancel', self.actions_taken))

    def confirm_result(self, observation, action):
        if self.pending != action:
            return False
        self.events.append(('confirm', self.actions_taken, observation.state))
        self.actions_taken += 1
        self.diagnostics['adaptation_seconds'] += .1
        self.pending = None
        return True


class OnlineExperimentTests(unittest.TestCase):
    def test_unlimited_defaults_continue_past_eighty_until_win(self):
        events = []
        policy = Policy(None, events)
        calls = []

        def step(action, **kwargs):
            calls.append(action.value)
            return frame(state=GameState.WIN if len(calls) == 81 else GameState.NOT_FINISHED)

        with patch('model.online_lora_experiment.time.monotonic', return_value=10000.):
            result = run_game(SimpleNamespace(observation_space=frame(), step=step), policy,
                started=0., emit=lambda e: None, reconstruct=lambda *a: None,
                predict=lambda *a: (None, {}))
        self.assertEqual(result['actions'], 81)
        self.assertEqual(result['confirmations'], 81)
        self.assertEqual(result['stop_reason'], 'win')
        self.assertEqual(result['wall_overrun_seconds'], 0.)
        self.assertIsNone(result['adaptation_limits']['total_seconds'])

    def test_cancellation_during_learning_keeps_already_confirmed_real_action(self):
        policy = Policy(None, [])
        original = policy.confirm_result

        def confirm(observation, action):
            original(observation, action)
            raise KeyboardInterrupt

        with patch.object(policy, 'confirm_result', side_effect=confirm), \
             patch('model.online_lora_experiment.time.monotonic', return_value=0.):
            result = run_game(SimpleNamespace(observation_space=frame(), step=lambda *a, **k: frame(1)),
                policy, started=0., emit=lambda e: None, reconstruct=lambda *a: None,
                predict=lambda *a: (None, {}))
        self.assertEqual(result['stop_reason'], 'cancelled')
        self.assertEqual((result['actions'], result['confirmations']), (1, 1))
        self.assertTrue(result['truncated'])

    def test_delta_categories_and_empty_denominator(self):
        # Missed change, invented change, wrong new color, correct change, keep.
        before = np.array([[0, 0, 0, 0, 0]])
        actual = np.array([[1, 0, 2, 3, 0]])
        predicted = np.array([[0, 4, 5, 3, 0]])
        result = aggregate_counts([counts(before, actual, predicted)])
        self.assertEqual(result, dict(wrong=3, union=4, changed=3, missed=1,
            invented=1, wrong_color=1, correct=1, samples=1,
            unchanged_samples=0, error_percent=75.))
        self.assertIsNone(aggregate_counts([counts(before, before, before)])['error_percent'])

    def test_prediction_and_measurement_precede_learning_of_real_result(self):
        events, saved = [], []
        policy = Policy(1, events)

        def predict(policy, observation, action):
            events.append(('predict', policy.actions_taken))
            return np.array([[0]]), {}

        def reconstruct(policy, observation):
            events.append(('reconstruct', policy.actions_taken, int(observation.final[0, 0])))

        def step(action, **kwargs):
            events.append(('sdk', action.value))
            return frame(1)

        with patch('model.online_lora_experiment.time.monotonic', return_value=0.):
            result = run_game(SimpleNamespace(observation_space=frame(), step=step), policy,
                started=0., emit=saved.append, predict=predict, reconstruct=reconstruct)
        self.assertEqual(events, [('reconstruct', 0, 0), ('choose', 0), ('predict', 0),
            ('sdk', 4), ('reconstruct', 0, 1), ('confirm', 0, 'NOT_FINISHED')])
        self.assertEqual(result['prediction']['missed'], 1)
        self.assertEqual(result['prediction']['union'], 1)
        self.assertEqual(result['actions'], 1)
        self.assertEqual(result['confirmations'], 1)
        self.assertEqual([e['event'] for e in saved],
            ['initial', 'request', 'sdk_result', 'measured_before_learning', 'action'])

    def test_reset_noop_and_terminal_at_budget_all_count_and_confirm(self):
        events = []
        policy = Policy(3, events)
        responses = iter([frame(state=GameState.GAME_OVER), frame(),
                          frame(1, state=GameState.WIN, levels=2)])
        executed = []

        def step(action, **kwargs):
            executed.append(action.value)
            return next(responses)

        with patch('model.online_lora_experiment.time.monotonic', return_value=0.):
            result = run_game(SimpleNamespace(observation_space=frame(), step=step), policy,
                started=0., emit=lambda e: None, reconstruct=lambda *a: None,
                predict=lambda *a: (np.array([[0]]), {}))
        self.assertEqual(executed, [4, 0, 4])
        self.assertEqual((result['actions'], result['confirmations'], result['resets']), (3, 3, 1))
        self.assertEqual(result['actions_by_level']['1'],
                         dict(actions=3, resets=1, ordinary_actions=2, no_visual_effect=2))
        self.assertEqual(result['prediction']['samples'], 1)
        self.assertTrue(result['win'])
        self.assertEqual(result['stop_reason'], 'win')

    def test_deadline_after_decision_cancels_and_after_sdk_preserves_result(self):
        for during_decision in (True, False):
            with self.subTest(during_decision=during_decision):
                now, events = [0.], []
                policy = Policy(2, events)
                executed = []

                def predict(*args):
                    if during_decision:
                        now[0] = 901.
                    return np.array([[0]]), {}

                def step(action, **kwargs):
                    executed.append(action.value)
                    now[0] = 901.
                    return frame(1)

                with patch('model.online_lora_experiment.time.monotonic', side_effect=lambda: now[0]):
                    result = run_game(SimpleNamespace(observation_space=frame(), step=step), policy,
                        started=0., emit=lambda e: None, seconds=900., predict=predict, reconstruct=lambda *a: None)
                self.assertEqual(result['stop_reason'], 'wall_timeout')
                self.assertEqual(result['wall_overrun_seconds'], 1.)
                self.assertEqual(result['confirmations'], 0 if during_decision else 1)
                self.assertEqual(len(executed), 0 if during_decision else 1)
                self.assertIsNone(policy.pending)

    def test_missing_sdk_response_never_trains_or_hides_attempt(self):
        events = []
        policy = Policy(2, events)
        with patch('model.online_lora_experiment.time.monotonic', return_value=0.), \
             self.assertLogs(level='ERROR'):
            result = run_game(SimpleNamespace(observation_space=frame(), step=lambda *a, **k: None),
                policy, started=0., emit=lambda e: None, predict=lambda *a: (None, {}),
                reconstruct=lambda *a: None)
        self.assertEqual(result['stop_reason'], 'error')
        self.assertEqual((result['requests'], result['actions'], result['confirmations']), (1, 0, 0))
        self.assertEqual(policy.diagnostics['adaptation_seconds'], 0.)
        self.assertIsNone(policy.pending)


if __name__ == '__main__':
    unittest.main()
