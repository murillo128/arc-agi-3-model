"""Runner contracts on scripted public SDK responses, never gameplay scores."""
from types import SimpleNamespace
from pathlib import Path
import json
import logging
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from arcengine import ActionInput, FrameData, FrameDataRaw, GameAction, GameState

from model.data import Action
from model.adaptation import AdaptationConfig
from model.online_lora_experiment import (aggregate_counts, counts, run_game, parse_args,
    verified_checkpoint, convergence_report)
from model.online_lora_reporting import render_report, publish_report


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
    def test_checkpoint_arguments_keep_8246_defaults_and_accept_8458_without_caps(self):
        required = ['--checkpoint', 'base.pt', '--output-dir', 'unused']
        _, original = parse_args(required)
        self.assertEqual(original.checkpoint_batch, 8246)
        self.assertFalse(original.sdk_recording)
        self.assertEqual(original.checkpoint_sha256,
            '4d125575cdc126044ec9b2e773252237cb9743c36032a08d4857f4d876150db7')
        _, current = parse_args(required + ['--checkpoint-batch', '8458', '--checkpoint-sha256',
            '918016ed000669f7bd17b241565f816ad6861aea484cb6c8876791440cf50787',
            '--checkpoint-release', 'https://github.com/murillo128/arc-agi-3-model/releases/tag/lora-base-b8458-20261028',
            '--sdk-recording'])
        self.assertEqual(current.checkpoint_batch, 8458)
        self.assertEqual(current.checkpoint_sha256,
            '918016ed000669f7bd17b241565f816ad6861aea484cb6c8876791440cf50787')
        self.assertIsNone(current.max_actions)
        self.assertIsNone(current.max_game_seconds)
        self.assertIsNone(current.report_issue)
        self.assertTrue(current.sdk_recording)

    def test_sdk_recording_keeps_initial_frames_noop_and_terminal_action_locally(self):
        from arc_agi.models import EnvironmentInfo
        from arc_agi.wrapper import EnvironmentWrapper
        from arc3.envs.sdk import open_arcade

        # Exercise the installed SDK writer with public response fixtures only;
        # no game instance, model, GPU or remote API is needed.
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'sdk-recordings'
            arcade = open_arcade('offline', str(Path(directory) / 'empty-environments'),
                                 recordings_dir=str(target))
            self.assertEqual(arcade.recordings_dir, str(target))
            wrapper = EnvironmentWrapper(EnvironmentInfo(game_id='recording-fixture'),
                logging.getLogger('recording-test'), 'test-card', save_recording=True,
                include_frame_data=True, recordings_dir=arcade.recordings_dir)
            wrapper._guid = 'test-run'
            wrapper._setup_recording_file()
            responses = [
                FrameData(frame=[[[0]], [[1]]], full_reset=True,
                          action_input=ActionInput(id=GameAction.RESET)),
                FrameData(frame=[[[1]]], action_input=ActionInput(id=GameAction.ACTION6, data={'x': 0, 'y': 0})),
                FrameData(frame=[[[2]]], state=GameState.WIN, levels_completed=1, win_levels=1,
                          action_input=ActionInput(id=GameAction.ACTION4)),
            ]
            for response in responses:
                raw = FrameDataRaw(state=response.state, action_input=response.action_input,
                    full_reset=response.full_reset, levels_completed=response.levels_completed,
                    win_levels=response.win_levels)
                raw.frame = [np.array(layer, dtype=np.uint8) for layer in response.frame]
                wrapper._set_last_response(raw)
            paths = list(target.rglob('*.jsonl'))
            self.assertEqual(len(paths), 1)
            records = [json.loads(line)['data'] for line in paths[0].read_text().splitlines()]
            self.assertEqual([r['frame'] for r in records], [[[[0]], [[1]]], [[[1]]], [[[2]]]])
            self.assertEqual([r['action_input']['id'] for r in records], ['RESET', 'ACTION6', 'ACTION4'])
            self.assertEqual(records[1]['action_input']['data'], {'x': 0, 'y': 0})
            self.assertTrue(records[0]['full_reset'])
            self.assertEqual(records[2]['state'], 'WIN')
            self.assertEqual(records[2]['levels_completed'], 1)

    def test_identity_checks_real_file_hash_then_portable_metadata(self):
        # Standard SHA-256 test vector; model deserialization has its own integration test.
        digest = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
        sentinel = SimpleNamespace(runtime=SimpleNamespace(metadata={'source_batches': 8458}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'base.pt'
            path.write_bytes(b'abc')
            identity = dict(expected_sha256=digest, expected_source_batches=8458)
            with patch('model.online_lora_experiment.load', return_value=sentinel) as loader:
                self.assertIs(verified_checkpoint(path, identity, config=AdaptationConfig(), max_actions=None), sentinel)
                self.assertEqual(identity['source_batches'], 8458)
                self.assertEqual(identity['sha256_before'], digest)
                loader.reset_mock()
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    verified_checkpoint(path, dict(expected_sha256='0'*64, expected_source_batches=8458),
                                        config=AdaptationConfig(), max_actions=None)
                loader.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'source batch mismatch'):
                    verified_checkpoint(path, dict(expected_sha256=digest, expected_source_batches=8246),
                                        config=AdaptationConfig(), max_actions=None)

    def test_pending_thresholds_keep_no_change_unavailable_and_do_not_mutate_diagnostics(self):
        diagnostics = dict(convergence={
            'goal': dict(reason='plateau', final=dict(metrics=dict(color_error=.2, mask_error=.05))),
            'predictor': dict(reason='no_change', final=dict(metrics=dict(delta_error=None, latest_delta_error=None)))})
        result = convergence_report(diagnostics, AdaptationConfig())
        self.assertEqual(result['goal']['pending_thresholds'], {'color_error': {'value': .2, 'threshold': .1}})
        self.assertEqual(result['predictor']['pending_thresholds'], {})
        self.assertIsNone(result['predictor']['final']['metrics']['delta_error'])
        self.assertNotIn('pending_thresholds', diagnostics['convergence']['goal'])

        diagnostics['convergence']['predictor'] = dict(reason='plateau', final=dict(metrics=dict(
            delta_error=.04, latest_delta_error=.03,
            rollout_delta_error=.14, latest_rollout_delta_error=.2)))
        result = convergence_report(diagnostics, AdaptationConfig())
        self.assertEqual(result['predictor']['pending_thresholds'], {
            'rollout_delta_error': {'value': .14, 'threshold': .1},
            'latest_rollout_delta_error': {'value': .2, 'threshold': .1}})
        self.assertNotIn('pending_thresholds', diagnostics['convergence']['predictor'])

    def test_failed_publication_retains_scalar_report_and_pending_games(self):
        report = dict(games=[], aggregate={'games_recorded': 0}, status='failed', commit='test',
            checkpoint=dict(expected_source_batches=8458, expected_sha256='a'*64, release='release-8458'))
        body = render_report(report)
        self.assertIn('batch **8458**', body)
        self.assertIn('ar25-0c556536', body)
        self.assertNotIn('arc3-lora-smoke-base', body)
        with tempfile.TemporaryDirectory() as directory, self.assertLogs(level='ERROR'), \
             patch('model.online_lora_reporting.subprocess.run', return_value=SimpleNamespace(returncode=1)):
            publish_report(Path(directory), report, 157)
            self.assertEqual((Path(directory) / 'issue-final.md').read_text(), body)
            self.assertFalse(json.loads((Path(directory) / 'issue-final.publication.json').read_text())['published'])

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
        self.assertEqual(result['prediction_exclusions'], {'reset': 1, 'level_change': 1})
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
