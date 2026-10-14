"""Dashboard contracts use literal log records, real files and loopback HTTP."""
import json
import os
import shutil
import subprocess
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from model.dashboard import LogMonitor, BehaviorMonitor, CampaignMonitor, MAX_LINE, MAX_READ, make_server


class DashboardTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js required for dashboard JavaScript rendering contract')
    def test_full_javascript_poll_and_render(self):
        for script in ('dashboard_runtime.cjs', 'campaign_runtime.cjs'):
            with self.subTest(script=script):
                result = subprocess.run(['node', str(Path(__file__).with_name(script))],
                                        text=True, capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'run.log'
        self.monitor = LogMonitor(self.path, max_points=3)

    def append(self, *rows):
        with self.path.open('a') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')

    def test_post_first_action_statistics_share_one_origin_across_consumers(self):
        self.append({'event': 'initializing', 'statistics_origin': 'after_first_action'},
                    {'event': 'resumed', 'actions': 9802, 'updates': 1220, 'episodes': 443})
        warming = self.monitor.snapshot()
        self.assertIsNone(warming['progress'].get('actions'))
        first = {'event': 'progress', 'actions': 9803, 'updates': 1221, 'episodes': 444,
                 'level': 1, 'level_index': 0, 'level_actions': 9803,
                 'statistics': {'actions': 0, 'updates': 0, 'episodes': 0, 'level_actions': 0,
                                'elapsed_seconds': 0, 'actions_per_second': None}}
        second = {'event': 'progress', 'actions': 9805, 'updates': 1223, 'episodes': 445,
                  'level': 1, 'level_index': 0, 'level_actions': 9805,
                  'statistics': {'actions': 2, 'updates': 2, 'episodes': 1, 'level_actions': 2,
                                 'elapsed_seconds': 4, 'actions_per_second': .5}}
        self.append(first, second, {'event': 'validation', 'level': 1, 'actions': 80,
                                   'episodes': 8, 'delta_error': .4},
                    {'event': 'checkpoint', 'actions': 9805})
        data = self.monitor.snapshot()
        self.assertEqual([p['actions'] for p in data['progress_points']], [0, 2])
        self.assertEqual(data['progress']['updates'], 2)
        self.assertEqual(data['progress']['episodes'], 1)
        self.assertEqual(data['progress']['actions_per_second'], .5)
        self.assertEqual(data['progress']['elapsed_seconds'], 4)
        self.assertEqual(data['validations'][0]['actions'], 2)
        self.assertEqual(data['validations'][0]['validation_actions'], 80)
        self.assertEqual(data['events'][-1]['actions'], 2)
        self.assertEqual(data['run']['statistics_origin'], 'after_first_action')
        # A restarted dashboard can begin at any complete progress record.
        self.path.write_text(json.dumps(second) + '\n')
        cold = LogMonitor(self.path).snapshot()
        self.assertEqual(cold['progress']['actions'], 2)
        self.assertEqual(cold['run']['statistics_origin'], 'after_first_action')
        self.append({'event': 'initializing', 'statistics_origin': 'after_first_action'},
                    {'event': 'resumed', 'actions': 12000})
        restarted = self.monitor.snapshot()
        self.assertIsNone(restarted['progress'].get('actions'))
        self.assertEqual(restarted['progress_points'], [])
        self.assertEqual(restarted['validations'], [])

    def test_incremental_partial_noise_and_validation_counters(self):
        self.append({'event': 'initializing', 'game': 'demo', 'levels': [1, 2],
                     'training': {'delta_error': .01, 'false_change_weight': 2., 'freeze_representation': True}},
                    {'event': 'progress', 'actions': 512, 'updates': 60, 'episodes': 23,
                     'level': 2, 'loss': {'decoder_ce': 2.5}, 'actions_per_second': 3.5})
        with self.path.open('ab') as stream:
            stream.write(b'INFO: SDK output\n{not json}\n[1,2]\n{"event":"progress","actions":')
        first = self.monitor.snapshot()
        self.assertEqual(first['progress']['actions'], 512)
        self.assertEqual(len(first['progress_points']), 1)
        self.assertEqual(self.monitor.snapshot()['revision'], first['revision'])
        with self.path.open('ab') as stream:
            stream.write(b'544,"updates":64,"level":2}\n')
        for level, error in ((1, .2), (2, .4)):
            self.append({'event': 'validation', 'level': level, 'actions': 17, 'episodes': 8,
                         'delta_error': error, 'decoder_delta_error': .1, 'passes': False,
                         'extra_changes': 12, 'missed_changes': 0,
                         'wrong_replacement_colors': 3, 'correct_changes': 40})
        result = self.monitor.snapshot()
        self.assertEqual(result['progress']['actions'], 544)
        self.assertEqual(result['progress']['episodes'], 23)
        self.assertEqual([(r['level'], r['actions'], r['updates'], r['delta_error'])
                          for r in result['validations']], [(1, 544, 64, .2), (2, 544, 64, .4)])
        self.assertEqual(result['validations'][1]['validation_actions'], 17)
        self.assertEqual(result['validations'][1]['validation_episodes'], 8)
        self.assertEqual({k: result['validations'][1][k] for k in (
            'extra_changes', 'missed_changes', 'wrong_replacement_colors', 'correct_changes')},
            {'extra_changes': 12, 'missed_changes': 0, 'wrong_replacement_colors': 3, 'correct_changes': 40})
        self.assertEqual(result['run']['training']['delta_error'], .01)
        self.assertEqual(result['run']['training']['false_change_weight'], 2.)
        self.assertIs(result['run']['training']['freeze_representation'], True)

    def test_gate_mode_and_evidence_survive_tail_only_restart(self):
        self.append({'event': 'initializing', 'training': {'auto_representation': True,
                     'freeze_representation': False, 'representation_unfreeze_error': .02,
                     'representation_freeze_error': .01, 'representation_gate_patience': 2}})
        gate = dict(auto_representation=True, freeze_representation=True, eligible=True,
                    reconstruction_error=.005, open_streak=0, close_streak=0,
                    unfreeze_error=.02, freeze_error=.01, patience=2,
                    changed=True, reason='reconstruction_low')
        self.append({'event': 'representation_gate', **gate})
        data = self.monitor.snapshot()
        self.assertTrue(data['run']['training']['freeze_representation'])
        self.assertEqual(data['run']['representation_gate']['reason'], 'reconstruction_low')
        row = {'event': 'progress', 'actions': 20, 'freeze_representation': True,
               'representation_gate': gate}
        self.path.write_text(json.dumps(row) + '\n')
        cold = LogMonitor(self.path).snapshot()
        self.assertTrue(cold['run']['training']['auto_representation'])
        self.assertTrue(cold['run']['training']['freeze_representation'])
        self.assertEqual(cold['run']['representation_gate']['reconstruction_error'], .005)
        self.append({'event': 'representation_gate', **gate, 'freeze_representation': False,
                     'reconstruction_error': .05, 'reason': 'reconstruction_high'})
        self.assertFalse(self.monitor.snapshot()['run']['training']['freeze_representation'])

    def test_restart_append_truncate_regrow_and_file_replacement(self):
        for method in ('initializing', 'truncate', 'regrow', 'replace'):
            with self.subTest(method=method):
                self.path.write_text('')
                monitor = LogMonitor(self.path)
                self.append({'event': 'initializing', 'game': 'old'},
                            {'event': 'progress', 'actions': 1000},
                            {'event': 'validation', 'level': 1, 'delta_error': .5})
                old = monitor.snapshot()
                fresh = '{"event":"initializing","game":"new"}\n{"event":"progress","actions":32}\n'
                if method == 'initializing':
                    with self.path.open('a') as stream:
                        stream.write(fresh)
                elif method == 'replace':
                    replacement = self.path.with_suffix('.new')
                    replacement.write_text(fresh)
                    replacement.replace(self.path)
                else:
                    self.path.write_text(fresh + ('noise\n' * 100 if method == 'regrow' else ''))
                new = monitor.snapshot()
                self.assertGreater(new['generation'], old['generation'])
                self.assertEqual(new['run']['game'], 'new')
                self.assertEqual(new['progress']['actions'], 32)
                self.assertEqual(new['validations'], [])
                self.assertEqual(len(new['progress_points']), 1)

    def test_missing_stale_stopped_and_nonfinite_metrics(self):
        self.assertFalse(self.monitor.snapshot()['source']['available'])
        self.append({'event': 'progress', 'actions': 64, 'loss': {'loss': float('nan')}},
                    {'event': 'stopped', 'actions': 70, 'updates': 4, 'reason': 'encoder_collapsed'})
        old = time.time() - 120
        os.utime(self.path, (old, old))
        data = self.monitor.snapshot()
        self.assertGreaterEqual(data['source']['age_seconds'], 119)
        self.assertEqual(data['phase'], 'stopped')
        self.assertEqual(data['reason'], 'encoder_collapsed')
        self.assertEqual(data['progress']['actions'], 70)
        self.assertIsNone(data['progress']['loss']['loss'])
        json.dumps(data, allow_nan=False)
        self.path.unlink()
        missing = self.monitor.snapshot()
        self.assertFalse(missing['source']['available'])
        self.assertEqual(missing['progress']['actions'], 70)

    def test_bounded_retention_oversize_and_initial_tail(self):
        self.path.write_bytes(b'x' * (MAX_READ + 100) + b'\n')
        for action in range(10):
            self.append({'event': 'progress', 'actions': action},
                        {'event': 'validation', 'level': None, 'delta_error': .5})
        data = self.monitor.snapshot()
        self.assertTrue(data['source']['tail_limited'])
        self.assertEqual([p['actions'] for p in data['progress_points']], [7, 8, 9])
        self.assertEqual([p['actions'] for p in data['validations']], [7, 8, 9])
        self.assertIsNone(data['validations'][0]['level'])
        with self.path.open('ab') as stream:
            stream.write(b'x' * (MAX_LINE + 10))
        self.monitor.snapshot()
        with self.path.open('ab') as stream:
            stream.write(b'"event":"progress","actions":900}\n')
        self.append({'event': 'progress', 'actions': 10})
        self.assertEqual(self.monitor.snapshot()['progress']['actions'], 10)

    def test_actual_http_assets_metrics_and_fixed_read_only_routes(self):
        self.append({'event': 'progress', 'actions': 256, 'updates': 28})
        behavior_path = self.path.with_name('behavior.log')
        behavior_path.write_text('{"event":"behavior_collection","actions":7,"episodes":2}\n')
        campaign_path = self.path.with_name('campaign.log')
        campaign_path.write_text('{"event":"campaign_progress","variant":"rollout","seed":2,"actions":9,"total_actions":100}\n')
        server = make_server(('127.0.0.1', 0), self.monitor, BehaviorMonitor(behavior_path), CampaignMonitor(campaign_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_port}'
        for route, mime in (('/', 'text/html'), ('/dashboard.js', 'text/javascript'),
                            ('/campaign.html', 'text/html'), ('/campaign.js', 'text/javascript'),
                            ('/dashboard.css', 'text/css'), ('/api/state', 'application/json')):
            with urlopen(base + route) as response:
                self.assertEqual(response.headers.get_content_type(), mime)
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
                body = response.read()
                self.assertTrue(body)
                if route == '/api/state':
                    data = json.loads(body)
                    self.assertEqual(data['progress']['actions'], 256)
                    self.assertEqual(data['behavior']['progress']['collection']['actions'], 7)
                    self.assertEqual(data['campaign']['progress']['actions'], 9)
                    self.assertEqual(data['campaign']['progress']['seed'], 2)
        self.append({'event': 'progress', 'actions': 288})
        with urlopen(base + '/api/state') as response:
            self.assertEqual(json.load(response)['progress']['actions'], 288)
        for route in ('/run.log', '/latest.pt', '/../config.py', '/%2e%2e/config.py', '/api/start'):
            with self.assertRaises(HTTPError) as error:
                urlopen(base + route)
            self.assertEqual(error.exception.code, 404)

    def test_campaign_inner_evaluation_progress_keeps_training_counters_and_context(self):
        monitor = CampaignMonitor(self.path)
        progress = {'event': 'behavior_evaluation_progress', 'role': 'campaign', 'split': 'dev',
                    'level': 1, 'episodes': 16, 'wins': 7, 'policy': 'value_mpc', 'horizon': 2}
        self.append(progress)
        self.assertNotIn('evaluation', monitor.snapshot()['progress'], 'orphan events have no reliable variant identity')
        self.append({'event': 'campaign_phase', 'variant': 'rollout2', 'seed': 0,
                     'stage': 'behavior_evaluation', 'actions': 16384}, progress)
        snapshot = monitor.snapshot()
        self.assertEqual(snapshot['progress']['actions'], 16384)
        self.assertEqual(snapshot['progress']['evaluation']['wins'], 7)
        self.assertEqual(snapshot['executions'][0]['evaluation']['episodes'], 16)
        self.assertEqual(snapshot['evaluations'], [], 'partial evaluation is not a completed campaign result')
        self.append(dict(progress, episodes=0, wins=0, actions=3, current_episode=1,
                         total_episodes=32, episode_actions=3, episode_budget=128,
                         seconds=2.25, episode_seconds=2.25, game='si01-v1', pair_index=6, pair_count=10))
        live = monitor.snapshot()['progress']
        self.assertEqual(live['actions'], 16384)
        self.assertEqual(live['evaluation']['actions'], 3)
        self.assertEqual(live['evaluation']['current_episode'], 1)
        self.assertEqual(live['evaluation']['pair_count'], 10)
        self.assertEqual(live['evaluation']['game'], 'si01-v1')
        self.assertEqual(live['evaluation']['seconds'], 2.25)
        self.append(dict(progress, event='behavior_evaluation', episodes=32, wins=14,
                         actions=3000, decision_ms=120, actions_per_second=8))
        self.assertEqual(monitor.snapshot()['progress']['evaluation']['decision_ms'], 120)
        self.append(dict(progress, level=2, wins=0))
        current = monitor.snapshot()['progress']['evaluation']
        self.assertEqual(current['wins'], 0)
        self.assertNotIn('decision_ms', current, 'a new level cannot inherit the previous latency')
        self.append(dict(progress, split='final', wins=15))
        self.assertEqual(monitor.snapshot()['progress']['evaluation']['wins'], 0)
        self.append({'event': 'campaign_evaluation', 'variant': 'rollout2', 'seed': 0, 'actions': 16384,
                     'split': 'dev', 'behavior': {'win_rate': .4}})
        self.assertNotIn('evaluation', monitor.snapshot()['progress'])
        self.append({'event': 'campaign_phase', 'variant': 'rollout2', 'seed': 1, 'stage': 'initializing', 'actions': 0}, progress)
        self.assertNotIn('evaluation', monitor.snapshot()['progress'], 'previous seed progress cannot leak')

    def test_campaign_report_refresh_retention_and_read_only_summary(self):
        path = self.path.with_name('report.json')
        monitor = CampaignMonitor(self.path, max_points=2, report_path=path)
        self.assertFalse(monitor.snapshot()['report']['available'])

        report = {'status': 'running', 'selected': None,
            'config': {'variants': ['world_reference', 'rollout2'], 'seeds': [0, 1, 2], 'actions': 100},
            'cells': [{'variant': 'world_reference', 'seed': 0,
                'world': {'delta_error': .2, 'per_pair': [{'private_trace': 'not exposed'}]},
                'behavior': {'win_rate': .5, 'wins': 2, 'episodes': 4, 'results': ['not exposed']},
                'training': {'actions': 100, 'per_pair': ['not exposed']}}]}
        path.write_text(json.dumps(report))
        original = path.read_bytes()
        snapshot = monitor.snapshot()['report']
        self.assertTrue(snapshot['available'])
        self.assertEqual(snapshot['cells'][0]['behavior']['win_rate'], .5)
        self.assertNotIn('results', snapshot['cells'][0]['behavior'])
        self.assertNotIn('per_pair', snapshot['cells'][0]['world'])
        self.assertEqual(path.read_bytes(), original)
        for action in range(5):
            self.append({'event': 'campaign_progress', 'variant': 'rollout2', 'seed': 1, 'actions': action})
        self.assertEqual(len(monitor.snapshot()['report']['cells']), 1)
        # The trainer atomically replaces its report; a cold dashboard can recover it too.
        report.update(status='complete', selected='world_reference', final={
            'selected': {'win_rate': .75, 'results': ['not exposed']}, 'reference': {'win_rate': .5}})
        replacement = path.with_suffix('.tmp')
        replacement.write_text(json.dumps(report)); replacement.replace(path)
        snapshot = monitor.snapshot()['report']
        self.assertEqual(snapshot['status'], 'complete')
        self.assertEqual(snapshot['final']['selected'], {'win_rate': .75})
        self.assertEqual(CampaignMonitor(self.path, report_path=path).snapshot()['report'], snapshot)
        path.write_text('{broken')
        self.assertFalse(monitor.snapshot()['report']['available'], 'do not silently reuse stale results')
        path.unlink()
        self.assertFalse(monitor.snapshot()['report']['available'])

    def test_campaign_counters_and_evaluation_are_isolated_by_variant_and_seed(self):
        monitor = CampaignMonitor(self.path, max_points=3)
        self.append({'event': 'campaign_started', 'config': {'variants': ['baseline', 'rollout'],
                    'seeds': [0, 1], 'actions_per_variant': 100}},
                    {'event': 'campaign_progress', 'variant': 'baseline', 'seed': 0, 'actions': 100,
                     'total_actions': 100, 'stage': 'world', 'loss': {'rollout_mse': 0}},
                    {'event': 'campaign_evaluation', 'variant': 'baseline', 'seed': 0, 'actions': 100,
                     'split': 'development', 'world': {'delta_error': .2, 'rollouts': {
                         '2': {'samples': 3, 'latent_mse': .4, 'identity_latent_mse': .7}}},
                     'behavior': {'win_rate': .5, 'actions': 38, 'decision_ms': 2}},
                    {'event': 'campaign_progress', 'variant': 'rollout', 'seed': 1, 'actions': 2,
                     'total_actions': 100, 'stage': 'world', 'loss': {'rollout_mse': float('nan')}})
        data = monitor.snapshot()
        self.assertEqual(data['progress']['actions'], 2)
        self.assertEqual(data['run']['config']['seeds'], [0, 1])
        self.assertEqual([row['actions'] for row in data['progress_points']], [100, 2])
        self.assertEqual(data['evaluations'][0]['actions'], 100)
        self.assertEqual(data['evaluations'][0]['behavior']['actions'], 38)
        self.assertEqual(data['evaluations'][0]['world']['rollouts']['2']['latent_mse'], .4)
        self.assertIsNone(data['progress']['loss']['rollout_mse'])
        json.dumps(data, allow_nan=False)
        self.append({'event': 'campaign_error', 'message': 'explicit failure'})
        self.assertEqual(monitor.snapshot()['reason'], 'explicit failure')
        self.append({'event': 'campaign_started', 'config': {'variants': ['local']}},
                    {'event': 'campaign_finished', 'status': 'complete', 'selected': 'local'})
        restarted = monitor.snapshot()
        self.assertEqual(restarted['progress_points'], [])
        self.assertEqual(restarted['evaluations'], [])
        self.assertIsNone(restarted['reason'])
        self.assertEqual(restarted['run']['selected'], 'local')

    def test_campaign_missing_tail_and_bounded_history(self):
        monitor = CampaignMonitor(self.path, max_points=3)
        self.assertFalse(monitor.snapshot()['source']['available'])
        for action in range(5):
            self.append({'event': 'campaign_progress', 'variant': 'x', 'seed': 0, 'actions': action},
                        {'event': 'campaign_evaluation', 'variant': 'x', 'seed': 0, 'actions': action})
        data = monitor.snapshot()
        self.assertEqual([r['actions'] for r in data['progress_points']], [2, 3, 4])
        self.assertEqual([r['actions'] for r in data['evaluations']], [2, 3, 4])
        self.path.write_text('{"event":"campaign_progress","variant":"new","seed":2,"actions":1}\n')
        self.assertEqual(monitor.snapshot()['evaluations'], [])

    def test_campaign_summaries_survive_curve_eviction_without_mixing_runs(self):
        monitor = CampaignMonitor(self.path, max_points=2)
        self.append({'event': 'campaign_progress', 'variant': 'reference', 'seed': 0,
                     'actions': 100, 'updates': 12, 'loss': {'latent_mse': .2}},
                    {'event': 'campaign_evaluation', 'variant': 'reference', 'seed': 0,
                     'actions': 100, 'world': {'delta_error': .3, 'per_pair': [
                         {'game': 'a', 'level': 1, 'latent_std': .6, 'extra_changes': 7},
                         {'game': 'b', 'level': 1, 'latent_std': .9, 'extra_changes': 0}]}})
        for action in (1, 2, 3):
            self.append({'event': 'campaign_progress', 'variant': 'rollout', 'seed': 1,
                         'actions': action, 'updates': action, 'loss': {'latent_mse': .5}})
        data = monitor.snapshot()
        self.assertEqual([p['actions'] for p in data['progress_points']], [2, 3])
        self.assertEqual([(p['variant'], p['seed'], p['actions'], p['updates'])
                          for p in data['executions']], [('reference', 0, 100, 12), ('rollout', 1, 3, 3)])
        self.assertEqual(data['executions'][0]['loss']['latent_mse'], .2)
        self.assertEqual(data['evaluations'][0]['world']['per_pair'], [
            {'game': 'a', 'level': 1, 'latent_std': .6, 'extra_changes': 7},
            {'game': 'b', 'level': 1, 'latent_std': .9, 'extra_changes': 0}])
        # Restarting the dashboard reconstructs summaries from the same log.
        self.assertEqual(CampaignMonitor(self.path, max_points=2).snapshot()['executions'], data['executions'])
        self.append({'event': 'campaign_started', 'config': {'actions': 200}})
        self.assertEqual(monitor.snapshot()['executions'], [])

    def test_large_campaign_evaluation_keeps_metrics_but_not_episode_payload(self):
        self.append({'event': 'campaign_evaluation', 'variant': 'reference', 'seed': 0,
                     'actions': 16384, 'world': {'delta_error': .2},
                     'behavior': {'win_rate': .5, 'actions_per_second': 12., 'wilson95': [.4, .6],
                                  'results': [{'payload': 'x' * (MAX_LINE + 1)}]}})
        data = CampaignMonitor(self.path).snapshot()
        self.assertEqual(data['evaluations'][0]['world']['delta_error'], .2)
        self.assertEqual(data['evaluations'][0]['behavior'],
                         {'win_rate': .5, 'actions_per_second': 12., 'wilson95': [.4, .6]})
        self.assertNotIn('payload', json.dumps(data))

    def test_campaign_phase_changes_status_without_fabricating_progress_points(self):
        monitor = CampaignMonitor(self.path)
        self.append({'event': 'campaign_started', 'config': {'actions': 40}},
                    {'event': 'campaign_phase', 'variant': 'reference', 'seed': 0,
                     'stage': 'initializing', 'actions': 0, 'total_actions': 40,
                     'message': 'Preparing compiled kernels'})
        data = monitor.snapshot()
        self.assertEqual(data['phase'], 'campaign_phase')
        self.assertEqual(data['progress']['stage'], 'initializing')
        self.assertEqual(data['progress']['message'], 'Preparing compiled kernels')
        self.assertEqual(data['progress_points'], [])
        self.append({'event': 'campaign_progress', 'variant': 'reference', 'seed': 0,
                     'stage': 'world', 'actions': 10, 'actions_per_second': 5},
                    {'event': 'campaign_phase', 'variant': 'reference', 'seed': 0,
                     'stage': 'head_calibration', 'actions': 10})
        data = monitor.snapshot()
        self.assertEqual(data['progress']['stage'], 'head_calibration')
        self.assertNotIn('actions_per_second', data['progress'])
        self.assertEqual([p['actions'] for p in data['progress_points']], [10])
        self.append({'event': 'campaign_phase', 'variant': 'rollout', 'seed': 1,
                     'stage': 'initializing', 'actions': 0})
        data = monitor.snapshot()
        self.assertEqual(data['progress']['actions'], 0)
        self.assertEqual(len(data['progress_points']), 1)

    def test_campaign_deferred_imagination_is_not_reported_as_training(self):
        monitor = CampaignMonitor(self.path)
        self.append({'event': 'campaign_started'},
                    {'event': 'campaign_imagination_gate', 'variant': 'actor_imagined', 'seed': 0, 'passed': False},
                    {'event': 'campaign_variant_deferred', 'variant': 'actor_imagined',
                     'reason': 'incomplete_h2_admission', 'fallback': 'actor_imitation'},
                    {'event': 'campaign_finished', 'selected': 'mc', 'status': 'complete'})
        data = monitor.snapshot()
        self.assertFalse(data['run']['imagination_gate']['passed'])
        self.assertEqual(data['run']['deferred']['actor_imagined']['fallback'], 'actor_imitation')
        self.assertEqual(data['progress_points'], [])
        self.assertEqual(data['evaluations'], [])

    def test_behavior_logs_isolate_splits_rounds_missing_values_and_restarts(self):
        monitor = BehaviorMonitor(self.path, max_points=3)
        self.append({'event': 'behavior_started', 'world_trainable': 0, 'level': 2},
                    {'event': 'behavior_collection', 'round': 1, 'actions': 20, 'episodes': 3},
                    {'event': 'behavior_fit', 'round': 1, 'epoch': 1, 'loss': 2., 'event_ce': .8, 'value_bce': .6, 'q_bce': .6},
                    {'event': 'behavior_evaluation', 'split': 'development', 'policy': 'random', 'episodes': 10, 'wins': 2, 'actions': 90},
                    {'event': 'behavior_evaluation', 'split': 'final', 'policy': 'mpc', 'role': 'initial', 'episodes': 20, 'wins': 7, 'wilson95': [.18, .57]},
                    {'event': 'behavior_finished', 'world_unchanged': True})
        data = monitor.snapshot()
        self.assertEqual(data['progress']['collection']['actions'], 20)
        self.assertEqual([r['split'] for r in data['evaluations']], ['development', 'final'])
        self.assertEqual(data['evaluations'][1]['wilson95'], [.18, .57])
        self.assertEqual(data['evaluations'][1]['role'], 'initial')
        self.assertNotIn('decision_ms', data['evaluations'][1])
        self.assertEqual(data['phase'], 'behavior_finished')
        self.assertTrue(data['run']['world_unchanged'])
        self.append({'event': 'behavior_started', 'world_trainable': 0},
                    {'event': 'behavior_fit', 'epoch': 1, 'loss': 0.},
                    {'event': 'behavior_evaluation', 'wins': 0, 'episodes': 2})
        data = monitor.snapshot()
        self.assertNotIn('collection', data['progress'])
        self.assertEqual(data['fits'][0]['loss'], 0.)
        self.assertNotIn('event_ce', data['fits'][0])
        self.assertEqual(data['evaluations'][0]['split'], 'unspecified')
        for i in range(6):
            self.append({'event': 'behavior_fit', 'epoch': i + 2, 'loss': 1.})
        self.assertEqual([r['epoch'] for r in monitor.snapshot()['fits']], [5, 6, 7])
        # Replacement without a start event also establishes a new lifetime.
        self.path.write_text('{"event":"behavior_collection","actions":1}\n')
        data = monitor.snapshot()
        self.assertEqual(data['fits'], [])
        self.assertEqual(data['evaluations'], [])
        self.assertEqual(data['progress']['collection']['actions'], 1)

    def test_behavior_multilevel_metrics_keep_aggregate_and_level_rows_distinct(self):
        monitor = BehaviorMonitor(self.path)
        self.append({'event': 'behavior_started', 'levels': [3, 4, 5], 'level': None, 'world_trainable': 0},
                    {'event': 'behavior_collection', 'actions': 12, 'level': 5,
                     'per_level': {'3': {'episodes': 2, 'actions': 4}, '4': {'episodes': 2, 'actions': 4},
                                   '5': {'episodes': 2, 'actions': 4}}},
                    {'event': 'behavior_evaluation', 'split': 'development', 'level': 3, 'wins': 2, 'episodes': 2},
                    {'event': 'behavior_evaluation', 'split': 'development', 'level': 4, 'wins': 0, 'episodes': 2},
                    {'event': 'behavior_evaluation', 'split': 'development', 'level': 5, 'wins': 1, 'episodes': 2},
                    {'event': 'behavior_evaluation', 'split': 'development', 'level': None, 'wins': 3, 'episodes': 6},
                    {'event': 'behavior_world_verified', 'world_unchanged': True})
        data = monitor.snapshot()
        self.assertEqual(data['run']['levels'], [3, 4, 5])
        self.assertTrue(data['run']['world_unchanged'])
        self.assertEqual([r['level'] for r in data['evaluations']], [3, 4, 5, None])
        self.assertEqual(data['progress']['collection']['per_level']['5']['episodes'], 2)


if __name__ == '__main__':
    unittest.main()
