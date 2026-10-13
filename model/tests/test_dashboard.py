"""Dashboard contracts use literal log records, real files and loopback HTTP."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from model.dashboard import LogMonitor, BehaviorMonitor, MAX_LINE, MAX_READ, make_server


class DashboardTests(unittest.TestCase):
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
        server = make_server(('127.0.0.1', 0), self.monitor, BehaviorMonitor(behavior_path))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_port}'
        for route, mime in (('/', 'text/html'), ('/dashboard.js', 'text/javascript'),
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
        self.append({'event': 'progress', 'actions': 288})
        with urlopen(base + '/api/state') as response:
            self.assertEqual(json.load(response)['progress']['actions'], 288)
        for route in ('/run.log', '/latest.pt', '/../config.py', '/%2e%2e/config.py', '/api/start'):
            with self.assertRaises(HTTPError) as error:
                urlopen(base + route)
            self.assertEqual(error.exception.code, 404)

    def test_behavior_logs_isolate_splits_rounds_missing_values_and_restarts(self):
        monitor = BehaviorMonitor(self.path, max_points=3)
        self.append({'event': 'behavior_started', 'world_trainable': 0, 'level': 2},
                    {'event': 'behavior_collection', 'round': 1, 'actions': 20, 'episodes': 3},
                    {'event': 'behavior_fit', 'round': 1, 'epoch': 1, 'loss': 2., 'event_ce': .8, 'value_bce': .6, 'q_bce': .6},
                    {'event': 'behavior_evaluation', 'split': 'development', 'policy': 'random', 'episodes': 10, 'wins': 2, 'actions': 90},
                    {'event': 'behavior_evaluation', 'split': 'final', 'policy': 'mpc', 'episodes': 20, 'wins': 7, 'wilson95': [.18, .57]},
                    {'event': 'behavior_finished', 'world_unchanged': True})
        data = monitor.snapshot()
        self.assertEqual(data['progress']['collection']['actions'], 20)
        self.assertEqual([r['split'] for r in data['evaluations']], ['development', 'final'])
        self.assertEqual(data['evaluations'][1]['wilson95'], [.18, .57])
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


if __name__ == '__main__':
    unittest.main()
