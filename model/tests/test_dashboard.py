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

from model.dashboard import LogMonitor, MAX_LINE, MAX_READ, make_server


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

    def test_incremental_partial_noise_and_validation_counters(self):
        self.append({'event': 'initializing', 'game': 'demo', 'levels': [1, 2],
                     'training': {'delta_error': .01}},
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
                         'delta_error': error, 'decoder_delta_error': .1, 'passes': False})
        result = self.monitor.snapshot()
        self.assertEqual(result['progress']['actions'], 544)
        self.assertEqual(result['progress']['episodes'], 23)
        self.assertEqual([(r['level'], r['actions'], r['updates'], r['delta_error'])
                          for r in result['validations']], [(1, 544, 64, .2), (2, 544, 64, .4)])
        self.assertEqual(result['validations'][1]['validation_actions'], 17)
        self.assertEqual(result['validations'][1]['validation_episodes'], 8)
        self.assertEqual(result['run']['training']['delta_error'], .01)

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
        server = make_server(('127.0.0.1', 0), self.monitor)
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
                    self.assertEqual(json.loads(body)['progress']['actions'], 256)
        self.append({'event': 'progress', 'actions': 288})
        with urlopen(base + '/api/state') as response:
            self.assertEqual(json.load(response)['progress']['actions'], 288)
        for route in ('/run.log', '/latest.pt', '/../config.py', '/%2e%2e/config.py', '/api/start'):
            with self.assertRaises(HTTPError) as error:
                urlopen(base + route)
            self.assertEqual(error.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
