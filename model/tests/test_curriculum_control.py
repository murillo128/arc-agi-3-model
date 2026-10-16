"""Loopback HTTP -> pinned child process, without touching a live campaign."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from model.dashboard import make_server
from model.curriculum_control import CurriculumControl


CHILD = '''
import signal, sys
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--output') + 1])
def received(number, frame):
    with (root / 'signals').open('a') as f:
        f.write(str(number) + '\\n')
if '--no-handlers' not in sys.argv:
    signal.signal(signal.SIGUSR2, received)
    signal.signal(signal.SIGHUP, received)
(root / 'ready').touch()
while True:
    signal.pause()
'''


@unittest.skipUnless(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'Linux pidfd required')
class CurriculumControlTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        package = root / 'model'
        package.mkdir()
        (package / '__init__.py').write_text('')
        (package / 'subgoal_training.py').write_text(CHILD)
        self.output = root / 'run with spaces'
        self.output.mkdir()
        self.report = self.output / 'report.json'
        self.data = dict(kind='synthetic_subgoals', status='running',
                         config={}, progress=dict(actions=10, total_actions=100,
                             representation_level=dict(game='sc01-v1', level=1)),
                         curriculum=dict(levels=[dict(game='sc01-v1', level=1, unlocked=True, planner=False),
                                                dict(game='sc01-v1', level=2, unlocked=False)]),
                         manual_controls=[], control_signals=dict(checkpoint='SIGUSR1', phase='SIGUSR2', level='SIGHUP'))
        self.write_report()
        self.child = subprocess.Popen([sys.executable, '-u', '-m', 'model.subgoal_training', '--output', str(self.output)], cwd=root)
        self.addCleanup(self.stop_child)
        self.wait_for(lambda: (self.output / 'ready').exists())
        self.enterContext(patch('model.dashboard.GpuMonitor.snapshot', return_value={'devices': []}))
        self.systemctl = self.enterContext(patch('model.curriculum_control.subprocess.run',
            return_value=subprocess.CompletedProcess([], 0, str(self.child.pid))))
        self.server = make_server(('127.0.0.1', 0), None, subgoal_report=self.report, curriculum_service='test-curriculum.service')
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.server.shutdown)
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.token = self.state()['token']

    def stop_child(self):
        if self.child.poll() is None:
            self.child.terminate()
        self.child.wait(timeout=5)

    def wait_for(self, condition):
        deadline = time.monotonic() + 3
        while not condition():
            self.assertLess(time.monotonic(), deadline, 'Child did not observe signal')
            time.sleep(.01)

    def write_report(self):
        temp = self.report.with_suffix('.tmp')
        temp.write_text(json.dumps(self.data))
        temp.replace(self.report)

    def state(self):
        with urlopen(self.base + '/api/state') as response:
            self.assertIsNone(response.headers.get('Access-Control-Allow-Origin'))
            return json.load(response)['curriculum_control']

    def post(self, operation, *, token=None):
        request = Request(self.base + '/api/curriculum', data=json.dumps({'operation': operation}).encode(),
                          headers={'Content-Type': 'application/json', 'X-Curriculum-Token': self.token if token is None else token})
        return urlopen(request)

    def test_phase_and_level_are_pending_until_trainer_ack_and_duplicates_are_blocked(self):
        for operation, expected in (('phase', signal.SIGUSR2), ('level', signal.SIGHUP)):
            with self.post(operation) as response:
                self.assertEqual(response.status, 202)
                state = json.load(response)
                self.assertEqual(state['pending'], dict(operation=operation))
                self.assertFalse(state['phase'])
                self.assertFalse(state['level'])
            self.wait_for(lambda: (self.output / 'signals').exists() and
                          len((self.output / 'signals').read_text().splitlines()) == (1 if operation == 'phase' else 2))
            self.assertEqual(int((self.output / 'signals').read_text().splitlines()[-1]), expected)
            with self.assertRaises(HTTPError) as error:
                self.post(operation)
            self.assertEqual(error.exception.code, 409)
            self.assertIsNone(self.state()['result'])
            # Only the trainer acknowledgement changes pending into applied.
            event = dict(actions=12, operation=operation, result='advanced', game='sc01-v1', level=1, phase='predictor')
            self.data['manual_controls'].append(event)
            self.write_report()
            state = self.state()
            self.assertIsNone(state['pending'])
            self.assertEqual(state['result'], dict(status='applied', event=event))
        self.assertEqual((self.output / 'signals').read_text().splitlines(), [str(signal.SIGUSR2), str(signal.SIGHUP)])
        self.systemctl.assert_called_with(['systemctl', '--user', 'show', '--property=MainPID', '--value', '--', 'test-curriculum.service'],
                                          capture_output=True, text=True, timeout=2, check=True)

    def test_rejects_bad_token_operation_wrong_run_stopped_and_unavailable_advances(self):
        for operation, token, code in (('phase', '', 403), ('phase', 'wrong', 403), ('kill', self.token, 400)):
            with self.assertRaises(HTTPError) as error:
                self.post(operation, token=token)
            self.assertEqual(error.exception.code, code)
        self.data['curriculum']['levels'][0]['planner'] = True
        self.data['curriculum']['levels'][1]['unlocked'] = True
        self.write_report()
        state = self.state()
        self.assertFalse(state['phase'])
        self.assertFalse(state['level'])
        with self.assertRaises(HTTPError) as error:
            self.post('level')
        self.assertEqual(error.exception.code, 409)
        wrong = CurriculumControl('test-curriculum.service', self.report.with_name('other.json'))
        self.assertFalse(wrong.snapshot(self.data)['phase'])
        with self.assertRaisesRegex(ValueError, 'otra campaña'):
            wrong.request('phase', self.data)
        # The configured service may not signal an unrelated current process.
        self.systemctl.return_value.stdout = str(os.getpid())
        self.assertFalse(self.state()['phase'])
        with self.assertRaises(HTTPError):
            self.post('phase')
        self.systemctl.return_value.stdout = '0'
        self.assertFalse(self.state()['level'])
        self.assertFalse((self.output / 'signals').exists())

    def test_old_trainer_without_handlers_is_never_signalled(self):
        self.stop_child()
        (self.output / 'ready').unlink()
        self.child = subprocess.Popen([sys.executable, '-m', 'model.subgoal_training', '--output', str(self.output),
                                       '--no-handlers'], cwd=self.output.parent)
        self.wait_for(lambda: (self.output / 'ready').exists())
        self.systemctl.return_value.stdout = str(self.child.pid)
        self.assertFalse(self.state()['phase'])
        with self.assertRaises(HTTPError) as error:
            self.post('phase')
        self.assertEqual(error.exception.code, 409)
        self.assertIsNone(self.child.poll(), 'Default SIGUSR2 disposition would kill this child')

    def test_process_failure_reports_uncertain_request_without_retry(self):
        self.post('phase').close()
        self.wait_for(lambda: (self.output / 'signals').exists())
        self.stop_child()
        state = self.state()
        self.assertIsNone(state['pending'])
        self.assertEqual(state['result']['status'], 'error')
        self.assertFalse(state['phase'])
        self.assertEqual((self.output / 'signals').read_text().splitlines(), [str(signal.SIGUSR2)])

    def test_controls_remain_disabled_without_explicit_service(self):
        with make_server(('127.0.0.1', 0), None, subgoal_report=self.report) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f'http://127.0.0.1:{server.server_port}'
                with urlopen(base + '/api/state') as response:
                    self.assertNotIn('curriculum_control', json.load(response))
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base + '/api/curriculum', data=b'{}'))
                self.assertEqual(error.exception.code, 404)
            finally:
                server.shutdown()
                thread.join()
