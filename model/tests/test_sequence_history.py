"""Derived dashboard history: real files, process restarts and HTTP boundaries."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from model.dashboard import make_server
from model.sequence_history import SequenceHistory


def row(batch):
    return dict(batch=batch, actions=batch*512, loss=.25, measurement_source='training_pre_update',
                quality={'a': dict(wrong=2, union=10, transitions=8)},
                frames=[[[9]*64]*64], diagnostics=dict(reconstruction_error=0),
                sessions=[dict(game='a', steps=512)])


class SequenceHistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.report, self.archive = self.root/'report.json', self.root/'history.json'

    def write(self, batches, identity='run-1', games=('a',)):
        payload = dict(kind='game_sequences', games=list(games), optimization_identity=identity,
                       status='running', history=[row(b) for b in batches])
        pending = self.report.with_suffix('.pending')
        pending.write_text(json.dumps(payload))
        pending.replace(self.report)
        return payload

    def test_numeric_history_survives_restart_deduplicates_and_retains_only_the_bound(self):
        self.write([1,2])
        history = SequenceHistory(self.report, self.archive, 3)
        original = self.report.read_bytes()
        body, revision = history.snapshot()
        self.assertNotIn('frames', json.loads(body)['history'][0])
        self.assertEqual(json.loads(body)['history'][0]['diagnostics']['reconstruction_error'], 0)
        history.refresh()
        self.assertEqual(history.snapshot()[1], revision)
        self.assertEqual(self.report.read_bytes(), original)
        self.write([2,3,4])
        history.refresh()
        restored = SequenceHistory(self.report, self.archive, 3)
        self.assertEqual([r['batch'] for r in json.loads(restored.snapshot()[0])['history']], [2,3,4])
        self.assertEqual(restored.metadata()['points'], 3)
        self.write([4,5])
        restored.refresh()
        self.assertEqual([r['batch'] for r in json.loads(restored.snapshot()[0])['history']], [3,4,5])

    def test_run_cohort_and_rollback_cannot_reuse_unrelated_or_future_records(self):
        self.write([8,9])
        history = SequenceHistory(self.report, self.archive)
        for batches, identity, games in [([1,2],'run-1',('a',)),([3,4],'run-2',('a',)),([5,6],'run-2',('b',))]:
            self.write(batches,identity,games)
            history.refresh()
            self.assertEqual([r['batch'] for r in json.loads(history.snapshot()[0])['history']],batches)

    def test_unavailable_or_foreign_archive_keeps_live_metrics_and_never_overwrites_report(self):
        self.write([1,2])
        with self.assertRaises(ValueError):
            SequenceHistory(self.report,self.report)
        self.archive.write_text('unrelated content')
        history=SequenceHistory(self.report,self.archive)
        self.assertEqual(self.archive.read_text(),'unrelated content')
        self.assertTrue(history.metadata()['error'])
        self.assertEqual(len(json.loads(history.snapshot()[0])['history']),2)
        blocked=self.root/'file';blocked.write_text('not a directory')
        failed=SequenceHistory(self.report,blocked/'archive.json')
        self.assertTrue(failed.metadata()['error'])
        self.assertEqual(len(json.loads(failed.snapshot()[0])['history']),2)
        self.report.write_text('{partial')
        failed.refresh()
        self.assertEqual(len(json.loads(failed.snapshot()[0])['history']),2)

    def test_http_conditional_history_and_background_collection_without_a_browser(self):
        self.write([1,2])
        server=make_server(('127.0.0.1',0),None,sequence_report=self.report,
                           sequence_history=self.archive,max_points=4)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            base=f'http://127.0.0.1:{server.server_port}'
            with urlopen(base+'/api/sequence-history') as response:
                revision=response.headers['ETag'];self.assertEqual(len(json.load(response)['history']),2)
            with self.assertRaises(HTTPError) as unchanged:
                urlopen(Request(base+'/api/sequence-history',headers={'If-None-Match':revision}))
            self.assertEqual(unchanged.exception.code,304)
            self.write([2,3])
            # No HTTP request triggers collection; the service itself watches the report.
            deadline=time.monotonic()+7
            while time.monotonic()<deadline and json.loads(self.archive.read_text())['history'][-1]['batch']!=3:
                time.sleep(.05)
            self.assertEqual(json.loads(self.archive.read_text())['history'][-1]['batch'],3)
            with urlopen(base+'/api/state') as response:
                state=json.load(response)
            self.assertEqual(state['sequence_history']['points'],3)
            self.assertEqual(len(state['sequences']['history']),2,'live state is not expanded with the archive')
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)
        self.assertFalse(any(t.name=='sequence-history' and t.is_alive() for t in threading.enumerate()))

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for numerical JavaScript contracts')
    def test_window_statistics(self):
        result=subprocess.run(['node',str(Path(__file__).with_name('sequence_trends_runtime.cjs'))],
                              capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
