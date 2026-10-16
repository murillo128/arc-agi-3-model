"""Literal executed-frame, sampling and wall-clock contracts; no GPU required."""
import base64
import json
from pathlib import Path
import random
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

from model.data import Action, Episode, Observation
from model.dashboard import make_server
from model.training_telemetry import PlannerTrajectory, RecentThroughput


class TelemetryTests(unittest.TestCase):
    def test_recent_rate_startup_cutoff_idle_and_new_process(self):
        now = [100.]
        rate = RecentThroughput(clock=lambda: now[0])
        self.assertIsNone(rate.value())
        self.assertIsNone(rate.record())
        now[0] = 101.; self.assertEqual(rate.record(), 1.)
        now[0] = 102.; self.assertEqual(rate.record(), 1.)
        now[0] = 130.; self.assertAlmostEqual(rate.value(), 2 / 30)
        now[0] = 161.; self.assertAlmostEqual(rate.value(), 1 / 60)
        now[0] = 162.; self.assertEqual(rate.value(), 0.)
        now[0] = 170.; self.assertAlmostEqual(rate.record(), 1 / 60)
        self.assertIsNone(RecentThroughput(clock=lambda: now[0]).value())

    def test_complete_animation_sampling_atomic_failure_and_size_bound(self):
        def observation(frames):
            return Observation(np.array(frames, dtype=np.uint8), (4,), 'NOT_FINISHED', 0, 1)
        episode = Episode([observation([[[1, 2]]]), observation([[[2, 3]], [[3, 4]]])], [Action(4)])
        records = [dict(action_source='planner', goal='Cruzar')]
        result = dict(success=False, reason='budget')
        now = [0.]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'planner_trajectory.json'
            sample = PlannerTrajectory(path, clock=lambda: now[0])
            rng = random.getstate()
            sample.add(episode, records, result, game='test', level=1)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['frame_count'], 3)
            self.assertEqual(saved['steps'][0]['action'], {'id': 4, 'data': {}})
            self.assertEqual(saved['steps'][0]['source'], 'planner')
            self.assertEqual(base64.b64decode(saved['initial']['pixels']), bytes([1, 2]))
            self.assertEqual(saved['steps'][0]['frames']['shape'], [2, 1, 2])
            self.assertEqual(base64.b64decode(saved['steps'][0]['frames']['pixels']), bytes([2, 3, 3, 4]))
            self.assertEqual(random.getstate(), rng)
            episode.observations[-1].frames[:] = 9
            now[0] = 1.99; sample.add(episode, records, result)
            self.assertEqual(json.loads(path.read_text()), saved)
            now[0] = 2.; sample.add(episode, records, result)
            self.assertNotEqual(sample.metadata['id'], saved['id'])
            self.assertEqual(base64.b64decode(json.loads(path.read_text())['steps'][0]['frames']['pixels']), bytes([9]*4))
            previous = path.read_bytes(); metadata = sample.metadata
            now[0] = 4.
            with patch.object(Path, 'replace', side_effect=OSError('disk failure')):
                sample.add(episode, records, result)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(sample.metadata, metadata)
            self.assertEqual(sample.error, 'disk failure')
            now[0] = 6.; sample.max_bytes = 5; sample.add(episode, records, result)
            self.assertIn('límite', sample.error)
            self.assertEqual(path.read_bytes(), previous)

    def test_fixed_http_route_rejects_stale_missing_and_oversize_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'report.json'
            sidecar = path.with_name('planner_trajectory.json')
            report = dict(kind='synthetic_subgoals', planner_trajectory={'id': 'new'})
            path.write_text(json.dumps(report))
            with make_server(('127.0.0.1', 0), None, subgoal_report=path) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
                def get(route='/api/trajectory'):
                    with urlopen(f'http://127.0.0.1:{server.server_port}'+route) as response:
                        return json.load(response)['trajectory']
                try:
                    self.assertIsNone(get())
                    sidecar.write_text(json.dumps(dict(kind='planner_trajectory', id='old')))
                    self.assertIsNone(get())
                    sample = dict(kind='planner_trajectory', id='new', actions=1)
                    sidecar.write_text(json.dumps(sample))
                    self.assertEqual(get(), sample)
                    self.assertEqual(get('/api/trajectory?file=/etc/passwd'), sample)
                    with self.assertRaises(HTTPError) as error:
                        get('/planner_trajectory.json')
                    self.assertEqual(error.exception.code, 404)
                    with patch('model.dashboard.MAX_READ', len(path.read_bytes()) + 1):
                        sidecar.write_text(json.dumps(sample) + ' '*200)
                        self.assertIsNone(get())
                    path.write_text(json.dumps(dict(kind='synthetic_subgoals')))
                    self.assertIsNone(get(), 'resume cannot expose a previous run sample')
                finally:
                    server.shutdown(); thread.join()


if __name__ == '__main__':
    unittest.main()
