"""Literal executed-frame, sampling and wall-clock contracts; no GPU required."""
import base64
from contextlib import nullcontext
import json
from pathlib import Path
import random
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import torch

from model.data import Action, Episode, Observation
from model.dashboard import make_server
from model.training_telemetry import PlannerTrajectory, RecentThroughput


class TelemetryTests(unittest.TestCase):
    def test_selected_latents_are_captured_before_updates_and_exploration_has_no_stale_plan(self):
        # Literal decoder: each latent is a palette index for one pixel. No
        # dynamics/search method exists on this seam, so it cannot re-plan.
        def decode(z, height, width):
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual((height, width), (1, 1))
            return torch.nn.functional.one_hot(z.long().reshape(-1, 1, 1), 16).float()
        planner = SimpleNamespace(last_plan=[Action(4)], last_states=[torch.tensor([12.])],
            last_forecast_plan=[Action(2), Action(4)],
            last_forecast_states=[torch.tensor([3.]), torch.tensor([7.])],
            last_forecast_source='searched_alternative',
            last_estimate=dict(success_probability=.91,successful_actions=8,remaining_budget=128),
            last_forecast_estimate=dict(success_probability=.90,successful_actions=6,remaining_budget=128),
            goal=dict(frame=np.array([[9]], dtype=np.uint8), pixel_mask=np.array([[False]])))
        agent = SimpleNamespace(planner=planner, last_diagnostics={'path': {'score': 1}},
            engine=SimpleNamespace(decode=decode, autocast=nullcontext), state=SimpleNamespace(geometry=(1, 1)))
        observations = [Observation(np.full((1, 1, 1), c, dtype=np.uint8), (2, 4), 'NOT_FINISHED', 0, 1)
                        for c in (1, 3, 8)]
        episode = Episode(observations, [Action(4), Action(2)])
        records = [dict(action_source=s, goal='Cruzar') for s in ('planner', 'exploration')]
        now = [0.]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'planner_trajectory.json'
            sample = PlannerTrajectory(path, clock=lambda: now[0])
            capture = sample.begin()
            rng, tensor_rng = random.getstate(), torch.get_rng_state().clone()
            capture(agent, Action(4), False)
            # Later learning/goal replacement must not rewrite the captured route.
            planner.last_forecast_states[0].fill_(12)
            planner.goal['frame'][:] = 2
            capture(agent, Action(2), True)
            sample.add(episode, records, dict(success=False, reason='budget'))
            saved = json.loads(path.read_text())
            self.assertEqual(saved['format_version'], 2)
            steps = saved['steps']
            self.assertEqual([p['action']['id'] for p in steps[0]['imagination']], [2, 4])
            self.assertEqual(steps[0]['imagination_source'], 'searched_alternative')
            self.assertEqual(steps[0]['selected_estimate'],dict(success_probability=.91,successful_actions=8,remaining_budget=128))
            self.assertEqual(steps[0]['forecast_estimate'],dict(success_probability=.90,successful_actions=6,remaining_budget=128))
            self.assertIsNone(steps[1]['selected_estimate'])
            self.assertIsNone(steps[1]['forecast_estimate'])
            self.assertEqual([base64.b64decode(p['frames']['pixels']) for p in steps[0]['imagination']], [b'\x03', b'\x07'])
            self.assertEqual(base64.b64decode(steps[0]['goal_frame']['pixels']), b'\x10')
            self.assertEqual(steps[1]['imagination'], [])
            self.assertEqual(steps[1]['imagination_status'], 'exploration')
            self.assertEqual(random.getstate(), rng)
            self.assertTrue(torch.equal(torch.get_rng_state(), tensor_rng))
            now[0] = 1.99
            self.assertIsNone(sample.begin(), 'throttled episodes must not decode any route')
            previous = path.read_bytes()
            now[0] = 2.
            capture = sample.begin()
            capture(agent, Action(4), False)
            sample.add(episode, records, dict(success=False, reason='budget'))
            self.assertIn('Faltan predicciones', sample.error)
            self.assertEqual(path.read_bytes(), previous)
            now[0] = 4.
            capture = sample.begin()
            sample.max_bytes = 6  # 4 captured bytes + 3 real pixels exceed the bound.
            capture(agent, Action(4), False); capture(agent, Action(2), True)
            sample.add(episode, records, dict(success=False, reason='budget'))
            self.assertIn('límite', sample.error)
            self.assertEqual(path.read_bytes(), previous)

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
