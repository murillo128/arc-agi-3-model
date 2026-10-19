"""Numerical, ownership and transport contracts; no visual/browser checks."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import torch
from torch import nn

from model.optimization_telemetry import OptimizationTelemetry, block_rms
from model.optimization_projection import ParameterPlane
from model.optimization_probe import memory_probe
from model.optimization_control import OptimizationControl
from model.tests.test_sequence_training import scalar_engine, ScalarLearner, session
from model.game_sequences import EncodedSessions
from model.config import SequenceConfig


class VisualScalarLearner(ScalarLearner):
    def _terms(self, *args):
        value = super()._terms(*args)[:, 0]
        return torch.stack((value * 0, value), -1)


class OptimizationTests(unittest.TestCase):
    def test_adam_update_is_observed_separately_from_gradient_and_does_not_change_training(self):
        model = nn.Module()
        for name in ('encoder', 'decoder', 'dynamics'):
            setattr(model, name, nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            model.dynamics.weight.copy_(torch.tensor([[1., -2.], [3., -4.]]))
        baseline = copy.deepcopy(model)
        a = torch.optim.AdamW(model.dynamics.parameters(), lr=.1, weight_decay=.05)
        b = torch.optim.AdamW(baseline.dynamics.parameters(), lr=.1, weight_decay=.05)
        with tempfile.TemporaryDirectory() as d:
            telemetry = OptimizationTelemetry(model, d, every=1)
            telemetry.selected = 'dynamics.weight'
            rng = torch.get_rng_state().clone()
            telemetry.start_batch(1)
            for network, opt in ((model, a), (baseline, b)):
                network.dynamics.weight.grad = torch.tensor([[3., 4.], [0., 0.]])
                before = network.dynamics.weight.detach().clone()
                capture = telemetry.before_step('predictor', opt) if network is model else None
                norm = torch.nn.utils.clip_grad_norm_(network.dynamics.parameters(), 1.)
                opt.step()
                if network is model:
                    telemetry.after_step(capture, norm, 2.)
                    actual = network.dynamics.weight.detach() - before
            torch.testing.assert_close(model.dynamics.weight, baseline.dynamics.weight, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            for key, value in a.state_dict()['state'][0].items():
                torch.testing.assert_close(value, b.state_dict()['state'][0][key], rtol=0, atol=0)
            event = telemetry.events[-1]
            row = event['matrices']['dynamics.weight']
            self.assertAlmostEqual(row['gradient_rms'], 2.5)
            self.assertAlmostEqual(event['clip_scale'], .2, places=6)
            self.assertNotIn('encoder.weight', event['matrices'])
            # Adam's sign-normalized first step plus decoupled weight decay.
            expected = torch.tensor([[-.105, -.09], [-.015, .02]])
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-7)
            self.assertAlmostEqual(row['update_rms'], float(expected.square().mean().sqrt()), places=6)
            torch.testing.assert_close(torch.tensor(event['detail']['gradient']), torch.tensor([[3., 4.], [0., 0.]]))
            torch.testing.assert_close(torch.tensor(event['detail']['update']), expected.abs(), rtol=1e-4, atol=1e-7)

    def test_matrix_rms_has_no_signed_cancellation(self):
        matrix = torch.ones(32, 32)
        matrix[::2] *= -1
        torch.testing.assert_close(block_rms(matrix), torch.ones(16, 16))

    def test_fixed_plane_detects_motion_outside_projection_and_recalibrates_all_points(self):
        plane = ParameterPlane()
        for point in ([0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0]):
            plane.append(point)
        basis = plane.basis.copy()
        plane.append([1, 2, 3])
        np.testing.assert_array_equal(plane.basis, basis)
        self.assertAlmostEqual(plane.project()[-1]['coverage'], 0.)
        self.assertAlmostEqual(plane.project()[-1]['angle'], 90.)
        self.assertAlmostEqual(plane.project()[1]['coverage'], 1.)
        plane.calibrate()
        self.assertEqual(plane.basis_id, 2)
        self.assertEqual(len(plane.project()), 5)
        self.assertGreater(plane.project()[-1]['coverage'], .8)
        still = ParameterPlane()
        for _ in range(5):
            still.append([0, 0])
        self.assertIsNone(still.project()[-1]['coverage'])

    def test_temporal_probe_matches_analytic_derivative_and_preserves_state(self):
        e, row = scalar_engine(), session()
        learner = VisualScalarLearner(e, SequenceConfig(block_steps=1, context_steps=1))
        encoded = EncodedSessions(e, [row])
        flags = [p.requires_grad for p in e.model.parameters()]
        e.model.dynamics.rho.grad = torch.tensor(7.)
        rng = torch.get_rng_state().clone()
        result = memory_probe(learner, encoded)
        # rho=.5; prefix h1=1; h2=2.5; L=(rho*h2+3-1)^2.
        # dL/dh2=3.25; dL/dh1=1.625; prefix before t=1 is detached.
        self.assertEqual([x['t'] for x in result['steps']], [1, 2])
        np.testing.assert_allclose([x['rms'] for x in result['steps']], [1.625, 3.25])
        self.assertAlmostEqual(result['loss'], 10.5625)
        self.assertAlmostEqual(result['cleared_loss'], 4.)
        self.assertEqual(result['context_start'], 1)
        self.assertEqual(result['start'], 2)
        self.assertEqual(flags, [p.requires_grad for p in e.model.parameters()])
        self.assertEqual(float(e.model.dynamics.rho.grad), 7.)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertFalse(learner.optimizer.state)

    def test_bounded_samples_resume_and_checkpoint_rollback(self):
        e = scalar_engine()
        with tempfile.TemporaryDirectory() as d:
            t = OptimizationTelemetry(e.model, d, identity='test')
            for i in range(1, 100):
                t.batch = i
                t.finish_batch(i/100, i*8)
            self.assertEqual(len(t.points), 96)
            restored = OptimizationTelemetry(e.model, d, restored_batch=99, identity='test')
            self.assertEqual(restored.points, t.points)
            self.assertEqual(restored.planes['dynamics'].basis_id, t.planes['dynamics'].basis_id)
            self.assertFalse(OptimizationTelemetry(e.model, d, restored_batch=98, identity='test').points)
            self.assertFalse(OptimizationTelemetry(e.model, d, restored_batch=99, identity='other').points)
            (Path(d)/'optimization.json').write_text('{bad')
            self.assertFalse(OptimizationTelemetry(e.model, d, restored_batch=99, identity='test').points)

    def test_real_cpu_sequence_run_with_telemetry_matches_uninstrumented_weights_and_resume(self):
        from model.sequence_training import run
        from model.tests.test_training_config import config
        from model import checkpoint
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d)
            path, values = config(root, [dict(source='synthetic', game='sc01-v1', levels=[1, 2])],
                sequence=dict(batch_games=4, session_steps=4, block_steps=2, context_steps=1, level_steps=2, representation_updates=1))
            values['settings']['actions'] = 32
            path.write_text(json.dumps(values))
            def args(name, optimization, resume=False, max_batches=None):
                return SimpleNamespace(training_config=path, output=root/name, source=None, resume=resume,
                    device='cpu', compile=False, max_batches=max_batches, optimization=optimization, optimization_every=1)
            run(args('plain', False))
            run(args('observed', True, max_batches=1))
            first = json.loads((root/'observed/optimization.json').read_text())
            self.assertEqual(first['probe']['status'], 'ready')
            self.assertEqual(first['points'][0]['batch'], 1)
            self.assertTrue(first['events'])
            run(args('observed', True, resume=True))
            plain, observed = [checkpoint.load(root/p/'latest.pt') for p in ('plain', 'observed')]
            for name, expected in plain['weights'].items():
                torch.testing.assert_close(observed['weights'][name], expected, rtol=0, atol=0)
            torch.testing.assert_close(plain['torch_rng'], observed['torch_rng'], rtol=0, atol=0)
            for optimizer in ('predictor_optimizer', 'representation_optimizer'):
                for key, expected in plain[optimizer]['state'].items():
                    for field, value in expected.items():
                        torch.testing.assert_close(observed[optimizer]['state'][key][field], value, rtol=0, atol=0)
            report = json.loads((root/'observed/report.json').read_text())
            self.assertTrue(report['optimization_identity'])
            self.assertTrue(np.isfinite(report['history'][-1]['quality']['reference']['nll']))

    def test_http_capture_scope_token_validation_missing_and_malformed_data(self):
        from model.dashboard import make_server
        with tempfile.TemporaryDirectory() as d:
            report = Path(d)/'report.json'
            report.write_text(json.dumps(dict(kind='game_sequences', status='running', optimization_identity='test',
                config=dict(batch_games=4), progress=dict(batches=4))))
            state = dict(kind='optimization', version=1, identity='test', batch=4, updated_at=time.time(),
                catalog=[dict(name='dynamics.weight')], events=[], planes={}, points=[])
            snapshot = Path(d)/'optimization.json'
            snapshot.write_text(json.dumps(state))
            server = make_server(('127.0.0.1', 0), None, sequence_report=report)
            worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
            try:
                base = f'http://127.0.0.1:{server.server_port}'
                def read():
                    return json.load(urlopen(base+'/api/optimization'))
                token = read()['control']['token']
                response = urlopen(base+'/api/optimization')
                with self.assertRaises(HTTPError) as unchanged:
                    urlopen(Request(base+'/api/optimization', headers={'If-None-Match':response.headers['ETag']}))
                self.assertEqual(unchanged.exception.code,304)
                def post(body, header=token):
                    return urlopen(Request(base+'/api/optimization', data=json.dumps(body).encode(),
                        headers={'Content-Type':'application/json', 'X-Optimization-Token':header}))
                for body, header, code in (({'operation':'probe','session':0},'wrong',403),
                         ({'operation':'matrix','matrix':'../../secret'},token,400),
                         ({'operation':'probe','session':True},token,400)):
                    with self.assertRaises(HTTPError) as raised:
                        post(body,header)
                    self.assertEqual(raised.exception.code,code)
                self.assertEqual(post({'operation':'probe','session':2}).status,202)
                self.assertEqual(json.loads((Path(d)/'optimization_request.json').read_text())['session'],2)
                with self.assertRaises(HTTPError) as pending:
                    post({'operation':'recalibrate'})
                self.assertEqual(pending.exception.code,409)
                self.assertIn('optimization.js',urlopen(base+'/optimization.html').read().decode())
                snapshot.write_text('{"kind":"optimization","updated_at":NaN}')
                self.assertEqual(read()['status'],'waiting')
                snapshot.unlink()
                self.assertEqual(read()['status'],'waiting')
            finally:
                server.shutdown();server.server_close();worker.join()

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for data/interaction contract')
    def test_javascript_live_pin_selections_and_control_requests(self):
        result = subprocess.run(['node', str(Path(__file__).with_name('optimization_runtime.cjs'))],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
