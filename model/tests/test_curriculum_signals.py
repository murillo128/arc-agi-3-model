"""Real signal lifecycle with CPU learning and atomic checkpoint persistence."""
from pathlib import Path
import signal
import subprocess
import sys
import unittest


SCENARIO = r'''
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from unittest.mock import patch
import torch
from model import agent_checkpoint, checkpoint, subgoal_training as training
from model.config import NetworkConfig
from model.curriculum import Curriculum
from model.environment import Environment
from tests.square_cross_support import fixture

failure = sys.argv[1] == 'failure'
handlers = {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGUSR2, signal.SIGHUP)}
real_play, real_serialize = training.play, torch.save
real_quality = training.evaluate_recording
real_make = training.Factory.make
episodes, captures = [], []
sent_eval = sent_final = False
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    config = root / 'config.json'
    config.write_text(json.dumps({'network': asdict(NetworkConfig.small())}))
    output = root / 'run'
    latest = output / 'latest.pt'
    args = ['--init-config', str(config), '--output', str(output), '--device', 'cpu',
            '--steps', '2', '--evaluate-every', '8', '--checkpoint-every', '100',
            '--evaluation-episodes', '1', '--representation-learning-rate', '3e-5',
            '--encoder-learning-rate-after-predictor', '3e-6']

    def serialize(payload, stream, *pos, **kw):
        if payload.get('format') != 'arc3-agent':
            return real_serialize(payload, stream, *pos, **kw)
        p = payload['training_progress']
        if p['actions'] == 4:
            if failure:
                stream.write(b'partial')
                raise OSError('manual checkpoint failure')
            # Arrives DURING the planner activation save; must wait until the
            # next episode ends, not mutate the snapshot being serialized.
            os.kill(os.getpid(), signal.SIGHUP)
            assert p['curriculum']['unlocked']['sc01-v1'] == 1
        if p['actions'] == 6:
            assert p['curriculum']['preferred_pair'] == ('sc01-v1', 2)
            restored = Curriculum({'sc01-v1': 2})
            restored.restore(p['curriculum'])
            assert restored.next_pair() == ('sc01-v1', 2)
        captures.append(p['actions'])
        return real_serialize(payload, stream, *pos, **kw)

    def make(factory, seed, level=None):
        env = real_make(factory, seed, level)
        # The SDK owns a separate loaded module. Fix geometry there, leaving
        # actual actions/observations/learning intact and episodes two steps long.
        with patch.dict(factory.teacher(env).on_set_level.__globals__,
                        {'generate_scene': lambda seed, level: fixture()}):
            factory._isolate_synthetic_level(env.wrapper, level)
            return Environment(env.wrapper)

    def play(agent, factory, level, *pos, **kw):
        representation = kw.get('representation')
        if representation:
            episodes.append((level, representation.frozen, bool(kw.get('world')),
                             dict(representation.learning_rates)))
            number = len(episodes)
            original_progress = kw['progress']
            def progress(row):
                if row['action'] == 1 and number in (1, 2, 5):
                    old = checkpoint.load(latest)['training_progress']
                    os.kill(os.getpid(), signal.SIGUSR2)
                    # No checkpoint/phase work in the signal handler.
                    assert checkpoint.load(latest)['training_progress'] == old
                original_progress(row)
            kw['progress'] = progress
        return real_play(agent, factory, level, *pos, **kw)

    def quality(*pos, **kw):
        global sent_eval, sent_final
        result = real_quality(*pos, **kw)
        if len(episodes) == 4 and not sent_eval:
            # Quality now replays the highest unlocked level, including when
            # its random cohort is already cached and play() is not called.
            os.kill(os.getpid(), signal.SIGUSR2)
            os.kill(os.getpid(), signal.SIGHUP)  # already at last level
            sent_eval = True
            assert checkpoint.load(latest)['training_progress']['manual_controls'][-1]['operation'] == 'level'
        elif len(episodes) == 6 and not sent_final:
            os.kill(os.getpid(), signal.SIGUSR2)
            sent_final = True
        return result

    log = io.StringIO()
    with redirect_stdout(log), \
         patch.dict(training.SYNTHETIC_LEVELS, {'sc01-v1': 2}, clear=True), \
         patch.object(training.Factory, 'make', make), \
         patch.object(training, 'summarize_quality', return_value=({'prediction_error': None, 'samples': 0}, [])), \
         patch.object(training, 'play', play), \
         patch.object(training, 'evaluate_recording', quality), patch.object(torch, 'save', serialize):
        try:
            training.main([*args, '--actions', '12', '--freeze-representation-after-planner'])
        except OSError as error:
            assert failure and str(error) == 'manual checkpoint failure'
        else:
            assert not failure
    assert {s: signal.getsignal(s) for s in handlers} == handlers
    assert not latest.with_suffix('.tmp').exists()
    saved = checkpoint.load(latest)['training_progress']
    report = json.loads((output / 'report.json').read_text())
    if failure:
        assert saved['actions'] == 2 and saved['episodes'] == 1
        assert len(saved['manual_controls']) == 1
        assert not saved['curriculum']['levels'][0]['planner']
        assert report['status'] == 'failed'
        assert captures == [0, 2]
    else:
        assert sent_eval and sent_final
        assert {r['level'] for r in report['evaluations']
                if r['actions'] == 8 and r['policy'] == 'quality_probe'} == {2}
        assert [(n, frozen, world) for n, frozen, world, lr in episodes] == [
            (1, False, False), (1, False, True), (1, True, True),
            (2, False, False), (1, True, True), (1, True, True)], (episodes, report['episodes'], saved['manual_controls'])
        assert episodes[0][3] == episodes[3][3] == dict(encoder=3e-5, decoder=3e-5)
        assert episodes[1][3] == dict(encoder=3e-6, decoder=3e-5)
        assert episodes[2][3] == dict(encoder=None, decoder=None)
        controls = saved['manual_controls']
        assert [(r['actions'], r['operation'], r['result'], r['level'], r['phase']) for r in controls] == [
            (2, 'phase', 'advanced', 1, 'predictor'),
            (4, 'phase', 'advanced', 1, 'planner'),
            (6, 'level', 'advanced', 2, 'representation'),
            (8, 'phase', 'advanced', 2, 'predictor'),
            (8, 'level', 'last_level', 2, 'predictor'),
            (10, 'phase', 'already_planner', 1, 'planner'),
            (12, 'phase', 'budget_exhausted', 1, 'planner')], controls
        assert saved['representation_updates'] == 6
        assert saved['updates'] == 8
        assert saved['path_updates'] > 0
        assert saved['episodes'] == 6
        assert all(r['prediction_passes'] == r['representation_passes'] == r['promotion_passes'] == 0
                   for r in saved['curriculum']['levels'])
        assert report['manual_controls'] == controls
        # Checkpoint owns controls on resume, even if report has newer data.
        report['manual_controls'].append(dict(actions=12, operation='phase', result='fake'))
        (output / 'report.json').write_text(json.dumps(report))
        with redirect_stdout(log), \
             patch.dict(training.SYNTHETIC_LEVELS, {'sc01-v1': 2}, clear=True), \
             patch.object(training.Factory, 'make', make), \
             patch.object(training, 'summarize_quality', return_value=({'prediction_error': None, 'samples': 0}, [])), \
             patch.object(training, 'play', play), \
             patch.object(training, 'evaluate_recording', quality):
            training.main([*args, '--resume', '--actions', '14'])
        assert episodes[-1][:3] == (2, False, True), episodes[-1]
        assert episodes[-1][3] == dict(encoder=3e-6, decoder=3e-5)
        resumed = checkpoint.load(latest)['training_progress']
        assert resumed['manual_controls'] == controls
        assert resumed['representation_updates'] == 8
        assert resumed['freeze_representation_after_planner'] is True
        assert {s: signal.getsignal(s) for s in handlers} == handlers
'''


@unittest.skipUnless(all(hasattr(signal, s) for s in ('SIGUSR2', 'SIGHUP')), 'POSIX signals required')
class CurriculumSignalTests(unittest.TestCase):
    def test_safe_manual_phase_level_resume_and_noops(self):
        self.run_scenario('continue')

    def test_failed_manual_save_keeps_previous_permissions_and_restores_handlers(self):
        self.run_scenario('failure')

    def run_scenario(self, mode):
        result = subprocess.run([sys.executable, '-c', SCENARIO, mode],
                                cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
