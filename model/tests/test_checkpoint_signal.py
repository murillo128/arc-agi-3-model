"""Exercise real SIGUSR1 delivery in a child running the CPU training entrypoint."""
from pathlib import Path
import signal
import subprocess
import sys
import unittest


# Isolate signal delivery: a missing handler must fail this child, not kill the
# test runner. Real learning, episode bookkeeping and checkpoint IO stay active.
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
from tests.square_cross_support import fixture

failure = sys.argv[1] == 'failure'
previous_handler = signal.getsignal(signal.SIGUSR1)
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    config = root / 'config.json'
    config.write_text(json.dumps({'network': asdict(NetworkConfig.small())}))
    output = root / 'run'
    latest = output / 'latest.pt'
    captures, initial_bytes = [], None
    real_update = training.OnlineRepresentation.update
    real_save, real_serialize = agent_checkpoint.save, torch.save
    real_quality = training.evaluate_recording
    sent_during_write = sent_during_evaluation = False

    def update(learner, *args, **kwargs):
        result = real_update(learner, *args, **kwargs)
        if learner.updates == 1:
            # Repeated requests before the safe boundary coalesce. The first
            # action has learned, but episode bookkeeping has not finished.
            os.kill(os.getpid(), signal.SIGUSR1)
            os.kill(os.getpid(), signal.SIGUSR1)
            assert checkpoint.load(latest)['training_progress']['actions'] == 0
        return result

    def serialize(payload, stream, *args, **kwargs):
        global sent_during_write
        if payload.get('format') != 'arc3-agent':
            # Fixed evaluation recordings share atomic serialization, but are
            # not agent snapshots and have no training-progress counters.
            return real_serialize(payload, stream, *args, **kwargs)
        if payload['training_progress']['actions'] == 2:
            if failure:
                stream.write(b'incomplete snapshot')
                raise OSError('simulated disk failure')
            assert not sent_during_write, 'recursive/duplicate checkpoint'
            sent_during_write = True
            os.kill(os.getpid(), signal.SIGUSR1)
            assert checkpoint.load(latest)['training_progress']['actions'] == 0
        return real_serialize(payload, stream, *args, **kwargs)

    def save(path, agent, **kwargs):
        global initial_bytes
        real_save(path, agent, **kwargs)
        payload = checkpoint.load(path)
        progress = payload['training_progress']
        assert progress == kwargs['training_progress']
        assert progress['representation_updates'] == progress['actions']
        assert progress['episodes'] * 2 == progress['actions']
        # Saved tensors are the actual trained tensors, not just fresh counters.
        for name, tensor in agent.engine.model.state_dict().items():
            assert torch.equal(payload['world']['weights'][name], tensor.cpu())
        captures.append(progress['actions'])
        if progress['actions'] == 0:
            initial_bytes = path.read_bytes()

    def quality(*args, **kwargs):
        global sent_during_evaluation
        result = real_quality(*args, **kwargs)
        if captures[-1] == 6 and not sent_during_evaluation:
            sent_during_evaluation = True
            os.kill(os.getpid(), signal.SIGUSR1)
        return result

    log = io.StringIO()
    try:
        with redirect_stdout(log), \
             patch('games.square_cross.v1.sc01.generate_scene', return_value=fixture()), \
             patch.object(training.OnlineRepresentation, 'update', update), \
             patch.object(agent_checkpoint, 'save', save), \
             patch.object(torch, 'save', serialize), \
             patch.object(training, 'evaluate_recording', quality):
            training.main(['--init-config', str(config), '--output', str(output),
                           '--device', 'cpu', '--actions', '6', '--steps', '2',
                           '--evaluate-every', '100', '--checkpoint-every', '100',
                           '--evaluation-episodes', '1'])
    except OSError as error:
        assert failure and str(error) == 'simulated disk failure'
    else:
        assert not failure, 'save failure was swallowed'

    assert signal.getsignal(signal.SIGUSR1) == previous_handler
    report = json.loads((output / 'report.json').read_text())
    events = [row for line in log.getvalue().splitlines() if line.startswith('{"event":')
              if (row := json.loads(line))['event'] == 'checkpoint_saved']
    assert not latest.with_suffix('.tmp').exists()
    if failure:
        assert latest.read_bytes() == initial_bytes, 'failed save destroyed previous weights'
        assert report['status'] == 'failed'
        assert report['checkpoint_action'] == 0
        assert captures == [0]
        assert len(events) == 1, 'failed save reported success'
    else:
        assert sent_during_write and sent_during_evaluation
        # First signal is served after episode 1; the signal during IO after
        # episode 2; training reaches its budget and a final eval also saves.
        assert captures == [0, 2, 4, 6, 6], captures
        assert [(e['actions'], e['reason']) for e in events] == [
            (0, 'initial'), (2, 'signal'), (4, 'signal'), (6, 'scheduled'), (6, 'signal')]
        assert report['status'] == 'complete'
        assert report['progress']['actions'] == 6
        assert report['last_checkpoint']['reason'] == 'signal'
        assert report['checkpoint_signal'] == 'SIGUSR1'
        assert latest.read_bytes() != initial_bytes
        assert checkpoint.load(latest)['training_progress']['episodes'] == 3
'''


@unittest.skipUnless(hasattr(signal, 'SIGUSR1'), 'SIGUSR1 is a POSIX signal')
class CheckpointSignalTests(unittest.TestCase):
    def run_scenario(self, scenario):
        result = subprocess.run([sys.executable, '-c', SCENARIO, scenario],
                                cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_signal_saves_at_safe_boundary_and_training_continues(self):
        self.run_scenario('continue')

    def test_failed_signal_save_preserves_previous_checkpoint_and_handler(self):
        self.run_scenario('failure')


if __name__ == '__main__':
    unittest.main()
