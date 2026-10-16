"""The profiler operates on a disposable copy and restores runtime hooks."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from model import agent_checkpoint, subgoal_training
from model.profile_training import main
from model.tests.test_subgoals import path_agent


class ProfileTests(unittest.TestCase):
    def test_output_cannot_modify_or_overwrite_source(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            source = Path(directory)
            for output in (source, source / 'child'):
                with self.assertRaises(SystemExit):
                    main(['--source-run', str(source), '--output', str(output)])
            self.assertEqual(list(source.iterdir()), [])

    def test_cpu_copy_counts_measured_actions_and_preserves_source_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            agent_checkpoint.save(root / 'agent.pt', path_agent())
            args = ['--source', str(root / 'agent.pt'), '--device', 'cpu', '--steps', '2',
                    '--evaluate-every', '4', '--evaluation-episodes', '1']
            source = root / 'source'
            subgoal_training.main([*args, '--output', str(source), '--actions', '4'])

            def hashes():
                return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in source.rglob('*') if p.is_file()}

            before = hashes()
            original_play, original_backward = subgoal_training.play, torch.autograd.backward
            output = root / 'profile'
            main(['--source-run', str(source), '--output', str(output), '--actions', '4',
                  '--warmup', '1', '--', *args])
            result = json.loads((output / 'timing.json').read_text())
            self.assertEqual(result['completed'], 4)
            self.assertEqual(result['measured_actions'], 3)
            self.assertEqual(result['peak_gpu_allocated'], 0)
            self.assertGreater(result['wall_seconds'], 0)
            self.assertIn('training/representation', result['categories'])
            self.assertEqual(before, hashes())
            self.assertIs(subgoal_training.play, original_play)
            self.assertIs(torch.autograd.backward, original_backward)
