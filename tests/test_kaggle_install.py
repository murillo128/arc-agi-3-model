"""Generated offline installation arguments, CLI wiring and actionable failures.

These tests execute the install cell with a subprocess boundary double. Actual
wheel resolution and the competition framework are separate integration evidence.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from arc3.kaggle import build


OFFICIAL_WHEELS = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels'
EXTRA_WHEELS = '/kaggle/input/private-base'


class KaggleInstallTests(unittest.TestCase):
    def execute_install(self, notebook):
        # Installation behavior stops at the pip process. Installed-version and
        # mount checks are isolated from this machine's available dependencies.
        with patch('pathlib.Path.is_dir', return_value=True), \
                patch('subprocess.run') as run, \
                patch('importlib.metadata.version', return_value='0.9.9'):
            exec(notebook['cells'][1]['source'], {})
        return run

    def test_generated_install_finds_companion_wheels_and_preserves_sdk_pin(self):
        notebook = build.build_notebook(extra_wheel_dir=EXTRA_WHEELS)
        run = self.execute_install(notebook)
        run.assert_called_once_with([
            sys.executable, '-m', 'pip', 'install', '--no-index',
            '--find-links', OFFICIAL_WHEELS, '--find-links', EXTRA_WHEELS,
            'arc-agi==0.9.9', 'python-dotenv',
        ], check=True)
        self.assertFalse(notebook['metadata']['kaggle']['isInternetEnabled'])

    def test_default_install_keeps_official_source_and_never_downgrades(self):
        run = self.execute_install(build.build_notebook())
        run.assert_called_once_with([
            sys.executable, '-m', 'pip', 'install', '--no-index',
            '--find-links', OFFICIAL_WHEELS, 'arc-agi==0.9.9', 'python-dotenv',
        ], check=True)

    def test_wheel_mount_validation_is_explicit_and_does_not_probe_host_paths(self):
        for directory in ('', '/kaggle/input', '/kaggle/input/', '/tmp/wheels',
                          '/kaggle/input/private-base/../wheels', 'https://example.com/wheels',
                          '/kaggle/input/private-base/arc_agi-0.9.9-py3-none-any.whl', 7):
            with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, 'wheel directory'):
                build.build_notebook(extra_wheel_dir=directory)
        with patch('pathlib.Path.is_dir', side_effect=AssertionError('Build must not probe Kaggle mounts')):
            notebook = build.build_notebook(extra_wheel_dir='/kaggle/input/private-base/wheels/')
        self.assertIn('/kaggle/input/private-base/wheels', notebook['cells'][1]['source'])

    def test_missing_companion_mount_fails_before_pip(self):
        notebook = build.build_notebook(extra_wheel_dir=EXTRA_WHEELS)
        with patch('pathlib.Path.is_dir', lambda path: str(path) == OFFICIAL_WHEELS), \
                patch('subprocess.run') as run:
            with self.assertRaisesRegex(FileNotFoundError, 'Missing offline wheel directory: /kaggle/input/private-base'):
                exec(notebook['cells'][1]['source'], {})
        run.assert_not_called()

    def test_missing_offline_dependency_fails_without_online_retry(self):
        notebook = build.build_notebook(extra_wheel_dir=EXTRA_WHEELS)
        failure = subprocess.CalledProcessError(1, 'pip')
        with patch('pathlib.Path.is_dir', return_value=True), \
                patch('subprocess.run', side_effect=failure) as run:
            with self.assertRaisesRegex(RuntimeError, 'Offline installation failed: attach arc-agi==0.9.9') as raised:
                exec(notebook['cells'][1]['source'], {})
        self.assertIs(raised.exception.__cause__, failure)
        self.assertEqual(run.call_count, 1)
        self.assertIn('--no-index', run.call_args.args[0])
        self.assertNotIn('--index-url', run.call_args.args[0])

    def test_cli_writes_configured_extra_source_without_embedding_wheel_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'submission.ipynb'
            with patch.object(sys, 'argv', ['build', '--output', str(output),
                                           '--extra-wheel-dir', EXTRA_WHEELS]), \
                    redirect_stdout(io.StringIO()):
                build.main()
            notebook = json.loads(output.read_text())
            run = self.execute_install(notebook)
            self.assertIn(EXTRA_WHEELS, run.call_args.args[0])
            self.assertEqual(list(Path(directory).iterdir()), [output])


if __name__ == '__main__':
    unittest.main()
