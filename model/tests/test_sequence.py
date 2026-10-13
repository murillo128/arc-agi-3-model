"""Practice continuation, isolated Git refresh and held-out SDK boundaries."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model import checkpoint
from model.config import NetworkConfig, TrainConfig
from model.data import Action, Observation
from model.environment import Factory, training_boundary
from model.learning import Engine, Learner
from model.networks import WorldModel
from model.runner import train, validation
from model.sequence import advance, check_stage, converged, enroll, git, owner, public_split, refresh_checkout, supervise

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(2)


def plan(root):
    return {'games': ['sc01-v1', 'si01-v1', 'ls20-public'], 'evaluation_games': ['ft09-public'],
            'source_checkout': str(root), 'refresh_checkout': str(root / 'private'),
            'sdk_environments_dir': str(root / 'sdk'), 'refresh_branch': 'training/test', 'revision': 'unused'}


def payload(root):
    return {'weights': {'weight': torch.tensor([1., 2.])},
            'optimizers': [{'state': {0: {'exp_avg': torch.tensor([.1, .2])}}}],
            'rng': random.Random(14).getstate(), 'training': {'freeze_representation': True},
            'run': {'game': 'sc01-v1', 'environments_dir': str(root / 'games'), 'mode': 'offline',
                    'levels': [1, 2], 'seed': 17},
            'progress': {'level_index': 1, 'actions': 213, 'level_actions': 13, 'updates': 45,
                         'passes': 2, 'representation_gate': {'close_streak': 1}}}


class SequenceTests(unittest.TestCase):
    def test_output_ownership_survives_supervisor_loss_until_trainer_exits(self):
        # Real supervisor/launch/lock; only the expensive neural trainer is a
        # temporary process with a controlled save boundary.
        for stop_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            with self.subTest(signal=stop_signal), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / 'output'
                output.mkdir()
                package = root / 'model'
                package.mkdir()
                (package / '__init__.py').touch()
                (package / '__main__.py').write_text('''
import os
from pathlib import Path
import signal
import sys
import time

output = Path(sys.argv[sys.argv.index('--output') + 1])
stopping = False
def interrupt(signum, frame):
    global stopping
    stopping = True
signal.signal(signal.SIGINT, interrupt)
(output / 'trainer.tmp').write_text(str(os.getpid()))
(output / 'trainer.tmp').replace(output / 'trainer.pid')
while not stopping:
    time.sleep(.01)
(output / 'saving').touch()
while not (output / 'release-save').exists():
    time.sleep(.01)
(output / 'saved').touch()
''')
                checkpoint.write(output / 'latest.pt', payload(root))
                policy = plan(root)
                (root / 'plan.json').write_text(json.dumps(policy))
                supervisor = subprocess.Popen(
                    [sys.executable, '-c',
                     'import json, sys; from pathlib import Path; '
                     'from model.sequence import supervise; '
                     'supervise(Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), "cpu")',
                     str(output), str(root / 'plan.json')],
                    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                trainer_pid = None
                def await_condition(condition):
                    deadline = time.monotonic() + 10
                    while not condition():
                        if time.monotonic() >= deadline:
                            self.fail('Timed out waiting for temporary trainer lifecycle')
                        time.sleep(.01)
                try:
                    await_condition(lambda: (output / 'trainer.pid').exists() or supervisor.poll() is not None)
                    self.assertIsNone(supervisor.poll())
                    trainer_pid = int((output / 'trainer.pid').read_text())
                    with self.assertRaisesRegex(RuntimeError, 'Another supervisor'):
                        with owner(output):
                            self.fail('Live trainer allowed another output owner')
                    supervisor.send_signal(stop_signal)
                    if stop_signal == signal.SIGKILL:
                        self.assertEqual(supervisor.wait(timeout=10), -signal.SIGKILL)
                        os.kill(trainer_pid, 0)  # Trainer remains alive without its parent.
                    else:
                        await_condition(lambda: (output / 'saving').exists() or supervisor.poll() is not None)
                        self.assertIsNone(supervisor.poll(), 'Supervisor must wait for the trainer save')
                        self.assertFalse((output / 'saved').exists())
                        # Repeated stop requests must not interrupt an atomic save.
                        supervisor.send_signal(stop_signal)
                    with self.assertRaisesRegex(RuntimeError, 'Another supervisor'):
                        with owner(output):
                            self.fail('Restart acquired output while previous trainer remained alive')
                    self.assertEqual(checkpoint.load(output / 'latest.pt')['run']['sequence']['stage'], 0)
                    if stop_signal == signal.SIGKILL:
                        os.kill(trainer_pid, signal.SIGINT)
                    (output / 'release-save').touch()
                    if stop_signal != signal.SIGKILL:
                        stdout, stderr = supervisor.communicate(timeout=10)
                        self.assertEqual(supervisor.returncode, 0, stderr.decode())
                        self.assertIn('"reason": "interrupted"', stdout.decode())
                    await_condition(lambda: (output / 'saved').exists())
                    def available():
                        try:
                            with owner(output):
                                return True
                        except RuntimeError:
                            return False
                    await_condition(available)
                finally:
                    (output / 'release-save').touch()
                    if trainer_pid is not None:
                        try:
                            os.kill(trainer_pid, signal.SIGINT)
                        except ProcessLookupError:
                            pass
                    if supervisor.poll() is None:
                        supervisor.kill()
                    supervisor.communicate(timeout=10)

    def test_transfer_preserves_learned_tensors_and_rng_and_resets_only_game_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p, policy = payload(root), plan(root)
            self.assertTrue(enroll(p, policy))
            self.assertEqual(p['run']['levels'], [1, 2, 3, 4, 5])
            self.assertEqual(p['progress']['actions'], 213)
            self.assertEqual(p['progress']['passes'], 2)
            self.assertFalse(enroll(p, policy))
            p['progress'].update(level_index=5, stop_reason='curriculum_converged')
            original = deepcopy(p)
            refreshed = []
            def refresh(plan):
                refreshed.append(plan)
                return plan['refresh_checkout']
            self.assertTrue(advance(p, policy, refresh))
            self.assertEqual(len(refreshed), 1)
            self.assertEqual(p['run']['game'], 'si01-v1')
            self.assertEqual(p['run']['levels'], [1, 2, 3, 4, 5])
            self.assertEqual(p['progress']['actions'], 0)
            self.assertNotIn('representation_gate', p['progress'])
            path = root / 'latest.pt'
            checkpoint.write(path, p)
            restored = checkpoint.load(path)
            self.assertEqual(restored['run']['sequence']['stage'], 1)
            self.assertEqual(restored['rng'], original['rng'])
            self.assertEqual(restored['training'], original['training'])
            torch.testing.assert_close(restored['weights']['weight'], original['weights']['weight'], rtol=0, atol=0)
            torch.testing.assert_close(restored['optimizers'][0]['state'][0]['exp_avg'],
                                       original['optimizers'][0]['state'][0]['exp_avg'], rtol=0, atol=0)
            # Replay the boundary after a restart: stage 1 is not a completed stage 0.
            with self.assertRaisesRegex(ValueError, 'fully converged'):
                advance(restored, policy, refresh)
            restored['progress'].update(level_index=5, stop_reason='curriculum_converged')
            advance(restored, policy, refresh)
            self.assertEqual(len(refreshed), 1)
            self.assertEqual(restored['run']['levels'], [None])
            self.assertTrue(restored['run']['reserve_final_level'])

    def test_stop_reason_and_full_curriculum_both_required(self):
        for reason, index in [('interrupted', 5), ('encoder_collapsed', 2), ('error', 5),
                              ('action_limit', 2), ('curriculum_converged', 2)]:
            with self.subTest(reason=reason):
                p, policy = payload(ROOT), plan(ROOT)
                enroll(p, policy)
                p['progress'].update(stop_reason=reason, level_index=index)
                self.assertFalse(converged(p))
                with self.assertRaises(ValueError):
                    advance(p, policy)
        # Old 2-level convergence only extends the current game, not the sequence.
        p = payload(ROOT)
        p['progress'].update(stop_reason='curriculum_converged', level_index=2)
        enroll(p, plan(ROOT))
        self.assertFalse(converged(p))
        self.assertEqual(p['progress']['level_index'], 2)

    def test_mismatched_identity_and_concurrent_owner_are_rejected(self):
        p, policy = payload(ROOT), plan(ROOT)
        enroll(p, policy)
        p['run']['game'] = 'si01-v1'
        with self.assertRaisesRegex(ValueError, 'does not match'):
            check_stage(p, policy)
        with tempfile.TemporaryDirectory() as directory, owner(Path(directory)):
            with self.assertRaisesRegex(RuntimeError, 'Another supervisor'):
                with owner(Path(directory)):
                    self.fail('Duplicate owner acquired lock')

    def test_supervisor_does_not_advance_after_interruption_or_failed_child(self):
        for exit_code in (None, 1, 0):
            with self.subTest(code=exit_code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                p, policy = payload(root), plan(root)
                p['progress']['stop_reason'] = 'interrupted'
                checkpoint.write(root / 'latest.pt', p)
                with patch('model.sequence.launch', return_value=exit_code) as launch, redirect_stdout(io.StringIO()):
                    if exit_code == 1:
                        with self.assertRaises(RuntimeError):
                            supervise(root, policy, 'cpu')
                    else:
                        supervise(root, policy, 'cpu')
                self.assertEqual(launch.call_count, 1)
                self.assertEqual(checkpoint.load(root / 'latest.pt')['run']['sequence']['stage'], 0)

    def test_public_split_is_stable_disjoint_and_requires_complete_catalog(self):
        games = [f'g{i:03}-public' for i in range(25)]
        training, evaluation = public_split(games)
        self.assertEqual(evaluation, ['g000-public', 'g003-public', 'g008-public', 'g020-public', 'g023-public'])
        self.assertEqual(len(training), 20)
        self.assertFalse(set(training) & set(evaluation))
        self.assertEqual(public_split(list(reversed(games))), (training, evaluation))
        for invalid in (games[:3], games[:-1] + ['g000-other']):
            with self.assertRaises(ValueError):
                public_split(invalid)

    def test_small_cpu_trainer_completes_supervised_curriculum(self):
        # Deliberately permissive gates exercise process/checkpoint wiring, not
        # predictive quality. Real SDK responses and real optimizer updates.
        config = replace(TrainConfig(), validate_every=1, validation_episodes=1,
                         validation_steps=4, min_validation_transitions=1,
                         min_changed_transitions=1, consecutive_passes=1,
                         delta_error=1, latent_delta_mse=1000, min_latent_std=0)
        learner = Learner(Engine(WorldModel(NetworkConfig.small()), config))
        run = {'game': 'sc01-v1', 'levels': [1, 2, 3, 4, 5], 'seed': 0,
               'mode': 'offline', 'environments_dir': str(ROOT / 'games')}
        progress = dict(actions=0, updates=0, episodes=0, level_index=4,
                        level_actions=0, passes=0, best_error=1.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = plan(ROOT) | {'games': ['sc01-v1']}
            checkpoint.save(root / 'latest.pt', learner, progress, run, random.Random(0))
            supervise(root, policy, 'cpu')
            saved = checkpoint.load(root / 'latest.pt')
        self.assertTrue(converged(saved))
        self.assertEqual(saved['progress']['level_index'], 5)
        self.assertGreater(saved['progress']['updates'], 0)
        self.assertEqual(saved['run']['sequence']['stage'], 0)

    def test_refresh_rebases_private_branch_and_aborts_conflict_without_touching_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote, source, target = root / 'remote.git', root / 'source', root / 'private'
            subprocess.run(['git', 'init', '--bare', '--initial-branch=main', str(remote)], check=True, capture_output=True)
            subprocess.run(['git', 'clone', str(remote), str(source)], check=True, capture_output=True)
            git(source, 'config', 'user.email', 'test@example.com')
            git(source, 'config', 'user.name', 'Test')
            (source / 'base').write_text('base\n')
            git(source, 'add', 'base')
            git(source, 'commit', '-m', 'base')
            git(source, 'push', 'origin', 'main')
            git(source, 'switch', '-c', 'codex/issue-test')
            (source / 'feature').write_text('feature\n')
            git(source, 'add', 'feature')
            git(source, 'commit', '-m', 'feature')
            head = git(source, 'rev-parse', 'HEAD')
            git(source, 'switch', 'main')
            (source / 'new').write_text('new main\n')
            git(source, 'add', 'new')
            git(source, 'commit', '-m', 'main advance')
            git(source, 'push', 'origin', 'main')
            git(source, 'switch', 'codex/issue-test')
            policy = plan(source) | {'refresh_checkout': str(target), 'revision': head}
            refresh_checkout(policy)
            self.assertEqual((target / 'new').read_text(), 'new main\n')
            self.assertEqual((target / 'feature').read_text(), 'feature\n')
            self.assertEqual(git(source, 'rev-parse', 'HEAD'), head)
            rebased = git(target, 'rev-parse', 'HEAD')
            refresh_checkout(policy)
            self.assertEqual(git(target, 'rev-parse', 'HEAD'), rebased)
            (target / 'base').write_text('dirty\n')
            with self.assertRaisesRegex(ValueError, 'cleanliness'):
                refresh_checkout(policy)
            git(target, 'restore', 'base')
            # Both sides add different contents to feature: real rebase conflict.
            git(source, 'switch', 'main')
            (source / 'feature').write_text('conflicting main\n')
            git(source, 'add', 'feature')
            git(source, 'commit', '-m', 'conflict')
            git(source, 'push', 'origin', 'main')
            git(source, 'switch', 'codex/issue-test')
            with self.assertRaises(subprocess.CalledProcessError):
                refresh_checkout(policy)
            self.assertEqual(git(target, 'rev-parse', 'HEAD'), rebased)
            self.assertEqual(git(target, 'status', '--porcelain'), '')
            self.assertEqual(git(source, 'rev-parse', 'HEAD'), head)


class ReservedLevelTests(unittest.TestCase):
    def test_synthetic_interaction_level_five_is_supported(self):
        factory = Factory('si01-v1', ROOT / 'games')
        self.addCleanup(factory.arcade.close_scorecard)
        env = factory.make(123, 5)
        self.assertEqual(env.wrapper._game.level_index, 4)
        self.assertEqual(env.observation.win_levels, 1)
        with self.assertRaises(ValueError):
            factory.make(123, 6)

    def test_reserved_response_never_enters_encoder_replay_or_validation_metrics(self):
        first = Observation(np.ones((1, 4, 4), dtype=np.uint8), (1,), 'NOT_FINISHED', 0, 2)
        valid = replace(first, frames=np.full((1, 4, 4), 2, dtype=np.uint8))
        reserved = replace(first, frames=np.full((1, 4, 4), 15, dtype=np.uint8), levels_completed=1)
        class Environment:
            observation = first
            def __init__(self):
                self.responses = iter([valid, reserved])
            def step(self, action):
                return next(self.responses)
        class Environments:
            reserve_final_level = True
            def make(self, seed, level):
                return Environment()
        config = replace(TrainConfig(), max_steps=4, train_every=8, validate_every=99,
                         validation_episodes=1, validation_steps=4)
        engine = Engine(WorldModel(NetworkConfig.small()), config)
        learner = Learner(engine)
        original_update, seen = learner.update, []
        def update(episodes):
            seen.extend(o for e in episodes for o in e.observations)
            return original_update(episodes)
        progress = dict(actions=0, updates=0, episodes=0, level_index=0, level_actions=0, passes=0, best_error=1.)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), patch.object(learner, 'update', update):
            train(engine, learner, Environments(), {'levels': [None]}, progress, random.Random(0), Path(directory))
        self.assertEqual(progress['actions'], 4)
        self.assertTrue(seen)
        self.assertTrue(all(o.levels_completed == 0 for o in seen))
        encoded, initial, observe = [], engine.initial, engine.observe
        def traced_initial(obs):
            encoded.append(obs)
            return initial(obs)
        def traced_observe(state, action, obs):
            encoded.append(obs)
            return observe(state, action, obs)
        with redirect_stdout(io.StringIO()), patch.object(engine, 'initial', traced_initial), patch.object(engine, 'observe', traced_observe):
            report = validation(engine, Environments(), [None], 0)['None']
        self.assertEqual(report['transitions'], 1)
        self.assertEqual(report['decoder_transitions'], 1)
        self.assertEqual(report['actions'], 2)
        self.assertEqual(report['reserved_level_boundaries'], 1)
        self.assertTrue(all(o.levels_completed == 0 for o in encoded))
        self.assertFalse(report['passes'])
        with self.assertRaises(ValueError):
            training_boundary(Environments(), replace(first, win_levels=1))

    def test_good_first_level_does_not_approve_unseen_native_levels(self):
        first = Observation(np.ones((1, 4, 4), dtype=np.uint8), (1,), 'NOT_FINISHED', 0, 3)
        class Environment:
            observation = first
            def step(self, action):
                return replace(first, frames=np.full((1, 4, 4), 2, dtype=np.uint8))
        class Environments:
            reserve_final_level = True
            def make(self, seed, level):
                return Environment()
        config = replace(TrainConfig(), validation_episodes=1, validation_steps=1,
                         min_validation_transitions=1, min_changed_transitions=1,
                         delta_error=1, latent_delta_mse=1000, min_latent_std=0)
        engine = Engine(WorldModel(NetworkConfig.small()), config)
        with redirect_stdout(io.StringIO()):
            report = validation(engine, Environments(), [None], 0)['None']
        self.assertTrue(report['native_levels']['1']['passes'])
        self.assertEqual(report['native_levels']['2']['transitions'], 0)
        self.assertFalse(report['passes'])
