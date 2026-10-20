"""Ordered training, public-only SDK access, resume and release boundaries.

Expected orders/thresholds/counts below are literal user-facing contracts.
Tiny real CPU weights exercise scratch/resume and export; SDK fakes expose only
observations and step, never private level selection or synthetic teacher rules.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from model import agent_checkpoint, checkpoint
from model.curriculum import CurriculumConfig, OrderedCurriculum
from model.environment import Factory, NativeTrainingFactory, SyntheticWorldFactory, synthetic_levels
from model.subgoal_training import main, play
from model.training_config import load_plan, run_owner
from model.training_release import prepare, file_sha256, final_quality_passes
from model.tests.test_agent import obs
from model.tests.test_subgoals import path_agent


GOOD = dict(reconstruction_error=.01, prediction_error=.03, rollout_error=.04,
            latent_drift=.02, samples=16, rollout_samples=8, changed=8,
            rollout_changed=4, completed_episodes=1)


def config(root, entries=None, **overrides):
    agent = path_agent()
    (root / 'initial.json').write_text(json.dumps(dict(network=asdict(agent.engine.model.config),
        training=asdict(agent.engine.config), outcome_width=4)))
    value = dict(version=1, init_config='initial.json',
        settings=dict(actions=2, steps=2, evaluate_every=2, evaluation_episodes=1,
                      checkpoint_every=2, path_horizon=1),
        curriculum=entries or [dict(source='synthetic', game='sc01-v1', levels=[2, 1])])
    value.update(overrides)
    path = root / 'training.json'
    path.write_text(json.dumps(value))
    return path, value


class TrainingConfigTests(unittest.TestCase):
    def test_all_synthetic_sequence_config_collects_every_game_and_level(self):
        from model.config import SequenceConfig
        from model.sequence_training import factories_for, close_factories, collect_batch
        path = Path(__file__).parents[1]/'sequence_all_synthetic.example.json'
        plan = load_plan(path)
        expected_games = {'sc01-v1', 'si01-v1', 'mc01-v1'} | {
            f'bf{i:02d}-v1' for i in range(1, 9)} | {f'sy{i:02d}-v1' for i in range(1, 61)}
        self.assertEqual({s['game'] for s in plan.stages}, expected_games)
        self.assertEqual(len(plan.stages), 424)
        self.assertEqual({s['objective'] for s in plan.stages}, {'world_model'})
        self.assertEqual(plan.payload['sequence_control']['window_batches'], 18)
        self.assertEqual(plan.payload['sequence_control']['diagnostic_samples'], 384)
        c = SequenceConfig(**(plan.payload['sequence'] | {'session_steps': 192}))
        factories = factories_for(plan)
        seen, sizes = set(), []
        try:
            for index in range(9):
                sessions = collect_batch(factories, c, index, plan.settings['seed'])
                sizes.append(len(sessions))
                seen.update((s.game, t.level_number) for s in sessions for t in s.transitions if t.supervised)
            self.assertEqual(sizes, [8] * 8 + [7])
            self.assertEqual(seen, {(s['game'], s['level']) for s in plan.stages})
        finally:
            close_factories(factories)

    def test_complete_synthetic_catalog_and_real_sdk_level_isolation(self):
        self.assertEqual(len(synthetic_levels()), 71)
        self.assertEqual(synthetic_levels()['mc01-v1'], 6)
        self.assertEqual(synthetic_levels()['bf01-v1'], 6)
        self.assertEqual(synthetic_levels()['sy60-v1'], 6)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = config(root, [dict(source='synthetic', game='bf01-v1', levels=[6, 1]),
                                   dict(source='synthetic', game='sc01-v1', levels=[1], objective='world_model')])
            plan = load_plan(path)
            self.assertEqual([s['objective'] for s in plan.stages], ['world_model'] * 3)
            factory = SyntheticWorldFactory('bf01-v1', Path('games'))
            try:
                result = play(path_agent(), factory, 6, 10, 2, random_policy=True)
                self.assertEqual(result['actions'], 2)
                self.assertEqual(result['segments'], [])
                self.assertEqual([r['goal'] for r in result['records']], ['Dinámica del entorno'] * 2)
            finally:
                factory.arcade.close_scorecard()

    def test_maze_world_model_registration_isolation_and_no_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, values = config(root, [dict(source='synthetic', game='mc01-v1', levels=[6, 3, 1])],
                                  sequence=dict(batch_games=4, session_steps=16, level_steps=4))
            plan = load_plan(path)
            self.assertEqual([s['objective'] for s in plan.stages], ['world_model']*3)
            self.assertEqual([s['level'] for s in plan.stages], [6, 3, 1])
            factory = SyntheticWorldFactory('mc01-v1', Path('games'))
            try:
                env = factory.make(42, 6)
                self.assertEqual(env.wrapper._game.number, 6)
                self.assertEqual(env.observation.final.shape, (64, 64))
                with self.assertRaisesRegex(ValueError, 'do not use a subgoal teacher'):
                    factory.teacher(env)
            finally:
                factory.arcade.close_scorecard()
            values['curriculum'][0]['objective'] = 'subgoals'
            path.write_text(json.dumps(values))
            with self.assertRaises(ValueError): load_plan(path)

    def test_explicit_order_thresholds_relative_paths_and_pure_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = config(root, [dict(source='synthetic', game='si01-v1', levels=[4, 2],
                                         thresholds={'success': .9}),
                                    dict(source='synthetic', game='sc01-v1', levels=[3])],
                             thresholds={'prediction_open': .2, 'patience': 2})
            plan = load_plan(path)
            self.assertEqual([(s['game'], s['level']) for s in plan.stages],
                             [('si01-v1', 4), ('si01-v1', 2), ('sc01-v1', 3)])
            self.assertEqual([s['thresholds']['success'] for s in plan.stages], [.9, .9, .8])
            self.assertEqual([s['thresholds']['prediction_open'] for s in plan.stages], [.2] * 3)
            self.assertEqual(plan.init_config, root / 'initial.json')
            before = sorted(p.name for p in root.iterdir())
            with redirect_stdout(io.StringIO()) as stream, patch('model.subgoal_training.WorldModel', side_effect=AssertionError('No model in preview')):
                main(['--training-config', str(path), '--validate-config'])
            self.assertEqual(json.loads(stream.getvalue())['training_plan_id'], plan.identity)
            self.assertEqual(sorted(p.name for p in root.iterdir()), before)

    def test_invalid_configs_fail_before_any_run_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, base = config(root)
            invalid = [dict(version=True), dict(unknown=1), dict(curriculum=[]),
                       dict(thresholds={'success': float('nan')}), dict(thresholds={'patience': True}),
                       dict(settings={'steps': 129}), dict(settings={'prediction_delta_loss': 'guess'}),
                       dict(settings={'path_horizon': 0}), dict(settings={'learning_rate': '3e-5'})]
            invalid += [dict(curriculum=[dict(source='synthetic', game='sc01-v1', levels=levels)])
                        for levels in ([0], [6], [True], [1, 1])]
            invalid += [dict(curriculum=[dict(source='sdk', game='aa01', levels=[1])])]
            for changes in invalid:
                with self.subTest(changes=changes):
                    path.write_text(json.dumps(base | changes))
                    with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()):
                        main(['--training-config', str(path), '--output', str(root / 'run'), '--device', 'cpu'])
                    self.assertFalse((root / 'run').exists())
            path.write_text('{"version":1,"version":1}')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_plan(path)

    def test_sdk_development_split_caps_and_final_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'split.json').write_text(json.dumps(dict(training_games=['aa01-v1'],
                evaluation_games=['bb02'], training_level_caps={'aa01-v1': 2})))
            sdk = dict(environments_dir='sdk-games', split='split.json', access_steps=40)
            path, value = config(root, [dict(source='sdk', game='aa01-v1', levels=[2, 1])], sdk=sdk)
            plan = load_plan(path)
            self.assertEqual([s['level'] for s in plan.stages], [2, 1])
            self.assertEqual(plan.payload['sdk']['access_steps'], 40)
            for game, levels in [('bb02-v3', [1]), ('aa01-v1', [3])]:
                path.write_text(json.dumps(value | dict(curriculum=[dict(source='sdk', game=game, levels=levels)])))
                with self.assertRaises(ValueError):
                    load_plan(path)
            path.write_text(json.dumps(value | dict(phase='final', sdk={'environments_dir': 'sdk-games'})))
            self.assertEqual(load_plan(path).payload['phase'], 'final')

    def test_ordered_progress_resume_and_manual_steps_are_not_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = config(root, thresholds={'patience': 1})
            plan = load_plan(path)
            curriculum = OrderedCurriculum(plan.stages)
            self.assertEqual(curriculum.next_pair(), ('sc01-v1', 2))
            curriculum.advance_phase(('sc01-v1', 2))
            self.assertEqual(curriculum.readiness['sc01-v1', 2].promotion_passes, 0)
            self.assertEqual(curriculum.advance(), [])
            self.assertEqual(curriculum.advance_level('sc01-v1'), ('sc01-v1', 1))
            state = curriculum.snapshot()
            resumed = OrderedCurriculum(plan.stages)
            resumed.restore(state)
            self.assertEqual(resumed.next_pair(), ('sc01-v1', 1))
            resumed.readiness['sc01-v1', 1].observe(GOOD, 1., resumed.config)
            resumed.advance()
            self.assertTrue(resumed.completed)
            # Manual advancement did not qualify the skipped stage.
            self.assertEqual(resumed.readiness['sc01-v1', 2].promotion_passes, 0)
            with self.assertRaises(ValueError):
                OrderedCurriculum(tuple(reversed(plan.stages))).restore(state)

    def test_sdk_quality_does_not_enable_or_require_a_planner(self):
        stages = [dict(source='sdk', game='aa01', level=1,
                       thresholds=asdict(CurriculumConfig(patience=2)))]
        curriculum = OrderedCurriculum(stages)
        state = curriculum.readiness['aa01', 1]
        for _ in range(3):
            state.observe(GOOD, None, curriculum.config, world_only=True)
            curriculum.advance()
        self.assertTrue(curriculum.completed)
        self.assertTrue(state.predictor)
        self.assertFalse(state.planner)

    def test_run_lease_and_fresh_directory_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            with run_owner(output, False):
                with self.assertRaisesRegex(ValueError, 'Another trainer'):
                    with run_owner(output, True):
                        self.fail('Concurrent ownership admitted')
                (output / 'latest.pt').write_bytes(b'preserve')
            with self.assertRaisesRegex(ValueError, 'empty'):
                with run_owner(output, False):
                    self.fail('Fresh overwrite admitted')
            self.assertEqual((output / 'latest.pt').read_bytes(), b'preserve')

    def test_tiny_scratch_budget_resume_keeps_order_and_rejects_changed_contract(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path, value = config(root)
            run = root / 'run'
            args = ['--training-config', str(path), '--output', str(run), '--device', 'cpu']
            main(args)
            report = json.loads((run / 'report.json').read_text())
            self.assertEqual(report['status'], 'budget_exhausted')
            self.assertFalse(report['publication_eligible'])
            self.assertEqual(report['episodes'][0]['level'], 2)
            self.assertEqual(report['curriculum']['index'], 0)
            first = checkpoint.load(run / 'latest.pt')
            with self.assertRaises(ValueError):
                prepare(run, root / 'release')
            main([*args, '--resume', '--actions', '4'])
            saved = checkpoint.load(run / 'latest.pt')
            self.assertEqual(saved['training_progress']['actions'], 4)
            self.assertGreater(saved['training_progress']['representation_updates'], first['training_progress']['representation_updates'])
            self.assertEqual(saved['training_progress']['training_plan_id'], first['training_progress']['training_plan_id'])
            before = (run / 'latest.pt').read_bytes(), (run / 'report.json').read_bytes()
            value['curriculum'][0]['levels'] = [1, 2]
            path.write_text(json.dumps(value))
            with self.assertRaises(SystemExit), patch('sys.stderr', io.StringIO()):
                main([*args, '--resume', '--actions', '6'])
            self.assertEqual(((run / 'latest.pt').read_bytes(), (run / 'report.json').read_bytes()), before)

    def test_native_cpu_curriculum_converges_in_requested_order_and_validates_all_stages(self):
        class PublicEnvironment:
            def __init__(self, level):
                self.observation = obs(1, level=level - 1)
                self.actions = 0
            def step(self, action):
                self.actions += 1
                self.observation = replace(self.observation,
                    frames=obs(1 + self.actions % 2).frames, action_input=action)
                return self.observation
        for checkpoint_during_final in (False, True):
            with self.subTest(checkpoint_during_final=checkpoint_during_final), \
                 tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                if checkpoint_during_final and not hasattr(signal, 'SIGUSR1'):
                    self.skipTest('SIGUSR1 is a POSIX signal')
                signalled = []
                def make(seed, level):
                    # Only the final seed partition requests a checkpoint. Assert
                    # a handler before delivery so a regression cannot kill unittest.
                    if checkpoint_during_final and seed >= 8_000_001 and not signalled:
                        self.assertTrue(callable(signal.getsignal(signal.SIGUSR1)))
                        signal.raise_signal(signal.SIGUSR1)
                        signalled.append(level)
                    return PublicEnvironment(level)
                factory = SimpleNamespace(native_training=True, reserve_final_level=False,
                    game_id='aa01', access_actions=2, make=make)
                root = Path(tmp)
                path, value = config(root, [dict(source='sdk', game='aa01', levels=[2, 1])],
                    phase='final', sdk={'environments_dir': 'sdk'},
                    thresholds=dict(representation_open=1., prediction_open=1., latent_drift=1.,
                                    patience=1, minimum_samples=1, minimum_changed=1))
                value['settings'].update(actions=24, steps=3, evaluate_every=3)
                path.write_text(json.dumps(value))
                # Keep unrelated Git provenance stable, including in dirty test
                # worktrees; training, evaluation, checkpoint IO and export are real.
                with patch('model.subgoal_training.NativeTrainingFactory', return_value=factory), \
                     patch('subprocess.check_output', side_effect=['a' * 40, '']):
                    main(['--training-config', str(path), '--output', str(root / 'run'), '--device', 'cpu'])
                report = json.loads((root / 'run/report.json').read_text())
                self.assertEqual(signalled, [2] if checkpoint_during_final else [])
                self.assertEqual(report['last_checkpoint']['reason'], 'signal' if checkpoint_during_final else 'final')
                self.assertEqual(report['status'], 'converged')
                self.assertTrue(report['publication_eligible'])
                self.assertLess(report['checkpoint_action'], 24)
                levels = [r['level'] for r in report['episodes']]
                self.assertEqual(levels, sorted(levels, reverse=True))
                self.assertEqual(set(levels), {1, 2})
                self.assertEqual([r['level'] for r in report['final_validation']], [2, 1])
                self.assertTrue(all(r['passed'] for r in report['final_validation']))
                self.assertTrue(all(r['success'] is None for r in report['final_validation']))
                self.assertEqual(report['progress']['path_updates'], 0)
                self.assertEqual(report['progress']['access_actions'], 2 * len(levels))
                saved = checkpoint.load(root / 'run/latest.pt')
                self.assertTrue(saved['training_progress']['curriculum']['completed'])
                self.assertEqual(report['checkpoint_sha256'], hashlib.sha256((root / 'run/latest.pt').read_bytes()).hexdigest())
                manifest = prepare(root / 'run', root / 'release')
                self.assertEqual([r['level'] for r in manifest['validation']], [2, 1])
                agent = agent_checkpoint.load(root / 'release/agent.pt', device='cpu', compile_model=False)
                agent.observe(obs())
                self.assertIsNotNone(agent.state.z)


class PublicNativeTests(unittest.TestCase):
    def factory(self, access=3):
        factory = NativeTrainingFactory.__new__(NativeTrainingFactory)
        factory.game_id, factory.reserve_final_level = 'aa01', True
        factory.access_steps, factory.access_actions = access, 0
        return factory

    def test_public_traversal_and_unreachable_or_reserved_levels(self):
        factory = self.factory()
        env = SimpleNamespace(observation=obs(12, level=0))
        def step(action):
            env.observation = obs(7, level=1)
            return env.observation
        env.step = Mock(side_effect=step)
        with patch.object(Factory, 'make', return_value=env):
            selected = factory.make(10, 2)
        self.assertEqual(factory.access_actions, 1)
        self.assertEqual(selected.observation.levels_completed, 1)
        self.assertEqual(int(selected.observation.final[0, 0]), 7)
        self.assertIsNone(selected.observation.action_input)
        with patch.object(Factory, 'make', return_value=env), self.assertRaisesRegex(ValueError, 'reserved'):
            factory.make(10, 3)
        env.observation = obs(12)
        env.step = Mock(return_value=env.observation)
        with patch.object(Factory, 'make', return_value=env), self.assertRaisesRegex(ValueError, 'Could not reach'):
            factory.make(10, 2)
        self.assertEqual(env.step.call_count, 3)

    def test_native_boundary_never_enters_encoder_replay_or_recordings(self):
        agent = path_agent()
        initial, reserved = obs(2, level=1), obs(15, level=2)
        env = SimpleNamespace(observation=initial, step=Mock(return_value=reserved))
        factory = self.factory()
        representation = SimpleNamespace(frozen=False, update=Mock(side_effect=AssertionError('Reserved observation learned')))
        replay = SimpleNamespace(record=Mock(side_effect=AssertionError('Reserved observation replayed')))
        captured, progress = [], []
        with patch.object(factory, 'make', return_value=env), patch.object(agent.engine, 'observe', side_effect=AssertionError('Reserved frame encoded')):
            result = play(agent, factory, 2, 10, 4, random_policy=True,
                representation=representation, predictor_replay=replay, progress=progress.append,
                trajectory=lambda *args: captured.append(args))
        self.assertTrue(result['success'])
        self.assertEqual(result['actions'], 1)
        self.assertEqual(result['counts']['union'], 0)
        self.assertIsNone(progress[0]['observation'])
        self.assertEqual(captured, [])
        recordings = []
        with patch.object(factory, 'make', return_value=env):
            play(agent, factory, 2, 10, 4, random_policy=True,
                 evaluation_capture=lambda episode, result: recordings.append(episode))
        self.assertEqual(len(recordings[0].observations), 1)
        self.assertEqual(recordings[0].actions, [])
        self.assertEqual(int(recordings[0].observations[0].final[0, 0]), 2)


class TrainingReleaseTests(unittest.TestCase):
    def fixture(self, root):
        path, _ = config(root, [dict(source='synthetic', game='sc01-v1', levels=[1])],
                         thresholds={'patience': 1})
        plan = load_plan(path)
        curriculum = OrderedCurriculum(plan.stages)
        curriculum.readiness['sc01-v1', 1].observe(GOOD, 1., curriculum.config)
        curriculum.advance()
        run = root / 'run'
        run.mkdir()
        agent_checkpoint.save(run / 'latest.pt', path_agent(), training_progress=dict(actions=42,
            training_plan_id=plan.identity, curriculum=curriculum.snapshot(), manual_controls=[]))
        report = dict(status='converged', checkpoint_action=42, manual_controls=[],
            config=dict(training_plan=plan.payload, training_plan_id=plan.identity),
            curriculum=curriculum.snapshot(), checkpoint_sha256=file_sha256(run / 'latest.pt'),
            provenance=dict(revision='a' * 40, dirty=False),
            final_validation=[dict(game='sc01-v1', level=1, source='synthetic', objective='subgoals',
                thresholds=plan.stages[0]['thresholds'], quality=dict(GOOD, latent_drift=None), success=1.)])
        (run / 'report.json').write_text(json.dumps(report))
        return run, report

    def test_export_is_loadable_finite_data_free_and_checksums_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, report = self.fixture(root)
            with run_owner(run, True), self.assertRaisesRegex(ValueError, 'Another trainer'):
                prepare(run, root / 'release')
            result = prepare(run, root / 'release')
            exported = checkpoint.load(root / 'release/agent.pt')
            self.assertIsNone(exported['runtime'])
            self.assertEqual(exported['training_progress'], {})
            original = checkpoint.load(run / 'latest.pt')
            for key, value in original['world']['weights'].items():
                torch.testing.assert_close(exported['world']['weights'][key], value, rtol=0, atol=0)
            loaded = agent_checkpoint.load(root / 'release/agent.pt', device='cpu', compile_model=False)
            loaded.observe(obs())
            self.assertIsNotNone(loaded.state.z)
            self.assertNotIn(str(root), (root / 'release/manifest.json').read_text())
            for line in (root / 'release/SHA256SUMS').read_text().splitlines():
                expected, name = line.split()
                self.assertEqual(expected, hashlib.sha256((root / 'release' / name).read_bytes()).hexdigest())
            self.assertEqual(result['source_revision'], 'a' * 40)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                prepare(run, root / 'release')

    def test_incomplete_manual_stale_failed_or_missing_evidence_cannot_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, baseline = self.fixture(root)
            changes = [dict(status='budget_exhausted'), dict(manual_controls=[{'operation': 'level'}]),
                       dict(checkpoint_sha256='0' * 64), dict(checkpoint_action=43),
                       dict(final_validation=[]), dict(provenance={'revision': 'a' * 40, 'dirty': True})]
            bad_quality = deepcopy(baseline['final_validation'])
            bad_quality[0]['quality']['prediction_error'] = .4
            changes.append(dict(final_validation=bad_quality))
            for change in changes:
                with self.subTest(change=change):
                    (run / 'report.json').write_text(json.dumps(baseline | change))
                    with self.assertRaises(ValueError):
                        prepare(run, root / 'release')
                    self.assertFalse((root / 'release').exists())
            payload = checkpoint.load(run / 'latest.pt')
            next(iter(payload['world']['weights'].values())).fill_(float('nan'))
            checkpoint.write(run / 'latest.pt', payload)
            baseline['checkpoint_sha256'] = file_sha256(run / 'latest.pt')
            (run / 'report.json').write_text(json.dumps(baseline))
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                prepare(run, root / 'release')

    def test_final_quality_requires_evidence_and_success_only_for_synthetic_goals(self):
        gates = CurriculumConfig()
        self.assertTrue(final_quality_passes(dict(GOOD, latent_drift=None), None, gates, True))
        self.assertFalse(final_quality_passes(GOOD, .79, gates))
        self.assertTrue(final_quality_passes(GOOD, .8, gates))
        self.assertFalse(final_quality_passes(dict(GOOD, samples=7), 1., gates))
        self.assertFalse(final_quality_passes(dict(GOOD, rollout_error=float('nan')), 1., gates))


if __name__ == '__main__':
    unittest.main()
