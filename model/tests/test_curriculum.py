"""Incremental activation, cumulative levels and real-only online learning.

Threshold tests use literal measurements, optimizer tests compare module weights,
and rollout tests use a tiny arithmetic world whose three-step answer is 7.
"""
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.curriculum import Curriculum, CurriculumConfig, LevelReadiness, planner_exploration
from model import agent_checkpoint, checkpoint
from model.data import Action, Episode
from model.environment import Environment, Factory
from model.learning import State
from model.online_learning import OnlineRepresentation, OnlinePredictor
from model.path_planning import PathLearner, real_states
from model.planning import fingerprint
from model.subgoal_training import main, play, QualityProbe, summarize_quality, evaluate_recording
from model.fixed_evaluation import random_cohort
from model.tests.test_subgoals import path_agent
from model.tests.test_agent import obs
from tests.square_cross_support import fixture


GOOD = dict(reconstruction_error=.005, latent_drift=.01, prediction_error=.05,
            rollout_error=.08, samples=20, rollout_samples=10, changed=10, rollout_changed=5)


class CurriculumTests(unittest.TestCase):
    def test_fixed_recording_matches_live_quality_and_reencodes_current_weights(self):
        agent = path_agent()
        engine = agent.engine
        factory = Factory('sc01-v1', Path('games'))
        metadata = dict(game='sc01-v1', level=1, split='dev', seeds=[123], steps=4)
        captures = []
        weights = fingerprint(engine.model), fingerprint(agent.path_evaluator)
        def collect(seed):
            play(agent, factory, 1, seed, 4, random_policy=True, quality_horizon=2,
                 evaluation_capture=lambda episode, result: captures.append((episode, result)))
            return captures[-1]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'dev.pt'
            payload, recordings = random_cohort(path, metadata, collect, 7)
            episode, outcome = recordings[0]
            live = captures[0][1]['quality']
            cached = evaluate_recording(engine, episode, 2)
            self.assertEqual(cached['counts'], live['counts'])
            torch.testing.assert_close(cached['anchor'], live['anchor'], rtol=0, atol=0)
            self.assertEqual(episode.actions, captures[0][0].actions)
            for old, new in zip(captures[0][0].observations, episode.observations):
                np.testing.assert_array_equal(old.frames, new.frames)
                self.assertEqual(old.action_input, new.action_input)
            self.assertEqual((fingerprint(engine.model), fingerprint(agent.path_evaluator)), weights)
            self.assertTrue(all(p.grad is None for p in engine.model.parameters()))
            original = path.read_bytes()
            with torch.no_grad():
                for parameter in engine.model.encoder.parameters():
                    parameter.add_(.05)
            changed = fingerprint(engine.model)
            with patch.object(factory, 'make', side_effect=AssertionError('No repeated environment calls')):
                resumed, loaded = random_cohort(path, metadata, collect, 100)
                measured = evaluate_recording(engine, loaded[0][0], 2)
            self.assertEqual(len(captures), 1)
            self.assertEqual(resumed['collected_at_action'], 7)
            self.assertEqual(loaded[0][1], outcome)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(torch.equal(measured['anchor'], live['anchor']))
            self.assertEqual(fingerprint(engine.model), changed)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                random_cohort(path, dict(metadata, split='final'), collect, 100)
            with self.assertRaisesRegex(ValueError, 'must not update'):
                play(agent, factory, 1, 123, 4, representation=object(), evaluation_capture=lambda *_: None)

    def test_evaluation_reuses_random_cohort_on_resume_and_only_visits_latest_level(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            agent_checkpoint.save(root/'source.pt', path_agent())
            args = ['--source',str(root/'source.pt'),'--output',str(root/'run'),
                    '--device','cpu','--steps','2','--evaluate-every','2','--evaluation-episodes','1']
            main([*args,'--actions','4'])
            report = json.loads((root/'run/report.json').read_text())
            baseline = [r for r in report['evaluations'] if r.get('fixed_baseline') and r['split']=='dev']
            self.assertEqual([r['actions'] for r in baseline], [0])
            probes = [r for r in report['evaluations'] if r['policy']=='quality_probe' and r['split']=='dev']
            self.assertEqual([r['actions'] for r in probes], [0,2,4])
            self.assertTrue(all(r['real_actions']==0 and r['successes'] is None for r in probes))
            saved = checkpoint.load(root/'run/latest.pt')
            progress = saved['training_progress']
            # Simulate migration of a two-level legacy run with permissions retained.
            progress.pop('evaluation_protocol')
            progress['curriculum']['unlocked']['sc01-v1'] = 2
            for row in progress['curriculum']['levels'][:2]:
                row.update(predictor=True, planner=True, representation_passes=3,
                           prediction_passes=2, promotion_passes=1)
            checkpoint.write(root/'run/latest.pt', saved)
            before = agent_checkpoint.load(root/'run/latest.pt')
            weights = fingerprint(before.engine.model), fingerprint(before.path_evaluator)
            snapshots, calls = [], []
            save = agent_checkpoint.save
            def capture(path, agent, **kwargs):
                if kwargs['training_progress']['actions']==4:
                    self.assertEqual((fingerprint(agent.engine.model),fingerprint(agent.path_evaluator)),weights)
                    snapshots.append(kwargs['training_progress'])
                return save(path,agent,**kwargs)
            def record(*pos, **kw):
                if kw.get('representation') is None and kw.get('learner') is None and kw.get('world') is None:
                    calls.append((pos[2],pos[3],kw.get('evaluation_capture') is not None))
                return play(*pos,**kw)
            with patch('model.subgoal_training.play',side_effect=record), \
                 patch('model.subgoal_training.agent_checkpoint.save',side_effect=capture):
                main([*args,'--actions','8','--resume'])
            self.assertEqual(len(snapshots),1)
            self.assertTrue(all(r['predictor'] and r['planner'] for r in snapshots[0]['curriculum']['levels'][:2]))
            self.assertEqual(snapshots[0]['planner_actions'],progress['planner_actions'])
            self.assertTrue(all(level==2 for level,_,_ in calls))
            self.assertEqual(sum(collected for _,_,collected in calls),2)  # dev and final, once each.
            self.assertEqual(len({seed for _,seed,collected in calls if collected}),2)
            calls.clear()
            with patch('model.subgoal_training.play',side_effect=record):
                main([*args,'--actions','10','--resume'])
            self.assertTrue(calls)
            self.assertTrue(all(level==2 and not collected for level,_,collected in calls))
            report = json.loads((root/'run/report.json').read_text())
            fresh = [r for r in report['evaluations'] if r['actions']>4]
            self.assertTrue(all(r['level']==2 for r in fresh))
            self.assertEqual(len([r for r in fresh if r.get('fixed_baseline') and r['split']=='dev']),1)

    def test_latest_level_promotion_does_not_require_stale_earlier_evidence(self):
        c = Curriculum({'a':3}, CurriculumConfig(patience=2))
        c.advance_level('a')
        c.readiness['a',2].observe(GOOD,1.,c.config)
        self.assertEqual(c.advance(latest_only=True),[])
        c.readiness['a',2].observe(GOOD,1.,c.config)
        c.readiness['a',2].observe(GOOD,1.,c.config)
        self.assertEqual(c.advance(latest_only=True),[])
        c.readiness['a',2].observe(GOOD,1.,c.config)
        self.assertEqual(c.advance(),[])  # Legacy aggregate mode remains available.
        self.assertEqual(c.advance(latest_only=True),[('a',3)])
        self.assertEqual(c.pairs,[('a',1),('a',2),('a',3)])  # Training remains cumulative.

    def test_exploration_warmup_preserves_initial_and_mature_policy(self):
        for warmup, halfway, endpoint in ((8192,4096,8192),(2048,1024,2048)):
            self.assertEqual(planner_exploration(0,warmup),1.)
            self.assertAlmostEqual(planner_exploration(halfway,warmup),.6)
            self.assertAlmostEqual(planner_exploration(endpoint,warmup),.2)
            self.assertEqual(planner_exploration(36970,warmup),.2)
        self.assertAlmostEqual(planner_exploration(1024),.9)

    def test_resume_exploration_warmup_preserves_weights_permissions_and_inherits(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            agent_checkpoint.save(root/'source.pt',path_agent())
            args = ['--source',str(root/'source.pt'),'--output',str(root/'run'),
                    '--device','cpu','--steps','2','--evaluate-every','4096','--evaluation-episodes','1']
            main([*args,'--actions','4'])
            saved = checkpoint.load(root/'run/latest.pt')
            self.assertEqual(saved['training_progress']['path_exploration_warmup'],8192)
            progress = saved['training_progress']
            progress['curriculum']['levels'][0].update(predictor=True,planner=True,
                representation_passes=3,prediction_passes=2,promotion_passes=1)
            progress['planner_actions'][0]['actions'] = 1024
            checkpoint.write(root/'run/latest.pt',saved)
            old = agent_checkpoint.load(root/'run/latest.pt')
            expected = fingerprint(old.engine.model),fingerprint(old.path_evaluator)
            snapshots = []
            save = agent_checkpoint.save
            def capture(path,agent,**kwargs):
                state = kwargs['training_progress']
                if state['actions'] == 4:
                    self.assertEqual((fingerprint(agent.engine.model),fingerprint(agent.path_evaluator)),expected)
                    snapshots.append(state)
                return save(path,agent,**kwargs)
            with patch('model.subgoal_training.agent_checkpoint.save',side_effect=capture):
                main([*args,'--actions','6','--resume','--path-exploration-warmup','2048'])
            self.assertEqual(len(snapshots),1)
            state = snapshots[0]
            self.assertEqual(state['path_exploration_warmup'],2048)
            self.assertEqual(state['planner_actions'],progress['planner_actions'])
            self.assertEqual(state['curriculum'],progress['curriculum'])
            for key in ('updates','representation_updates','path_updates','episodes'):
                self.assertEqual(state[key],progress[key])
            report_path = root/'run/report.json'
            report = json.loads(report_path.read_text())
            self.assertEqual(report['resumed_from']['path_exploration_warmup'],dict(old=8192,new=2048))
            self.assertAlmostEqual(report['progress']['epsilon'],.6)
            main([*args,'--actions','8','--resume'])
            report = json.loads(report_path.read_text())
            self.assertEqual(report['config']['path_exploration_warmup'],2048)
            self.assertNotIn('path_exploration_warmup',report['resumed_from'])
            self.assertAlmostEqual(report['progress']['epsilon'],.59921875)
            self.assertEqual(checkpoint.load(root/'run/latest.pt')['training_progress']['path_exploration_warmup'],2048)
            before = report_path.read_bytes()
            with self.assertRaises(SystemExit), patch('sys.stderr',new=io.StringIO()):
                main([*args,'--actions','10','--resume','--path-exploration-warmup','0'])
            self.assertEqual(report_path.read_bytes(),before)

    def test_planner_freeze_skips_representation_then_new_level_trains_again(self):
        agent = path_agent()
        e = agent.engine
        e.config = replace(e.config, freeze_representation=False)
        representation = OnlineRepresentation(e, 3e-5, encoder_learning_rate_after_predictor=3e-6)
        predictor, planner = OnlinePredictor(e, None, horizon=2), PathLearner(agent.planner)
        # Prime Adam moments, then freeze without changing their state.
        representation.update([obs(1), obs(3)])
        moments = {p: {k: v.clone() for k, v in s.items()}
                   for p, s in representation.optimizer.state.items()}
        episode = Episode([obs(1), obs(3)], [Action(1)])
        latent = real_states(e, episode)[-1].z.clone()
        epoch = e._latent_cache_epoch
        original = {n: fingerprint(getattr(e.model, n)) for n in ('encoder', 'decoder', 'dynamics')}
        value = fingerprint(agent.path_evaluator)
        representation.select_level(predictor=True, planner=True)
        self.assertEqual(representation.learning_rates, dict(encoder=None, decoder=None))
        # Frozen update is a no-op even for direct callers; play must skip it.
        self.assertIsNone(representation.update([]))
        with patch.object(representation, 'update', side_effect=AssertionError('frozen forward')), \
             patch('games.square_cross.v1.sc01.generate_scene', return_value=fixture()):
            result = play(agent, Factory('sc01-v1', Path('games')), 1, 123, 2,
                          representation=representation, world=predictor, learner=planner, random_policy=True)
        self.assertEqual(representation.updates, 1)
        self.assertEqual(e._latent_cache_epoch, epoch)
        self.assertTrue(torch.equal(real_states(e, episode)[-1].z, latent))
        self.assertTrue(all(r['representation_loss'] is None for r in result['records']))
        for name in ('encoder', 'decoder'):
            self.assertEqual(fingerprint(getattr(e.model, name)), original[name])
            self.assertTrue(all(p.grad is None for p in getattr(e.model, name).parameters()))
        self.assertNotEqual(fingerprint(e.model.dynamics), original['dynamics'])
        self.assertNotEqual(fingerprint(agent.path_evaluator), value)
        representation.select_level(predictor=False, planner=False)
        self.assertEqual(representation.learning_rates, dict(encoder=3e-5, decoder=3e-5))
        self.assertFalse(representation.sigreg_enabled)
        for p, state in moments.items():
            for key, tensor in state.items():
                self.assertTrue(torch.equal(representation.optimizer.state[p][key], tensor))
        representation.update([obs(1), obs(3)])
        for name in ('encoder', 'decoder'):
            self.assertNotEqual(fingerprint(getattr(e.model, name)), original[name])
        self.assertGreater(e._latent_cache_epoch, epoch)
        representation.select_level(predictor=True, planner=False)
        self.assertEqual(representation.learning_rates, dict(encoder=3e-6, decoder=3e-5))
        explicit = OnlineRepresentation(e, freeze_encoder=True)
        explicit.select_level(predictor=True, planner=True)
        explicit.select_level(predictor=False, planner=False)
        self.assertTrue(explicit.freeze_encoder)
        self.assertEqual(explicit.learning_rates, dict(encoder=None, decoder=3e-6))

    def test_planner_freeze_admission_resume_override_and_legacy_compatibility(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
             patch('games.square_cross.v1.sc01.generate_scene', return_value=fixture()):
            root = Path(temp)
            config = root / 'config.json'
            config.write_text(json.dumps({'network': asdict(path_agent().engine.model.config)}))
            args = ['--init-config', str(config), '--output', str(root / 'run'), '--device', 'cpu',
                    '--steps', '2', '--evaluate-every', '2', '--evaluation-episodes', '1', '--patience', '1']
            phases = []
            def capture(*pos, **kw):
                if kw.get('representation'):
                    phases.append(kw['representation'].frozen)
                return play(*pos, **kw)
            pending = dict(GOOD, latent_drift=None)
            with patch('model.subgoal_training.play', side_effect=capture), \
                 patch('model.subgoal_training.summarize_quality',
                       side_effect=[(pending, []), (GOOD, []), (GOOD, []), (GOOD, [])]):
                main([*args, '--actions', '4', '--freeze-representation-after-planner'])
            self.assertEqual(phases, [False, True])
            latest = root / 'run/latest.pt'
            saved = checkpoint.load(latest)
            self.assertEqual(saved['training_progress']['representation_updates'], 2)
            before = agent_checkpoint.load(latest, device='cpu')
            original_save = agent_checkpoint.save
            def check_save(path, agent, **kw):
                if kw['training_progress']['actions'] == 6:
                    # Override save before any extra learning must keep weights/counters.
                    self.assertEqual(fingerprint(agent.engine.model), fingerprint(before.engine.model))
                    self.assertEqual(kw['training_progress']['representation_updates'], 2)
                    self.assertEqual(kw['training_progress']['path_updates'], saved['training_progress']['path_updates'])
                original_save(path, agent, **kw)
            phases.clear()
            with patch('model.subgoal_training.play', side_effect=capture):
                main([*args, '--resume', '--actions', '6'])
            self.assertEqual(phases, [True])
            saved = checkpoint.load(latest)
            before = agent_checkpoint.load(latest, device='cpu')
            self.assertTrue(saved['training_progress']['freeze_representation_after_planner'])
            with patch.object(agent_checkpoint, 'save', side_effect=check_save):
                main([*args, '--resume', '--actions', '8', '--no-freeze-representation-after-planner'])
            self.assertEqual(checkpoint.load(latest)['training_progress']['representation_updates'], 4)
            # An old checkpoint without the option keeps representation trainable.
            saved = checkpoint.load(latest)
            saved['training_progress'].pop('freeze_representation_after_planner')
            checkpoint.write(latest, saved)
            main([*args, '--resume', '--actions', '10'])
            self.assertEqual(checkpoint.load(latest)['training_progress']['representation_updates'], 6)

    def test_encoder_rate_resume_override_inherits_checkpoint_without_losing_weights(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            agent_checkpoint.save(root / 'source.pt', path_agent())
            args = ['--source', str(root / 'source.pt'), '--output', str(root / 'run'),
                    '--device', 'cpu', '--steps', '2', '--evaluate-every', '4',
                    '--checkpoint-every', '2', '--evaluation-episodes', '1',
                    '--representation-learning-rate', '3e-5']
            main([*args, '--actions', '4'])
            filename = root / 'run/latest.pt'
            saved = checkpoint.load(filename)
            saved['training_progress'].pop('encoder_learning_rate_after_predictor')
            checkpoint.write(filename, saved)
            report = json.loads((root / 'run/report.json').read_text())
            report['config'].pop('encoder_learning_rate_after_predictor')
            (root / 'run/report.json').write_text(json.dumps(report))
            main([*args, '--actions', '6', '--resume'])
            report = json.loads((root / 'run/report.json').read_text())
            self.assertEqual(report['progress']['representation_learning_rates'], dict(encoder=3e-5, decoder=3e-5))
            saved = checkpoint.load(filename)
            saved['training_progress']['curriculum']['levels'][0].update(
                predictor=True, representation_passes=3, prediction_passes=2, promotion_passes=1)
            checkpoint.write(filename, saved)
            old = agent_checkpoint.load(filename)
            expected = fingerprint(old.engine.model), fingerprint(old.path_evaluator)
            snapshots = []
            save = agent_checkpoint.save
            def capture(path, agent, **kwargs):
                p = kwargs['training_progress']
                if p['actions'] == 6:
                    self.assertEqual((fingerprint(agent.engine.model), fingerprint(agent.path_evaluator)), expected)
                    self.assertEqual((p['representation_updates'], p['updates']), (6, 0))
                    self.assertEqual(p['encoder_learning_rate_after_predictor'], 3e-6)
                    self.assertTrue(p['curriculum']['levels'][0]['predictor'])
                    self.assertEqual(p['curriculum']['levels'][0]['prediction_passes'], 0)
                    snapshots.append(p)
                return save(path, agent, **kwargs)
            with patch('model.subgoal_training.agent_checkpoint.save', side_effect=capture):
                main([*args, '--actions', '8', '--resume', '--encoder-learning-rate-after-predictor', '3e-6'])
            self.assertEqual(len(snapshots), 1)
            report = json.loads((root / 'run/report.json').read_text())
            self.assertEqual(report['progress']['representation_learning_rates'], dict(encoder=3e-6, decoder=3e-5))
            self.assertEqual(report['resumed_from']['encoder_learning_rate_after_predictor'], dict(old=None, new=3e-6))
            report['config']['encoder_learning_rate_after_predictor'] = None
            (root / 'run/report.json').write_text(json.dumps(report))
            main([*args, '--actions', '10', '--resume'])
            report = json.loads((root / 'run/report.json').read_text())
            self.assertEqual(report['config']['encoder_learning_rate_after_predictor'], 3e-6)
            self.assertEqual(report['progress']['representation_learning_rates'], dict(encoder=3e-6, decoder=3e-5))
            self.assertNotIn('encoder_learning_rate_after_predictor', report['resumed_from'])
            self.assertEqual((report['progress']['representation_updates'], report['progress']['predictor_updates']), (10, 4))

    def test_encoder_rate_rejects_nonpositive_nonfinite_or_increased_values(self):
        for rate in ('0', '-1', '0.000031', 'nan', 'inf'):
            with self.subTest(rate=rate), redirect_stdout(io.StringIO()), \
                 patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
                main(['--init-config', 'unused.json', '--representation-learning-rate', '3e-5',
                      '--encoder-learning-rate-after-predictor', rate])

    def test_resume_freezes_only_encoder_inherits_mode_and_can_unfreeze(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            agent_checkpoint.save(root / 'source.pt', path_agent())
            args = ['--source', str(root / 'source.pt'), '--output', str(root / 'run'),
                    '--device', 'cpu', '--steps', '2', '--evaluate-every', '4',
                    '--checkpoint-every', '2', '--evaluation-episodes', '1']
            main([*args, '--actions', '4'])
            filename = root / 'run/latest.pt'
            saved = checkpoint.load(filename)
            # Legacy progress has no freeze setting. Its default remains trainable.
            saved['training_progress'].pop('freeze_encoder')
            saved['training_progress']['curriculum']['levels'][0].update(
                predictor=True, representation_passes=3, prediction_passes=2, promotion_passes=1)
            checkpoint.write(filename, saved)
            old = agent_checkpoint.load(filename)
            initial = {name: fingerprint(getattr(old.engine.model, name))
                       for name in ('encoder', 'decoder', 'dynamics')}
            full = fingerprint(old.engine.model), fingerprint(old.path_evaluator)
            snapshots = []
            save = agent_checkpoint.save

            def capture(path, agent, **kwargs):
                progress = kwargs['training_progress']
                if progress['actions'] == 4:
                    self.assertEqual((fingerprint(agent.engine.model), fingerprint(agent.path_evaluator)), full)
                    self.assertEqual(progress['representation_updates'], 4)
                    self.assertEqual(progress['updates'], 0)
                    self.assertTrue(progress['freeze_encoder'])
                    snapshots.append(progress)
                return save(path, agent, **kwargs)

            with patch('model.subgoal_training.agent_checkpoint.save', side_effect=capture):
                main([*args, '--actions', '6', '--resume', '--freeze-encoder'])
            self.assertEqual(len(snapshots), 1)
            permissions = snapshots[0]['curriculum']['levels'][0]
            self.assertTrue(permissions['predictor'])
            self.assertFalse(permissions['planner'])
            self.assertEqual([permissions[k] for k in
                ('representation_passes', 'prediction_passes', 'promotion_passes')], [0, 0, 0])
            report = json.loads((root / 'run/report.json').read_text())
            self.assertEqual(report['resumed_from']['freeze_encoder'], dict(old=False, new=True))
            self.assertTrue(report['progress']['encoder_frozen'])
            # Omitting the flag inherits the checkpoint, even if the report disagrees.
            report['config']['freeze_encoder'] = False
            (root / 'run/report.json').write_text(json.dumps(report))
            main([*args, '--actions', '8', '--resume'])
            current = agent_checkpoint.load(filename)
            self.assertEqual(fingerprint(current.engine.model.encoder), initial['encoder'])
            for name in ('decoder', 'dynamics'):
                self.assertNotEqual(fingerprint(getattr(current.engine.model, name)), initial[name])
            report = json.loads((root / 'run/report.json').read_text())
            self.assertTrue(report['config']['freeze_encoder'])
            self.assertNotIn('freeze_encoder', report['resumed_from'])
            self.assertEqual(report['progress']['representation_updates'], 8)
            self.assertEqual(report['progress']['predictor_updates'], 4)
            self.assertTrue(checkpoint.load(filename)['training_progress']['freeze_encoder'])
            main([*args, '--actions', '10', '--resume', '--no-freeze-encoder'])
            current = agent_checkpoint.load(filename)
            self.assertNotEqual(fingerprint(current.engine.model.encoder), initial['encoder'])
            self.assertFalse(checkpoint.load(filename)['training_progress']['freeze_encoder'])
            report = json.loads((root / 'run/report.json').read_text())
            self.assertEqual(report['resumed_from']['freeze_encoder'], dict(old=True, new=False))
            self.assertFalse(report['progress']['encoder_frozen'])

    def test_resume_selects_predictive_loss_preserving_weights_counters_and_admissions(self):
        for mode in ('union', 'focal', 'recall_focal'):
            with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
                root = Path(temp)
                agent_checkpoint.save(root / 'source.pt', path_agent())
                args = ['--source', str(root / 'source.pt'), '--output', str(root / 'run'),
                        '--device', 'cpu', '--steps', '2', '--evaluate-every', '4',
                        '--checkpoint-every', '2', '--evaluation-episodes', '1']
                main([*args, '--actions', '4'])
                filename = root / 'run/latest.pt'
                saved = checkpoint.load(filename)
                # Old checkpoints predate the selector and must still load as balanced.
                saved['world']['training'].pop('prediction_delta_loss')
                saved['training_progress']['curriculum']['levels'][0].update(
                    predictor=True, representation_passes=3, prediction_passes=2, promotion_passes=1)
                checkpoint.write(filename, saved)
                old = agent_checkpoint.load(filename)
                self.assertEqual(old.engine.config.prediction_delta_loss, 'balanced')
                expected = fingerprint(old.engine.model), fingerprint(old.path_evaluator)
                snapshots = []
                save = agent_checkpoint.save

                def capture(path, agent, **kwargs):
                    progress = kwargs['training_progress']
                    if progress['actions'] == 4:
                        self.assertEqual((fingerprint(agent.engine.model), fingerprint(agent.path_evaluator)), expected)
                        self.assertEqual(agent.engine.config.prediction_delta_loss, mode)
                        self.assertEqual(progress['representation_updates'], 4)
                        self.assertEqual(progress['updates'], 0)
                        snapshots.append(progress)
                    return save(path, agent, **kwargs)

                with patch('model.subgoal_training.agent_checkpoint.save', side_effect=capture):
                    main([*args, '--actions', '6', '--resume', '--prediction-delta-loss', mode])
                self.assertEqual(len(snapshots), 1)
                permissions = snapshots[0]['curriculum']['levels'][0]
                self.assertTrue(permissions['predictor'])
                self.assertFalse(permissions['planner'])
                self.assertEqual([permissions[k] for k in
                    ('representation_passes', 'prediction_passes', 'promotion_passes')], [0, 0, 0])
                report = json.loads((root / 'run/report.json').read_text())
                self.assertEqual(report['resumed_from']['prediction_delta_loss'], dict(old='balanced', new=mode))
                self.assertEqual(report['resumed_from']['actions'], 4)
                self.assertEqual(report['progress']['representation_updates'], 6)
                self.assertEqual(report['progress']['predictor_updates'], 2)
                self.assertEqual(checkpoint.load(filename)['world']['training']['prediction_delta_loss'], mode)
                # Omitting the override on another resume inherits the persisted choice.
                main([*args, '--actions', '8', '--resume'])
                report = json.loads((root / 'run/report.json').read_text())
                self.assertEqual(report['config']['prediction_delta_loss'], mode)
                self.assertNotIn('prediction_delta_loss', report['resumed_from'])
                self.assertEqual(report['progress']['representation_updates'], 8)
                self.assertEqual(report['progress']['predictor_updates'], 4)

    def test_sigreg_stops_globally_on_predictor_admission_and_stays_off_on_resume(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            config = root / 'config.json'
            config.write_text(json.dumps({'network': asdict(path_agent().engine.model.config),
                                          'training': {'variance_weight': 0.}}))
            args = ['--init-config', str(config), '--output', str(root / 'run'), '--device', 'cpu',
                    '--steps', '2', '--evaluate-every', '2', '--evaluation-episodes', '1', '--patience', '1',
                    '--representation-learning-rate', '3e-5', '--encoder-learning-rate-after-predictor', '3e-6']
            training_phases = []
            def capture(agent, factory, level, *positional, **kwargs):
                if kwargs.get('representation') is not None:
                    training_phases.append((level, kwargs['representation'].sigreg_enabled,
                                            kwargs.get('world') is not None, kwargs['representation'].learning_rates))
                return play(agent, factory, level, *positional, **kwargs)
            # Literal quality inputs isolate the controller from convergence.
            # Initial drift is absent; the first learning episode then qualifies
            # representation only. Real learner/optimizer/checkpoint paths run.
            pending = dict(GOOD, latent_drift=None)
            ready = dict(GOOD, prediction_error=.8, rollout_error=.8)
            with patch('model.subgoal_training.play', side_effect=capture), \
                 patch('games.square_cross.v1.sc01.generate_scene', return_value=fixture()), \
                 patch('model.subgoal_training.summarize_quality',
                       side_effect=[(pending, []), (ready, []), (ready, []), (ready, [])]):
                main([*args, '--actions', '4'])
            self.assertEqual(training_phases, [(1, True, False, dict(encoder=3e-5, decoder=3e-5)),
                                               (1, False, True, dict(encoder=3e-6, decoder=3e-5))])
            report = json.loads((root / 'run/report.json').read_text())
            self.assertTrue(all('sigreg' in r['representation_terms'] for r in report['curves'][:2]))
            self.assertTrue(all('sigreg' not in r['representation_terms'] for r in report['curves'][2:]))
            self.assertFalse(report['progress']['sigreg_enabled'])
            saved = checkpoint.load(root / 'run/latest.pt')
            curriculum = saved['training_progress']['curriculum']
            # A later level starts representation-only, but shares the already
            # stabilized encoder. It must not restart SIGReg after a resume.
            curriculum['unlocked']['sc01-v1'] = 2
            curriculum['levels'][1]['unlocked'] = True
            checkpoint.write(root / 'run/latest.pt', saved)
            training_phases.clear()
            with patch('model.subgoal_training.play', side_effect=capture), \
                 patch('games.square_cross.v1.sc01.generate_scene', return_value=fixture()):
                main([*args, '--resume', '--actions', '6'])
            self.assertEqual(training_phases, [(2, False, False, dict(encoder=3e-6, decoder=3e-5))])
            report = json.loads((root / 'run/report.json').read_text())
            self.assertFalse(report['progress']['sigreg_enabled'])
            self.assertNotIn('sigreg', report['curves'][-1]['representation_terms'])

    def test_predictor_admission_needs_ten_percent_and_stability_three_times(self):
        for reconstruction, drift, expected in ((.099, .14, True), (.10, .15, True),
                                                 (.101, .14, False), (.099, .151, False)):
            with self.subTest(reconstruction=reconstruction, drift=drift):
                state = LevelReadiness()
                quality = dict(GOOD, reconstruction_error=reconstruction, latent_drift=drift)
                for _ in range(2):
                    state.observe(quality, None, CurriculumConfig())
                    self.assertFalse(state.predictor)
                state.observe(quality, None, CurriculumConfig())
                self.assertEqual(state.predictor, expected)
                self.assertFalse(state.planner)

    def test_activation_is_incremental_and_regression_blocks_only_promotion(self):
        state, config = LevelReadiness(), CurriculumConfig(patience=2)
        poor_world = dict(GOOD, prediction_error=.8, rollout_error=.9)
        state.observe(poor_world, None, config)
        self.assertFalse(state.predictor)
        state.observe(poor_world, None, config)
        self.assertTrue(state.predictor)
        self.assertFalse(state.planner)
        state.observe(GOOD, None, config)
        self.assertFalse(state.planner)
        state.observe(GOOD, None, config)
        self.assertTrue(state.planner)
        state.observe(GOOD, .9, config)
        self.assertEqual(state.promotion_passes, 1)
        state.observe(dict(GOOD, reconstruction_error=.5), .9, config)
        self.assertTrue(state.predictor)
        self.assertTrue(state.planner)
        self.assertEqual(state.promotion_passes, 0)

    def test_missing_unstable_and_one_step_only_evidence_never_opens_planner(self):
        for changed in ({'samples': 0}, {'latent_drift': None}, {'latent_drift': .2},
                        {'rollout_samples': 0}, {'rollout_error': .5}, {'rollout_changed': 0},
                        {'prediction_error': float('nan')}):
            with self.subTest(changed=changed):
                state = LevelReadiness()
                for _ in range(5):
                    state.observe(dict(GOOD, **changed), 1., CurriculumConfig())
                self.assertFalse(state.planner)
                self.assertEqual(state.promotion_passes, 0)

    def test_curriculum_retains_old_levels_and_requires_each_game_level(self):
        c = Curriculum({'a': 3, 'b': 2}, CurriculumConfig(patience=1))
        self.assertEqual(c.pairs, [('a', 1), ('b', 1)])
        self.assertEqual([c.next_pair() for _ in range(4)], [('a', 1), ('b', 1)] * 2)
        c.readiness['a', 1].observe(GOOD, 1., c.config)
        self.assertEqual(c.advance(), [('a', 2)])
        self.assertEqual(c.pairs, [('a', 1), ('a', 2), ('b', 1)])
        self.assertEqual([c.next_pair() for _ in range(3)], c.pairs)
        c.readiness['a', 2].observe(GOOD, 1., c.config)
        c.readiness['a', 1].observe(GOOD, .2, c.config)
        self.assertEqual(c.advance(), [])  # Regression in the previous level.
        c.readiness['a', 1].observe(GOOD, 1., c.config)
        self.assertEqual(c.advance(), [('a', 3)])
        self.assertEqual(c.unlocked['b'], 1)

    def test_event_only_games_need_real_completion_not_an_assumed_change(self):
        state, config = LevelReadiness(), CurriculumConfig(patience=1)
        no_pixels = dict(GOOD, changed=0, rollout_changed=0, prediction_error=0., rollout_error=0.)
        state.observe(no_pixels, 0., config)
        self.assertFalse(state.planner)
        state.observe(dict(no_pixels,completed_episodes=2),0.,config)
        self.assertTrue(state.planner)  # Observed terminal events, same pixels.
        self.assertEqual(state.promotion_passes,0)  # Still needs planner success.

    def test_quality_uses_three_predicted_successors_and_keeps_weights_read_only(self):
        # z' = z+h+1, h'=h+1: starting (1,0) gives z=2,4,7.
        # Real intermediate latents are deliberately different (3 and 5).
        engine = SimpleNamespace(autocast=nullcontext, model=SimpleNamespace(config=SimpleNamespace(patch=1)),
            actions=lambda actions: torch.tensor([[a.id] for a in actions]),
            predict=lambda z,h,a,*shape: (z+h[:,:,None]+1,h+1,torch.zeros_like(z)),
            decode=lambda z,*shape: torch.nn.functional.one_hot(z.long().reshape(-1,1,1),16).float())
        states = [State(torch.tensor([[float(z)]]),torch.tensor([[float(h)]]),(1,1))
                  for z,h in ((1,0),(3,1),(5,2),(7,3))]
        observations = [replace(obs(z), frames=np.full((1,1,1),z,dtype=np.uint8)) for z in (1,3,5,7)]
        probe = QualityProbe(engine, 3, states[0])
        for i in range(1,4):
            probe.observe(Episode(observations[:i+1],[Action(1)]*i),states[i-1],states[i],
                          dict(wrong=1,union=1,changed=1))
        self.assertEqual(probe.counts['rollout'],dict(wrong=0,union=1,samples=1,changed=1))
        quality, anchors = summarize_quality([{'quality':probe.result()}])
        self.assertIsNone(quality['latent_drift'])
        self.assertEqual(summarize_quality([{'quality':probe.result()}],anchors)[0]['latent_drift'],0.)
        self.assertEqual(quality['latent']['sample_count'],4)
        self.assertAlmostEqual(quality['latent']['std_mean'],5 ** .5)
        self.assertAlmostEqual(quality['latent']['effective_rank'],1.)
        self.assertEqual(quality['latent']['changed_rms'],2.)
        self.assertIsNone(quality['latent']['unchanged_rms'])

    def test_all_admitted_learners_continue_and_refresh_episode_and_goal(self):
        agent = path_agent()
        e = agent.engine
        e.config = replace(e.config, freeze_representation=False)
        representation, predictor = OnlineRepresentation(e), OnlinePredictor(e,None,horizon=2)
        planner = PathLearner(agent.planner)
        factory = Factory('sc01-v1',Path('games'))
        original = {n:fingerprint(getattr(e.model,n)) for n in ('encoder','decoder','dynamics')}
        value = fingerprint(agent.path_evaluator)
        samples = []
        expected_reconstructions = []
        predict = e.prediction
        def predict_and_capture(state, action, reference):
            with torch.no_grad(), e.autocast():
                expected_reconstructions.append(e.decode(state.z[None], *reference.shape).argmax(-1)[0].cpu().numpy())
            return predict(state, action, reference)
        def sample(before, actual, predicted, **meta):
            # The first preview must still use the original weights, even when
            # all three learners are updating in this same episode.
            if not samples:
                for name in original:
                    self.assertEqual(fingerprint(getattr(e.model, name)), original[name])
            np.testing.assert_array_equal(meta['reconstruction'],expected_reconstructions[-1])
            samples.append((before.copy(), actual.copy(), predicted.copy(), meta))
        # All three receive data in the SAME episode; no mutually-exclusive phase.
        with patch.object(e,'prediction',side_effect=predict_and_capture):
            result = play(agent,factory,1,123,2,representation=representation,
                          world=predictor,learner=planner,random_policy=True,sample=sample)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0][3]['episode_action'], 1)
        self.assertEqual(samples[0][3]['game'], 'sc01-v1')
        np.testing.assert_array_equal(samples[0][1], samples[1][0])
        self.assertEqual((representation.updates,predictor.updates),(2,2))
        self.assertGreater(planner.updates,0)
        for name in original:
            self.assertNotEqual(fingerprint(getattr(e.model,name)),original[name])
        self.assertNotEqual(fingerprint(agent.path_evaluator),value)
        self.assertTrue(all(r['representation_loss'] and r['predictor_loss'] for r in result['records']))
        self.assertIsNotNone(result['records'][-1]['predictor_loss']['rollout'])
        goal = agent.planner.goal
        with torch.no_grad(), e.autocast():
            encoded = e.encode([replace(obs(),frames=goal['frame'][None])])[0]
        torch.testing.assert_close(goal['z'],encoded)
        weights = fingerprint(e.model)
        with patch.object(e,'prediction',side_effect=predict_and_capture):
            play(agent,factory,1,123,3,random_policy=True,quality_horizon=2,sample=sample)
        self.assertEqual(fingerprint(e.model),weights)
        with self.assertRaisesRegex(ValueError,'must not update'):
            play(agent,factory,1,123,2,representation=representation,quality_horizon=2)

    def test_representation_alone_leaves_dynamics_and_path_unchanged_and_invalidates_cache(self):
        agent = path_agent(); e = agent.engine
        e.config = replace(e.config, freeze_representation=False)
        episode = Episode([obs(1),obs(3)],[Action(1)])
        cached = real_states(e,episode)[-1].z.clone()
        original = fingerprint(e.model.dynamics),fingerprint(agent.path_evaluator)
        OnlineRepresentation(e).update(episode.observations)
        self.assertEqual((fingerprint(e.model.dynamics),fingerprint(agent.path_evaluator)),original)
        refreshed = real_states(e,episode)[-1].z
        self.assertFalse(torch.equal(cached,refreshed))
        with torch.no_grad():
            expected = e.encode(episode.observations)[-1]
        torch.testing.assert_close(refreshed,expected)

    def test_three_real_subgoals_keep_one_episode_and_finish_on_last_action(self):
        agent = path_agent()
        scene = fixture(target=(40,30),obstacle=(26,27,3,7),distance=8)
        route = [Action(n) for n in (1,1,4,4,4,4,2,2)]
        factory = Factory('sc01-v1',Path('games'))
        env = factory.make(123,4)
        # SDK loads the game in its own module namespace. Install the literal
        # geometry there, then let the real wrapper construct its observation.
        with patch.dict(factory.teacher(env).on_set_level.__globals__,
                        {'generate_scene':lambda seed,level:scene}):
            factory._isolate_synthetic_level(env.wrapper,4)
            env = Environment(env.wrapper)
        with patch.object(factory,'make',return_value=env), \
             patch('model.subgoal_training.random_action',side_effect=route), \
             patch.object(agent,'reset',wraps=agent.reset) as reset:
            result = play(agent,factory,4,123,8,random_policy=True)
        self.assertTrue(result['success'])
        self.assertEqual([s['actions'] for s in result['segments']],[2,4,2])
        self.assertTrue(all(s['success'] for s in result['segments']))
        self.assertEqual(reset.call_count,1)
        self.assertEqual(agent.actions_taken,8)

    def test_native_win_accounts_for_goals_already_satisfied_out_of_order(self):
        factory = Factory('si01-v1',Path('games'))
        env = factory.make(19,2)
        players = factory.teacher(env).players
        route = [Action(6,{'x':players[i].x,'y':players[i].y}) for i in (1,0)]
        with patch.object(factory,'make',return_value=env), \
             patch('model.subgoal_training.random_action',side_effect=route):
            result = play(path_agent(),factory,2,19,2,random_policy=True)
        self.assertTrue(result['success'])
        self.assertEqual([s['actions'] for s in result['segments']],[2,0])
        self.assertEqual([s['success'] for s in result['segments']],[True,True])

    def test_entrypoint_trains_and_evaluates_only_the_selected_game(self):
        for game, selection in (('sc01-v1', []), ('si01-v1', ['--game', 'si01-v1'])):
            with self.subTest(game=game), tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
                root = Path(temp)
                agent_checkpoint.save(root/'source.pt',path_agent())
                with patch('model.subgoal_training.Factory', wraps=Factory) as factories:
                    main(['--source',str(root/'source.pt'),'--output',str(root/'run'),
                          '--device','cpu','--actions','4','--steps','2','--evaluate-every','4',
                          '--evaluation-episodes','1', *selection])
                self.assertEqual([call.args[0] for call in factories.call_args_list], [game])
                report = json.loads((root/'run/report.json').read_text())
                self.assertEqual(report['status'],'complete')
                self.assertEqual(report['progress']['actions'],4)
                self.assertEqual(report['progress']['representation_updates'],4)
                self.assertEqual(report['progress']['predictor_updates'],0)
                self.assertEqual(report['progress']['path_updates'],0)
                self.assertEqual(report['config']['game'],game)
                self.assertEqual(report['curriculum']['unlocked'],{game:1})
                for key in ('curves', 'episodes', 'evaluations'):
                    self.assertEqual({r['game'] for r in report[key]}, {game})
                self.assertEqual(report['prediction_sample']['game'],game)
                self.assertEqual(sum(r['real_actions'] for r in report['evaluations']),4)
                self.assertTrue(all(r['quality']['latent_drift'] is None
                                    for r in report['evaluations'] if r['split']=='final' and r['quality']))
                for row in (r for r in report['evaluations'] if r['quality']):
                    diagnostics = row['quality']['latent']
                    self.assertEqual(diagnostics['sample_count'], 3)
                    self.assertEqual(diagnostics['transitions'], 2)
                    self.assertEqual(diagnostics['skipped_transitions'], 0)
                    self.assertIsNotNone(diagnostics['std_mean'])
                    self.assertGreaterEqual(diagnostics['effective_rank'], 0.)
                    self.assertLessEqual(diagnostics['effective_rank'], diagnostics['max_rank'])
                loaded = agent_checkpoint.load(root/'run/latest.pt')
                self.assertEqual(loaded.config.mode,'path_mpc')

    def test_resume_changes_evaluation_interval_without_losing_learning(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            agent_checkpoint.save(root/'source.pt',path_agent())
            args = ['--source',str(root/'source.pt'),'--output',str(root/'run'),
                    '--device','cpu','--steps','2','--evaluate-every','4',
                    '--checkpoint-every','2','--evaluation-episodes','1']
            main([*args,'--actions','4'])
            saved = checkpoint.load(root/'run/latest.pt')
            saved['training_progress']['curriculum']['levels'][0].update(
                predictor=True, planner=True, representation_passes=3,
                prediction_passes=2, promotion_passes=1)
            checkpoint.write(root/'run/latest.pt',saved)
            old = agent_checkpoint.load(root/'run/latest.pt')
            expected = fingerprint(old.engine.model), fingerprint(old.path_evaluator)
            report_path = root/'run/report.json'
            unchanged = report_path.read_bytes()
            with self.assertRaises(SystemExit), patch('sys.stderr', new=io.StringIO()) as stderr:
                main([*args,'--actions','8','--resume','--evaluate-every','2'])
            self.assertIn('Cannot change evaluate_every',stderr.getvalue())
            self.assertEqual(report_path.read_bytes(),unchanged)
            snapshots = []
            save = agent_checkpoint.save
            def capture(path, agent, **kwargs):
                progress = kwargs['training_progress']
                if progress['actions'] == 4:
                    self.assertEqual((fingerprint(agent.engine.model), fingerprint(agent.path_evaluator)),expected)
                    snapshots.append(progress)
                return save(path,agent,**kwargs)
            with patch('model.subgoal_training.agent_checkpoint.save',side_effect=capture):
                main([*args,'--actions','8','--resume','--evaluate-every','2','--update-evaluation-interval'])
            self.assertEqual(len(snapshots),1)
            permissions = snapshots[0]['curriculum']['levels'][0]
            self.assertTrue(permissions['predictor'] and permissions['planner'])
            self.assertEqual([permissions[k] for k in
                              ('representation_passes','prediction_passes','promotion_passes')],[0,0,0])
            report = json.loads(report_path.read_text())
            checks = [r for r in report['evaluations'] if r['split']=='dev' and r['policy']=='quality_probe']
            self.assertEqual([(r['actions'],r['evaluate_every']) for r in checks],[(0,4),(4,4),(6,2),(8,2)])
            self.assertIsNone(checks[-2]['quality']['latent_drift'])
            self.assertIsNotNone(checks[-1]['quality']['latent_drift'])
            self.assertEqual(report['config']['evaluate_every'],2)
            self.assertEqual(report['resumed_from']['evaluation_interval'],dict(old=4,new=2))
            self.assertEqual(report['resumed_from']['actions'],4)
            self.assertEqual(report['progress']['representation_updates'],8)
            self.assertEqual(report['progress']['predictor_updates'],4)
            self.assertGreater(report['progress']['path_updates'],0)

    def test_resume_uses_saved_weights_gates_counters_and_discards_unsaved_measurements(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            agent_checkpoint.save(root/'source.pt',path_agent())
            args = ['--source',str(root/'source.pt'),'--output',str(root/'run'),
                    '--device','cpu','--steps','2','--evaluate-every','4','--evaluation-episodes','1',
                    '--latent-drift','0.05']
            main([*args,'--actions','4','--representation-error','0.01'])
            saved = checkpoint.load(root/'run/latest.pt')
            row = saved['training_progress']['curriculum']['levels'][0]
            row.update(predictor=True, planner=True, representation_passes=3, prediction_passes=3)
            recovery = dict(training_episodes=10, training_actions=25, path_updates=40)
            saved['training_progress']['path_recovery'] = recovery
            checkpoint.write(root/'run/latest.pt',saved)
            old = agent_checkpoint.load(root/'run/latest.pt')
            expected = fingerprint(old.engine.model), fingerprint(old.path_evaluator)
            report_path = root/'run/report.json'
            report = json.loads(report_path.read_text())
            report['curves'].append({'actions':999,'predictor_loss':999})
            report['progress']['actions'] = 999
            report_path.write_text(json.dumps(report))
            unchanged = report_path.read_bytes()
            with self.assertRaises(SystemExit), patch('sys.stderr', new=io.StringIO()) as stderr:
                main([*args,'--actions','6','--resume','--representation-error','0.10'])
            self.assertIn('Cannot change representation_error',stderr.getvalue())
            self.assertEqual(report_path.read_bytes(),unchanged)
            starts = []
            resumed_permissions = []
            saved_action = 4
            save = agent_checkpoint.save
            def capture_save(path, agent, **kwargs):
                progress = kwargs['training_progress']
                if progress['actions'] == saved_action:
                    resumed_permissions.append(progress['curriculum'])
                return save(path,agent,**kwargs)
            def observe_start(agent, *positional, **kwargs):
                starts.append((fingerprint(agent.engine.model), fingerprint(agent.path_evaluator)))
                return play(agent,*positional,**kwargs)
            with patch('model.subgoal_training.play',side_effect=observe_start), \
                 patch('model.subgoal_training.agent_checkpoint.save',side_effect=capture_save):
                main([*args,'--actions','6','--resume','--representation-error','0.10',
                      '--update-representation-threshold','--path-hindsight','--path-predicted-inputs'])
            self.assertEqual(starts[0],expected)
            report = json.loads(report_path.read_text())
            self.assertEqual(report['progress']['actions'],6)
            self.assertEqual(report['path_recovery'], recovery)
            self.assertTrue(report['progress']['path_replay']['hindsight'])
            self.assertTrue(report['progress']['path_replay']['predicted_inputs'])
            self.assertEqual(report['progress']['representation_updates'],6)
            self.assertEqual(report['progress']['predictor_updates'],2)
            self.assertGreater(report['progress']['path_updates'],0)
            self.assertTrue(all(r['actions']<=6 for r in report['curves']))
            self.assertEqual(report['resumed_from']['actions'],4)
            self.assertEqual(report['resumed_from']['representation_threshold'],dict(old=.01,new=.10))
            self.assertEqual(resumed_permissions[0]['thresholds']['representation_open'],.10)
            permissions = resumed_permissions[0]['levels'][0]
            self.assertTrue(permissions['predictor'] and permissions['planner'])
            self.assertEqual([permissions[k] for k in
                              ('representation_passes','prediction_passes','promotion_passes')],[0,0,0])
            self.assertTrue(report['curriculum']['levels'][0]['planner'])
            self.assertIsNotNone(report['prediction_sample'])
            # A saved single-game run must not silently become another campaign.
            unchanged = report_path.read_bytes()
            with self.assertRaises(SystemExit), patch('sys.stderr', new=io.StringIO()) as stderr:
                main([*args,'--actions','8','--resume','--game','si01-v1'])
            self.assertIn('Cannot change game',stderr.getvalue())
            self.assertEqual(report_path.read_bytes(),unchanged)
            with self.assertRaises(SystemExit), patch('sys.stderr', new=io.StringIO()) as stderr:
                main([*args,'--actions','8','--resume','--latent-drift','0.20',
                      '--update-representation-threshold'])
            self.assertIn('Cannot change latent_drift',stderr.getvalue())
            self.assertEqual(report_path.read_bytes(),unchanged)
            # Relax drift explicitly without resetting learned modules or permissions.
            old = agent_checkpoint.load(root/'run/latest.pt')
            expected = fingerprint(old.engine.model), fingerprint(old.path_evaluator)
            starts.clear()
            resumed_permissions.clear()
            saved_action = 6
            with patch('model.subgoal_training.play',side_effect=observe_start), \
                 patch('model.subgoal_training.agent_checkpoint.save',side_effect=capture_save):
                main([*args,'--actions','8','--resume','--representation-error','0.10',
                      '--latent-drift','0.15','--update-latent-drift-threshold'])
            self.assertEqual(starts[0],expected)
            report = json.loads(report_path.read_text())
            self.assertEqual(report['progress']['actions'],8)
            self.assertEqual(report['progress']['representation_updates'],8)
            self.assertEqual(report['progress']['predictor_updates'],4)
            self.assertEqual(report['resumed_from']['latent_drift_threshold'],dict(old=.05,new=.15))
            self.assertEqual(resumed_permissions[0]['thresholds']['latent_drift'],.15)
            permissions = resumed_permissions[0]['levels'][0]
            self.assertTrue(permissions['predictor'] and permissions['planner'])
            self.assertEqual([permissions[k] for k in
                              ('representation_passes','prediction_passes','promotion_passes')],[0,0,0])
            saved = checkpoint.load(root/'run/latest.pt')
            self.assertEqual(saved['training_progress']['path_learning'], dict(hindsight=True, predicted_inputs=True))
            self.assertEqual(saved['training_progress']['path_recovery'], recovery)
            self.assertNotIn('path_learning', report['resumed_from'])
            self.assertEqual(saved['training_progress']['curriculum']['thresholds'],report['config']['curriculum'])
            unchanged = report_path.read_bytes()
            with self.assertRaises(SystemExit), patch('sys.stderr', new=io.StringIO()) as stderr:
                main([*args,'--actions','10','--resume','--representation-error','0.20',
                      '--latent-drift','0.15','--update-latent-drift-threshold'])
            self.assertIn('Cannot change representation_error',stderr.getvalue())
            self.assertEqual(report_path.read_bytes(),unchanged)

    def test_scratch_initialization_never_loads_weights_and_only_trains_representation(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            config = root/'config.json'
            config.write_text(json.dumps({'network':asdict(path_agent().engine.model.config)}))
            initial = {}
            save = agent_checkpoint.save
            def capture(path, agent, **kwargs):
                progress = kwargs['training_progress']
                if progress['actions'] == 0:
                    initial.update({n:fingerprint(getattr(agent.engine.model,n))
                                    for n in ('encoder','decoder','dynamics')})
                    initial['path'] = fingerprint(agent.path_evaluator)
                    self.assertEqual(progress['representation_updates'],0)
                    self.assertEqual(progress['updates'],0)
                    self.assertEqual(progress['path_updates'],0)
                    self.assertEqual(progress['episodes'],0)
                    self.assertTrue(all(not r['predictor'] and not r['planner']
                                        for r in progress['curriculum']['levels']))
                return save(path,agent,**kwargs)
            args = ['--init-config',str(config),'--output',str(root/'run'),
                    '--device','cpu','--steps','2','--evaluate-every','4','--evaluation-episodes','1']
            with patch('model.subgoal_training.agent_checkpoint.load',side_effect=AssertionError('No pretrained weights')), \
                 patch('model.subgoal_training.agent_checkpoint.save',side_effect=capture):
                main([*args,'--actions','2'])
            self.assertTrue(initial,'the zero-action checkpoint is published before any evaluation/learning')
            loaded = agent_checkpoint.load(root/'run/latest.pt')
            for name in ('encoder','decoder'):
                self.assertNotEqual(fingerprint(getattr(loaded.engine.model,name)),initial[name])
            self.assertEqual(fingerprint(loaded.engine.model.dynamics),initial['dynamics'])
            self.assertEqual(fingerprint(loaded.path_evaluator),initial['path'])
            # Resume still restores the saved modules even for a randomly initialized run.
            main([*args,'--actions','4','--resume'])
            report = json.loads((root/'run/report.json').read_text())
            self.assertEqual(report['config']['initialization'],'random')
            self.assertEqual(report['resumed_from']['actions'],2)
            self.assertEqual(report['progress']['representation_updates'],4)
            self.assertEqual(report['progress']['predictor_updates'],0)
            self.assertEqual(report['progress']['path_updates'],0)


if __name__ == '__main__':
    unittest.main()
