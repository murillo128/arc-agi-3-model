"""Teacher goals train detached heads; only real outcomes label path success."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from model import checkpoint
from model.config import SequenceConfig
from model.environment import Factory
from model.game_sequences import GameSession, Transition, SequenceLearner, collect_session
from model.sequence_control import SequencePhaseControl, SequenceControlConfig
from model.sequence_planning import SequencePlanning, SequencePlanningConfig, goal_loss, path_loss
from model.sequence_training import (collect_batch, load_engine, representation_for, restore_run,
                                     run, save_run)
from model.tests.test_agent import make_agent, obs
from model.tests.test_training_config import config
from model.training_config import load_plan
from model.data import Action
from model.optimization_telemetry import OptimizationTelemetry
from model.optimization_control import OptimizationControl


class SequencePlanningTests(unittest.TestCase):
    def test_configuration_requires_teacher_scope_and_bounded_exploration(self):
        with tempfile.TemporaryDirectory() as directory:
            path, values = config(Path(directory), sequence={}, sequence_planning={})
            with self.assertRaisesRegex(ValueError, 'controlled sequences'):
                load_plan(path)
            values['sequence_control'] = {}
            path.write_text(json.dumps(values))
            self.assertEqual(load_plan(path).payload['sequence_planning']['horizon'], 2)
            for invalid in ({'unknown': 1}, {'exploration': .1}, {'learned_fraction': 1.},
                            {'learning_rate': 0}, {'samples': 1, 'minibatch': 1}):
                values['sequence_planning'] = invalid
                path.write_text(json.dumps(values))
                with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                    load_plan(path)
            values['sequence_planning'] = {}
            values['curriculum'] = [dict(source='synthetic', game='bf01-v1', levels=[1])]
            path.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, 'teacher games'):
                load_plan(path)

    def heads(self, **options):
        return SequencePlanning(make_agent().engine, SequencePlanningConfig(
            **(dict(width=8, horizon=1, beam=2, samples=4, minibatch=2, passes=1) | options)))

    def collection(self, heads, *, learned=False):
        goal = SimpleNamespace(frame=np.ones((4, 4), np.uint8), mask=np.ones((4, 4), bool))
        teacher = SimpleNamespace(training_subgoals=lambda: [goal], training_goal_reached=lambda _: False)
        factory = SimpleNamespace(game_id='sc01-v1', reserve_final_level=False, teacher=lambda _: teacher)
        session = GameSession('sc01-v1', 1)
        collector = heads.collector(factory, session)
        collector.start(SimpleNamespace())
        heads.path_updates = heads.config.warmup_updates
        if learned:
            heads.goal_updates = heads.config.warmup_updates + heads.config.ramp_updates
        collector.planner.choose = Mock(return_value=Action(1))
        return collector, session, SimpleNamespace(random=lambda: .4)

    def step(self, collector, session, rng, before, after, remaining=8):
        action = collector.action(before, remaining, rng, lambda: Action(2))
        session.transitions.append(Transition(before, action, after))
        collector.observe(before, action, after)

    def test_sparse_goal_pixel_has_equal_region_weight_and_correct_gradient(self):
        before = torch.zeros(1, 8, 8, dtype=torch.long)
        target = before.clone(); target[0, 0, 0] = 1
        mask = torch.ones_like(before, dtype=torch.bool)
        logits = torch.zeros(1, 8, 8, 16, requires_grad=True)
        masks = torch.zeros(1, 8, 8, requires_grad=True)
        loss = goal_loss(logits, masks, before, target, mask)
        loss.backward()
        # One changed pixel receives half of the color objective, exactly as
        # all 63 background pixels combined. This cannot collapse to copying.
        self.assertAlmostEqual(float(logits.grad[0, 0, 0].abs().sum()),
                               float(logits.grad.abs().sum())/2, places=6)
        self.assertLess(float(logits.grad[0, 0, 0, 1]), 0.)
        self.assertGreater(float(logits.grad[0, 0, 0, 0]), 0.)
        self.assertTrue(torch.isfinite(masks.grad).all())
        sparse = torch.zeros_like(mask); sparse[0, 0, 0] = True
        masks.grad = None
        goal_loss(logits, masks, before, target, sparse).backward()
        self.assertAlmostEqual(float(masks.grad[0, 0, 0].abs()),
                               float(masks.grad.abs().sum())/2, places=6)

    def test_failed_attempt_never_teaches_short_duration(self):
        logits = torch.zeros(2, requires_grad=True)
        cost = torch.tensor([2., 2.], requires_grad=True)
        path_loss(logits, cost, torch.tensor([1., 0.]), torch.tensor([0., 0.])).backward()
        self.assertLess(float(logits.grad[0]), 0.)
        self.assertGreater(float(logits.grad[1]), 0.)
        self.assertGreater(float(cost.grad[0]), 0.)
        self.assertEqual(float(cost.grad[1]), 0.)

    def test_goal_lifetime_decreases_budget_and_censoring_is_not_failure(self):
        heads = self.heads(lifetime=2)
        collector, session, rng = self.collection(heads)
        before = obs(0)
        self.step(collector, session, rng, before, before)
        active = collector.active
        self.step(collector, session, rng, before, before)
        self.assertEqual([c.kwargs['remaining_budget'] for c in collector.planner.choose.call_args_list], [2, 1])
        self.assertEqual(session.planning['segments'][0]['success'], False)
        self.assertEqual(session.planning['segments'][0]['frame'].tolist(), active['frame'].tolist())
        self.step(collector, session, rng, before, before)
        collector.finish(None)
        self.assertIsNone(session.planning['segments'][-1]['success'])
        metrics, _ = heads.fit([session])
        self.assertEqual(metrics['teacher_attempts'], 1)
        self.assertEqual(metrics['censored'], 1)
        self.assertEqual(metrics['path_examples'], 3, 'Only the completed two-action segment is supervised')

    def test_learned_pixel_match_is_not_a_task_win_and_goal_stays_fixed(self):
        heads = self.heads()
        collector, session, rng = self.collection(heads, learned=True)
        heads.propose = Mock(return_value=(np.full((4, 4), 2, np.uint8), np.ones((4, 4), bool)))
        self.step(collector, session, rng, obs(0), obs(0))
        self.step(collector, session, rng, obs(0), obs(2))
        heads.propose.assert_called_once()
        counts = session.planning['counts']
        self.assertEqual(counts['learned_reached'], 1)
        self.assertEqual(counts['learned_task_progress'], 0)
        self.assertEqual(counts.get('level_successes', 0), 0)
        self.assertEqual(session.planning['segments'][0]['source'], 'learned')

    def test_empty_or_already_satisfied_proposal_falls_back_and_new_level_keeps_memory(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                heads = self.heads()
                collector, session, rng = self.collection(heads, learned=True)
                heads.propose = Mock(return_value=(np.zeros((4, 4), np.uint8), np.full((4, 4), not empty, bool)))
                self.step(collector, session, rng, obs(0), obs(0))
                self.assertEqual(collector.active['source'], 'teacher')
                self.assertEqual(session.planning['counts']['fallbacks'], 1)
                memory = collector.state.h.clone()
                collector.start(SimpleNamespace())
                expected = heads.engine.initial(obs(3), memory=memory).h
                collector.action(obs(3), 4, rng, lambda: Action(2))
                torch.testing.assert_close(collector.state.h, expected, rtol=0, atol=0)
                self.assertEqual(session.planning['counts']['censored'], 1)

    def test_real_collection_learns_both_heads_without_world_gradients_or_weight_changes(self):
        heads = self.heads()
        factory = Factory('sc01-v1', Path(__file__).parents[2]/'games')
        try:
            sessions = [collect_session(factory, [1, 2], 123,
                SequenceConfig(batch_games=1, session_steps=8, level_steps=4, block_steps=4), planning=heads)]
        finally:
            factory.arcade.close_scorecard()
        self.assertEqual(len(sessions[0].transitions), 8)
        self.assertEqual(sum(t.reset_before for t in sessions[0].transitions), 1)
        for step in sessions[0].transitions:
            self.assertIn(step.action.id, step.before.available_actions)
        engine = heads.engine
        engine.model.requires_grad_(True)
        engine.model.zero_grad(set_to_none=True)
        before = deepcopy(engine.model.state_dict())
        goal_before, path_before = deepcopy(heads.goal.state_dict()), deepcopy(heads.path.state_dict())
        observed = deepcopy(heads)
        rng_before = torch.get_rng_state().clone()
        measured_versions = []
        versions = tuple(p._version for p in heads.goal.parameters())
        def measured(module, inputs):
            if not torch.is_grad_enabled():
                measured_versions.append(tuple(p._version for p in module.parameters()))
        hook = heads.goal.register_forward_pre_hook(measured)
        metrics, sample = heads.fit(sessions)
        rng_after = torch.get_rng_state().clone()
        hook.remove()
        self.assertEqual(measured_versions, [versions, versions],
                         'All diagnostic minibatches must use the same pre-update head')
        self.assertGreater(metrics['goal_updates'], 0)
        self.assertGreater(metrics['path_updates'], 0)
        self.assertEqual(sample['game'], 'sc01-v1')
        for name, old in before.items():
            torch.testing.assert_close(old, engine.model.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in engine.model.parameters()))
        for module, old in ((heads.goal, goal_before), (heads.path, path_before)):
            self.assertTrue(any(not torch.equal(old[k], v) for k, v in module.state_dict().items()))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None))
        self.assertIsNone(heads.collector(SimpleNamespace(game_id='sdk', reserve_final_level=True), GameSession('sdk', 0)))
        self.assertIsNone(heads.collector(SimpleNamespace(game_id='sc01-v1', reserve_final_level=True), GameSession('sc01-v1', 0)))
        with tempfile.TemporaryDirectory() as directory:
            modules = dict(goal_planner=observed.goal, path_planner=observed.path)
            telemetry = OptimizationTelemetry(observed.engine.model, directory, every=2, extra_modules=modules)
            telemetry.start_batch(1)
            telemetry.finish_batch(.25, 0)
            telemetry.start_batch(2)
            torch.set_rng_state(rng_before)
            measured, _ = observed.fit(sessions, telemetry=telemetry)
            torch.testing.assert_close(torch.get_rng_state(), rng_after, rtol=0, atol=0)
            self.assertEqual(observed.rng.getstate(), heads.rng.getstate())
            for name in ('goal', 'path'):
                for key, value in getattr(heads, name).state_dict().items():
                    torch.testing.assert_close(getattr(observed, name).state_dict()[key], value, rtol=0, atol=0)
                expected = getattr(heads, name+'_optimizer').state_dict()
                actual = getattr(observed, name+'_optimizer').state_dict()
                self.assertEqual(actual['param_groups'], expected['param_groups'])
                for key, state in expected['state'].items():
                    for field, value in state.items():
                        torch.testing.assert_close(actual['state'][key][field], value, rtol=0, atol=0)
            losses = {name+'_planner': measured[name+'_loss'] for name in ('goal', 'path')}
            telemetry.finish_batch(.2, 8, module_losses=losses)
            for batch in (3, 4):
                telemetry.start_batch(batch)
                telemetry.finish_batch(.2, 8, module_losses=dict.fromkeys(modules))
            self.assertEqual({e['phase'] for e in telemetry.events}, set(modules))
            for name in modules:
                self.assertTrue(any(e['matrices'] and all(k.startswith(name+'.') for k in e['matrices'])
                                    for e in telemetry.events if e['phase'] == name))
                self.assertGreater(telemetry.planes[name].project()[1]['step_norm'], 0)
                self.assertEqual(telemetry.points[1]['module_losses'][name], metrics[name.split('_')[0]+'_loss'])
                self.assertIsNone(telemetry.points[-1]['module_losses'][name])
            restored = OptimizationTelemetry(observed.engine.model, directory, every=2, restored_batch=4, extra_modules=modules)
            self.assertEqual(restored.points, telemetry.points)
            for name in modules:
                np.testing.assert_array_equal(restored.planes[name].samples, telemetry.planes[name].samples)
            report = Path(directory)/'report.json'
            report.write_text(json.dumps(dict(kind='game_sequences',status='running',optimization_identity=telemetry.identity,
                config=dict(batch_games=1),progress=dict(batches=4))))
            self.assertEqual(OptimizationControl(report).snapshot()['status'], 'ready')

    def test_evaluation_collection_never_receives_teacher_or_head(self):
        planning = Mock()
        fake = GameSession('game', 0, [Transition(obs(1), Action(1), obs(2))])
        with patch('model.sequence_training.collect_session', return_value=fake) as collect:
            collect_batch({'game': (object(), [1])}, SequenceConfig(batch_games=1), 0, 1,
                          planning=planning, evaluation=True)
            self.assertNotIn('planning', collect.call_args.kwargs)
            collect_batch({'game': (object(), [1])}, SequenceConfig(batch_games=1), 0, 1, planning=planning)
            self.assertIs(collect.call_args.kwargs['planning'], planning)

    def test_planner_respects_undo_exclusion_without_mutating_sdk_observation(self):
        collector, _, rng = self.collection(self.heads())
        before = obs(0, actions=(1, 6, 7))
        collector.action(before, 8, rng, lambda: Action(1), allow_undo=False)
        permitted = collector.planner.choose.call_args.args[1]
        self.assertEqual(permitted.available_actions, (1, 6))
        self.assertEqual(before.available_actions, (1, 6, 7))

    def test_explicit_adoption_preserves_every_old_state_and_resumes_new_heads(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, values = config(root, [dict(source='synthetic', game='sc01-v1', levels=[1])],
                sequence=dict(batch_games=1, session_steps=4, block_steps=2, level_steps=4, representation_updates=1),
                sequence_control=dict(diagnostic_samples=4),
                thresholds=dict(representation_open=1., latent_drift=1., patience=1, minimum_samples=1, minimum_changed=1))
            values['settings']['actions'] = 64
            path.write_text(json.dumps(values))
            args = dict(training_config=path, output=root/'run', source=None, device='cpu', compile=False, max_batches=1)
            run(SimpleNamespace(**args, resume=False))
            run(SimpleNamespace(**args, resume=True, freeze_representation=True))
            saved = checkpoint.load(root/'run/latest.pt')
            values['sequence_planning'] = dict(width=8, horizon=1, beam=2, samples=4, minibatch=2, passes=1)
            path.write_text(json.dumps(values))
            plan = load_plan(path)
            engine = load_engine(plan, root/'run/latest.pt')
            learner = SequenceLearner(engine, SequenceConfig(**plan.payload['sequence']))
            rep = representation_for(engine, plan)
            control = SequencePhaseControl(plan.stages, plan.settings, SequenceControlConfig(**plan.payload['sequence_control']))
            heads = SequencePlanning(engine, SequencePlanningConfig(**plan.payload['sequence_planning']))
            with self.assertRaises(ValueError):
                restore_run(saved, plan, learner, rep, control, planning=heads)
            self.assertEqual(restore_run(saved, plan, learner, rep, control, planning=heads, adopt_planning=True), (2, 8))
            self.assertTrue(rep.frozen)
            for key, old in saved['weights'].items():
                torch.testing.assert_close(old, engine.model.state_dict()[key], rtol=0, atol=0)
            for actual, key in ((learner.optimizer, 'predictor_optimizer'), (rep.optimizer, 'representation_optimizer')):
                self.assertEqual(actual.state_dict()['param_groups'], saved[key]['param_groups'])
                for tensor, state in saved[key]['state'].items():
                    for name, old in state.items():
                        torch.testing.assert_close(actual.state_dict()['state'][tensor][name], old, rtol=0, atol=0)
            for change in ('encoder_learning_rate_after_predictor', 'seed'):
                changed = deepcopy(values); changed['settings'][change] = 1e-8 if 'learning_rate' in change else 7
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    restore_run(saved, load_plan(path), learner, rep, control, planning=heads, adopt_planning=True)
            path.write_text(json.dumps(values))
            save_run(root/'adopted.pt', plan, learner, rep, 2, 8, control, planning=heads)
            adopted = checkpoint.load(root/'adopted.pt')
            with self.assertRaises(ValueError):
                restore_run(adopted, plan, learner, rep, control, planning=heads, adopt_planning=True)
            nonfrozen = deepcopy(saved); nonfrozen['sequence_control']['representation_frozen_at_batch'] = None
            with self.assertRaises(ValueError):
                restore_run(nonfrozen, plan, learner, rep, control, planning=heads, adopt_planning=True)
            run(SimpleNamespace(**args, resume=True, adopt_planning=True))
            trained = checkpoint.load(root/'run/latest.pt')
            self.assertGreater(trained['planning']['goal_updates'], 0)
            self.assertGreater(trained['planning']['path_updates'], 0)
            self.assertGreater(trained['progress']['updates'], saved['progress']['updates'])
            fresh = SequencePlanning(engine, heads.config)
            fresh.restore(trained['planning'])
            self.assertEqual(fresh.status(), {k: trained['planning'][k] for k in fresh.status()})
            self.assertEqual(fresh.rng.getstate(), trained['planning']['rng'])
            for module in ('goal', 'path'):
                for key, old in trained['planning'][module].items():
                    torch.testing.assert_close(getattr(fresh, module).state_dict()[key], old, rtol=0, atol=0)
                actual = getattr(fresh, module+'_optimizer').state_dict()
                expected = trained['planning'][module+'_optimizer']
                self.assertEqual(actual['param_groups'], expected['param_groups'])
                for key, state in expected['state'].items():
                    for name, old in state.items():
                        torch.testing.assert_close(actual['state'][key][name], old, rtol=0, atol=0)
            run(SimpleNamespace(**args, resume=True))
            resumed = checkpoint.load(root/'run/latest.pt')
            self.assertGreater(resumed['planning']['goal_updates'], trained['planning']['goal_updates'])
            self.assertEqual(resumed['planning']['adopted_batch'], 2)
            for key, old in saved['weights'].items():
                if key.startswith(('encoder.', 'decoder.')):
                    torch.testing.assert_close(resumed['weights'][key], old, rtol=0, atol=0)
            report = json.loads((root/'run/report.json').read_text())
            self.assertEqual(report['planning_sample']['batch'], 4)
            self.assertEqual(report['history'][-1]['planning']['identity'], '2')


if __name__ == '__main__':
    unittest.main()
