"""Incremental activation, cumulative levels and real-only online learning.

Threshold tests use literal measurements, optimizer tests compare module weights,
and rollout tests use a tiny arithmetic world whose three-step answer is 7.
"""
from contextlib import nullcontext, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.curriculum import Curriculum, CurriculumConfig, LevelReadiness
from model import agent_checkpoint, checkpoint
from model.data import Action, Episode
from model.environment import Environment, Factory
from model.learning import State
from model.online_learning import OnlineRepresentation, OnlinePredictor
from model.path_planning import PathLearner, real_states
from model.planning import fingerprint
from model.subgoal_training import main, play, QualityProbe, summarize_quality
from model.tests.test_subgoals import path_agent
from model.tests.test_agent import obs
from tests.square_cross_support import fixture


GOOD = dict(reconstruction_error=.005, latent_drift=.01, prediction_error=.05,
            rollout_error=.08, samples=20, rollout_samples=10, changed=10, rollout_changed=5)


class CurriculumTests(unittest.TestCase):
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
        engine = SimpleNamespace(autocast=nullcontext,
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
        def sample(before, actual, predicted, **meta):
            # The first preview must still use the original weights, even when
            # all three learners are updating in this same episode.
            if not samples:
                for name in original:
                    self.assertEqual(fingerprint(getattr(e.model, name)), original[name])
            samples.append((before.copy(), actual.copy(), predicted.copy(), meta))
        # All three receive data in the SAME episode; no mutually-exclusive phase.
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
        play(agent,factory,1,123,3,random_policy=True,quality_horizon=2)
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

    def test_entrypoint_trains_representation_while_unqualified_dependents_wait(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            agent_checkpoint.save(root/'source.pt',path_agent())
            main(['--source',str(root/'source.pt'),'--output',str(root/'run'),
                  '--device','cpu','--actions','4','--steps','2','--evaluate-every','4',
                  '--evaluation-episodes','1'])
            report = json.loads((root/'run/report.json').read_text())
            self.assertEqual(report['status'],'complete')
            self.assertEqual(report['progress']['actions'],4)
            self.assertEqual(report['progress']['representation_updates'],4)
            self.assertEqual(report['progress']['predictor_updates'],0)
            self.assertEqual(report['progress']['path_updates'],0)
            self.assertEqual(report['curriculum']['unlocked'],{'sc01-v1':1,'si01-v1':1})
            self.assertEqual(sum(r['real_actions'] for r in report['evaluations']),12)
            self.assertTrue(all(r['quality']['latent_drift'] is None
                                for r in report['evaluations'] if r['split']=='final'))
            loaded = agent_checkpoint.load(root/'run/latest.pt')
            self.assertEqual(loaded.config.mode,'path_mpc')

    def test_resume_uses_saved_weights_gates_counters_and_discards_unsaved_measurements(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            agent_checkpoint.save(root/'source.pt',path_agent())
            args = ['--source',str(root/'source.pt'),'--output',str(root/'run'),
                    '--device','cpu','--steps','2','--evaluate-every','4','--evaluation-episodes','1']
            main([*args,'--actions','4'])
            saved = checkpoint.load(root/'run/latest.pt')
            row = saved['training_progress']['curriculum']['levels'][0]
            row.update(predictor=True, planner=True, representation_passes=3, prediction_passes=3)
            checkpoint.write(root/'run/latest.pt',saved)
            old = agent_checkpoint.load(root/'run/latest.pt')
            expected = fingerprint(old.engine.model), fingerprint(old.path_evaluator)
            report_path = root/'run/report.json'
            report = json.loads(report_path.read_text())
            report['curves'].append({'actions':999,'predictor_loss':999})
            report['progress']['actions'] = 999
            report_path.write_text(json.dumps(report))
            starts = []
            def observe_start(agent, *positional, **kwargs):
                starts.append((fingerprint(agent.engine.model), fingerprint(agent.path_evaluator)))
                return play(agent,*positional,**kwargs)
            with patch('model.subgoal_training.play',side_effect=observe_start):
                main([*args,'--actions','6','--resume'])
            self.assertEqual(starts[0],expected)
            report = json.loads(report_path.read_text())
            self.assertEqual(report['progress']['actions'],6)
            self.assertEqual(report['progress']['representation_updates'],6)
            self.assertEqual(report['progress']['predictor_updates'],2)
            self.assertGreater(report['progress']['path_updates'],0)
            self.assertTrue(all(r['actions']<=6 for r in report['curves']))
            self.assertEqual(report['resumed_from']['actions'],4)
            self.assertTrue(report['curriculum']['levels'][0]['planner'])
            self.assertIsNotNone(report['prediction_sample'])


if __name__ == '__main__':
    unittest.main()
