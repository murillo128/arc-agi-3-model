"""Synthetic teacher, real-transition learning and portable goal control contracts."""
from dataclasses import replace
from contextlib import nullcontext, redirect_stdout
import io
import json
import math
from pathlib import Path
import random
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np
import torch
from torch import nn

from games.square_cross.v1.sc01 import Sc01
from games.square_cross_interactions.v1.si01 import Si01
from model import agent_checkpoint, checkpoint
from model.agent import Agent, AgentConfig
from model.data import Action, Episode
from model.dashboard import subgoal_snapshot, make_server
from model.environment import Factory
from model.learning import State
from model.path_planning import PathConfig, PathPlanner, PathValue, PathLearner, rank_routes
from model.planning import fingerprint
from model.subgoal_training import OnlinePredictor, play
from model.training_telemetry import PlannerTrajectory
from model.tests.test_agent import make_agent, obs
from tests.square_cross_support import act, frame, fixture, center


class TeacherTests(unittest.TestCase):
    def test_literal_route_and_goal_render_do_not_change_environment(self):
        with patch('games.square_cross.v1.sc01.generate_scene', return_value=fixture()):
            game = Sc01(2)
            game.is_last_level = lambda: True
            act(game)
            before = frame(game).copy()
            goals = game.training_subgoals()
            self.assertEqual(len(goals), 1)
            goal = goals[0]
            np.testing.assert_array_equal(frame(game), before)
            self.assertEqual(center(game), (20, 30))
            self.assertEqual(game.training_goal_distance(goal), 3)
            for remaining in (2, 1, 0):
                act(game, 4)
                self.assertEqual(game.training_goal_distance(goal), remaining)
            self.assertTrue(game.training_goal_reached(goal))
            np.testing.assert_array_equal(frame(game)[goal.mask], goal.frame[goal.mask])
            self.assertFalse(hasattr(goal, 'actions'))
            act(game, 0)
            new = game.training_subgoals()[0]
            self.assertIsNot(new, goal)
            self.assertFalse(game.training_goal_reached(new))

    def test_obstacle_requires_three_straight_segments_and_eight_actions(self):
        scene = fixture(target=(40, 30), obstacle=(26, 27, 3, 7), distance=8)
        with patch('games.square_cross.v1.sc01.generate_scene', return_value=scene):
            game = Sc01(0)
            game.is_last_level = lambda: True
            act(game)
            goals = game.training_subgoals()
            self.assertEqual([g.position for g in goals], [(20, 20), (40, 20), (40, 30)])
            for goal, commands in zip(goals, ((1, 1), (4, 4, 4, 4), (2, 2)), strict=True):
                self.assertEqual(game.training_goal_distance(goal), len(commands))
                for action in commands:
                    act(game, action)
                self.assertTrue(game.training_goal_reached(goal))

    def test_click_goals_preserve_selection_and_count_free_clicks(self):
        game = Si01(19)
        game.set_level(1)
        game.is_last_level = lambda: True
        goals = game.training_subgoals()
        budget = game.remaining
        self.assertEqual([g.selected for g in goals], [(0,), (0, 1)])
        self.assertEqual(game.training_goal_distance(goals[0]), 1)
        for i, goal in enumerate(goals):
            x, y = game.players[i].x, game.players[i].y
            act(game, 6, {'x': x, 'y': y})
            self.assertTrue(game.training_goal_reached(goal))
            self.assertEqual(game.remaining, budget)
        # Generation varies with the seed, never a fixed board or colour map.
        other = Si01(20)
        self.assertFalse(np.array_equal(other.training_subgoals()[0].frame, Si01(19).training_subgoals()[0].frame))

    def test_goal_mask_does_not_constrain_an_unrelated_selection(self):
        game = Si01(19)
        game.set_level(1)
        game.is_last_level = lambda: True
        goal = game.training_subgoals()[0]
        for index in (1, 0):
            p = game.players[index]
            act(game, 6, {'x': p.x, 'y': p.y})
        self.assertTrue(game.training_goal_reached(goal))
        # Goal 1 asks only for piece 1 to be selected; piece 2 may also be selected.
        np.testing.assert_array_equal(frame(game)[goal.mask], goal.frame[goal.mask])

    def test_pinned_sdk_teacher_is_separate_from_observations(self):
        factory = Factory('sc01-v1', Path('games'))
        env = factory.make(12, 4)
        teacher = factory.teacher(env)
        self.assertEqual(len(teacher.training_subgoals()), 3)
        self.assertFalse(hasattr(env.observation, 'subgoals'))
        factory.reserve_final_level = True
        with self.assertRaisesRegex(ValueError, 'only available'):
            factory.teacher(env)


def path_agent():
    original = make_agent(frozen=False)
    original.engine.config = replace(original.engine.config, freeze_representation=True, cache_frozen_latents=True)
    value = PathValue(original.engine.model.config.latent, original.engine.model.config.memory, width=4)
    return Agent(original.engine, original.heads, AgentConfig(mode='path_mpc', max_actions=8,
        path=PathConfig(horizon=2, beam=2, clicks=2)), frozen=False, path_evaluator=value)


class PathTests(unittest.TestCase):
    def test_tuning_metrics_keep_failures_and_common_winners(self):
        from model.path_tuning import summarize
        baseline = [dict(seed=s, success=won, actions=n) for s,won,n in
                    ((1,True,10),(2,True,20),(3,False,30),(4,False,40))]
        candidate = [dict(seed=s, success=won, actions=n) for s,won,n in
                     ((1,True,6),(2,False,40),(3,True,4),(4,False,35))]
        row = summarize(candidate, baseline)
        self.assertEqual(row, dict(episodes=4, successes=2, total_actions=85,
            successful_actions=10, mean_successful_actions=5., common_successes=1,
            common_actions=6, baseline_common_actions=10, gained=[3], lost=[2]))
        failed = [dict(seed=r['seed'],success=False,actions=30) for r in baseline]
        self.assertIsNone(summarize(failed, baseline)['mean_successful_actions'])
        with self.assertRaisesRegex(ValueError, 'identical episode seeds'):
            summarize(candidate[:-1], baseline)
        with self.assertRaisesRegex(ValueError, 'identical episode seeds'):
            summarize(candidate, baseline[:-1]+[baseline[0]])

    def test_tuning_uses_same_seeds_and_preserves_checkpoint(self):
        from model.path_tuning import main
        agent = path_agent()
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            source = root/'source.pt'
            agent_checkpoint.save(source, agent)
            original = source.read_bytes()
            with patch('model.path_tuning.play', wraps=play) as playing:
                main(['--source',str(source),'--output',str(root/'comparison'),
                      '--levels','1','--episodes','1','--margins','.02','.05',
                      '--horizons','2','1','--epsilons','.2','.1','--device','cpu'])
            self.assertEqual([c.kwargs['epsilon'] for c in playing.call_args_list], [.2,.1]*4)
            report = json.loads((root/'comparison/report.json').read_text())
            rows = report['evaluations']
            self.assertEqual([r['margin'] for r in rows], [.02,.02,.05,.05]*2)
            self.assertEqual([r['horizon'] for r in rows], [2,2,2,2,1,1,1,1])
            self.assertEqual([r['results'][0]['seed'] for r in rows], [7420001]*8)
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse((root/'comparison/latest.pt').exists())
            with self.assertRaises(SystemExit), patch('sys.stderr',new=io.StringIO()):
                main(['--source',str(source),'--output',str(root/'invalid'),'--margins','nan'])
            self.assertFalse((root/'invalid').exists())

    def test_recovery_experiment_separates_evaluation_and_preserves_world(self):
        from model.path_recovery import main
        agent = path_agent()
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            source = root/'source.pt'
            agent_checkpoint.save(source, agent)
            common = ['--source',str(source),'--device','cpu','--evaluation-episodes','1']
            main(common+['--output',str(root/'control'),'--episodes','0'])
            control = json.loads((root/'control/report.json').read_text())
            self.assertEqual(control['training'], [])
            self.assertEqual(len(control['evaluations']), 2)
            main(common+['--output',str(root/'candidate'),'--episodes','1','--evaluate-every','1',
                         '--policy','planner','--epsilon','1','--learning-rate','3e-5',
                         '--replay-capacity','2','--replay-batch','2'])
            trained = json.loads((root/'candidate/report.json').read_text())
            self.assertEqual(len(trained['training']), 1)
            self.assertEqual(len(trained['evaluations']), 3)
            train_seeds = {e['seed'] for e in trained['training']}
            dev_seeds = {e['seed'] for row in trained['evaluations'] for e in row['results']}
            self.assertFalse(train_seeds & dev_seeds)
            self.assertEqual(trained['training_actions'], sum(e['actions'] for e in trained['training']))
            self.assertEqual(trained['replay']['capacity'], 2)
            self.assertEqual(trained['replay']['batch_segments'], 2)
            result = agent_checkpoint.load(root/'candidate/latest.pt')
            self.assertEqual(fingerprint(result.engine.model), fingerprint(agent.engine.model))
            self.assertNotEqual(fingerprint(result.path_evaluator), fingerprint(agent.path_evaluator))

    def test_full_forecast_keeps_short_decision_and_labels_pruned_alternatives(self):
        # State digits record the literal action prefix. A cheap one-step
        # decision must remain selected even when its full preview costs more.
        class Value(nn.Module):
            def __init__(self, costs):
                super().__init__()
                self.costs = costs
            def forward(self, z, h, goal, mask, rows, columns, remaining_budget):
                return torch.zeros(len(z)), torch.tensor([math.log1p(self.costs.get(int(v), 50)) for v in z.flatten()])

        calls = []
        def predict(z, h, actions, height, width):
            calls.append(len(z))
            following = z * 10 + actions[:, :1, None]
            return following, h + 1, following - z

        engine = SimpleNamespace(device=torch.device('cpu'), model=nn.Identity(),
            actions=lambda actions: torch.tensor([[a.id] for a in actions]), predict=predict, autocast=nullcontext)
        engine.model.config = SimpleNamespace(patch=1)
        initial = State(torch.tensor([[0.]]), torch.tensor([[0.]]), (1, 1))
        observation = replace(obs(1), available_actions=(1, 2))
        cases = [
            # The best deepest route is 211, but 111 continues the chosen 1.
            ({1: 0, 2: 4, 11: 4, 21: 2, 111: 8, 211: 1, 212: 15}, [1, 1, 1], 'searched_continuation'),
            ({1: 0, 2: 4, 21: 2, 22: 3, 211: 3, 212: 4}, [2, 1, 1], 'searched_alternative'),
        ]
        for costs, expected_path, source in cases:
            planner = PathPlanner(engine, Value(costs), PathConfig(horizon=3, beam=2))
            planner.goal = dict(z=torch.tensor([[0.]]), mask=torch.ones(1, 1, 1), frame=np.zeros((1, 1)))
            rng_states = []
            for diagnostics in (None, {}):
                calls.clear()
                rng = random.Random(0)
                self.assertEqual(planner.choose(initial, observation, rng, diagnostics=diagnostics), Action(1))
                self.assertEqual(planner.last_plan, [Action(1)])
                self.assertEqual([a.id for a in planner.last_forecast_plan], expected_path)
                self.assertEqual([int(z.item()) for z in planner.last_forecast_states],
                                 [1, 11, 111] if source == 'searched_continuation' else [2, 21, 211])
                self.assertEqual(planner.last_forecast_source, source)
                self.assertEqual(calls, [2, 4, 4], 'the preview adds no dynamics calls')
                rng_states.append(rng.getstate())
            self.assertEqual(rng_states[0], rng_states[1])
            calls.clear()
            planner.choose(initial, observation, random.Random(0), remaining_budget=1)
            self.assertEqual(planner.last_forecast_plan, [Action(1)])
            self.assertEqual(planner.last_forecast_source, 'selected_route')
            self.assertEqual(calls, [2])
            planner.clear_paths()
            self.assertEqual(planner.last_forecast_states, [])
            self.assertIsNone(planner.last_forecast_source)
            self.assertIsNotNone(planner.goal)
            planner.reset()
            self.assertIsNone(planner.goal)

    def test_two_step_search_can_choose_an_initially_worse_action_without_pixels(self):
        # Literal tiny world: ACTION1 goes to -1, then reaches goal 2;
        # ACTION2 goes to a permanent trap at 1. Greedy predicted cost picks 2,
        # but two-step search finds the cheaper route through -1.
        class Value(nn.Module):
            def forward(self, z, h, goal, mask, rows, columns, remaining_budget):
                return torch.zeros(len(z)), (2 * (goal - z).abs().flatten(1).mean(1)).log1p()

        def predict(z, h, actions, height, width):
            following = torch.where(z == 0, torch.where(actions[:, :1, None] == 1, -1., 1.),
                torch.where((z == -1) & (actions[:, :1, None] == 1), 2., z))
            return following, h + 1, following - z

        engine = SimpleNamespace(device=torch.device('cpu'), model=nn.Identity(),
            actions=lambda actions: torch.tensor([[a.id] for a in actions]), predict=predict,
            autocast=nullcontext)
        engine.model.config = SimpleNamespace(patch=1)
        initial = State(torch.tensor([[0.]]), torch.tensor([[0.]]), (1, 1))
        choices = replace(obs(1), available_actions=(1, 2))
        chosen = []
        for horizon in (1, 2):
            planner = PathPlanner(engine, Value(), PathConfig(horizon=horizon, beam=2))
            planner.goal = dict(z=torch.tensor([[2.]]), mask=torch.ones(1, 1, 1), frame=np.zeros((1, 1)))
            chosen.append(planner.choose(initial, choices, random.Random(0)).id)
        self.assertEqual(chosen, [2, 1])
        self.assertEqual([a.id for a in planner.last_plan], [1, 1])
        self.assertEqual(initial.z.item(), 0)
        self.assertEqual(initial.h.item(), 0)

    def test_success_band_precedes_cost_and_ties_are_seeded(self):
        # A cheap but unreliable path must lose. C is within 2pp of B.
        self.assertEqual(rank_routes([.5, .91, .90], [3, 8, 6], random.Random(0), .02), [2, 1, 0])
        self.assertEqual(rank_routes([.5, .91, .90], [3, 8, 6], random.Random(0), 0), [1, 2, 0])
        # Bands anchor to the maximum: do not chain pairwise "similarity".
        self.assertEqual(rank_routes([.91, .895, .88], [8, 6, 1], random.Random(0), .02), [1, 0, 2])
        ranks = [rank_routes([.5] * 4, [3, 3 + 1e-8, 3, 3], random.Random(seed), .02) for seed in range(32)]
        self.assertEqual({r[0] for r in ranks}, {0, 1, 2, 3})
        self.assertEqual(ranks[0], rank_routes([.5] * 4, [3, 3 + 1e-8, 3, 3], random.Random(0), .02))

    def test_success_priority_in_beam_and_across_depths_with_budget(self):
        calls = []
        class Value(nn.Module):
            def forward(self, z, h, goal, mask, rows, columns, remaining_budget):
                calls.append((z.flatten().tolist(), remaining_budget.tolist()))
                # 2 wins the first beam despite costing more; 21 then wins globally.
                p = {1:.5, 2:.8, 21:.95, 22:.6}
                cost = {1:0, 2:2, 21:5, 22:0}
                return (torch.tensor([p[int(v)] for v in z.flatten()]).logit(),
                        torch.tensor([math.log1p(cost[int(v)]) for v in z.flatten()]))
        engine = SimpleNamespace(device=torch.device('cpu'), model=nn.Identity(),
            actions=lambda actions: torch.tensor([[a.id] for a in actions]), autocast=nullcontext,
            predict=lambda z,h,a,height,width: (z*10+a[:,:1,None],h,z))
        engine.model.config = SimpleNamespace(patch=1)
        planner = PathPlanner(engine, Value(), PathConfig(horizon=2, beam=1))
        planner.goal = dict(z=torch.tensor([[0.]]), mask=torch.ones(1,1,1), frame=np.zeros((1,1)))
        chosen = planner.choose(State(torch.tensor([[0.]]),torch.zeros(1,1),(1,1)),
            replace(obs(1),available_actions=(1,2)), random.Random(0), remaining_budget=2)
        self.assertEqual(chosen, Action(2))
        self.assertEqual(planner.last_plan, [Action(2),Action(1)])
        self.assertEqual(calls, [([1.,2.],[1,1]),([21.,22.],[0,0])])
        self.assertAlmostEqual(planner.last_estimate['success_probability'], .95, places=6)
        self.assertAlmostEqual(planner.last_estimate['successful_actions'], 7, places=6)

    def test_real_outcomes_train_success_and_conditional_duration_with_episode_budget(self):
        class Value(nn.Module):
            def __init__(self):
                super().__init__()
                self.success = nn.Parameter(torch.tensor(0.))
                self.duration = nn.Parameter(torch.tensor(0.))
                self.budgets = []
            def forward(self,z,h,goal,mask,rows,columns,remaining_budget):
                self.budgets.append(remaining_budget.tolist())
                return self.success.expand(len(z)), self.duration.expand(len(z))
        episode = Episode([obs(1),obs(2),obs(3)], [Action(1),Action(2)])
        states = [State(torch.zeros(1,1),torch.zeros(1,1),(1,1)) for _ in range(3)]
        for success in (True,False):
            value = Value()
            engine = path_agent().engine
            planner = SimpleNamespace(engine=engine,evaluator=value,
                goal=dict(frame=obs(1).final, pixel_mask=np.ones((4,4),dtype=bool)))
            learner = PathLearner(planner)
            with patch('model.path_planning.real_states',return_value=states):
                loss = learner.update(episode,1,success,2,passes=1)
            expected = math.log(2) + ((.5*math.log(2)**2)/2 if success else 0)
            self.assertAlmostEqual(loss, expected, places=6)
            self.assertEqual(sorted(value.budgets[0]), [0,1])
            self.assertGreater(float(value.success.detach()) * (1 if success else -1), 0)
            if success:
                self.assertGreater(float(value.duration.detach()), 0)
            else:
                self.assertIsNone(value.duration.grad)
                self.assertEqual(float(value.duration.detach()), 0)

    def test_explicit_legacy_path_migration_preserves_world_and_features(self):
        agent = path_agent()
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp)/'legacy.pt'
            agent_checkpoint.save(filename,agent)
            saved = checkpoint.load(filename)
            value = saved['components']['path_evaluator']
            del value['config']['objective_version']
            for key in ('budget.weight','success.weight','success.bias','normalization.weight','normalization.bias'):
                del value['weights'][key]
            value['weights']['readout.2.weight'].fill_(9)
            value['weights']['readout.2.bias'].fill_(8)
            checkpoint.write(filename,saved)
            with self.assertRaisesRegex(ValueError,'explicit migrate_path_value'):
                agent_checkpoint.load(filename)
            migrated = agent_checkpoint.load(filename,migrate_path_value=True)
            self.assertEqual(fingerprint(migrated.engine.model),fingerprint(agent.engine.model))
            self.assertEqual(fingerprint(migrated.path_evaluator.spatial),fingerprint(agent.path_evaluator.spatial))
            self.assertEqual(torch.count_nonzero(migrated.path_evaluator.success.weight).item(),0)
            self.assertEqual(torch.count_nonzero(migrated.path_evaluator.readout[-1].weight).item(),0)
            agent_checkpoint.save(filename,migrated)
            reloaded=agent_checkpoint.load(filename)
            self.assertEqual(fingerprint(reloaded.path_evaluator),fingerprint(migrated.path_evaluator))

    def test_version_two_control_loads_exactly_and_upgrade_is_explicit(self):
        agent = path_agent()
        agent.path_evaluator = PathValue(agent.engine.model.config.latent,
            agent.engine.model.config.memory, width=4, objective_version=2)
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp)/'v2.pt'
            agent_checkpoint.save(filename, agent)
            control = agent_checkpoint.load(filename)
            self.assertEqual(fingerprint(control.path_evaluator), fingerprint(agent.path_evaluator))
            upgraded = agent_checkpoint.load(filename, migrate_path_value=True)
            self.assertEqual(upgraded.path_evaluator.config['objective_version'], 3)
            self.assertEqual(fingerprint(upgraded.engine.model), fingerprint(agent.engine.model))
            self.assertEqual(fingerprint(upgraded.path_evaluator.project), fingerprint(agent.path_evaluator.project))
            self.assertIsInstance(upgraded.path_evaluator.normalization, nn.LayerNorm)
            agent_checkpoint.save(filename, upgraded)
            reloaded = agent_checkpoint.load(filename)
            self.assertEqual(fingerprint(reloaded.path_evaluator), fingerprint(upgraded.path_evaluator))

    def test_goal_contrast_preserves_relative_changes_and_ignores_shared_offset(self):
        torch.manual_seed(23)
        value = PathValue(4,4,width=4,objective_version=4)
        nn.init.normal_(value.success.weight)
        nn.init.normal_(value.readout[-1].weight)
        z, goal = torch.randn(1,4,4), torch.randn(1,4,4)
        h, mask, budget = torch.zeros(1,4), torch.tensor([[[1.],[1.],[0.],[1.]]]), torch.tensor([8])
        evaluate = lambda a,b: torch.stack(value(a,h,b,mask,2,2,budget))
        original = evaluate(z,goal)
        # Shared per-channel background offsets and common scales should not
        # drown the state-goal difference. A goal-only change must remain visible.
        offset = torch.tensor([10.,-5.,4.,3.])
        torch.testing.assert_close(evaluate(z*3+offset,goal*3+offset),original,rtol=1e-5,atol=1e-5)
        other = z.clone();other[:,2]=1000
        torch.testing.assert_close(evaluate(other,goal),original,rtol=0,atol=0)
        self.assertGreater(float((evaluate(z,goal+offset)-original).abs().max()),1e-3)
        zero = torch.zeros_like(z)
        self.assertTrue(torch.isfinite(evaluate(zero,zero)).all())
        self.assertGreater(float((evaluate(zero,torch.ones_like(goal))-evaluate(zero,zero)).abs().max()),1e-3)

    def test_completed_replay_mixes_outcomes_and_owns_bounded_raw_experience(self):
        class Value(nn.Module):
            def __init__(self):
                super().__init__()
                self.logit = nn.Parameter(torch.tensor(0.))
                self.duration = nn.Parameter(torch.tensor(0.))
            def forward(self,z,h,goal,mask,rows,columns,budget):
                self.inputs = z.detach().clone(), goal.detach().clone()
                return self.logit.expand(len(z)), self.duration.expand(len(z))
        agent = path_agent()
        value = Value()
        planner = SimpleNamespace(engine=agent.engine, evaluator=value,
            goal=dict(frame=obs(1).final.copy(), pixel_mask=np.ones((4,4),dtype=bool)))
        learner = PathLearner(planner, learning_rate=0., capacity=2, batch_segments=2)
        episode = Episode([obs(1),obs(2)], [Action(1)])
        learner.update(episode,0,True,8,passes=1)
        saved_frame = learner.segments[0]['episode'].observations[0].frames.copy()
        episode.observations[0].frames.fill(7)
        planner.goal['frame'].fill(4)
        np.testing.assert_array_equal(learner.segments[0]['episode'].observations[0].frames, saved_frame)
        self.assertTrue((learner.segments[0]['frame'] == 1).all())
        # One success and one failure with equal segment weight have cancelling
        # BCE gradients at p=.5. Last-episode-only training instead pushes down.
        # A changed encoder now returns different, literal feature values.
        # Both stored observations AND their stored goal must be re-encoded.
        def changed_encoding(observations, **kwargs):
            return [torch.full((1,agent.engine.model.config.latent),100.+float(o.final[0,0]))
                    for o in observations]
        with patch.object(agent.engine, 'encode', side_effect=changed_encoding):
            learner.update(Episode([obs(3),obs(4),obs(5)], [Action(1),Action(2)]),0,False,8,passes=1)
        features, goals = value.inputs
        self.assertEqual(set(zip(features[:,0,0].tolist(),goals[:,0,0].tolist())),
                         {(101.,101.),(102.,101.),(103.,104.),(104.,104.),(105.,104.)})
        self.assertAlmostEqual(float(value.logit.grad), 0., places=6)
        self.assertAlmostEqual(learner.last_terms['success_bce'], math.log(2), places=6)
        self.assertLess(float(value.duration.grad), 0.)
        self.assertEqual(learner.status()['successes'], 1)
        self.assertTrue(all(s['episode'].device_bytes == 0 for s in learner.segments))
        learner.update(Episode([obs(6)], []),0,False,8,passes=1)
        self.assertEqual(learner.status()['segments'], 2)
        self.assertEqual(learner.status()['successes'], 0)

    def test_hindsight_uses_reached_nonfailure_frames_without_relabeling_original(self):
        class Value(nn.Module):
            def __init__(self):
                super().__init__()
                self.logit = nn.Parameter(torch.tensor(0.))
                self.duration = nn.Parameter(torch.tensor(0.))
            def forward(self,z,h,goal,mask,rows,columns,budget):
                return self.logit.expand(len(z)), self.duration.expand(len(z))
        agent = path_agent()
        value = Value()
        planner = SimpleNamespace(engine=agent.engine, evaluator=value,
            goal=dict(frame=obs(9).final.copy(), pixel_mask=np.ones((4,4),dtype=bool)))
        learner = PathLearner(planner, learning_rate=0., hindsight=True)
        # The original goal 9 failed. Frame 2 was really reached in one action;
        # GAME_OVER frame 3 must not become a successful auxiliary target.
        episode = Episode([obs(1),obs(2),replace(obs(3),state='GAME_OVER')], [Action(1),Action(2)])
        loss = learner.update(episode,0,False,8,passes=1)
        self.assertAlmostEqual(loss, math.log(2)+math.log(2)**2/4, places=6)
        self.assertAlmostEqual(float(value.logit.grad), 0., places=6)
        self.assertEqual(learner.status()['successes'], 0)
        self.assertEqual(learner.status()['hindsight_examples'], 1)
        self.assertFalse(learner.segments[0]['success'])
        self.assertTrue((learner.segments[0]['frame'] == 9).all())
        # An already reached auxiliary goal supplies no evidence of movement.
        learner = PathLearner(planner, learning_rate=0., hindsight=True)
        learner.update(Episode([obs(1),obs(1),replace(obs(1),state='GAME_OVER')],
                               [Action(1),Action(2)]),0,False,8,passes=1)
        self.assertEqual(learner.status()['hindsight_examples'], 0)

    def test_predicted_inputs_use_executed_prefixes_but_real_labels_and_no_world_gradients(self):
        class Value(nn.Module):
            def __init__(self):
                super().__init__()
                self.logit = nn.Parameter(torch.tensor(0.))
                self.batch = None
            def forward(self,z,h,goal,mask,rows,columns,budget):
                self.batch = (z.detach().clone(),budget.tolist())
                return self.logit.expand(len(z)), torch.zeros(len(z))
        agent = path_agent()
        value = Value()
        planner = SimpleNamespace(engine=agent.engine, evaluator=value,
            goal=dict(frame=obs(9).final, pixel_mask=np.ones((4,4),dtype=bool)))
        learner = PathLearner(planner,learning_rate=0.,predicted_inputs=True)
        episode = Episode([obs(1),obs(2),obs(3)],[Action(1),Action(2)])
        states = [State(torch.full((1,1),float(i*10)),torch.zeros(1,1),(4,4)) for i in range(3)]
        def predict(z,h,actions,height,width):
            self.assertFalse(torch.is_grad_enabled())
            return z+actions[:,:8].argmax(-1)[:,None,None],h,z
        with patch('model.path_planning.real_states',return_value=states), patch.object(agent.engine,'predict',side_effect=predict):
            loss = learner.update(episode,0,False,8,passes=1)
        z,budgets = value.batch
        self.assertEqual(len(z),6)
        self.assertEqual(budgets[:3],budgets[3:])
        paired = {8-budget:(float(real),float(predicted)) for real,predicted,budget in zip(z[:3],z[3:],budgets[:3])}
        self.assertEqual(paired[0],(0.,0.))
        self.assertEqual(paired[1],(10.,1.))
        self.assertIn(paired[2],((20.,12.),(20.,3.)))
        self.assertAlmostEqual(loss,math.log(2),places=6)
        self.assertAlmostEqual(float(value.logit.grad),.5,places=6)
        self.assertTrue(all(p.grad is None for p in agent.engine.model.parameters()))

    def test_gate_updates_dynamics_only_and_goal_learning_keeps_world(self):
        agent = path_agent()
        before, after = obs(1), obs(3)
        agent.observe(before)
        state = agent.state
        with torch.no_grad():
            target = agent.engine.initial(after).z
        online = OnlinePredictor(agent.engine, .1)
        original = {name: fingerprint(getattr(agent.engine.model, name)) for name in ('encoder', 'decoder', 'dynamics')}
        self.assertIsNone(online.update(state, Action(1), before.final, after, target, .1))
        self.assertEqual(online.updates, 0)
        result = online.update(state, Action(1), before.final, after, target, .8)
        self.assertTrue(np.isfinite(result['total']))
        self.assertEqual(online.updates, 1)
        self.assertNotEqual(fingerprint(agent.engine.model.dynamics), original['dynamics'])
        for name in ('encoder', 'decoder'):
            self.assertEqual(fingerprint(getattr(agent.engine.model, name)), original[name])
        agent.set_goal(after.final, np.ones_like(after.final, dtype=bool))
        episode = Episode([before, after], [Action(1)])
        weights = fingerprint(agent.engine.model)
        old_value = fingerprint(agent.path_evaluator)
        loss = PathLearner(agent.planner).update(episode, 0, True, 8)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(fingerprint(agent.engine.model), weights)
        self.assertNotEqual(fingerprint(agent.path_evaluator), old_value)

    def test_save_reload_goal_and_decision_use_same_agent(self):
        agent = path_agent()
        agent.observe(obs(1))
        agent.set_goal(obs(3).final, np.ones((4, 4), dtype=bool))
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp) / 'agent.pt'
            agent_checkpoint.save(filename, agent, include_runtime=True)
            loaded = agent_checkpoint.load(filename, resume_runtime=True)
            self.assertEqual(loaded.config.mode, 'path_mpc')
            self.assertEqual(agent.act(), loaded.act())
            self.assertEqual(fingerprint(agent.path_evaluator), fingerprint(loaded.path_evaluator))

    def test_real_sdk_cutoff_is_censored_and_frozen_evaluation_is_read_only(self):
        agent = path_agent()
        factory = Factory('sc01-v1', Path('games'))
        weights = fingerprint(agent.engine.model), fingerprint(agent.path_evaluator)
        result = play(agent, factory, 4, 22, 8, random_policy=True, action_limit=1)
        self.assertEqual(result['actions'], 1)
        self.assertEqual(result['reason'], 'interrupted')
        self.assertEqual(result['segments'], [])
        self.assertEqual((fingerprint(agent.engine.model), fingerprint(agent.path_evaluator)), weights)

    def test_executed_planner_preview_preserves_actions_and_excludes_random_evaluation(self):
        agent = path_agent()
        factory = Factory('sc01-v1', Path('games'))
        weights = fingerprint(agent.engine.model), fingerprint(agent.path_evaluator)
        captures = []
        capture = lambda episode, records, result: captures.append((episode, records, result))
        with patch.object(agent, 'act', return_value=Action(4)):
            result = play(agent, factory, 1, 22, 8, action_limit=1, trajectory=capture)
        self.assertEqual(len(captures), 1)
        episode, records, captured = captures[0]
        self.assertIs(captured, result)
        self.assertEqual(episode.actions, [Action(4)])
        self.assertEqual(records[0]['action_source'], 'planner')
        np.testing.assert_array_equal(episode.observations[-1].final, agent.observation.final)
        play(agent, factory, 1, 23, 8, action_limit=1, random_policy=True, trajectory=capture)
        self.assertEqual(len(captures), 1)
        play(agent, factory, 1, 24, 8, action_limit=1, epsilon=1., trajectory=capture)
        self.assertEqual(captures[-1][1][0]['action_source'], 'exploration')
        self.assertEqual((fingerprint(agent.engine.model), fingerprint(agent.path_evaluator)), weights)

    def test_real_sdk_imagination_capture_preserves_policy_and_weights(self):
        agent = path_agent()
        factory = Factory('sc01-v1', Path('games'))
        weights = fingerprint(agent.engine.model), fingerprint(agent.path_evaluator)
        captured = []
        play(agent, factory, 1, 22, 8, action_limit=2,
             trajectory=lambda episode, records, result: captured.append(episode))
        expected_actions = captured[0].actions
        tensor_rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'planner_trajectory.json'
            preview = PlannerTrajectory(path)
            callback = preview.begin()
            before_actions = []
            def capture_step(current, action, exploring):
                self.assertIs(current, agent)
                self.assertEqual(current.planner.last_plan[0], action)
                # The state is still the origin; env.step/observe have not run.
                before_actions.append(current.observation.final.copy())
                callback(current, action, exploring)
            play(agent, factory, 1, 22, 8, action_limit=2, trajectory_step=capture_step,
                 trajectory=lambda episode, records, result: preview.add(episode, records, result))
            data = json.loads(path.read_text())
            self.assertEqual([s['action'] for s in data['steps']],
                             [dict(id=a.id, data=dict(a.data)) for a in expected_actions])
            self.assertTrue(all(s['imagination'] for s in data['steps']))
            for original, before in zip(captured[0].observations, before_actions):
                np.testing.assert_array_equal(before, original.final)
        self.assertEqual((fingerprint(agent.engine.model), fingerprint(agent.path_evaluator)), weights)
        self.assertTrue(torch.equal(torch.get_rng_state(), tensor_rng))

    def test_dashboard_selects_only_new_report_and_redacts_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'report.json'
            file.write_text(json.dumps({'kind': 'synthetic_subgoals', 'status': 'running',
                'config': {'source': '/private/model.pt', 'init_config': '/private/config.json',
                           'output': '/private/run', 'seed': 0, 'initialization': 'random'},
                'progress': {'actions': 1}, 'curves': []}))
            result = subgoal_snapshot(file)
            self.assertEqual(result['config'], {'seed': 0, 'initialization': 'random'})
            self.assertEqual(result['progress']['actions'], 1)
            self.assertNotIn('campaign', result)
            # Actual transport and assets: old campaign bookmark resolves to
            # this run as well, and no legacy monitor is polled.
            with make_server(('127.0.0.1', 0), None, subgoal_report=file) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    base = f'http://127.0.0.1:{server.server_port}'
                    with urlopen(base + '/api/state') as response:
                        state = json.load(response)
                        self.assertEqual(set(state), {'subgoals', 'gpu'})
                        self.assertEqual(state['subgoals']['progress']['actions'], 1)
                    for route in ('/', '/campaign.html'):
                        with urlopen(base + route) as response:
                            html = response.read().decode()
                            self.assertIn('subgoals.js', html)
                            self.assertNotIn('style=', html)
                finally:
                    server.shutdown()
                    thread.join()

    def test_architecture_has_connected_ports_and_valid_module_references(self):
        document = json.loads(Path('model/architecture.json').read_text())
        nodes = {n['id']: n for n in document['nodes']}
        self.assertEqual(len(nodes), len(document['nodes']))
        ports = {(n['id'], p['id']) for n in nodes.values() for p in n['ports']}
        endpoints = {(edge[side]['node_id'], edge[side]['port_id'])
                     for edge in document['edges'] for side in ('source', 'target')}
        self.assertEqual(ports, endpoints)
        for n in nodes.values():
            if n.get('parent_id'):
                self.assertIn(n['id'], nodes[n['parent_id']]['children'])
            for child in n.get('children', []):
                self.assertEqual(nodes[child]['parent_id'], n['id'])
        for template in document['templates']:
            for instance in template['instances']:
                self.assertIn(instance['node_id'], nodes)
                for ref in instance['nodes']:
                    self.assertIn(ref['node_id'], nodes)
        for repetition in document['repetitions']:
            self.assertIn(repetition['parent_id'], nodes)
            for instance in repetition['instances']:
                self.assertIn(instance['node_id'], nodes)
        parameters = {p['id'] for p in document['parameters']}
        self.assertTrue(all(p in parameters for n in nodes.values() for p in n['parameter_ids']))


if __name__ == '__main__':
    unittest.main()
