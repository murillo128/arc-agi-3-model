"""Synthetic teacher, real-transition learning and portable goal control contracts."""
from dataclasses import replace
from contextlib import nullcontext
import json
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
from model import agent_checkpoint
from model.agent import Agent, AgentConfig
from model.data import Action, Episode
from model.dashboard import subgoal_snapshot, make_server
from model.environment import Factory
from model.learning import State
from model.path_planning import PathConfig, PathPlanner, PathValue, PathLearner, segment_cost
from model.planning import fingerprint
from model.subgoal_training import OnlinePredictor, play
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
    def test_two_step_search_can_choose_an_initially_worse_action_without_pixels(self):
        # Literal tiny world: ACTION1 goes to -1, then reaches goal 2;
        # ACTION2 goes to a permanent trap at 1. Greedy predicted cost picks 2,
        # but two-step search finds the cheaper route through -1.
        class Value(nn.Module):
            def forward(self, z, h, goal, mask, rows, columns):
                return (2 * (goal - z).abs().flatten(1).mean(1)).log1p()

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

    def test_cost_targets_success_failure_and_last_action(self):
        self.assertEqual([segment_cost(n, True, 128) for n in (3, 2, 1, 0)], [3, 2, 1, 0])
        self.assertEqual([segment_cost(n, False, 128) for n in (3, 2, 1)], [131, 130, 129])
        self.assertEqual(segment_cost(1, True, 1), 1)

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
                        self.assertEqual(set(state), {'subgoals'})
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
