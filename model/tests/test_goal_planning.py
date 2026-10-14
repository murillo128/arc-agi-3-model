"""Independent distance oracle, lifecycle and exact diagnostic tensor contracts."""
from contextlib import nullcontext
from dataclasses import asdict, replace
import math
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io
import tempfile
from pathlib import Path

import torch
from torch import nn

from model.data import Action
from model.goal_planning import GoalConfig, GoalPlanner, distance
from model.learning import State
from model.tests.test_planning import obs


class Readout(nn.Module):
    def forward(self, z, h, action, rows, columns):
        outputs = torch.zeros(len(z), 4)
        outputs[:, 0] = 30.  # continuation; SDK V always prefers right
        return z[:, 0, 0], outputs


class Dynamics:
    def __init__(self):
        self.model = nn.Linear(1, 1)
        self.model.config = SimpleNamespace(patch=1)
        self.decoded = None
        self.memories = []

    def autocast(self):
        return nullcontext()

    def actions(self, choices):
        return torch.tensor([a.features() for a in choices])

    def predict(self, z, h, action, height, width):
        self.memories.append(h.clone())
        d = (action[:, 2] - action[:, 1])[:, None, None]
        return z + d, h + d[:, 0, :], d

    def decode(self, z, height, width):
        self.decoded = z.clone()
        colors = (z[:, 0, 0] + 4).long()
        return torch.nn.functional.one_hot(colors, 16)[:, None, None].float()


class GoalPlanningTests(unittest.TestCase):
    def setUp(self):
        self.engine = Dynamics()
        self.planner = GoalPlanner(self.engine, Readout(), GoalConfig(proposal_horizon=2, horizon=2, beam=2, lifetime=2))
        self.state = State(torch.zeros(1, 1), torch.zeros(1, 1), (1, 1))
        self.obs = replace(obs(), available_actions=(1, 2))

    def choose(self, diagnostics=None):
        return self.planner.choose(self.state, self.obs, random.Random(5), diagnostics=diagnostics)

    def test_opposing_targets_causally_change_actions_and_final_distance_rank(self):
        # Two steps reach {-2,0,+2}; distance to target -1.5 is .25, 2.25,12.25.
        # V favours +2, while the explicit distance objective must choose -2.
        for target, action in ((-1.5, 1), (1.5, 2)):
            self.planner.set_goal(torch.tensor([[target]]), self.state)
            diagnostic = {}
            self.assertEqual(self.choose(diagnostic).id, action)
            plan = diagnostic['planning']
            self.assertEqual([s['action']['id'] for s in plan['steps']], [action, action])
            self.assertAlmostEqual(plan['selected_score'], math.exp(-.25), places=6)
            self.assertEqual([s['goal_distance'] for s in plan['steps']], [.25, .25])
            self.assertEqual(plan['goal']['distance'], 2.25)
        self.assertEqual(self.state.z.item(), 0)
        self.assertEqual(self.state.h.item(), 0)
        self.assertEqual(self.engine.memories[0].flatten().tolist(), [0, 0])
        self.assertEqual(self.engine.memories[1].flatten().tolist(), [-1, -1, 1, 1])

    def test_proposal_predates_action_and_diagnostics_decode_exact_ranked_tensors(self):
        self.assertEqual(self.choose().id, 2)
        goal = self.planner.goal
        self.assertEqual(goal.z.item(), 2)
        self.assertEqual(goal.created_position, 0)
        self.assertAlmostEqual(goal.proposal_score, .95**2 / (1 + math.exp(-2)), places=6)
        self.assertIsNone(self.engine.decoded)
        trace = {}
        self.assertEqual(self.choose(trace).id, 2)
        self.assertIs(self.planner.goal, goal)
        self.assertEqual(self.engine.decoded.flatten().tolist(), [2, 1, 2])
        self.assertEqual(trace['planning']['goal']['frame']['data'], bytes([6]))
        self.assertEqual([s['frame']['data'] for s in trace['planning']['steps']], [bytes([5]), bytes([6])])
        self.assertEqual([s['goal_distance'] for s in trace['planning']['steps']], [1, 0])
        self.assertEqual(trace['planning']['selected_score'], 1)

    def test_real_action_ttl_episode_reset_sdk_boundary_and_geometry(self):
        self.choose()
        goal = self.planner.goal
        self.planner.after_real_action(Action(2), self.obs, self.obs)
        self.choose()
        self.assertIs(self.planner.goal, goal)
        self.planner.after_real_action(Action(2), self.obs, self.obs)
        self.assertIsNone(self.planner.goal)
        self.choose()
        self.assertEqual(self.planner.goal.created_position, 2)
        for after in (replace(self.obs, levels_completed=1), replace(self.obs, state='GAME_OVER')):
            self.planner.after_real_action(Action(2), self.obs, after)
            self.assertIsNone(self.planner.goal)
            self.choose()
        self.planner.after_real_action(Action(0), self.obs, self.obs)
        self.assertEqual(self.planner.position, 0)
        self.assertIsNone(self.planner.goal)
        self.choose()
        self.planner.reset()
        self.assertIsNone(self.planner.goal)
        self.choose()
        changed = State(torch.zeros(2, 1), torch.zeros(1, 1), (1, 2))
        self.planner.choose(changed, self.obs, random.Random(5))
        self.assertEqual(self.planner.goal.geometry, (1, 2))

    def test_only_final_depth_rank_chooses_branch_despite_better_earlier_distance(self):
        self.planner.set_goal(torch.tensor([[.75]]), self.state)
        diagnostic = {}
        self.assertEqual(self.choose(diagnostic).id, 2)
        plan = diagnostic['planning']
        self.assertEqual([s['action']['id'] for s in plan['steps']], [2, 1])
        self.assertEqual([s['goal_distance'] for s in plan['steps']], [.0625, .5625])
        self.assertAlmostEqual(plan['selected_score'], math.exp(-.5625), places=6)

    def test_distance_means_all_entries_in_fp32_and_ties_are_stable(self):
        z = torch.tensor([[[3., 1.], [0., 2.]]], dtype=torch.bfloat16)
        target = torch.tensor([[0., 2.], [1., 2.]], dtype=torch.bfloat16)
        # Four squared differences: 9, 1, 1, 0. Mean, not sum or pooled features.
        result = distance(z, target)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.item(), 2.75)
        self.planner = GoalPlanner(self.engine, Readout(), GoalConfig(horizon=1))
        self.planner.set_goal(torch.zeros(1, 1), self.state)
        self.assertEqual(self.choose().id, 1)  # exact tie: candidate order
        extreme = torch.full((1, 1, 1), torch.finfo(torch.float32).max)
        with self.assertRaisesRegex(ValueError, 'Nonfinite goal distance'):
            distance(extreme, -extreme[0])

    def test_cpu_cli_explicit_mode_overrides_saved_q_and_preserves_source_files(self):
        from model.behavior_trace import main
        from model import checkpoint
        from model.config import NetworkConfig, TrainConfig
        from model.networks import WorldModel
        from model.planning import GoalValue, PlanConfig, fingerprint
        from model.behavior_experiment import save_heads
        from arc3.traces import read_attempt
        torch.set_num_threads(2)
        torch.manual_seed(52)
        net = WorldModel(NetworkConfig.small())
        heads = GoalValue(16, 32)
        with tempfile.TemporaryDirectory() as directory:
            world, utility, output = (Path(directory) / name for name in ('world.pt', 'heads.pt', 'out.arc3'))
            checkpoint.write(world, {'network': asdict(net.config), 'training': asdict(TrainConfig()), 'weights': net.state_dict()})
            save_heads(utility, heads, fingerprint(net), PlanConfig(), {'selected': {'policy': 'q'}})
            before = (world.read_bytes(), utility.read_bytes())
            args = ['behavior_trace', '--checkpoint', str(world), '--heads', str(utility),
                    '--output', str(output), '--device', 'cpu', '--goal-planning',
                    '--goal-horizon', '2', '--plan-horizon', '2', '--goal-lifetime', '1',
                    '--seed', '9426301', '--level', '3', '--steps', '2']
            with patch('sys.argv', args), redirect_stdout(io.StringIO()):
                main()
            attempt = read_attempt(output)
            self.assertEqual(attempt['metadata']['config']['policy'], 'goal_mpc')
            self.assertEqual(len(attempt['steps']), 2)
            for i, step in enumerate(attempt['steps']):
                plan = step['planning']
                self.assertEqual(plan['mode'], 'goal_mpc')
                self.assertEqual(plan['goal']['created_position'], i)
                self.assertEqual(plan['horizon'], 2)
                self.assertEqual(len(plan['goal']['frame']['data']), 64 * 64)
                self.assertEqual(plan['steps'][0]['action'], step['action'])
            self.assertEqual((world.read_bytes(), utility.read_bytes()), before)

    def test_value_goal_mode_ignores_visual_reward_and_keeps_first_progress_objective(self):
        class ProgressReadout(nn.Module):
            def forward(self, z, h, a, rows, columns):
                output = torch.full((len(z), 4), -100.)
                progress = a[:, 2] > .5
                output[progress, 1] = torch.log(torch.tensor(.8))
                output[progress, 2] = torch.log(torch.tensor(.2))
                output[~progress, 0] = 0
                return torch.full((len(z),), -100.), output
        planner = GoalPlanner(self.engine, ProgressReadout(), GoalConfig(horizon=2))
        planner.set_goal(torch.tensor([[-2.]]), self.state)  # nearest path is left.
        diagnostics = {}
        result = planner.choose(self.state, self.obs, random.Random(1), 'goal_value_mpc',
                                 diagnostics, remaining_budget=2)
        self.assertEqual(result.id, 2)  # progress beats visually perfect left goal.
        self.assertAlmostEqual(diagnostics['planning']['selected_score'], .8, places=6)
        self.assertEqual(diagnostics['planning']['objective'], 'sdk_level_progress')
        self.assertEqual(diagnostics['planning_v2']['endpoint_role'], 'predicted_plan_result')
        self.assertIsNone(self.engine.decoded)  # no readout in imagined control loop.

    def test_warm_click_suffix_is_recomputed_even_when_uniform_samples_miss_it(self):
        class ClickDynamics(Dynamics):
            def predict(self, z, h, action, height, width):
                # Generic coordinate-dependent oracle, independent of any game.
                next_z = action[:, 8, None, None] * 3
                return next_z, h + 1, next_z - z
        planner = GoalPlanner(ClickDynamics(), Readout(), GoalConfig(horizon=1, clicks=1))
        retained = Action(6, {'x': 63, 'y': 0})
        planner.last_plan = [retained]
        diagnostic = {}
        result = planner.choose(self.state, replace(self.obs, available_actions=(6,)), random.Random(5),
                                 'goal_value_mpc', diagnostic, remaining_budget=2)
        self.assertEqual(result, retained)
        self.assertEqual(planner.last_endpoint.item(), 3)
        self.assertEqual(diagnostic['planning_v2']['selected_route_source'], 'retained')
        planner.after_real_action(retained, self.obs, self.obs)
        self.assertEqual(planner.last_plan, [])
        self.assertEqual(self.state.z.item(), 0)

    def test_actor_route_can_contribute_without_click_actions_and_still_loses_if_worse(self):
        class Proposal:
            calls = 0
            def propose_route(self, engine, state, available, rng, **kwargs):
                self.calls += 1
                return [Action(1)]
        planner = GoalPlanner(self.engine, Readout(), GoalConfig(horizon=1))
        proposer = Proposal()
        planner.proposal_policy = proposer
        choice = planner.choose(self.state, self.obs, random.Random(1), 'goal_value_mpc', remaining_budget=2)
        self.assertEqual(choice.id, 2)  # actor is a candidate, never an override.
        self.assertEqual(proposer.calls, 1)

    def test_bad_bounds_states_and_targets_fail_explicitly(self):
        for setting in ({'horizon': 65}, {'proposal_beam': 0}, {'lifetime': 1025}, {'discount': float('nan')}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                GoalConfig(**setting)
        for target in (torch.ones(2, 1), torch.tensor([[float('nan')]])):
            with self.assertRaises(ValueError):
                self.planner.set_goal(target, self.state)
        for state in (replace(self.state, h=torch.tensor([[float('inf')]])),
                      replace(self.state, geometry=(2, 2))):
            with self.assertRaises(ValueError):
                self.planner.choose(state, self.obs, random.Random())


if __name__ == '__main__':
    unittest.main()
