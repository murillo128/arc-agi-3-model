"""Whole-agent inference, lifecycle and offline packaging contracts on tiny CPU weights."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from arc3.core.policy import Observation as PublicObservation
from arc3.kaggle.build import build_notebook
from model import agent_checkpoint, checkpoint
from model.agent import Agent, AgentConfig
from model.config import NetworkConfig, TrainConfig
from model.data import Action, Observation
from model.learning import Engine, Learner
from model.networks import WorldModel
from model.planning import GoalValue, PlanConfig, fingerprint


torch.set_num_threads(2)


def obs(value=1, *, state='NOT_FINISHED', level=0, empty=False, actions=(1, 2, 6)):
    frames = np.full((1, 4, 4), value, dtype=np.uint8)
    if empty:
        frames = np.empty((0, 0, 0), dtype=np.uint8)
    return Observation(frames, actions, state, level, 3)


def public(observation):
    return PublicObservation('example', tuple(tuple(tuple(map(int, row)) for row in frame)
                                              for frame in observation.frames),
                             observation.state, observation.levels_completed,
                             observation.win_levels, observation.available_actions)


def make_agent(mode='random', budget=8, seed=17, frozen=True):
    torch.manual_seed(13)
    net = WorldModel(NetworkConfig.small())
    engine = Engine(net, TrainConfig())
    heads = GoalValue(net.config.latent, net.config.memory, width=4)
    config = AgentConfig(mode=mode, max_actions=budget, planner=PlanConfig(horizon=2, beam=2, clicks=2))
    return Agent(engine, heads, config, seed=seed, frozen=frozen)


class AgentTests(unittest.TestCase):
    def test_budget_counts_confirmed_reset_and_noop_and_level_reinitializes(self):
        agent = make_agent(budget=3)
        agent.observe(obs(state='NOT_PLAYED', empty=True))
        self.assertEqual(agent.act(), Action(0))
        self.assertEqual(agent.remaining_budget, 3)  # A selection is not an SDK result.
        agent.observe(obs(2))
        self.assertEqual(agent.remaining_budget, 2)
        self.assertTrue(torch.equal(agent.state.h, torch.zeros_like(agent.state.h)))
        agent.act()
        # Identical pixels still confirm a real action and advance recurrent state.
        agent.observe(obs(2))
        self.assertEqual(agent.remaining_budget, 1)
        self.assertGreater(float(agent.state.h.abs().sum()), 0)
        agent.act()
        agent.observe(obs(3, level=1))
        self.assertEqual(agent.remaining_budget, 0)
        self.assertTrue(torch.equal(agent.state.h, torch.zeros_like(agent.state.h)))
        with self.assertRaisesRegex(ValueError, 'budget exhausted'):
            agent.act()

    def test_imagination_and_diagnostics_do_not_assimilate_or_consume_real_actions(self):
        agent = make_agent()
        agent.observe(obs())
        before_z, before_h = agent.state.z.clone(), agent.state.h.clone()
        rng = agent.rng.getstate()
        imagined = agent.imagine([Action(1), Action(2)])
        self.assertEqual(len(imagined), 2)
        imagined[-1].h.fill_(5)
        torch.testing.assert_close(agent.state.z, before_z, rtol=0, atol=0)
        torch.testing.assert_close(agent.state.h, before_h, rtol=0, atol=0)
        self.assertEqual(agent.rng.getstate(), rng)
        self.assertEqual(agent.actions_taken, 0)
        action = agent.act()
        for _ in range(3):
            agent.trace_diagnostics(public(obs()), action)
        self.assertEqual(agent.actions_taken, 0)
        with self.assertRaisesRegex(ValueError, 'pending action'):
            agent.act()
        agent.observe(obs(empty=True))
        self.assertEqual(agent.actions_taken, 1)
        torch.testing.assert_close(agent.state.z, before_z, rtol=0, atol=0)
        expected_h, _ = agent.engine.advance(before_z[None], before_h, agent.engine.actions([action]))
        torch.testing.assert_close(agent.state.h, expected_h)

    def test_learner_owns_same_modules_and_inference_never_changes_weights(self):
        agent = make_agent(frozen=False)
        learner = Learner(agent.engine)
        self.assertIs(learner.model, agent.engine.model)
        self.assertIs(learner.primary[0], next(agent.engine.model.encoder.parameters()))
        before = fingerprint(learner.model)
        agent.observe(obs())
        agent.act()
        agent.observe(obs(2))
        self.assertEqual(fingerprint(learner.model), before)
        self.assertTrue(all(p.requires_grad for p in learner.primary))
        frozen = make_agent()
        self.assertFalse(any(p.requires_grad for p in frozen.engine.model.parameters()))
        self.assertFalse(any(p.requires_grad for p in frozen.heads.parameters()))

    def test_whole_checkpoint_resumes_pending_action_rng_carry_and_budget(self):
        agent = make_agent(mode='value_mpc')
        agent.choose_action(public(obs()))
        agent.choose_action(public(obs(2)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agent.pt'
            agent_checkpoint.save(path, agent, include_runtime=True)
            restored = agent_checkpoint.load(path, resume_runtime=True)
            self.assertEqual(restored.pending_action, agent.pending_action)
            self.assertEqual(restored.actions_taken, 1)
            self.assertEqual(restored.rng.getstate(), agent.rng.getstate())
            for response in (obs(3), obs(empty=True), obs(4, level=1)):
                expected = agent.choose_action(public(response))
                actual = restored.choose_action(public(response))
                self.assertEqual(actual, expected)
                torch.testing.assert_close(restored.state.h, agent.state.h, rtol=0, atol=0)
                torch.testing.assert_close(restored.state.z, agent.state.z, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'override'):
                agent_checkpoint.load(path, resume_runtime=True, seed=99)

    def test_legacy_conversion_rejects_mismatched_weights_and_preserves_world(self):
        agent = make_agent()
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world, heads, result = (root / name for name in ('world.pt', 'heads.pt', 'agent.pt'))
            checkpoint.write(world, {'network': asdict(agent.engine.model.config),
                                    'training': asdict(agent.engine.config),
                                    'weights': agent.engine.model.state_dict()})
            payload = {'source_fingerprint': 'incorrect', 'weights': agent.heads.state_dict(),
                       'planner': asdict(agent.config.planner)}
            checkpoint.write(heads, payload)
            with self.assertRaisesRegex(ValueError, 'different world'):
                agent_checkpoint.convert_legacy(world, heads, result)
            self.assertFalse(result.exists())
            payload['source_fingerprint'] = fingerprint(agent.engine.model)
            checkpoint.write(heads, payload)
            agent_checkpoint.convert_legacy(world, heads, result)
            restored = agent_checkpoint.load(result)
            self.assertEqual(fingerprint(restored.engine.model), fingerprint(agent.engine.model))

    def test_active_proposal_policy_is_saved_and_shared_with_planner(self):
        from model.proposals import ProposalPolicy
        original = make_agent(mode='value_mpc')
        policy = ProposalPolicy(original.engine.model.config.latent, original.engine.model.config.memory,
                                width=4, patch=original.engine.model.config.patch)
        config = replace(original.config, planner=replace(original.config.planner,
                                                          clicks=16, click_proposals='actor'))
        agent = Agent(original.engine, original.heads, config, proposal_policy=policy)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agent.pt'
            agent_checkpoint.save(path, agent)
            restored = agent_checkpoint.load(path)
            self.assertIs(restored.proposal_policy, restored.planner.proposal_policy)
            self.assertEqual(fingerprint(restored.proposal_policy), fingerprint(policy))
            for response in (obs(), obs(2)):
                self.assertEqual(agent.choose_action(public(response)), restored.choose_action(public(response)))
            self.assertFalse(any(p.requires_grad for p in restored.proposal_policy.parameters()))

    def test_packaged_offline_agent_matches_local_actions_and_carry(self):
        agent = make_agent(mode='value_mpc', budget=6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'agent.pt'
            agent_checkpoint.save(asset, agent)
            local = agent_checkpoint.load(asset, seed='23:example', max_actions=6)
            observations = [obs(state='NOT_PLAYED', empty=True), obs(2), obs(2),
                            obs(empty=True), obs(3, level=1), obs(state='GAME_OVER')]
            expected = []
            for observation in observations:
                action = local.choose_action(public(observation))
                expected.append({'id': action.id, 'data': action.data,
                                 'h': local.state.h.tolist(), 'remaining': local.remaining_budget})
            notebook = build_notebook(seed=23, max_actions=6,
                                      agent_checkpoint='/kaggle/input/my-agent/agent.pt')
            scope = {}
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    compile(cell['source'], 'generated-cell', 'exec')
            exec(notebook['cells'][2]['source'], scope)
            packaged = scope['source_root']
            try:
                wire = [{'frame': observation.frames.tolist(), 'state': observation.state,
                         'levels_completed': observation.levels_completed,
                         'win_levels': observation.win_levels,
                         'available_actions': observation.available_actions} for observation in observations]
                (root / 'observations.json').write_text(json.dumps(wire))
                # Only the official transport interface is doubled. Both paths
                # execute the real, serialized neural modules and packaged code.
                script = '''import json, sys, types
from pathlib import Path
from types import SimpleNamespace
import torch
torch.set_num_threads(2)
base = types.ModuleType('agents.agent')
class FrameworkAgent:
    def __init__(self):
        self.game_id, self.action_counter = 'example', 0
base.Agent = FrameworkAgent
sys.modules['agents'] = types.ModuleType('agents')
sys.modules['agents.agent'] = base
from my_agent import MyAgent
import model.agent
assert Path(model.agent.__file__).resolve().is_relative_to(Path(sys.argv[3]).resolve())
MyAgent.AGENT_CHECKPOINT = sys.argv[1]
a = MyAgent()
rows = []
for item in json.loads(Path(sys.argv[2]).read_text()):
    item['state'] = SimpleNamespace(name=item['state'])
    raw = SimpleNamespace(game_id='example', **item)
    a.choose_action([], raw)
    rows.append({'id': a.selected.id, 'data': a.selected.data,
                 'h': a.policy.state.h.tolist(), 'remaining': a.policy.remaining_budget})
print(json.dumps(rows))
'''
                process = subprocess.run([sys.executable, '-c', script, str(asset),
                                          str(root / 'observations.json'), str(packaged)],
                                         cwd=root, env={**os.environ, 'PYTHONPATH': str(packaged)},
                                         capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(process.stdout), expected)
                self.assertFalse(notebook['metadata']['kaggle']['isInternetEnabled'])
                self.assertNotIn(str(asset), json.dumps(notebook))
            finally:
                shutil.rmtree(packaged)


if __name__ == '__main__':
    unittest.main()
