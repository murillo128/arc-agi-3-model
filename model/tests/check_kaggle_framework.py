"""Opt-in compatibility probe for a locally supplied competition framework.

Run with --framework DIR --output report.json. Agent and Swarm are loaded from
that exact directory, without fetching upstream source. The gateway supplies
scripted public SDK frames, so this is framework integration evidence, not game
performance. Missing recording/tracing helpers are explicit ancillary doubles.
"""
import argparse
import ast
from contextlib import chdir
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch
from arc_agi import OperationMode
from arcengine import FrameDataRaw, GameAction, GameState

from arc3.kaggle.build import build_notebook
from model.adaptation import AdaptationConfig, load
from model.adaptation_smoke import bank_digest, base_digest
from model.tests.test_kaggle_adaptation import portable_spatial_base


class _Recorder:
    """In-memory replacement only when the supplied recording helper is absent."""
    def __init__(self, prefix, filename=None):
        self.filename = filename or f'{prefix}.simulated.jsonl'
        self.records = []

    def record(self, value):
        self.records.append(value)


def _raw(game, index, terminal=False):
    frame = np.zeros((8, 8), dtype=np.uint8)
    x = 1 + index if game == 'terminal-fixture' else 5 - index
    frame[2:4, x:x+2] = 9
    result = FrameDataRaw(game_id=game,
        state=GameState.WIN if terminal else GameState.NOT_FINISHED,
        levels_completed=0, win_levels=2, available_actions=[6], guid=f'{game}-{index}')
    result.frame = [frame]
    return result


class _Environment:
    def __init__(self, game, barrier=None):
        self.game, self.barrier = game, barrier
        self.observation_space = _raw(game, 0)
        self.calls = []

    def step(self, action, *, data, reasoning):
        assert action == GameAction.ACTION6, action
        assert set(data) == {'x', 'y'}, data
        assert 0 <= data['x'] < 8 and 0 <= data['y'] < 8, data
        assert self.observation_space.frame[0][data['y'], data['x']] == 9, data
        if self.barrier is not None and not self.calls:
            self.barrier.wait(timeout=20)
        self.calls.append(dict(data))
        self.observation_space = _raw(self.game, len(self.calls), self.game == 'terminal-fixture')
        return self.observation_space


class _Arcade:
    operation_mode = OperationMode.OFFLINE

    def __init__(self):
        self.barrier = threading.Barrier(2)
        self.environments = {}
        self.closed = []

    def open_scorecard(self, *, tags):
        return 'simulated-scorecard'

    def close_scorecard(self, card_id):
        self.closed.append(card_id)
        return None  # No score is produced by this probe.

    def make(self, game, *, scorecard_id):
        assert scorecard_id == 'simulated-scorecard'
        environment = _Environment(game, self.barrier)
        self.environments[game] = environment
        return environment


class _LongEnvironment(_Environment):
    """A scripted WIN after action 81; rejects an extra request after termination."""
    def step(self, action, *, data, reasoning):
        assert self.observation_space.state != GameState.WIN
        assert action == GameAction.ACTION6
        self.calls.append(dict(data))
        self.observation_space = _raw(self.game, len(self.calls) % 3, len(self.calls) == 81)
        return self.observation_space


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check(framework):
    started = time.perf_counter()
    framework = Path(framework).resolve(strict=True)
    required = ('agents/agent.py', 'agents/swarm.py')
    hashes = {name: hashlib.sha256((framework / name).read_bytes()).hexdigest() for name in required}
    package = ModuleType('agents')
    package.__path__ = [str(framework / 'agents')]
    modules, doubled = {'agents': package}, []
    if not (framework / 'agents/recorder.py').is_file():
        recorder = ModuleType('agents.recorder')
        recorder.Recorder = _Recorder
        modules['agents.recorder'] = recorder
        doubled.append('agents.recorder: in-memory recording only')
    if not (framework / 'agents/tracing.py').is_file():
        tracing = ModuleType('agents.tracing')
        tracing.trace_agent_session = lambda function: function
        modules['agents.tracing'] = tracing
        doubled.append('agents.tracing: identity decorator only')

    def offline(*args, **kwargs):
        raise RuntimeError('Network forbidden in framework compatibility probe')

    errors = []
    with tempfile.TemporaryDirectory(prefix='arc3-framework-probe-') as directory:
        root = Path(directory)
        asset = portable_spatial_base(root)
        config = dict(stopping='steps', exploration=1., representation_steps=1, per_observation_seconds=30., total_seconds=120.)
        sentinel = load(asset, config=AdaptationConfig(**config), seed='sentinel')
        before, inactive_before = base_digest(sentinel.runtime), bank_digest(sentinel.bank)
        notebook = build_notebook(agent_checkpoint='/kaggle/input/probe/base.pt', adaptive=True,
                                  adaptation_config=config, max_actions=2, cpu_threads=2)
        sources = next(ast.literal_eval(node.value) for node in ast.parse(notebook['cells'][2]['source']).body
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id == 'sources')
        arcade = _Arcade()
        with (patch.dict(sys.modules, modules), chdir(root),
              patch.object(socket.socket, 'connect', offline),
              patch.object(socket, 'create_connection', offline),
              patch.object(threading, 'excepthook', lambda event: errors.append(str(event.exc_value)))):
            agent_module = _module('agents.agent', framework / 'agents/agent.py')
            swarm_module = _module('agents.swarm', framework / 'agents/swarm.py')
            namespace = {'__name__': 'agents.templates.my_agent'}
            exec(compile(sources['my_agent.py'], 'generated/my_agent.py', 'exec'), namespace)
            agent_class = namespace['MyAgent']
            agent_class.AGENT_CHECKPOINT = str(asset)  # Relocate the input mount only.
            package.AVAILABLE_AGENTS = {'arc3': agent_class}
            assert agent_class._convert_raw_frame_data is agent_module.Agent._convert_raw_frame_data
            assert agent_class.take_action is agent_module.Agent.take_action
            sentinel_payload = GameAction.ACTION6.action_type(x=63, y=63)
            with (patch.object(swarm_module, 'Arcade', return_value=arcade),
                  patch.object(GameAction.ACTION6, 'action_data', sentinel_payload)):
                swarm = swarm_module.Swarm('arc3', 'http://unused.invalid',
                                           ['terminal-fixture', 'budget-fixture'])
                assert swarm.main() is None
                assert not errors, errors
                assert GameAction.ACTION6.action_data.x == 63
                assert GameAction.ACTION6.action_data.y == 63
            assert len(swarm.threads) == 2 and all(not thread.is_alive() for thread in swarm.threads)
            assert arcade.closed == ['simulated-scorecard']
            terminal, budget = swarm.agents
            assert terminal.action_counter == terminal.policy.actions_taken == 1
            assert budget.action_counter == budget.policy.actions_taken == 2
            assert terminal.policy.diagnostics['confirmed_transitions'] == 1
            assert budget.policy.diagnostics['confirmed_transitions'] == 2
            assert terminal.policy.diagnostics['native_outcomes']['WIN'] == 1
            assert terminal.policy.observation.state == 'WIN'
            assert budget.policy.observation.state == 'NOT_FINISHED'
            assert all(agent.policy.pending_action is None for agent in swarm.agents)
            assert terminal.policy.runtime is budget.policy.runtime is sentinel.runtime
            assert terminal.policy.bank is not budget.policy.bank
            assert terminal.policy.history is not budget.policy.history
            assert not set(map(id, terminal.policy.bank.parameters())) & set(map(id, budget.policy.bank.parameters()))
            assert arcade.environments['terminal-fixture'].calls[0] != arcade.environments['budget-fixture'].calls[0]
            for agent in swarm.agents:
                assert sum(agent.policy.diagnostics['updates'].values()) > 0
                assert agent.policy.diagnostics['numerical_failures'] == 0
                # Same game/seed/real API inputs in isolation must match the
                # concurrent adapter result, independently of thread ordering.
                isolated = agent_class(card_id='isolated', game_id=agent.game_id,
                    agent_name='arc3', ROOT_URL='http://unused.invalid', record=False,
                    arc_env=_Environment(agent.game_id))
                isolated.main()
                assert bank_digest(isolated.policy.bank) == bank_digest(agent.policy.bank)
            # Exercise our unlimited loop against the actual supplied Agent's
            # initialization, conversion, take_action, recording and cleanup.
            # No optimizer work is needed to establish this transport contract.
            agent_class.MAX_ACTIONS = None
            agent_class.ADAPTATION_CONFIG = config | dict(representation_steps=0,
                predictor_steps=0, path_steps=0, goal_steps=0)
            long_environment = _LongEnvironment('unlimited-fixture')
            unlimited = agent_class(card_id='unlimited', game_id='unlimited-fixture',
                agent_name='arc3', ROOT_URL='http://unused.invalid', record=False,
                arc_env=long_environment)
            unlimited.main()
            assert unlimited.action_counter == unlimited.policy.actions_taken == 81
            assert unlimited.policy.diagnostics['confirmed_transitions'] == 81
            assert unlimited.policy.diagnostics['native_outcomes']['WIN'] == 1
            assert unlimited.policy.pending_action is None
            assert unlimited._cleanup is False
            assert len(unlimited.frames) == 1
            assert base_digest(sentinel.runtime) == before
            assert bank_digest(sentinel.bank) == inactive_before
            return dict(format='arc3-competition-framework-probe', version=1, passed=True,
                elapsed_seconds=time.perf_counter()-started, framework_sha256=hashes,
                adapter_source_sha256=hashlib.sha256(sources['my_agent.py'].encode()).hexdigest(),
                ancillary_doubles=doubled, gateway='scripted public SDK FrameDataRaw; no scores',
                network_disabled=True, real_sdk_transitions=0,
                simulated_confirmations=sum(len(env.calls) for env in arcade.environments.values()) + len(long_environment.calls),
                checks=dict(supplied_bounded_agent_main=True, unlimited_actions=81, supplied_swarm_threads=2,
                    exact_nonterminal_budget=True, terminal_result_committed_once=True,
                    click_payloads_isolated=True, shared_base=True, private_game_state=True,
                    concurrent_matches_isolated=True, base_immutable=True, inactive_bank_unchanged=True),
                games=[dict(game=agent.game_id, actions=agent.action_counter,
                            state=agent.policy.observation.state,
                            updates=agent.policy.diagnostics['updates']) for agent in swarm.agents],
                limitations=['No live competition gateway, network protocol or scoring validation',
                             'Scripted observations are not real game performance evidence'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = check(args.framework)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=report['passed'], output=str(args.output),
                          elapsed_seconds=report['elapsed_seconds'])))


if __name__ == '__main__':
    main()
