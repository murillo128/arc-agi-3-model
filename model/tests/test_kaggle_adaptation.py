"""Official transport commit boundary and isolated adaptive notebook execution."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from arcengine import FrameData, GameAction, GameState

from arc3.core.policy import Action
from arc3.kaggle.build import build_notebook
from model import adaptation_checkpoint
from model.adaptation_heads import VisualGoal
from model.config import NetworkConfig, TrainConfig
from model.networks import WorldModel
from model.path_planning import PathValue


torch.set_num_threads(2)


def frame(state=GameState.NOT_FINISHED):
    return FrameData(game_id='example', frame=[[[0, 1], [2, 0]]], state=state,
                     levels_completed=0, win_levels=2, available_actions=[6])


def transport_agent():
    """Double the external official base class; run our actual transport code."""
    module = ModuleType('agents.agent')

    class FrameworkAgent:
        def __init__(self, environment):
            self.game_id, self.arc_env, self.action_counter = 'example', environment, 0

        def _convert_raw_frame_data(self, raw):
            if raw is None:
                raise ValueError('Received None frame data from environment')
            return raw

    module.Agent = FrameworkAgent
    namespace = {}
    path = Path(__file__).parents[2] / 'src/arc3/kaggle/agent.py'
    with patch.dict(sys.modules, {'agents': ModuleType('agents'), 'agents.agent': module}):
        exec(compile(path.read_text(), str(path), 'exec'), namespace)
    return namespace['MyAgent']


def portable_spatial_base(root):
    # Production architecture switches with small dimensions: missing spatial
    # adapter code must fail in the packaged interpreter, even on a CPU.
    config = replace(NetworkConfig.small(), latent=32, pixel_latent=True,
                     local_dynamics=True, rotation_adapter_width=16)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        world = WorldModel(config)
        goal = VisualGoal(config.latent, config.memory, config.patch, 8)
        path = PathValue(config.latent, config.memory, 8)
    source, target = root / 'training.pt', root / 'base.pt'
    torch.save(dict(kind='game_sequences', version=1, network=asdict(config),
                    training=asdict(TrainConfig()), weights=world.state_dict(),
                    planning=dict(version=1, config={'width': 8}, goal=goal.state_dict(),
                                  path=path.state_dict()), progress={'batches': 8246}), source)
    adaptation_checkpoint.export(source, target)
    return target


class KaggleAdaptationTests(unittest.TestCase):
    def test_generated_defaults_separate_unlimited_adaptation_and_bounded_self_check(self):
        adaptive = build_notebook(adaptive=True, agent_checkpoint='/kaggle/input/probe/base.pt')
        scope = {}
        exec(adaptive['cells'][2]['source'], scope)
        try:
            module = scope['sources']['my_agent.py']
            self.assertIn('MyAgent.MAX_ACTIONS = None', module)
            self.assertIn("'stopping': 'convergence'", module)
            self.assertIn("'total_seconds': None", module)
            self.assertIn('model/adaptation_convergence.py', scope['sources'])
            self.assertIn('"stopping": "steps"', adaptive['cells'][3]['source'])
        finally:
            shutil.rmtree(scope['source_root'])
        scope = {}
        exec(build_notebook()['cells'][2]['source'], scope)
        try:
            self.assertIn('MyAgent.MAX_ACTIONS = 80', scope['sources']['my_agent.py'])
        finally:
            shutil.rmtree(scope['source_root'])

    def test_sdk_return_commits_terminal_and_keeps_click_payload_private(self):
        from model import adaptation
        cls = transport_agent()
        cls.ADAPTIVE, cls.AGENT_CHECKPOINT = True, 'base.pt'
        cls.MAX_ACTIONS = 2  # Explicit bounded mode still stops exactly here.
        confirmed = []
        selected = Action(6, {'x': 1, 'y': 0})
        policy = SimpleNamespace(choose_action=lambda observation: selected,
                                 confirm_result=lambda observation, action: confirmed.append((observation, action)))
        sent = []

        def step(action, **kwargs):
            self.assertEqual(confirmed, [])
            sent.append((action, kwargs))
            return frame(GameState.WIN)

        with patch.object(adaptation, 'load', return_value=policy):
            agent = cls(SimpleNamespace(step=step))
        action = agent.choose_action([], frame())
        agent.choose_action([], frame())
        self.assertEqual(confirmed, [])
        with patch.object(GameAction.ACTION6, 'action_data', GameAction.ACTION6.action_type(x=63, y=63)):
            result = agent.do_action_request(action)
        self.assertEqual(sent[0][1]['data'], {'x': 1, 'y': 0})
        self.assertIsNot(sent[0][1]['data'], selected.data)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0][0].state, 'WIN')
        self.assertEqual(confirmed[0][1], selected)
        self.assertTrue(agent.is_done([], result))
        agent.action_counter = agent.MAX_ACTIONS
        self.assertTrue(agent.is_done([], frame()))

    def test_failed_or_empty_sdk_result_cancels_without_confirmation(self):
        from model import adaptation
        cls = transport_agent()
        cls.ADAPTIVE, cls.AGENT_CHECKPOINT = True, 'base.pt'
        for raises in (True, False):
            with self.subTest(raises=raises):
                events = []
                policy = SimpleNamespace(choose_action=lambda observation: Action(6, {'x': 1, 'y': 0}),
                    confirm_result=lambda *args, **kwargs: events.append('confirmed'),
                    cancel_pending=lambda: events.append('cancelled'))

                def step(*args, **kwargs):
                    if raises:
                        raise OSError('SDK step failed')
                    return None

                with patch.object(adaptation, 'load', return_value=policy):
                    agent = cls(SimpleNamespace(step=step))
                selected = agent.choose_action([], frame())
                with self.assertRaises((OSError, ValueError)):
                    agent.do_action_request(selected)
                self.assertEqual(events, ['cancelled'])

    def test_adaptive_build_rejects_invalid_modes_budgets_and_paths(self):
        mounted = '/kaggle/input/private-base/base.pt'
        for arguments in ({'adaptive': True},
                          {'adaptation_config': {}},
                          {'cpu_threads': 0},
                          {'cpu_threads': 17},
                          {'adaptive': True, 'agent_checkpoint': mounted, 'adaptation_config': []},
                          {'adaptive': True, 'agent_checkpoint': mounted, 'adaptation_config': {'unknown': 1}},
                          {'adaptive': True, 'agent_checkpoint': mounted, 'smoke_output': '/tmp/smoke.json'}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                build_notebook(**arguments)

    def test_frozen_spatial_checkpoint_runs_from_isolated_generated_package(self):
        from arc3.core.policy import Observation
        from model import agent_checkpoint
        from model.agent import Agent, AgentConfig
        from model.learning import Engine
        from model.planning import GoalValue, PlanConfig

        config = replace(NetworkConfig.small(), latent=32, pixel_latent=True,
                         local_dynamics=True, rotation_adapter_width=16)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            model = WorldModel(config)
            heads = GoalValue(config.latent, config.memory, width=4)
        original = Agent(Engine(model, TrainConfig()), heads,
                         AgentConfig(mode='value_mpc', max_actions=2,
                                     planner=PlanConfig(horizon=1, beam=1, clicks=1)), seed=23)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'frozen.pt'
            agent_checkpoint.save(asset, original)
            notebook = build_notebook(agent_checkpoint='/kaggle/input/frozen/agent.pt')
            scope = {}
            exec(notebook['cells'][2]['source'], scope)
            package = scope['source_root']
            try:
                observation = Observation.from_public(frame())
                local = agent_checkpoint.load(asset, seed=23)
                action = local.choose_action(observation)
                expected = dict(id=action.id, data=action.data, h=local.state.h.tolist())
                # -I excludes the checkout/PYTHONPATH. Import only the generated
                # model package, with networking disabled, then run real MPC.
                script = '''import json, socket, sys
from pathlib import Path
sys.meta_path = [finder for finder in sys.meta_path
                 if not getattr(finder, '__module__', '').startswith('__editable__')]
sys.path.insert(0, sys.argv[1])
def offline(*args, **kwargs):
    raise RuntimeError('network forbidden')
socket.socket.connect = offline
socket.create_connection = offline
import torch
torch.set_num_threads(2)
from model import agent_checkpoint
from arc3.core.policy import Observation
policy = agent_checkpoint.load(sys.argv[2], seed=23)
from model import spatial_adapter
assert Path(spatial_adapter.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
observation = Observation('example', (((0, 1), (2, 0)),), 'NOT_FINISHED', 0, 2, (6,))
action = policy.choose_action(observation)
print(json.dumps(dict(id=action.id, data=action.data, h=policy.state.h.tolist())))
'''
                process = subprocess.run([sys.executable, '-I', '-c', script, str(package), str(asset)],
                                         cwd=root, capture_output=True, text=True, timeout=60)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertEqual(json.loads(process.stdout), expected)
                # Negative control proves this boundary detects the original
                # omitted spatial module, rather than resolving it from checkout.
                (package / 'model/spatial_adapter.py').unlink()
                shutil.rmtree(package / 'model/__pycache__', ignore_errors=True)
                missing = subprocess.run([sys.executable, '-I', '-c', script, str(package), str(asset)],
                                         cwd=root, capture_output=True, text=True, timeout=60)
                self.assertNotEqual(missing.returncode, 0)
                self.assertIn('spatial_adapter', missing.stderr)
            finally:
                shutil.rmtree(package)

    def test_generated_save_run_executes_spatial_adaptation_without_repo_or_network(self):
        mounted = '/kaggle/input/private-base/base.pt'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = portable_spatial_base(root)
            notebook = build_notebook(agent_checkpoint=mounted, adaptive=True, cpu_threads=2)
            scope = {}
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    compile(cell['source'], 'generated-cell', 'exec')
            exec(notebook['cells'][2]['source'], scope)
            package = scope['source_root']
            try:
                self.assertNotIn('model/sequence_planning.py', scope['sources'])
                self.assertFalse(any(name.startswith('games/') for name in scope['sources']))
                # Loaded automatically in the generated subprocess, so accidental
                # downloads fail instead of silently making the test online.
                (package / 'sitecustomize.py').write_text(
                    "import socket, sys\n"
                    "sys.meta_path = [finder for finder in sys.meta_path\n"
                    "    if not getattr(finder, '__module__', '').startswith('__editable__')]\n"
                    "def unavailable(*args, **kwargs):\n"
                    "    raise RuntimeError('network forbidden in packaged self-check')\n"
                    "socket.socket.connect = unavailable\n"
                    "socket.create_connection = unavailable\n")
                result = root / 'actual-smoke.json'
                run_cell = notebook['cells'][3]['source'].replace(repr(mounted), repr(str(asset)))
                run_cell = run_cell.replace(repr('/kaggle/working/adaptation-smoke.json'), repr(str(result)))
                environment = {**os.environ, 'PYTHONPATH': str(package), 'OMP_NUM_THREADS': '2',
                               'MKL_NUM_THREADS': '2', 'KAGGLE_IS_COMPETITION_RERUN': ''}
                # Execute the actual generated save-and-run cell. Only Kaggle
                # mount/output paths are relocated to this temporary sandbox.
                script = ('import sys, subprocess\nfrom pathlib import Path\n'
                          f'source_root = Path({str(package)!r})\n' + run_cell)
                completed = subprocess.run([sys.executable, '-c', script], cwd=root,
                                           env=environment, text=True, capture_output=True, timeout=120)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                report = json.loads(result.read_text())
                self.assertEqual(report['format'], 'arc3-adaptation-smoke')
                self.assertEqual(report['kind'], 'offline-fixture-self-check')
                self.assertTrue(report['passed'])
                self.assertEqual(report['real_sdk_transitions'], 0)
                self.assertGreater(report['optimizer_steps'], 0)
                self.assertEqual(report['runtime']['precision'], 'fp32')
                self.assertEqual(report['runtime']['checkpoint_loading'], 'staged_sequential_eager')
                self.assertGreater(report['timings']['load_seconds'], 0)
                events = [json.loads(line) for line in completed.stderr.splitlines() if line.startswith('{')]
                self.assertEqual([event['stage'] for event in events],
                                 ['loading', 'loaded', 'checking', 'learning', 'verifying', 'complete'])
                self.assertFalse(notebook['metadata']['kaggle']['isInternetEnabled'])
                self.assertNotIn(str(asset), json.dumps(notebook))
                # A failed self-check stops normal notebook execution before the
                # subsequent dummy-parquet cell can mask it as success.
                with patch.object(subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'smoke')):
                    with self.assertRaises(subprocess.CalledProcessError):
                        exec(run_cell, {'source_root': package, 'sys': sys, 'subprocess': subprocess})
            finally:
                shutil.rmtree(package)


if __name__ == '__main__':
    unittest.main()
