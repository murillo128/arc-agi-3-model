"""Generate an offline, self-contained notebook following the official starter."""

import argparse
import json
import re
from importlib.resources import files
from pathlib import Path, PurePosixPath
from textwrap import dedent

from arc3.envs.sdk import SDK_VERSION

COMPETITION = "arc-prize-2026-arc-agi-3"
INPUT_ROOT = f"/kaggle/input/competitions/{COMPETITION}"
ACCELERATORS = {
    "cpu": ("none", False),
    "t4": ("nvidiaTeslaT4", True),
    "p100": ("nvidiaTeslaP100", True),
    "rtx6000": ("nvidiaRtx6000", True),
}


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source}


def build_notebook(seed: int = 0, max_actions: int | None = None, *, accelerator: str = "cpu",
                   agent_checkpoint: str | None = None, adaptive: bool = False,
                   adaptation_config: dict | None = None, cpu_threads: int = 2,
                   smoke_output: str = '/kaggle/working/adaptation-smoke.json',
                   extra_wheel_dir: str | None = None) -> dict:
    if max_actions is not None and (type(max_actions) is not int or max_actions <= 0):
        raise ValueError("max_actions must be positive or None")
    if max_actions is None and not adaptive:
        max_actions = 80
    if accelerator not in ACCELERATORS:
        raise ValueError(f"Unsupported accelerator: {accelerator}")
    if type(adaptive) is not bool:
        raise ValueError('adaptive must be boolean')
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 16:
        raise ValueError('cpu_threads must be an integer in 1..16')
    if adaptive and agent_checkpoint is None:
        raise ValueError('Adaptive mode requires --agent-checkpoint with a portable base.pt')
    if adaptation_config is not None and not adaptive:
        raise ValueError('Adaptation configuration requires adaptive mode')
    if adaptive:
        from dataclasses import asdict
        from model.adaptation import AdaptationConfig

        if adaptation_config is not None and not isinstance(adaptation_config, dict):
            raise ValueError('Adaptation configuration must be a JSON object')
        try:
            adaptation_config = asdict(AdaptationConfig(**(adaptation_config or {})))
        except TypeError as exc:
            raise ValueError(f'Invalid adaptation configuration: {exc}') from exc
        smoke_path = PurePosixPath(smoke_output)
        if (not smoke_path.is_relative_to('/kaggle/working') or '..' in smoke_path.parts
                or smoke_path.suffix != '.json'):
            raise ValueError('Smoke output must be a /kaggle/working/.../*.json path')
    if agent_checkpoint is not None:
        mount = PurePosixPath(agent_checkpoint)
        if not mount.is_relative_to('/kaggle/input') or '..' in mount.parts or mount.suffix != '.pt':
            raise ValueError('Agent checkpoint must be a mounted /kaggle/input/.../*.pt asset')
    wheel_dirs = [f'{INPUT_ROOT}/arc_agi_3_wheels']
    if extra_wheel_dir is not None:
        if not isinstance(extra_wheel_dir, str) or '\x00' in extra_wheel_dir:
            raise ValueError('Extra wheel directory must be a mounted /kaggle/input/... directory')
        wheel_mount = PurePosixPath(extra_wheel_dir)
        if (not wheel_mount.is_relative_to('/kaggle/input')
                or wheel_mount == PurePosixPath('/kaggle/input') or '..' in wheel_mount.parts
                or wheel_mount.suffix.lower() == '.whl'):
            raise ValueError('Extra wheel directory must be a mounted /kaggle/input/... directory, not a wheel file')
        wheel_dirs.append(str(wheel_mount))
    accelerator_name, uses_gpu = ACCELERATORS[accelerator]
    # Save-and-run is an explicitly bounded engineering fixture, not gameplay
    # adaptation. The scored agent above keeps the requested convergence config.
    smoke_config = dict(adaptation_config or {}, stopping='steps')
    sources = {
        "arc3/__init__.py": "",
        "arc3/core/__init__.py": "",
        "arc3/core/policy.py": files("arc3.core").joinpath("policy.py").read_text(encoding="utf-8"),
        "my_agent.py": (
            files("arc3.kaggle").joinpath("agent.py").read_text(encoding="utf-8")
            + f"\nMyAgent.SEED = {seed!r}\nMyAgent.MAX_ACTIONS = {max_actions!r}\n"
            + f"MyAgent.AGENT_CHECKPOINT = {agent_checkpoint!r}\nMyAgent.DEVICE = {'cuda' if uses_gpu else 'cpu'!r}\n"
            + f"MyAgent.ADAPTIVE = {adaptive!r}\nMyAgent.ADAPTATION_CONFIG = {adaptation_config!r}\n"
            + (f"import torch\ntorch.set_num_threads({cpu_threads!r})\n" if agent_checkpoint else '')
        ),
    }
    if agent_checkpoint is not None:
        # Versioned Python only. The checkpoint remains a separately mounted
        # offline asset; no host path, training data or weights enter the notebook.
        names = ['__init__', 'checkpoint', 'config', 'data', 'learning', 'networks',
                 'metrics', 'planning', 'goal_observation', 'path_planning', 'spatial_adapter',
                 'object_dynamics', 'segmentation']
        if adaptive:
            names += ['adaptation', 'adaptation_convergence', 'adaptation_heads', 'adaptation_checkpoint',
                      'adaptation_smoke', 'lora', 'sequence_exploration']
        else:
            names += ['agent', 'agent_checkpoint', 'sequence', 'goal_planning', 'proposals']
        for name in names:
            source = files('model').joinpath(f'{name}.py')
            if name == 'proposals' and not source.is_file():
                continue
            sources[f'model/{name}.py'] = source.read_text(encoding='utf-8')
    install = dedent(f"""\
        import subprocess
        import sys
        from pathlib import Path
        offline_wheel_dirs = {wheel_dirs!r}
        for wheel_dir in offline_wheel_dirs:
            if not Path(wheel_dir).is_dir():
                raise FileNotFoundError(
                    f'Missing offline wheel directory: {{wheel_dir}}. Attach the competition input '
                    'and the configured --extra-wheel-dir asset before running this notebook.')
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--no-index',
                *[argument for directory in offline_wheel_dirs for argument in ('--find-links', directory)],
                'arc-agi=={SDK_VERSION}', 'python-dotenv'
            ], check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                'Offline installation failed: attach arc-agi=={SDK_VERSION} and all required dependency wheels. '
                'The competition bundle may contain an older SDK; use --extra-wheel-dir for the companion '
                'wheel asset. Internet installation is disabled.') from exc
        from importlib.metadata import version
        assert version('arc-agi') == '{SDK_VERSION}', 'Unexpected SDK version'
    """)
    if agent_checkpoint is not None:
        install += dedent(f"""\
            from pathlib import Path
            import torch
            import numpy
            assert Path({agent_checkpoint!r}).is_file(), 'Attach the configured offline agent checkpoint asset'
        """)
    embed = (
        "from pathlib import Path\n"
        "import tempfile\n"
        "source_root = Path(tempfile.mkdtemp(prefix='arc3-source-'))\n"
        f"sources = {sources!r}\n"
        "for relative_path, source in sources.items():\n"
        "    destination = source_root / relative_path\n"
        "    destination.parent.mkdir(parents=True, exist_ok=True)\n"
        "    destination.write_text(source, encoding='utf-8')\n"
    )
    run = dedent(f"""\
        import os
        import shutil
        import time
        import urllib.request

        if {adaptive!r} and not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Execute the embedded code and mounted weights before producing any
            # starter artifact. This fixture check is not a scored game run.
            environment = dict(os.environ)
            environment.update({{'PYTHONPATH': str(source_root),
                                'OMP_NUM_THREADS': {str(cpu_threads)!r},
                                'MKL_NUM_THREADS': {str(cpu_threads)!r}}})
            subprocess.run([
                sys.executable, '-m', 'model.adaptation_smoke', '--self-check',
                '--checkpoint', {agent_checkpoint!r}, '--device', {'cuda' if uses_gpu else 'cpu'!r},
                '--threads', {str(cpu_threads)!r}, '--config-json', {json.dumps(smoke_config)!r},
                '--output', {smoke_output!r},
            ], cwd=source_root, env=environment, check=True)

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # The competition gateway enforces scoring/lifetime restrictions.
            deadline = time.monotonic() + 600
            while True:
                try:
                    with urllib.request.urlopen('http://gateway:8001/api/games', timeout=10):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Kaggle gateway did not become ready')
                    time.sleep(5)

            framework = source_root / 'framework'
            shutil.copytree('{INPUT_ROOT}/ARC-AGI-3-Agents', framework, dirs_exist_ok=True)
            shutil.copy2(source_root / 'my_agent.py', framework / 'agents/templates/my_agent.py')
            registry = (
                'from .agent import Agent, Playback\\n'
                'from .swarm import Swarm\\n'
                'from .templates.my_agent import MyAgent\\n'
                "AVAILABLE_AGENTS = {{'arc3': MyAgent}}\\n"
            )
            (framework / 'agents/__init__.py').write_text(registry, encoding='utf-8')
            environment = dict(os.environ)
            environment.update({{
                'SCHEME': 'http', 'HOST': 'gateway', 'PORT': '8001',
                'ARC_API_KEY': 'test-key-123', 'ARC_BASE_URL': 'http://gateway:8001/',
                'OPERATION_MODE': 'online', 'ENVIRONMENTS_DIR': '',
                'RECORDINGS_DIR': '/tmp/arc3-server-recordings',
                'PYTHONPATH': str(source_root), 'MPLBACKEND': 'agg',
                'OMP_NUM_THREADS': {str(cpu_threads)!r}, 'MKL_NUM_THREADS': {str(cpu_threads)!r},
            }})
            # The official framework makes each game once. The gateway produces
            # submission.parquet; this notebook does not create competition scores.
            subprocess.run([sys.executable, 'main.py', '--agent', 'arc3'],
                           cwd=framework, env=environment, check=True)
    """)
    dummy = dedent("""\
        if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Starter save-and-run artifact only; never an evaluation result.
            import pandas as pd
            pd.DataFrame([['1_0', '1', True, 1]],
                         columns=['row_id', 'game_id', 'end_of_game', 'score']).to_parquet(
                '/kaggle/working/submission.parquet', index=False)
    """)
    return {
        "nbformat": 4, "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"isInternetEnabled": False, "isGpuEnabled": uses_gpu,
                       "accelerator": accelerator_name, "language": "python", "sourceType": "notebook"},
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": (
                ("# ARC-AGI-3 per-game LoRA smoke agent\n\n" if adaptive else
                 "# ARC-AGI-3 frozen latent agent\n\n" if agent_checkpoint else "# ARC-AGI-3 seeded random baseline\n\n")
                + f"Generated from versioned Python source. Seed: {seed}; action budget: {max_actions if max_actions is not None else 'environment-controlled'}. "
                + (f"Per-game adaptation is enabled. Save-and-run writes {smoke_output} from an offline fixture self-check; "
                   "this and the dummy parquet are not game scores. " if adaptive else
                   "This mode never updates model weights. ")
                + "Competition execution uses only "
                "the supplied offline wheels, official framework and gateway."
            )},
            code_cell(install), code_cell(embed), code_cell(run), code_cell(dummy),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/submission.ipynb")
    parser.add_argument("--username", help="Explicit Kaggle handle; enables kernel metadata generation")
    parser.add_argument("--accelerator", choices=sorted(ACCELERATORS), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, help='Optional action cap; adaptive default: no cap, frozen/random: 80')
    parser.add_argument("--agent-checkpoint", help="Mounted /kaggle/input/.../agent.pt; weights are never embedded")
    parser.add_argument("--agent-dataset", help="Optional owner/dataset reference for that mounted checkpoint")
    parser.add_argument('--extra-wheel-dir', help='Mounted /kaggle/input/... directory with companion offline wheels, including arc-agi 0.9.9')
    parser.add_argument('--adaptive', action='store_true', help='Opt in to per-game LoRA using a portable adaptation base')
    parser.add_argument('--adaptation-config', help='Local JSON object containing AdaptationConfig overrides')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--smoke-output', default='/kaggle/working/adaptation-smoke.json')
    args = parser.parse_args()
    output = Path(args.output)
    if output.suffix != ".ipynb":
        parser.error("--output must end in .ipynb")
    if args.username is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", args.username):
        parser.error("--username must be a nonempty Kaggle handle without a slash")
    if args.max_actions is not None and args.max_actions <= 0:
        parser.error("--max-actions must be positive")
    if args.agent_dataset is not None and (args.agent_checkpoint is None or
            not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', args.agent_dataset)):
        parser.error('--agent-dataset requires --agent-checkpoint and owner/dataset syntax')
    try:
        adaptation_config = (json.loads(Path(args.adaptation_config).read_text(encoding='utf-8'))
                             if args.adaptation_config else None)
        notebook = build_notebook(args.seed, args.max_actions, accelerator=args.accelerator,
                                  agent_checkpoint=args.agent_checkpoint, adaptive=args.adaptive,
                                  adaptation_config=adaptation_config, cpu_threads=args.cpu_threads,
                                  smoke_output=args.smoke_output, extra_wheel_dir=args.extra_wheel_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    if args.username is not None:
        artifact = 'adaptive-agent' if args.adaptive else 'frozen-agent' if args.agent_checkpoint else 'random-baseline'
        metadata = {
            "id": f"{args.username}/arc3-{artifact}", "title": f"ARC-AGI-3 {artifact.replace('-', ' ')}",
            "code_file": output.name, "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": ACCELERATORS[args.accelerator][1], "enable_tpu": False,
            "enable_internet": False, "dataset_sources": [args.agent_dataset] if args.agent_dataset else [], "kernel_sources": [],
            "competition_sources": [COMPETITION], "model_sources": [],
        }
        output.with_name("kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
        )
    print(output)


if __name__ == "__main__":
    main()
