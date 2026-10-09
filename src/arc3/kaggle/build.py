"""Generate an offline, self-contained notebook following the official starter."""

import argparse
import json
import re
from importlib.resources import files
from pathlib import Path
from textwrap import dedent

from arc3.envs.sdk import SDK_VERSION

COMPETITION = "arc-prize-2026-arc-agi-3"
INPUT_ROOT = f"/kaggle/input/competitions/{COMPETITION}"


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source}


def build_notebook(seed: int = 0, max_actions: int = 80) -> dict:
    if max_actions <= 0:
        raise ValueError("max_actions must be positive")
    sources = {
        "arc3/__init__.py": "",
        "arc3/core/__init__.py": "",
        "arc3/core/policy.py": files("arc3.core").joinpath("policy.py").read_text(encoding="utf-8"),
        "my_agent.py": (
            files("arc3.kaggle").joinpath("agent.py").read_text(encoding="utf-8")
            + f"\nMyAgent.SEED = {seed!r}\nMyAgent.MAX_ACTIONS = {max_actions!r}\n"
        ),
    }
    install = dedent(f"""\
        import subprocess
        import sys
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links',
            '{INPUT_ROOT}/arc_agi_3_wheels', 'arc-agi=={SDK_VERSION}', 'python-dotenv'
        ], check=True)
        from importlib.metadata import version
        assert version('arc-agi') == '{SDK_VERSION}', 'Unexpected SDK version'
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
            "kaggle": {"isInternetEnabled": False, "isGpuEnabled": False,
                       "accelerator": "none", "language": "python", "sourceType": "notebook"},
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": (
                "# ARC-AGI-3 seeded random baseline\n\n"
                f"Generated from versioned Python source. Seed: {seed}; action budget: {max_actions}. "
                "No online adaptation is implemented. Competition execution uses only "
                "the supplied offline wheels, official framework and gateway."
            )},
            code_cell(install), code_cell(embed), code_cell(run), code_cell(dummy),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/submission.ipynb")
    parser.add_argument("--username", help="Explicit Kaggle handle; enables kernel metadata generation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=80)
    args = parser.parse_args()
    output = Path(args.output)
    if output.suffix != ".ipynb":
        parser.error("--output must end in .ipynb")
    if args.username is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", args.username):
        parser.error("--username must be a nonempty Kaggle handle without a slash")
    if args.max_actions <= 0:
        parser.error("--max-actions must be positive")
    notebook = build_notebook(args.seed, args.max_actions)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    if args.username is not None:
        metadata = {
            "id": f"{args.username}/arc3-random-baseline", "title": "ARC-AGI-3 random baseline",
            "code_file": output.name, "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": [], "kernel_sources": [],
            "competition_sources": [COMPETITION], "model_sources": [],
        }
        output.with_name("kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
        )
    print(output)


if __name__ == "__main__":
    main()
