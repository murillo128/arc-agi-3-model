"""Build a Kaggle notebook from shared source; never submit automatically.

Follows the official ARC-AGI-3 Kaggle Starter's gateway/competition pattern.
The competition evaluates the notebook rerun, not local model predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
ACCELERATORS = {
    "cpu": ("none", False),
    "t4": ("nvidiaTeslaT4", True),
    "p100": ("nvidiaTeslaP100", True),
    "rtx6000": ("nvidiaRtx6000", True),
}


def _code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


def build_notebook(*, accelerator: str = "t4") -> dict:
    if accelerator not in ACCELERATORS:
        raise ValueError(f"Unsupported accelerator: {accelerator}")
    name, uses_gpu = ACCELERATORS[accelerator]
    relative_files = [
        "arc3/__init__.py",
        "arc3/core/__init__.py",
        "arc3/core/policy.py",
        "arc3/kaggle/__init__.py",
        "arc3/kaggle/agent.py",
    ]
    sources = {path: (SOURCE_ROOT / path).read_text(encoding="utf-8") for path in relative_files}
    # No model/checkpoint is bundled. Future trainable versions must attach weights
    # explicitly through a Kaggle dataset and validate that it exists at runtime.
    bootstrap = 'import sys\nsys.path.insert(0, "/tmp")\nfrom arc3.kaggle.agent import MyAgent\n'
    write_files = (
        "from pathlib import Path\n"
        f"sources = {sources!r}\n"
        "for relative, content in sources.items():\n"
        "    target = Path('/tmp') / relative\n"
        "    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    target.write_text(content, encoding='utf-8')\n"
        f"Path('/tmp/my_agent.py').write_text({bootstrap!r}, encoding='utf-8')\n"
    )
    competition_run = dedent('''\
        import os
        from pathlib import Path
        import shutil
        import subprocess
        import sys

        if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            subprocess.run([
                "curl", "--fail", "--retry", "999", "--retry-all-errors",
                "--retry-delay", "5", "--retry-max-time", "600",
                "http://gateway:8001/api/games",
            ], check=True)
            framework = Path("/kaggle/working/ARC-AGI-3-Agents")
            shutil.copytree(
                "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents",
                framework, dirs_exist_ok=True,
            )
            shutil.copyfile("/tmp/my_agent.py", framework / "agents/templates/my_agent.py")
            (framework / "agents/__init__.py").write_text(
                "from typing import Type\\n"
                "from dotenv import load_dotenv\\n"
                "from .agent import Agent, Playback\\n"
                "from .swarm import Swarm\\n"
                "from .templates.random_agent import Random\\n"
                "from .templates.my_agent import MyAgent\\n"
                "load_dotenv()\\n"
                "AVAILABLE_AGENTS: dict[str, Type[Agent]] = "
                "{'random': Random, 'myagent': MyAgent}\\n",
                encoding="utf-8",
            )
            (framework / ".env").write_text(
                "SCHEME=http\\nHOST=gateway\\nPORT=8001\\n"
                "ARC_API_KEY=test-key-123\\n"
                "ARC_BASE_URL=http://gateway:8001/\\n"
                "OPERATION_MODE=online\\n"
                "ENVIRONMENTS_DIR=\\n"
                "RECORDINGS_DIR=/kaggle/working/server_recording\\n",
                encoding="utf-8",
            )
            subprocess.run(
                [sys.executable, "main.py", "--agent", "myagent"],
                cwd=framework, check=True,
                env={**os.environ, "MPLBACKEND": "agg"},
            )
    ''')
    validation_run = dedent('''\
        import os
        if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            # This output only validates the Kaggle notebook commit.
            # The real parquet is produced by the gateway in the competition rerun.
            import pandas as pd
            pd.DataFrame(
                [["1_0", "1", True, 1]],
                columns=["row_id", "game_id", "end_of_game", "score"],
            ).to_parquet("/kaggle/working/submission.parquet", index=False)
    ''')
    return {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {
                "accelerator": name, "isInternetEnabled": False,
                "isGpuEnabled": uses_gpu, "language": "python", "sourceType": "notebook",
            },
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": "# ARC-AGI-3 baseline — generated from src/arc3"},
            _code("!pip install --no-index --find-links /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels arc-agi python-dotenv"),
            _code(write_files),
            _code(competition_run),
            _code(validation_run),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", choices=sorted(ACCELERATORS), default="t4")
    parser.add_argument("--username", help="Kaggle username; generates kernel-metadata.json")
    parser.add_argument("--slug", default="arc-agi-3-model")
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    notebook = args.output / "submission.ipynb"
    notebook.write_text(json.dumps(build_notebook(accelerator=args.accelerator), indent=2) + "\n", encoding="utf-8")
    if args.username:
        metadata = {
            "id": f"{args.username}/{args.slug}",
            "title": "ARC-AGI-3 Model — baseline",
            "code_file": notebook.name,
            "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": ACCELERATORS[args.accelerator][1],
            "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": [], "kernel_sources": [],
            "competition_sources": ["arc-prize-2026-arc-agi-3"], "model_sources": [],
        }
        (args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Notebook generated: {notebook}")
    if not args.username:
        print("Pass --username to create the Kaggle kernel metadata before pushing")


if __name__ == "__main__":
    main()
