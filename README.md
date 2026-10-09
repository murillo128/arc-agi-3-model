# ARC-AGI-3 Model

Research workspace for **agents that learn to act in previously unseen ARC-AGI-3 games**. The central question is whether an action-conditioned world model can be adapted from a few real interactions, then used for inexpensive imagined rollouts, planning, or policy improvement.

This is an **early-stage research repository**, not yet a trained model, a working agent, or a Kaggle-ready submission. Architecture choices and performance claims remain hypotheses until supported by tests.

## Research objective

1. Pretrain reusable visual/state representations and action-conditioned dynamics on public and/or procedurally generated games.
2. For a new game, collect allowed `(observation, action, next_observation)` transitions and adapt a **per-game** model without privileged access to that game's rules.
3. Evaluate candidate actions by planning or training policies on imagined trajectories, execute real actions conservatively, and correct the model using new observations.
4. Compare against identical-information baselines without online updates or without imagination. Measure solved levels, action efficiency, generalization to held-out games, prediction error, and compute time.

JEPA-style representations, Dreamer-style latent dynamics, planning algorithms, and model-update strategies are **candidates to compare**, not assumptions about implemented code. In particular, do not train the world model on its own unverified predictions as though they were real transitions.

## Official dependencies and external references

- [ARC-AGI Python SDK](https://github.com/arcprize/ARC-AGI) — game API, offline/local environment execution, scoring semantics.
- [ARC-AGI-3 Agents](https://github.com/arcprize/ARC-AGI-3-Agents) — reference agent interfaces and execution utilities.
- [ARC-AGI-3 Kaggle Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter) — reference notebook packaging and competition submission workflow.
- [ARC-AGI-3 Kaggle competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) — canonical submission rules and evaluation constraints.

Prefer the SDK as a **versioned dependency** rather than vendoring a Starter repository into the research code. Keep Kaggle-specific code at the submission boundary.

## Three separated execution paths

The first implemented agent is a **seeded random baseline**. JEPA/Dreamer training, online adaptation, and checkpoint loading are not yet implemented. The code shares a small policy in `src/arc3/core/`, while keeping three command entrypoints separate:

| Path | Purpose | Command |
| --- | --- | --- |
| `src/arc3/training/` | Capture real public-game transitions for later world-model training | `arc3-collect` |
| `src/arc3/evaluation/` | Evaluate locally on disjoint held-out games, without training | `arc3-evaluate` |
| `src/arc3/kaggle/` | Build a notebook with the same policy and competition-mode adapter | `arc3-build-kaggle` |

The `envs/` module imports the official SDK lazily. The Kaggle adapter implements the official framework's `MyAgent` interface without copying the policy.

## Install and run

Use **Python 3.12**. The baseline runs on CPU; no CUDA installation is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[kaggle]'
cp configs/splits.example.json configs/splits.local.json
```

Edit `configs/splits.local.json` before running. The versioned file is an **illustrative example only**, not the final deterministic 20-training/5-evaluation split. Training games need a positive `training_level_caps` value set from the actual number of levels you want to finish before entering the reserved final level. Do not guess it. The training collector excludes the boundary transition to avoid recording the new held-out level's starting observation.

**Training data collection (not model fitting):**

```bash
arc3-collect --split configs/splits.local.json --output data/transitions.jsonl --max-actions 80 --seed 42
```

**Local evaluation:**

```bash
arc3-evaluate --split configs/splits.local.json --output results/evaluation.json --max-actions 80 --seed 42
```

These commands use `arc-agi` in normal local mode and can download public games on first use. Pass `--offline` after caching games. The evaluator measures actions, levels completed and final state; it does not collect training records. Scoring *only the reserved last level* within training games still needs a dedicated implementation; no hidden Kaggle results are available locally.

**Generate a Kaggle notebook (without uploading):**

```bash
arc3-build-kaggle --username YOUR_KAGGLE_USERNAME --accelerator t4
```

For an intentional upload, authenticate the Kaggle CLI and run `kaggle kernels push -p notebooks/`. The generated notebook follows the official Starter's gateway pattern with the same shared policy. Kaggle then requires a separate **Submit to Competition** action; no upload, remote benchmark or official submission has been performed by this project bootstrap. Accelerators: `cpu`, `t4`, `p100`, `rtx6000`. Future trainable models must bundle their weights through approved Kaggle datasets, not by assuming local files exist.

## Minimal validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Four inexpensive tests with a fake environment cover run lifetime/transition recording, local split separation, held-out-level boundary and notebook source inclusion. They do **not** prove real-engine or Kaggle gateway compatibility. Actual game evaluation and GPU performance measurement are deliberately run outside CI; inherited Skillforge workflows remain unchanged.

## Evaluation and reproducibility

Separate **real environment steps** (scarce and potentially scored) from **imagined model steps** (compute-limited). Keep a per-game state and replay buffer, and explicitly define reset semantics for offline experiments versus competition runs. Maintain fixed seeds, SDK versions, recorded configuration, action counts, total wall-clock time, and evidence for comparative claims.

Do not use private/hidden games for training, do not inspect source code as an agent shortcut, and do not commit game caches, credentials, model weights, large generated datasets, or notebooks created for submission. Recheck official Kaggle rules before each submission, as competition constraints can change.

## Repository workflow

Read [AGENTS.md](AGENTS.md) for repository-wide invariants and the issue-driven implementation/review process. Local runner provisioning is optional and is **not part of this initialization**.
