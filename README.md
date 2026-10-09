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

## Current repository state

The repository was initialized from **Skillforge** and retains its issue/PR/review workflow infrastructure (see [AGENTS.md](AGENTS.md) and [skills/](skills/)). There is **no agent implementation or experiment runner yet**. Do not interpret planned module names as existing source files.

The first implementation milestone should deliver a minimal, deterministic local agent and environment adapter before introducing a trainable world model:

- Package/setup for Python 3.12 and the official `arc-agi` SDK, with local game setup documented.
- A small adapter that exposes observations, legal actions, level state, and transition recording without using private game implementation details.
- An executable baseline agent and a reproducible smoke test on public games, with fixed seeds and action budgets.
- Unit and integration checks, plus a clear split between public training games and held-out validation games.
- Only then: online dynamics adaptation, imagined planning/policy learning, performance benchmarks, and Kaggle notebook generation.

Each milestone should be tracked as a bounded GitHub issue and integrated via the existing Skillforge review process.

## Local development prerequisites

Use **Python 3.12**, a virtual environment, and the official [`arc-agi` package](https://pypi.org/project/arc-agi/) for SDK experiments. CUDA/PyTorch and GPU acceleration are optional until an actual trainable model is added. Development should remain testable on a CPU, RTX 4070 Ti (12 GB), or RTX A4500 (20 GB); future Kaggle hardware should not be hard-coded into model logic.

Local games can be exercised through the SDK, after downloading the publicly available environment files according to its documentation. This repository does not yet provide its own `make play` or `train` command.

## Evaluation and reproducibility

Separate **real environment steps** (scarce and potentially scored) from **imagined model steps** (compute-limited). Keep a per-game state and replay buffer, and explicitly define reset semantics for offline experiments versus competition runs. Maintain fixed seeds, SDK versions, recorded configuration, action counts, total wall-clock time, and evidence for comparative claims.

Do not use private/hidden games for training, do not inspect source code as an agent shortcut, and do not commit game caches, credentials, model weights, large generated datasets, or notebooks created for submission. Recheck official Kaggle rules before each submission, as competition constraints can change.

## Repository workflow

Read [AGENTS.md](AGENTS.md) for repository-wide invariants and the issue-driven implementation/review process. Local runner provisioning is optional and is **not part of this initialization**.
