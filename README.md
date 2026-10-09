# ARC-AGI-3 Model

Research workspace for **agents that learn to act in previously unseen ARC-AGI-3 games**. The central question is whether an action-conditioned world model can be adapted from a few real interactions, then used for inexpensive imagined rollouts, planning, or policy improvement.

This is an **early-stage research repository** with an installable seeded random baseline and separate transition collection, local evaluation, and Kaggle notebook generation paths. There is no trained world model or online adaptation, and no measured game performance or validated Kaggle submission. Architecture choices remain hypotheses.

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

The repository retains its **Skillforge** issue/PR/review infrastructure (see [AGENTS.md](AGENTS.md) and [skills/](skills/)). Application code lives in `src/arc3`: `core` owns the shared seeded policy and real-action loop, `envs` owns the thin SDK boundary and game split validation, and `training`, `evaluation`, and `kaggle` own their separate entrypoints. No private environment implementation is passed to the policy.

## Local development prerequisites

Use **Python 3.12**. The package pins the official SDK to `arc-agi==0.9.9`; the adapter checks that version at runtime. No GPU, PyTorch, Kaggle account or credentials are required for the smoke tests or notebook generator.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Local execution defaults to `--mode offline`, using public games previously provisioned in `environment_files/`. Obtain these through the [official SDK setup](https://github.com/arcprize/ARC-AGI#quick-start), honoring its license and access rules. With network access, `--mode normal` lets the SDK download and execute public games locally. The policy only sees public frame pixels, available actions, game state and level counts. Game caches remain ignored; do not use private games or inspect game rules to train/evaluate the agent.

## Game splits and transition collection

Copy [configs/game-split.example.json](configs/game-split.example.json) and assign only authorized public games. The example holds out `ft09` and `vc33` from collection; these are public local evaluation games, not the competition's hidden set.

```json
{
  "training": {"ls20": 1},
  "evaluation": ["ft09", "vc33"]
}
```

Both splits must be nonempty and disjoint, including version aliases such as `ls20` and `ls20-abc`. Each training value is a positive count of allowed levels, starting at level 1. Collection always reserves the game's last level as reported by public `win_levels`. Its effective cap is the smaller of the configured cap and `win_levels - 1`; an unknown count or a one-level game fails closed.

```sh
python -m arc3.training.collect --split configs/game-split.example.json \
  --seed 0 --max-actions 80 --output recordings/train.jsonl
```

The collector executes each training game once and writes JSONL containing `observation`, `action`, `next_observation`, `seed`, and `source: "real"`. Snapshots copy only the public observation fields. No model predictions enter these records. The step entering a disallowed level is real and counts against the action budget, but its entire transition is discarded before serialization and no further action is taken. With cap 1, for example, the response that first exposes level 2 is never recorded. SDK automatic recording is disabled.

This entrypoint collects data; it does not train weights. It prints a JSON run summary with the seed, SDK version, budget and configured caps. Collection refuses to overwrite or append to an existing output; use a new ignored `.jsonl` path for each run.

## Local evaluation without training

```sh
python -m arc3.evaluation.run --split configs/game-split.example.json \
  --seed 0 --max-actions 80 --output results/evaluation.json
```

Evaluation runs only the configured held-out games and writes JSON metrics without training imports, transition recordings or weight updates. Metrics include state, win status, completed levels, actual policy-issued actions (including resets), actions per completed level, stop reason and elapsed wall time. They are local baseline diagnostics, not an official competition score. SDK initialization performed by `make` is outside the policy action count. Both lifecycles use one `make` per game and stop on a win or the action budget; `GAME_OVER` causes the shared policy to choose a reset on its next turn. Evaluation requires a new `.json` output path.

## Generate a Kaggle notebook locally

```sh
python -m arc3.kaggle.build --seed 0 --max-actions 80 \
  --output artifacts/submission.ipynb
```

The generated notebook embeds the actual `core/policy.py` source and the [official Agents interface](https://github.com/arcprize/ARC-AGI-3-Agents/blob/4743e7d0aaae0ded0d98a89a7e282e63564cd58b/agents/agent.py) adapter. It needs no repository checkout, local game cache or network dependency installation during competition execution. Following the [official Starter pattern](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter/blob/eeb1535404f321d280a8f9194bbc1d7aca5f05fc/scripts/build_notebook.py), it installs from the competition's offline wheels, waits for the internal gateway on a competition rerun, and runs the supplied Agents framework. The gateway enforces competition restrictions and produces `submission.parquet`. During save-and-run, the notebook creates the starter's dummy parquet artifact; that artifact is not a measured score.

The notebook uses CPU and disables internet. Attach the official `arc-prize-2026-arc-agi-3` competition input, which must provide the pinned `arc-agi==0.9.9` wheel and its dependencies plus the Agents framework at the starter's paths. Compatibility with Kaggle's current wheel bundle and gateway must be checked there before any submission; the local smoke suite does not validate those services. Per-game online adaptation is a future extension at `kaggle/agent.py`, with no updates in the baseline.

Kernel metadata is generated **only** when a username is explicitly supplied:

```sh
python -m arc3.kaggle.build --username YOUR_KAGGLE_HANDLE \
  --output artifacts/submission.ipynb
```

This writes `kernel-metadata.json` next to the notebook. Generation never reads credentials, uploads, submits or invokes the Kaggle CLI. Generated notebooks, metadata, recordings, results, game caches, weights and secrets are ignored. Future source changes should be made in Python and followed by notebook regeneration.

## Smoke validation

After installing the package, run the small offline CPU suite:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

The suite uses concrete pinned SDK frame/action types with scripted public responses to check action counts, real transitions, game splits, the reserved-level recording boundary, lifecycle output separation, notebook policy execution, and explicit-username metadata. It downloads no games, makes no network requests and requires no credentials. It does not establish live-game performance, framework/gateway integration, or end-to-end Kaggle success. The inherited `.github` workflows are unchanged; there is no new automatic application CI workflow.

## Evaluation and reproducibility

Separate **real environment steps** (scarce and potentially scored) from **imagined model steps** (compute-limited). Keep a per-game state and replay buffer, and explicitly define reset semantics for offline experiments versus competition runs. Maintain fixed seeds, SDK versions, recorded configuration, action counts, total wall-clock time, and evidence for comparative claims.

Do not use private/hidden games for training, do not inspect source code as an agent shortcut, and do not commit game caches, credentials, model weights, large generated datasets, or notebooks created for submission. Recheck official Kaggle rules before each submission, as competition constraints can change.

## Repository workflow

Read [AGENTS.md](AGENTS.md) for repository-wide invariants and the issue-driven implementation/review process. Local runner provisioning is optional and is **not part of this initialization**.
