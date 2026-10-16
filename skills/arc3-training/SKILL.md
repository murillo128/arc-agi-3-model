---
name: arc3-training
description: Run a configured ARC-AGI-3 model curriculum from scratch, monitor or resume it, validate its final weights, and publish passing weights as GitHub Release assets when requested. Use for this repository's training runs, not architecture redesign or Kaggle submission.
---

# Train and publish an ARC-AGI-3 model

Use the repository's versioned JSON training interface. Read [the configuration contract](../../model/README.md#entrenamiento-configurable) and [training policy](../../docs/training-policy.md) before choosing games or interpreting success. This skill is intended to be read from this repository's `skills/arc3-training` directory.

## Establish the run

Confirm the requested games, level order, thresholds, development/final phase and output directory from the user's configuration. If absent, propose a concrete configuration from `model/training_config.example.json`; ask only for choices that materially affect the run. `model/training_config.sdk.example.json` demonstrates the SDK boundary; its example split is not the full official 25-game split. Use the actual accepted development split, with seed-42 game holdouts and final-level reservations. Never use private/held-out evaluation data for development training.

All 70 bundled synthetics support direct level isolation. Only `sc01-v1` and `si01-v1` have the trainer-owned visual subgoal teacher (`objective: subgoals`). The other 68 synthetics and public SDK entries use `objective: world_model`: representation/dynamics with random actions, without training the synthetic-goal planner. SDK levels must be reached through public native actions within `sdk.access_steps`; the SDK offers no portable direct level setter. Do not inspect SDK game internals, introduce a solver or silently replace an unreachable stage. Report the access failure and revisit the configured budget/sequence with the user.

Work from a clean, committed source revision and record its SHA. Keep weights, game files, recordings, logs and release packages in ignored run/artifact directories. Do not stop an existing service, delete its output, reset its weights or reuse its output without user authorization. Confirm the chosen device is available; retain the CPU path for small configurations. A full production configuration may be too large for CPU.

Validate without creating a model or starting a run:

```sh
.venv/bin/python -m model.subgoal_training --training-config model/training_config.example.json --validate-config
```

Paths inside a training file are relative to that file. Edit a copy deliberately; validation rejects unknown keys, invalid levels, duplicate pairs and inconsistent development splits. Settings and expanded stage order are recorded in the run. Stage thresholds override global thresholds. There is no dashboard configuration editor.

## Start, monitor and resume

For a scratch run use a fresh output, with no `--source` and no `--resume`:

```sh
.venv/bin/python -u -m model.subgoal_training \
  --training-config model/training_config.example.json \
  --output model/runs/my-training --device cuda
```

Choose a durable process/service mechanism available on the host for a long run; do not silently replace a running service definition. The output lease prevents concurrent configured trainers. Network initialization and all active components start from random weights. `phase: final` follows the policy's fresh-initialization rule, never a Phase 1 checkpoint transfer.

Monitor `report.json` for active stage, quality, native/synthetic success, access-action cost, recent throughput and errors; expose it through the existing dashboard's `--subgoal-report` argument if requested. SDK quality convergence is not autonomous game-solving success. The report distinguishes `converged`, `budget_exhausted` and `failed`. Manual phase/level advances are recorded and make the run ineligible for automatic successful-weight publication.

SIGUSR1 requests an atomic checkpoint at the next episode/evaluation boundary. Verify `last_checkpoint` advances before stopping an authorized trainer. Resume with the identical training file and output:

```sh
.venv/bin/python -u -m model.subgoal_training \
  --training-config model/training_config.example.json \
  --output model/runs/my-training --device cuda --resume --actions 524288
```

`--actions` is an optional explicit extension of the total action budget, not additional actions. Weights, cursor, counters and phase permissions survive; optimizer moments and RAM replay restart, as documented by the existing trainer. The plan/threshold contract cannot change on resume. Preserve the original source revision. Never treat a restart, exhausted budget, process exit zero or manual advance as convergence. If interrupted/failed, inspect the checkpoint/report boundary before retrying; preserve evidence and never delete the run to make a check pass.

## Validate and prepare weights

Only a fully converged plan can qualify. Every configured stage must pass final frozen validation using seeds separate from development. `subgoals` stages also require planner success; `world_model` stages require predictive quality and report native completion separately. Drift/patience are measured during development qualification, not fabricated from one frozen checkpoint. Final evaluation performs no updates or training replay writes.

Run the applicable repository-native CPU/model checks at the trained source revision. At minimum use the configuration/release owner plus the full model suite for a production release; inspect failures rather than ignoring them. Honor the user's no-visual-test preference. Confirm the source SHA exists on the intended GitHub repository and that local source/dependencies still match the recorded run. Retain concise test evidence for release notes.

```sh
PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s model/tests -v
.venv/bin/python -m model.training_release \
  --run model/runs/my-training --output model/runs/my-training-release
```

Preparation fails closed for incomplete/manual runs, missing or failed final evidence, changed plan/checkpoint identity, dirty source provenance, or nonfinite/unloadable weights. Do not bypass these gates or edit reports to qualify. It exports only `agent.pt` (portable inference configuration and weights, no live carry/training data), `manifest.json` (source, sanitized configuration, evidence) and `SHA256SUMS`. Inspect that allowlist and verify checksums before publication; no game frames, replay, environment source, caches, private local paths or credentials may be attached. Do not commit weight blobs to Git.

## Publish on GitHub when authorized

If the user's task includes publishing passing weights, that authorization covers publication after these gates; do not ask again. Otherwise finish local preparation and request publication approval only with the exact repository, new tag, files, metrics and limitations ready to inspect. Creating this skill or reviewing a configuration alone does not start training or authorize uploading weights.

Use an existing authenticated `gh`, the intended repository and a new release tag anchored to the **trained source SHA**. Confirm neither tag nor release exists; distinguish an actual 404 from authentication/network failures. Never use `--clobber`, replace a release or retarget an existing tag silently. Write multiline release notes to a file, including source/plan identities, configuration coverage, native vs teacher-assisted scope, validation results, action count and checksums.

Create a draft with the three prepared assets, verify the uploaded inventory/size and download to a separate temporary directory to check SHA256SUMS, then publish the verified draft. Typical commands (replace placeholders with verified values):

```sh
gh release create TAG ARTIFACT_DIR/agent.pt ARTIFACT_DIR/manifest.json ARTIFACT_DIR/SHA256SUMS \
  --repo OWNER/REPO --target SOURCE_SHA --title TITLE --notes-file NOTES_FILE --draft
gh release download TAG --repo OWNER/REPO --dir VERIFY_DIR
# Run sha256sum --check SHA256SUMS inside VERIFY_DIR and compare it to the local manifest.
gh release edit TAG --repo OWNER/REPO --draft=false
```

An upload/download/verification failure leaves the release draft; report the concrete error and resume only the same owned draft after inspecting its state. Do not repeatedly retry unknown failures or publish partial assets. Verify the final release is public/published as intended and provide its URL. These are local training weights, not a Kaggle submission or an independent competition benchmark.
