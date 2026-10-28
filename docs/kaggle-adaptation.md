# Per-game LoRA engineering smoke

Adaptive Kaggle mode shares one frozen world model and VisualGoal/PathValue
base across the official framework's game threads. Each game has private LoRA
parameters, Adam state, RNG, replay and goals. Updates use only confirmed public
observations from that game; ordinary `arc3.evaluation.run` remains frozen.
This delivery uses development weights and is not the final Phase 2 benchmark.

Use Python 3.12 with the repository dependencies, including `arc-agi==0.9.9` and
PyTorch. These commands generate local artifacts; none uploads or submits them.

1. Export a completed sequence-training snapshot to a separate portable asset:

   ```sh
   python -m model.adaptation_checkpoint \
     --source artifacts/kaggle-lora-smoke-20261028/base-source.pt \
     --output artifacts/kaggle-lora-smoke-20261028/base.pt --threads 2
   ```

   The source is read-only. The export contains world/head weights, strict
   inference configuration and compact provenance, without trainer optimizers,
   replay, recorded frames, RNG or paths. Optional ChangeHead export is not part
   of version 1. Replace `base.pt` with a future compatible export to change
   weights without changing notebook source.

2. Run the local public-SDK smoke on CPU, separately from any GPU trainer:

   ```sh
   OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m model.adaptation_smoke \
     --checkpoint artifacts/kaggle-lora-smoke-20261028/base.pt \
     --output artifacts/kaggle-lora-smoke-20261028/local-smoke.json \
     --device cpu --threads 2 --actions 4 --environments-dir games
   ```

   This uses only public observations from training-authorized `sc01-v1` and
   `si01-v1` native level 1. A result entering another level is excluded before
   adaptation. No oracle goals, private game implementations, held-out games or
   training recordings are read by the agent. Inspect `passed`, real transition
   counts, finite pre/post losses, adapter/base bytes and exact base/isolation
   checks in the JSON. Game progress and goal outcomes are separate diagnostics.

3. Generate the offline notebook using the weight dataset's intended mount:

   ```sh
   python -m arc3.kaggle.build --adaptive \
     --agent-checkpoint /kaggle/input/your-private-base/base.pt \
     --agent-dataset your-handle/your-private-base \
     --username your-handle --accelerator cpu --cpu-threads 2 \
     --max-actions 80 --output artifacts/adaptive-submission.ipynb
   ```

   Uploading a private weight dataset and notebook is a separate explicit
   operation owned by the chat controller. Attach that dataset and the official
   competition inputs in Kaggle. Verify the mounted filename matches the
   generated path. `--accelerator t4`, `p100` or `rtx6000` selects CUDA; the
   runtime uses BF16 only when the device supports it, otherwise FP32. CPU is
   supported. There are no runtime internet/package downloads: SDK installation
   uses the competition's local wheels with `--no-index`.

4. Run Kaggle **Save & Run All** and inspect
   `/kaggle/working/adaptation-smoke.json` (configurable with `--smoke-output`).
   Before writing the starter dummy parquet, the notebook launches its embedded
   source in a subprocess and actually adapts the mounted model on a tiny
   simulated palette fixture. A failed self-check fails notebook execution.
   Its `kind` is `offline-fixture-self-check` and `real_sdk_transitions` is zero:
   it validates packaging and optimization, not game performance. A dummy
   `submission.parquet` alone is never agent validation.

5. An optional scored competition rerun is a later explicit submission. On that
   rerun the supplied official framework and gateway own game lifetime and
   scores; this notebook does not generate competition scores. Do not report
   save-and-run or the local smoke as the final Phase 2 competition benchmark.

`--adaptation-config path/to/config.json` accepts a JSON object of
`model.adaptation.AdaptationConfig` overrides. Defaults include rank/alpha 4,
32 replay transitions, minibatches of at most 4, up to 2 representation steps
and 1 predictor/path/goal step per confirmed observation, a 10% color-macro
reconstruction early-stop threshold, 10 seconds per observation, 300 seconds
total per game, and 25% exploration. Step/time budgets are validated. Elapsed
limits are checked between optimizer steps; a running PyTorch kernel cannot be
preempted. Exhausted learning budgets still permit inference and exploration.

Adapters persist through RESET and level changes, but those scene boundaries
never become predictive pairs and clear the active spatial goal. Recurrent
context is rebuilt with current adapters from the bounded retained history,
starting at zero at its left edge and using boundary markers. Hindsight masks
come only from changed same-level states actually reached; they indicate
reachability, not proof of winning. Pending goal segments are censored rather
than treated as failures. Only group=1 Conv2d and Linear targets are adapted;
the smoke report lists supported targets and excluded grouped/depthwise layers.
Loss diagnostics retain the most recently measured pre/post values for each
phase; skipped phases do not imply an optimizer step. Nonfinite steps roll back
the affected adapters and Adam state. After two numerical failures, learning
stops for that game while legal exploration remains available.

Omit `--adaptive` to preserve the existing frozen `arc3-agent` checkpoint mode;
omit both it and `--agent-checkpoint` for the seeded random baseline. Portable
adaptation bases and frozen-agent checkpoints have different validated formats.

The transport follows the inspected official
[Agent lifecycle](https://github.com/arcprize/ARC-AGI-3-Agents/blob/4743e7d0aaae0ded0d98a89a7e282e63564cd58b/agents/agent.py)
and [threaded Swarm](https://github.com/arcprize/ARC-AGI-3-Agents/blob/4743e7d0aaae0ded0d98a89a7e282e63564cd58b/agents/swarm.py):
one agent per game, a commit immediately after a real SDK return (including
terminal results), and no mutation of the shared GameAction enum payload.
The two Kaggle phases follow the
[official starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter/tree/eeb1535404f321d280a8f9194bbc1d7aca5f05fc).

## Local engineering evidence, 2026-10-28

Batch 8246 exported to an 867,909,774-byte portable file. The source snapshot
SHA256 remained `3eca870ddf0428419c9c3cead2926da8b52a293dddea958283508730c1145222`
after validation. The CPU smoke used two threads, four requested actions per
game, 30 seconds per observation and 120 seconds total adaptation allowance per
game; all other adaptation settings were defaults. It completed in 82.85 seconds.

| Observation | Result |
| --- | --- |
| Shared world plus goal/path base | 867,745,928 bytes |
| Private adapters per game | 6,744,096 bytes, excluding Adam state |
| Real training-authorized transitions | sc01: 3; si01: 4 |
| Excluded next-level results | sc01: 1; si01: 0 |
| Optimizer steps | 14; no numerical failures |
| Base and inactive-game checks | Exact base digest, inactive adapter digest and inactive encoder output unchanged |
| Adapter targets | 198 Linear/group=1 Conv2d; excluded `predictor.rotation_adapter.spatial` (64 groups) |

The exact world/head digest before and after was
`9554d9ffcae09cda4a9cfb9b9018bed7dc04b737ea505e26123aab2cf5abd070`.
Latest sc01 predictor/path/goal losses were respectively
0.738342→0.725450, 4.526569→4.521645 and 3.695382→3.693964.
Latest si01 predictor/path losses were 0.004807→0.004675 and
0.021115→0.021100. Both games' measured color-macro reconstruction error was
zero, so the threshold correctly skipped representation steps. si01 had no
changed hindsight goal sample and reported that skip; its path label came from
the real active-goal timeout. These observations are not evidence of general
game-solving quality.

The generated source also ran against these production weights in a subprocess
with network access and editable-install fallback disabled: 5.13 seconds, six
optimizer steps, finite diagnostics, unchanged base/inactive game. Its two
simulated confirmations are explicitly separate from real SDK transitions.
The generated notebook, portable base and complete JSON reports remain outside
Git in `artifacts/kaggle-lora-smoke-20261028/` (`adaptive-submission.ipynb`,
`base.pt`, `production-smoke.json`, `packaged-smoke.json`). No Kaggle upload,
remote run or competition submission was performed.

Focused validation passed 65 tests in 10.36 seconds, followed by the nine affected
smoke/packaging tests after tightening numerical-failure reporting. This includes
zero-effect adapters, immutable bases, interleaved/concurrent game isolation,
real confirmation and terminal handling, replay/time limits, current-weight
memory, hindsight and censored labels, portable schema rejection, and isolated
random/frozen/adaptive packaging. The fixed-seed small sc01 real-SDK test exercises
all four learning phases with strictly decreasing losses. Existing sequence-head
and masked-goal tests remain in the same focused run:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest \
  model.tests.test_adaptation model.tests.test_lora \
  model.tests.test_adaptation_checkpoint model.tests.test_adaptation_smoke \
  model.tests.test_kaggle_adaptation model.tests.test_sequence_planning \
  model.tests.test_agent model.tests.test_masked_path_goal \
  model.tests.test_masked_path_training tests.test_smoke -v
```

Compilation and `git diff --check` passed. No browser tests or broad training
benchmark were run. Repository Actions only route executor workflow regressions
for infrastructure paths; none of those paths changes in this delivery, so no
application CI run is expected from the current workflow filters.
