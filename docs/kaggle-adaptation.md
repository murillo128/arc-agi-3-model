# Per-game LoRA adaptation

Adaptive Kaggle mode shares one frozen world model and VisualGoal/PathValue
base across the official framework's game threads. Each game has private LoRA
parameters, Adam state, RNG, replay and goals. Updates use only confirmed public
observations from that game; ordinary `arc3.evaluation.run` remains frozen.
This delivery uses development weights and is not the final Phase 2 benchmark.

## Adapt until convergence

The adaptive policy defaults to `stopping="convergence"`, without an action cap,
per-observation learning deadline, per-game learning deadline or maximum number
of optimizer updates. Kaggle/environment termination and explicit cancellation
own the outer lifetime. Each observation triggers representation → predictor →
Path → Goal fitting, using only that game's retained real evidence. The base
weights remain frozen. Adapters and Adam state persist across actions and levels.

Each stage fixes its examples for this observation and measures the entire set
before learning and after each minibatched sweep. Newest examples are included;
checks never use a changing random minibatch. Latents and causal memory are
rebuilt with the current adapters before each predictor update and measurement.
Encoder/decoder remain fixed during predictor/Path/Goal fitting. The default
thresholds are fractions (0.10 = 10%), independently configurable:

| Configuration key | Measured fitting error |
| --- | --- |
| `reconstruction_threshold` | Equal-weight-per-color reconstruction error; replay mean and latest frame must both pass |
| `delta_threshold` | Wrong pixels / union of actual and predicted changes; pooled replay and latest eligible transition must both pass |
| `goal_color_threshold` | Color error inside the target mask, balancing changed and unchanged regions |
| `goal_mask_threshold` | Mask classification error, balancing included and excluded pixels |
| `path_reachability_threshold` | Mean absolute reachability probability error against observed labels |
| `path_cost_threshold` | Mean absolute action-distance error / max(1, actual distance), successful labels only |

The goal stage needs both its thresholds; Path needs both when successful labels
exist. An empty delta union is reported as `no_change` with null delta error, not
as evidence of dynamic prediction accuracy. A stage without usable data reports
`no_data` and does not delay action selection. Hindsight uses one deterministic
reached target per replay origin, preferring evidence of native progress and
then the latest reachable state in that scene. Reachability is not a winning
label. Path also retains the active goal's observed success/failure outcomes.

There is no update-count ceiling in convergence mode. To avoid waiting forever
for inconsistent or insufficient evidence, `plateau_patience=8` full-replay checks
without `plateau_min_relative_improvement=0.001` (0.1%) improvement in the same
replay objective stop that stage. Improvements accumulate against the last
meaningful improvement; the best exact loss seen is tracked separately. On a
plateau, restore the best adapters **and Adam state**, report `plateau` and
`threshold_met=false`, and acquire another observation. A decreasing loss can
keep training beyond any fixed number of sweeps. This does not guarantee every
threshold is attainable or that a fitted transition will generalize.

`diagnostics.convergence` records each stage's stop reason, updates, full checks,
example count, initial/final fitting errors and loss, best loss, restoration and
time. These are fitting measurements on already-seen observations. The runner's
prediction-before-SDK-result measurements remain separate. Nonfinite updates
roll back; cooperative `policy.request_stop()` or Ctrl+C can interrupt fitting.
An already-admitted real result remains confirmed even if its SFT is cancelled.

Optional positive `max_actions`, `per_observation_seconds` and `total_seconds`
are still supported explicitly. `stopping="steps"` reproduces the earlier bounded
update schedule. In convergence mode, positive `*_steps` values do not cap work;
zero disables that stage. `goal_lifetime`, `horizon`, `beam` and `clicks` still
control goal/search scope and are not total game action budgets. The standalone
smoke and generated save-and-run fixture explicitly use `stopping="steps"`; the
scored adaptive agent uses the requested convergence configuration.

## Manual public held-out experiment

`model.online_lora_experiment` is a separate, explicitly authorized online
experiment, not `arc3.evaluation.run` or the smoke below. It runs only the
adaptive Kaggle policy, sequentially on the five evaluation games in
`configs/splits.public-development.json`, with seed 42. By default it runs until
WIN, cancellation or an external failure, with no artificial action/time caps.
The original experiment in issue #153 used 80 actions, 10 s/observation,
300 s/game adaptation and a 900-second wall deadline; its results remain a record
of that earlier protocol. The new defaults do not rerun or reinterpret it. There
is no automatic launch, submission, parameter search or transfer between games;
launching a new held-out experiment remains an explicit operation.

With the authorized batch-8246 portable checkpoint available locally:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u \
  -m model.online_lora_experiment \
  --checkpoint artifacts/online-lora-20261028/checkpoint/base.pt \
  --output-dir artifacts/online-lora-convergence/run \
  --environments-dir artifacts/online-lora-20261028/environment_files \
  --mode normal --threads 2
```

Use `--adaptation-config path/to/config.json` for threshold/plateau overrides.
Optional `--max-actions N` and `--max-game-seconds S` enable explicit experiment
caps; both default to no limit. To reproduce the previous bounded protocol,
combine `--max-actions 80 --max-game-seconds 900` with configuration
`{"stopping":"steps","per_observation_seconds":10,"total_seconds":300}`.

The runner rejects an existing output directory, changed split, a
checkpoint hash other than
`4d125575cdc126044ec9b2e773252237cb9743c36032a08d4857f4d876150db7`,
or missing CUDA. NORMAL mode provisions each game through the official SDK at
its single initialization. Only public observations enter the policy. Checkpoint
and base digests, pristine per-game optimizer/replay state, an unused isolated
adapter bank, effective precision and GPU peak allocation/reservation are recorded.
The unused bank never chooses actions or supplies a comparison score.

Predictions are materialized before `step`, and compared to the real return
before `confirm_result` learns from it. Delta error is the sum of missed,
invented and wrong-color changes divided by the union of real/predicted changed
pixels; an empty union is unavailable. RESET and scene boundaries are separate.
Reconstruction on each new observation is measured before admission; the existing
adaptation pre/post losses describe already-seen samples. First/last windows use
up to ten eligible transitions and expose action indices, levels and overlap.
They are descriptive, not a causal comparison of equally difficult transitions.

Every returned action is logged, including no-ops, failures, terminal results and
level changes. The SDK's official scorecard supplies RHAE and per-level action
counts, including failed attempts; unavailable references produce null RHAE.
SDK 0.9.9 counts partial RESET in both actions and resets; a full initialization
starts a scorecard run at zero actions. The experiment counts every agent request,
including RESET, while `make`'s initial reset is reported separately. When a wall
limit is explicitly supplied, time is checked before selection and dispatch. Already-running GPU/SDK operations and their
confirmation finish; measured overruns remain visible. Game time includes policy
creation, provisioning, diagnostics and learning; initial shared-base loading and
final integrity verification are separately timed. The adapter implementation's
learning timer excludes outer adapter transfers, so confirmation wall time is
also recorded. Optional exhausted adaptation budgets still permit inference.
Cancellation saves partial scalar results and prevents starting the next game.

JSON, CSV and flushed scalar JSONL evidence stay in ignored `artifacts/`. No
observations, adapters, weights or game sources are exported. Partial records
survive action/adaptation failures. These development-weight public-game results
are not official Kaggle scores.

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
   weights without changing notebook source. Portable runtime loading copies the
   read-only source in one sequential pass with a 1 MiB buffer into a local
   `TemporaryFile`, then deserializes that file eagerly on CPU. The mounted source
   needs neither seeking nor memory mapping. Strict key, shape, dtype and
   finite-value checks still precede use. Allow local temporary disk space roughly
   equal to the source size; temporary storage is closed/deleted on success and
   failure, without writes to the input directory or a persistent weight cache.
   The deserialized payload is released after copying into the frozen base; allow
   host memory for that payload plus one base and per-tensor validation scratch.
   Games continue to share one base keyed by the original source identity and
   device while their adapters remain independent.

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
   Flushed stderr events distinguish `loading`, `loaded`, `checking`, `learning`,
   `verifying` and `complete`; the public-SDK path identifies each learning game.
   The report's `runtime` records the selected device, PyTorch version, effective
   precision, native BF16 selection, parameter dtype and
   `checkpoint_loading="staged_sequential_eager"`.
   `timings.load_seconds` measures the first policy load, including sequential
   staging, deserialization, validation, device transfer and its adapter creation
   when the base is not already live.
   `startup_seconds` includes that load plus initial integrity/isolation checks;
   `interaction_seconds` covers inference, adaptation and SDK/game setup where
   applicable; `verification_seconds` covers the final integrity/isolation checks.
   Per-game `adaptation_seconds` remains the learning-budget diagnostic. These
   smoke timings exclude interpreter/import time and are not a training benchmark.

3. Generate the offline notebook using the weight dataset's intended mount:

   ```sh
   python -m arc3.kaggle.build --adaptive \
     --agent-checkpoint /kaggle/input/your-private-base/base.pt \
     --agent-dataset your-handle/your-private-base \
     --extra-wheel-dir /kaggle/input/your-private-base \
     --username your-handle --accelerator cpu --cpu-threads 2 \
     --output artifacts/adaptive-submission.ipynb
   ```

   Uploading a private weight dataset and notebook is a separate explicit
   operation owned by the chat controller. Attach that dataset and the official
   competition inputs in Kaggle. Verify the mounted filename matches the
   generated path. Include the official MIT-licensed
   `arc_agi-0.9.9-py3-none-any.whl` beside `base.pt` in that private asset.
   The inspected competition bundle contains SDK 0.9.8; it cannot satisfy the
   project's 0.9.9 pin by itself. `--extra-wheel-dir` names the mounted directory
   containing companion wheels (not an individual wheel). Installation searches
   both this directory and the competition's `arc_agi_3_wheels` using
   `--no-index --find-links ... --find-links ...`. Missing directories or required
   dependency wheels stop execution with an attachment error; there is no
   internet fallback or SDK downgrade. This option also works for random/frozen
   notebooks. It does not upload a dataset or alter its contents.
   `--accelerator t4`, `p100` or `rtx6000` selects CUDA; the
   adaptive runtime uses BF16 only with native support on the selected CUDA
   device, explicitly querying `is_bf16_supported(including_emulation=False)`.
   T4, P100 and CPU use FP32. The default PyTorch query also permits emulation;
   see [the API](https://docs.pytorch.org/docs/main/generated/torch.cuda.is_bf16_supported.html).
   This selection applies only at the adaptation boundary and leaves production
   training precision unchanged. There are no runtime internet/package downloads: SDK installation
   uses only the explicitly mounted local wheels.

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
32 replay transitions, minibatches of at most 4, convergence stopping with the
thresholds above, and 25% exploration. Timers default to null; when supplied,
they are checked between optimizer steps/check chunks and permit inference after
expiry. A running PyTorch kernel cannot be preempted. Frozen/random notebooks
retain their 80-action default. Adaptive notebooks omit the default cap and use
a compatible unlimited loop over the supplied framework's conversion, action
request, frame validation and recording methods; no large integer substitutes
for unlimited mode.

Adapters persist through RESET and level changes, but those scene boundaries
never become predictive pairs and clear the active spatial goal. Recurrent
context is rebuilt with current adapters from the bounded retained history,
starting at zero at its left edge and using boundary markers. Hindsight masks
come only from changed same-level states actually reached; they indicate
reachability, not proof of winning. Pending goal segments are censored rather
than treated as failures. Only group=1 Conv2d and Linear targets are adapted;
the smoke report lists supported targets and excluded grouped/depthwise layers.
Minibatch loss diagnostics retain the most recently measured pre/post values for each
phase; skipped phases do not imply an optimizer step. Nonfinite steps roll back
the affected adapters and Adam state. After two numerical failures, learning
stops for that game while legal exploration remains available.

Omit `--adaptive` to preserve the existing frozen `arc3-agent` checkpoint mode;
omit both it and `--agent-checkpoint` for the seeded random baseline. Portable
adaptation bases and frozen-agent checkpoints have different validated formats.

The earlier direct-eager loader was compared with the previous mapped path using
five alternating CPU runs per mode on a small deterministic fixture: all
world/goal/path tensors matched exactly. Median load plus
validation/materialization was 15.07 ms direct eager versus 14.69 ms mapped. These
historical local warm-cache timings do not measure staged loading or establish a
mounted-filesystem or production-weight speedup. Eager loading materializes tensor
storages before validation; see [PyTorch's loading semantics](https://docs.pytorch.org/docs/stable/generated/torch.load.html).
Sequential staging now keeps the deserializer's seek-based reads on the local
temporary file. Its focused boundary regressions use a source reader that forbids
seeking/mapping and records bounded reads, checking equivalent loaded weights,
source immutability and temporary-storage cleanup for valid, malformed, nonfinite
and truncated payloads.

Existing regressions retain portable schema rejection (including missing/extra
keys, shape, dtype, NaN and infinities), native versus emulated BF16 on a selected
device, CPU fallback and diagnostics in the isolated offline generated package.
Accelerator capability checks use CPU test doubles; actual remote
accelerator/mount checks remain a separate controller operation. The earlier evidence below describes the bounded protocol that was executed;
it is not evidence of convergence-mode game performance.

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

## Offline runtime compatibility correction

The companion-wheel installer was validated in a clean Python 3.12 environment
with socket connections disabled. The 31 development wheel fixtures were fetched
from PyPI and matched every filename and byte size in the supplied competition
inventory; the competition wheel archive itself was not available locally.
The separately supplied SDK wheel was used unchanged.

- The previous generated install cell failed because the official inventory
  provides only SDK 0.9.8, demonstrating the original deployment defect.
- The corrected cell installed SDK 0.9.9, arcengine 0.9.3, NumPy 2.4.4 and
  python-dotenv 1.2.2 using both local sources and no network.
- Removing arcengine from the local wheel fixture failed with the new explicit
  offline-dependency error. There was no fallback install attempt.
- Nineteen focused installer, baseline and adaptive/frozen packaging tests
  passed. The six installer tests cover generated arguments, the SDK pin, CLI
  wiring, mounted-path validation and missing mount/dependency diagnostics.

The supplied `agents/agent.py` and `agents/swarm.py` were executed directly with
the generated adaptive agent and the versions above. The probe used their actual
main loops, frame conversion and two Swarm threads. It confirmed one terminal
result exactly once, a second game's exact two-action budget, independent click
payloads, unchanged base weights, and matching concurrent/isolated adapter states.
Agent SHA256: `85655f7a11a105e50bba76efd00bebf8056d02ddbae827a8f8a5b46e849fc087`;
Swarm SHA256: `d9dc48f710f1b90a6552db0921293c7e89c8a925ed00a3faefa07ae19998ad39`.

The local supplied framework fixture lacks `recorder.py` and `tracing.py`; the
probe substitutes only those ancillary recording/tracing helpers and a scripted
SDK gateway. It reports three simulated confirmations and zero real game
transitions. It does not validate live gateway protocol, scoring or the absent
helpers. To reproduce with the supplied framework directory (or a complete
runtime bundle, whose helpers will be used when present):

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python \
  -m model.tests.check_kaggle_framework \
  --framework /path/to/competition-input/ARC-AGI-3-Agents \
  --output artifacts/kaggle-lora-smoke-20261028/runtime-compatibility/framework-report.json
```

The installer report, actual pip logs, focused test log and framework report are
retained in ignored `artifacts/kaggle-lora-smoke-20261028/runtime-compatibility/`.
Earlier production-weight adaptation evidence remains applicable: this correction
changes installation and its validation, not model weights or learning behavior.
