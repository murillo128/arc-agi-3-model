# Optional pretrained segmentation and object dynamics

This experimental sequence-training extension preserves the existing world model,
Goal and Path weights. It is **off by default** (`object_slots: 0`). It has not
demonstrated better game completion or held-out prediction quality. The live
training run was left unchanged during its initial audit.

## Data and model boundary

The teacher is frozen [SAM 2.1 Tiny](https://github.com/facebookresearch/sam2),
revision `2b90b9f5ceec907a1c18123530e92e794ad901a4`. Its Apache-2.0 weights are
external assets, never included in Git or downloaded during execution:

- Official asset: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt`
- SHA-256: `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`

SAM receives only real observed palette frames. Nearest-neighbor enlargement
preserves their edges. Its prompts combine a uniform grid and at most 32 points
inside small connected color regions; these regions are **prompts, not labels**.
The pretrained model still determines the masks. No game rules, hidden state or
reserved-level observations enter the teacher. The normal collector retains its
existing split and level boundaries.

Masks return to native pixel coordinates. Low-confidence, duplicate, empty and
almost-full-frame masks are filtered. When capacity is exceeded, selection spans
multiple area scales so that tiny parts cannot evict every larger object.
Overlapping and nested masks remain possible. Missing proposals are unknown;
neither the teacher nor its confidence establishes complete scene coverage.

The small learned head reads the pixel-structured latent, pools features inside
current masks, and conditions transport/refinement and presence on the real
action and causal recurrent memory. Empty slots have discovery queries. A
current-mask reader lets it operate without SAM during imagination. One-step and
three-step forecasts use their own predicted latents, masks and memory; later real
frames provide targets only. Windows cannot cross resets or level boundaries.

Supervision uses permutation-aware assignment, per-object Dice and separately
averaged foreground/background BCE. A one-pixel object therefore receives a
meaningful omission penalty. Matched presence is positive, while unmatched
presence has a weak weight of 0.05 because SAM can omit an object. An entirely
empty teacher result contributes no mask supervision. Mask order is not a stable
object identity. The usual visual objective is retained; segmentation is not a
replacement for pixel/color prediction.

## Consumers and warm-up

- The click helper proposes actual interior pixels, including in non-convex masks.
  The existing learned change head scores real proposals. Ordinary legal actions
  and uniform exploration remain available.
- After activation, future predicted masks can relocate branch-specific click
  candidates. At least a quarter of these clicks remain uniform. SAM is never
  called for an imagined branch.
- Predictor, Path and Goal receive an added object residual/context. Both fitting
  and planning apply the same context. Goal still outputs a final palette image
  and relevant-pixel mask; its supervision and Path's success/cost semantics are
  unchanged.

The new output projection and fusion gain start at zero. Migration retains every
old weight and binds existing Adam moments by parameter name. Auxiliary steps
freeze the original modules and update only the new head, using the same
optimizer's new parameter states at a temporary learning rate. Normal sequence
predictor learning continues at its original rate. Encoder/decoder rates and
checkpoint state are unchanged.

`activate_residual: false` keeps fusion disabled while collecting evidence.
With explicit activation requested, fusion can start only after the configured
number of auxiliary updates and current/one-step/three-step matched-mask IoU all
meet `activation_iou`. These are **fit metrics**, not a guarantee of generalization
or improved visual predictions. Keep a comparison run before promoting a model.

## Installation and configuration

Use a separate Python 3.12 environment for the optional dependency. Do not modify
the environment of a running trainer. Install PyTorch first, then the pinned SAM
source without build isolation; its optional CUDA extension is unnecessary here:

```sh
python3.12 -m venv .venv-objects
.venv-objects/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 setuptools
SAM2_BUILD_CUDA=0 .venv-objects/bin/python -m pip install --no-build-isolation '.[model,segmentation]'
```

Download the official checkpoint separately and verify the SHA above. The provider
also verifies it on startup. CPU is supported. With `device: "cuda:0"`, the frozen teacher is kept on CPU
when idle. Each uncached, budgeted call transfers it to the selected GPU, runs
BF16 inference, returns native CPU labels, clears predictor image features and
moves the teacher back to CPU in a `finally` block. Unused CUDA allocator memory
is released before and after the call. Cache hits and exhausted budgets do not
transfer the model. CUDA requires native BF16 support; there is no silent CPU or
precision fallback. Exceptions propagate after cleanup.

Labels are generated outside the world model's backward pass, but the model and
Adam still occupy VRAM. Check actual memory before adoption; temporary residency
does not guarantee that every GPU fits. Teacher timing includes transfer,
synchronization and cleanup. Its report identifies inference device, precision
and idle residency. CPU execution keeps its existing FP32 behavior. The pinned
SAM implementation also holds positional caches and rotary tensors outside
registered buffers: these are cleared/offloaded explicitly. Positional warm-up
is disabled for CUDA loading; upstream rotary initialization may briefly allocate
on the selected GPU, but no live teacher tensors remain there when construction
returns. The device probe measures its baseline before constructing SAM.
No held-out game is needed for setup or the audit.

Make a copy of the current training and initialization JSON files, preserving their
other settings and relative paths. Add `object_slots: 32` and `object_width: 16`
to the initialization's `network`. Add this top-level training section (a modest
CPU sampling allowance; tune after measuring):

```json
"sequence_objects": {
  "checkpoint": "/absolute/path/sam2.1_hiera_tiny.pt",
  "device": "cpu",
  "samples": 2,
  "passes": 2,
  "learning_rate": 0.001,
  "teacher_frames": 6,
  "click_frames": 1,
  "cache_frames": 256,
  "points_per_side": 8,
  "warmup_updates": 64,
  "activation_iou": 0.5,
  "activate_residual": false
}
```

The teacher has separate per-batch call allowances for collection and learning.
An exhausted allowance falls back to the existing click policy or skips an
unlabeled training example. Cached real frames cost no new inference calls. The
LR in this section applies only to auxiliary updates of new parameters.

To resume a **frozen-representation sequence checkpoint**, migrate into a new
directory, leaving the old run intact, and then use the normal resume path:

```sh
python -m model.object_upgrade --source /old-run/latest.pt \
  --training-config /configs/train-objects.json --output /new-object-run
python -m model.sequence_training --training-config /configs/train-objects.json \
  --output /new-object-run --resume --device cuda --unlimited
```

Migration rejects unrelated architecture, curriculum, phase and LR changes. To
request fusion later without discarding learned masks, change **only**
`activate_residual` to `true` in a copied config and migrate the object run into
another new directory with the same command. This retains its masks and Adam
states and requests the warm-up/IoU gate; it does not force activation.

To change the teacher device on an already-enabled run, copy its training config
and change **only** `sequence_objects.device`. Use the explicit migration flag:

```sh
python -m model.object_upgrade --source /old-object-run/latest.pt \
  --training-config /configs/train-objects-cuda.json --output /new-object-run \
  --change-teacher-device
```

This preserves object weights, gain, update counters, every optimizer and all
other learned/control state. It cannot simultaneously change activation, labels,
rates, budgets, games or architecture. Resume the new output normally. Keep the
source run and copy its metric history when switching the dashboard.

At a saved batch boundary with the trainer stopped, a one-frame device check can
use the report's real training sample. It writes a new local evidence file and
checks native masks, frozen weights, and release of GPU tensors:

```sh
python -m model.bench_sam_device --report /old-object-run/report.json \
  --device cuda:0 --output /tmp/sam-device-check.json
```

It measures teacher memory only; a completed resumed training batch is the
co-residency check. No held-out observations or comparative training are needed.

The dashboard adds an experimental panel only when object measurements exist:
real frames, SAM masks, predicted next masks, current/one/three-step IoU, auxiliary
loss and teacher cost. Preview colors are local mask indices, not cross-frame
identity. The architecture describes the optional branch; LMEX binds its tensor
inventory only when exporting an enabled checkpoint. Generated Kaggle packages
include learned inference code, without SAM dependencies or checkpoint downloads.
Packaging alone does not enable teacher adaptation in the competition agent.
Portable v1 bases saved before this extension still load with objects disabled;
only the missing `object_slots` and `object_width` fields receive defaults. Enabled
exports retain the scalar int64 auxiliary-update counter as well as learned
weights and fusion gain. Loading rejects negative counters, incorrect tensor
shapes/dtypes and nonfinite weights; it does not activate fusion.

## Initial audit — 2026-10-29

Four normal training observations (ft09, ka59, sb26 and mc01) were segmented with
the official checkpoint on CPU, four threads. A uniform 8×8 prompt grid took
3.19–3.48 s/frame. On mc01 it retained only one region. Adding small-region prompts
took 3.41–4.31 s/frame across this set; on mc01 it retained five masks, including
the five-pixel cross and nine-pixel piece. Other frames still contain grouped
regions and parts rather than perfect semantic objects. Counts are not accuracy.

The copy-only probe used saved batch **9021**, six real mc01 actions at level 1,
seed 178, and two sampled anchors. Over 24 auxiliary updates on these examples:

| Measurement | Before fitting | Final fit pass |
| --- | ---: | ---: |
| Auxiliary combined objective | 11.975 | 8.464 |
| Matched one-step mask IoU | 0% | 51.28% |
| Matched three-step mask IoU | 0% | 23.93% |
| Current-mask reader IoU | 0% | 40.27% |

The audit took 35.38 s after setup: five SAM calls accounted for 16.27 s, and
subsequent cached two-update passes took about 1.55 s. The initial predictor
output was bit-for-bit identical; every pre-existing weight remained identical
after fitting. Fusion remained at zero. These are small-sample fitting results,
not a visual delta improvement, held-out result or game-completion claim. They do
not justify activating this branch in the live trainer yet. GPU throughput and
multi-game generalization remain unmeasured.

Repeat the probe with conventional versioned Python; output must be a new local
file outside Git:

```sh
python -m model.bench_objects --source /old-run/latest.pt \
  --sam-checkpoint /assets/sam2.1_hiera_tiny.pt \
  --output /tmp/object-audit.json --passes 12
```

Focused CPU checks cover native geometry, tiny objects, mask ordering, translation,
rotation, disappearance, new-mask slots, three-step gradients, migration/Adam,
normal sequence resume, planner defaults, offline packaging and dashboard data.
