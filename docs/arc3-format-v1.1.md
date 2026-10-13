# .arc3 binary attempt format — version 1.1

This normative minor extension adds optional latent residual, decoded visual
change and component learning diagnostics to the frozen
[version 1.0 contract](arc3-format-v1.md). All v1.0 required fields, existing
meanings, real frame streams, action IDs, chronology, attempt/RESET accounting,
MessagePack/BIN and single checksummed Zstandard envelope, unknown-field policy,
512 MiB payload/window cap and lower reader caps remain unchanged. This document
defines export semantics; it does not adopt or implement a neural architecture.

## Version selection and compatibility

Writers MUST explicitly select minor 1 to produce 1.1. The Python boundaries are
`encode_attempt(attempt, *, format_minor=0)` and
`write_attempt(path, attempt, *, format_minor=0)`; the only supported writer
selections are integer 0 and 1. `EvaluationRecorder(..., format_minor=1)` is also
opt-in. Existing callers and the evaluation CLI still emit 1.0. Field names do
not automatically promote a recording. No CLI selection is added.

Readers pass the envelope's minor into semantic validation. For minor 0, the
names added here remain structurally validated unknown values, without 1.1
interpretation or semantic rejection. For minor >= 1, readers validate and may
render these known fields, continuing to accept other structurally valid unknown
optional keys. Old v1 readers can replay 1.1 using its unchanged required fields
and absolute predicted frames. The writer must retain those fields.

Every new group is optional and MUST be omitted rather than null when unavailable.
A present recognized group must contain actual data satisfying the rules below.
Never invent zero diagnostics. Tensor dimensions are file-defined; neither
1024/2048 nor any particular backbone is required. Tensors retain v1.0's
little-endian, row-major BIN encoding and bounds. Single-action state tensors have
no batch axis. Unknown optional keys within groups follow v1.0's structural
validation and preservation rules.

## Actual state and frozen residual prediction

| Optional field | Required contents and meaning |
| --- | --- |
| `initial_representation` | `{z, h}`: actual initial-observation encoding, before action or subsequent learning |
| `steps[i].representation` | `{z, h}`: encoding assimilated from this step's real result, before learning from it; never an imagined successor |
| `steps[i].prediction.dynamics` | `{z, h, delta_z, h_next}`: frozen pre-action inputs, predicted latent residual and predicted next recurrent memory |
| `steps[i].policy_mode` | `"random"` or `"learned"`, only when known; no candidate scores are implied |

Every z/h/delta_z/h_next above MUST be a finite `f32` rank-one tensor with a
positive length. Dynamics requires the existing `prediction.latent` and
`target: "post_action_last_frame"`. In that conditional profile,
`prediction.latent` MUST also be f32 rank one: z, delta_z and latent lengths match
D_z; h and h_next lengths match D_h. This does not restrict legacy standalone
`prediction.latent` tensors without dynamics.

The exporter owns and captures the residual equation:

```text
(z_t, h_t) + executed action → d_hat_(t+1)
z_hat_(t+1) = z_t + d_hat_(t+1)
```

`prediction.latent` stores the actual z_hat output, not a viewer reconstruction.
Shape/dtype violations are malformed. Numerical residual disagreement is valid
replay evidence: a viewer may check each element using
`abs(z_hat - (z + delta_z)) <= 1e-6 + 1e-5 * max(abs(z_hat), abs(z + delta_z))`,
label inconsistency, and retain the recorded output. This derived consistency
check is not a prediction-quality metric. The viewer bounds the check and norms
to 65,536 elements and previews at most 16 state values; no arbitrary projection,
weights, model lookup or decoder execution is introduced.

| Optional prediction error | Pair and units |
| --- | --- |
| `prediction.errors.latent_mse` | Mean squared difference of frozen `prediction.latent` and this result's actual `representation.z`; squared latent units |
| `prediction.errors.no_change_mse` | Mean squared difference of frozen `dynamics.z` and the same actual z; no-change baseline, squared latent units |

Both metrics MUST be finite, nonnegative and use a last-frame target. Each
requires both source and real tensors with matching dimensions. The recorder
omits a score when no permitted pair exists, including dimension mismatches;
unmatched raw states may still be recorded without that comparison. A real
representation requires a nonempty visual result. With no valid visual successor,
omit representation and latent scores, preserving the real action and any empty
SDK response. These keys never occupy legacy pixel MAE/MSE fields and are never
reported as replay-training losses.

All compared encodings MUST use the same model parameters, before an update.
Do not compare stale cached latents with a newly trained encoder, reconstruct an
old prediction after learning, or borrow tensors from another action/attempt.
A subsequent step's pre-action encoding may differ from the prior saved real
representation after permitted learning/re-encoding. Snapshots carry this causal
meaning without a revision counter or checkpoint/version-management subsystem.
The recorder uses stable float64 accumulation (`math.fsum`) on the exported f32
values, and merges its owned metric keys while preserving other validated keys.

## Decoded visual change and real animation

Optional `prediction.visual_delta` has exactly these required known fields:

| Field | Type / meaning |
| --- | --- |
| `reference_position` | Integer 0..i for step i: 0 is initial observation, j+1 is the result of step j |
| `reference_frame` | u8[H,W], exactly the last frame of that nonempty causal observation |
| `delta` | u8[H,W], with KEEP=16 or replacement color 0..15 |

It requires `target: "post_action_last_frame"` and the existing
`prediction.frames`, an absolute u8[H,W] decoded final board. Reference and
predicted colors MUST be 0..15. All three tensors have identical positive spatial
geometry. Applying each KEEP/replacement to the immutable reference MUST exactly
produce the absolute predicted board. Invalid codes, references, geometry or
materialized boards are malformed and rejected. The writer omits the bundle
when it cannot establish a valid causal reference. An intervening empty result
does not permit borrowing a future frame: a valid explicitly earlier reference
can still be used.

The learned decoder executes in the producer, before the real action. The static
viewer loads no weights and performs no neural decoding. This delta is not
subtraction, probabilities, a latent residual, or real animation. The viewer
labels its KEEP/color view separately from the decoded board and prediction vs
actual disagreement map. The final target is compared only at the last real
returned frame. Intermediate/empty results or different real geometry remain
valid evidence with unavailable comparison, without resizing or fabrication.

An observed animation change may be derived locally from consecutive recorded
real frames. For the first result frame, use the immediate pre-action
observation's last frame when nonempty and same-sized. Do not search backward
past an intervening empty response. With no valid predecessor, show unavailable,
not a zero change. Label it **observed / derived**, keeping it distinct from the
predicted decoder delta; no duplicate observed-delta stream goes on the wire.
If real palette bytes fall outside 0..15, retain them in replay and refuse this
KEEP/color derivation. Change views and disagreement work are bounded to 65,536
pixels. In the change views KEEP=16 is cyan, explicitly labelled as a change code;
replacement colors use the bundled palette. Raw codes remain visible.

## Learning diagnostics by component

Optional `learning.modules` is a nonempty map. Recognized entries are
`world_model`, `decoder`, `actor`, `critic`. Each present recognized entry MUST
have at least one of these fields with data:

| Optional entry field | Type / meaning |
| --- | --- |
| `updates` | Nonnegative integer, actual optimizer steps attributed to this real step, not cumulative |
| `loss` | Finite reported loss |
| `losses` | Nonempty map of nonempty names to finite component losses |
| `reset_reason` | Nonempty plain text for an actual actor/critic reset event only |

Recognized world-model component names are `latent_mse` and `sigreg`; decoder
recognizes `color_cross_entropy`. Other component loss names remain finite
reported values and are displayed literally in bounded previews. Negative loss
values are legal, including actor objectives. Never infer a reset from a spike.
Actor/critic entries stay absent until there are actual reports.

For multiple updates, reported losses are arithmetic means of the reported
per-update values associated with the real step. Public config MUST document
objective weights and units. Legacy `learning.updates/loss/replay_size` keep
their writer-reported meanings: do not manufacture an aggregate from unrelated
module losses, or add the legacy loss to module values.

The viewer provides separate module series/counters, missing-sample gaps and
actual reset markers, within the existing 2,048-step chart bound. A reset-only
chart marks recorded event presence as a derived 1, never as an optimizer step.
Random action selection can coexist with real world-model/decoder diagnostics;
random/no-model baseline files retain unavailable model panels. These fields
grant no training permission, success/reward inference or imagined rollout schema.

## Producer capture and failure boundary

1. Optionally snapshot the initial representation.
2. Select the real action. Freeze/deep-copy action, decision and prediction,
   including dynamics and decoded visuals, before `env.step`.
3. Receive the actual SDK result. Synchronously obtain its optional real
   representation using frozen weights and pair it with the active attempt/step.
   Attach and score before terminal finalization.
4. A permitted training producer may subsequently attach actual component
   reports. It MUST NOT rewrite the action or frozen input/prediction/result
   snapshots. Evaluation recording remains read-only and never invokes training.

The Python recorder retains `trace_diagnostics(observation, action)` and adds:

| Optional policy hook | Allowed returned top-level fields |
| --- | --- |
| `trace_initial_diagnostics(observation)` | `initial_representation` only |
| `trace_result_diagnostics(before, action, result)` | `representation`, `learning`, `timing`, `notes` only |

Hooks are synchronous, called only with tracing enabled, and identify the
current real observation/action/result. The initial hook runs on each attempt's
initial observation, including a RESET result. No result hook runs for RESET as
an ordinary step. Missing hooks or empty maps simply omit diagnostics. The
pre-action hook also permits `policy_mode`; recognized error metrics are owned
by post-result scoring, while structurally valid unrelated error keys survive.
Returned snapshots MUST contain only plain wire values, not mutable model/device
objects. Existing pre-action learning/timing reports are permitted; post-result
reports merge these maps and append notes, without altering frozen predictions.

Validate staged diagnostic values before attachment. Reject late attempts to
overwrite action, decision, prediction, index or observation. Retain the confirmed
real step before invoking its result hook. A hook failure preserves that result
and aligned pixel scores, propagates through the existing error/finalization
policy, and never emits an unconfirmed transition. Known WIN/GAME_OVER evidence
keeps its terminal reason even on a later diagnostic failure. Clear pending
diagnostics on RESET, failure, finalization and new attempts. All ordinary
evaluation/collection guards and training-level exclusions remain unchanged.

## Independent conformance examples

These are original synthetic values, not model predictions from a checkpoint:

* z=[1,2], delta_z=[0.5,-1], recorded z_hat=[1.5,1], real z=[2,1]:
  latent MSE **0.125**, no-change MSE **1**. Use f32/BIN little-endian bytes
  `0000803f00000040`, `0000003f000080bf`, `0000c03f0000803f`,
  `000000400000803f`, respectively. D_z=2 and D_h=3 exercise file-defined sizes.
* Reference `[[0,1],[2,3]]` plus codes `[[16,4],[16,0]]` yields recorded final
  `[[0,4],[2,0]]`. Real final `[[0,4],[5,0]]` disagrees at exactly **1/4** pixels.

Focused Python codec/recorder tests and the shared generated Python→TypeScript
vectors own wire/lifecycle validation. Frozen v1.0 goldens are retained unchanged.
Generated `.arc3`/JSON artifacts stay ignored. See the
[viewer manual sequence](../viewer/README.md#manual-v11-diagnostics-check) for
fixture paths and the user-run browser check; no browser check is implied by
codec/build success.
