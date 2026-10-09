# .arc3 binary attempt format — version 1.0

## Status and scope

This is the normative v1.0 wire specification adopted through
[issue #4](https://github.com/murillo128/arc-agi-3-model/issues/4). **MUST**, **MUST NOT**,
**SHOULD** and **MAY** express requirements, prohibitions, recommendations and
permissions. Version 1.0 is frozen on acceptance: implementations must use this
document without consulting issue discussion or chat. Section 10 governs changes.

One `.arc3` file contains one continuous attempt in one versioned game, beginning
with an initial or post-RESET observation and ending at WIN, GAME_OVER, the next
RESET, an action budget, timeout, error, interruption or training-level boundary.
Level changes remain inside the attempt. Multiple attempts MAY share a session
and preserve per-game online adaptation when the execution policy permits it;
the format neither resets model weights nor authorizes evaluation-time updates.
Every file MUST be independently replayable from its own bytes, with no backend,
database, separate frame files, external compression dictionary or asset lookup.

This document specifies storage, validation and provenance. It does not implement
a writer, viewer, training pipeline or competition submission. The current SDK
adapter, collection/evaluation entry points and [training policy](training-policy.md)
remain unchanged. A replay is evidence of recorded observations, not proof that
a game was reproducible, solved correctly, or evaluated under a particular policy.

## 1. Binary envelope and compression

Offsets use half-open byte ranges. There is no padding or trailing material.

| Bytes | Type | Required value or meaning |
| --- | --- | --- |
| `[0:4]` | Four bytes | ASCII `ARC3`, hex `41 52 43 33` |
| `[4:6]` | Unsigned 16-bit, little-endian | Major version `1` |
| `[6:8]` | Unsigned 16-bit, little-endian | Minor version `0` for this specification |
| `[8:16]` | Unsigned 64-bit, little-endian | `N`, exact decompressed MessagePack byte length |
| `[16:end]` | Zstandard frame | Exactly one ordinary frame containing exactly one MessagePack map |

`N` counts the serialized payload bytes, including MessagePack tags and lengths;
it is neither compressed length, tensor byte count nor a character count.
`N` MUST be positive and at most `536870912` (512 MiB). There is no extra `.arc3`
checksum field. The Zstandard frame MUST have its content checksum flag enabled,
and the checksum MUST be verified before the file is accepted. That checksum
detects corruption; it does not authenticate authorship.

Use ordinary Zstandard frame magic `28 b5 2f fd`. Skippable frames, concatenated
frames and bytes after the first frame's checksum MUST be rejected. Compression
MUST operate without a dictionary, even when a frame omits a dictionary ID;
readers MUST reject a nonzero dictionary ID and MUST NOT supply a dictionary.
Frame content size MAY be omitted; when present it MUST equal `N`. The declared
Zstandard window size MUST be no greater than 512 MiB. Readers MAY use a lower
resource cap as described in section 2. Compression level, block boundaries and
MessagePack map ordering are not canonical: interoperable writers can produce
different compressed bytes representing the same semantic payload.

These frame properties follow the
[Zstandard frame specification, RFC 8878 sections 3.1.1–3.1.1.1](https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1).
The stricter single-frame, checksum and dictionary rules above are `.arc3` rules.

## 2. Decode recipe and resource guards

A Python or browser-only reader MUST perform the following operations in order;
an API that silently accepts a prefix or ignores a checksum is insufficient.

1. Read the 16-byte header, requiring sufficient input. Verify magic and supported
   major, then decode minor and `N`. In JavaScript read the uint64 as `BigInt`
   (for example `DataView.getBigUint64(8, true)`), compare it to the cap, and only
   then convert to a `Number`. Never truncate a header value to 32 bits.
2. Set uncompressed cap `C` to 512 MiB by default. A reader MAY expose a positive
   lower cap, but MUST NOT silently raise the v1 maximum. Reject `N == 0` or
   `N > C` before allocating an output buffer or initializing a large decoder.
   Inspect the Zstandard frame header: require ordinary magic, checksum enabled,
   no dictionary dependency, window size at most `C`, and content size equal to
   `N` if it is declared. A single-segment frame uses its content size as its
   window size. Check reserved bits and invalid headers through the decoder.
3. Decompress with a hard output ceiling of `N` and a window-memory limit of `C`.
   Stop and reject before producing byte `N + 1`; a streaming decoder must bound
   cumulative output, not just each chunk. Require a completed frame, verified
   checksum, total output exactly `N`, and compressed bytes consumed exactly
   `file_length - 16`. Do not use an API's default concatenated-frame decoding.
   A browser MAY bundle a JavaScript or WebAssembly Zstandard decoder locally;
   native browser decompression support is not assumed and no service is needed.
4. Decode one MessagePack value from the bounded output. Require a map and that
   all `N` bytes are consumed. Apply section 3 while decoding, before collapsing
   duplicate keys into a language object. Reject malformed/truncated tags,
   invalid UTF-8, prohibited types and any second value or trailing byte.
5. Validate required fields and every present recognized optional field, then
   chronology, action legality, tensor lengths and termination/summary consistency.
   Validate each shape and byte count before creating reshaped arrays, typed
   float views, canvases, textures or GPU buffers. Expose the attempt only after
   the entire file passes; a partial decode MUST NOT be reported as valid.

Allocation from untrusted lengths is prohibited. MessagePack array/map counts,
string/BIN lengths and nesting MUST be checked against remaining input before
allocation: an array needs at least one encoded byte per child and a map at least
two per entry. Nesting depth MUST NOT exceed 64 containers (the top-level map is
depth 1). Unknown fields still undergo these structural checks. `C` limits raw
output, not the total memory of decoded objects; readers SHOULD parse without
unnecessary copies and MAY impose lower documented memory/time limits, reporting
a resource-limit refusal rather than claiming the input is malformed. They MUST
never fetch an asset or interpret metadata as code to finish decoding.

## 3. MessagePack profile and common types

Use the [MessagePack specification](https://github.com/msgpack/msgpack/blob/master/spec.md),
with STR for UTF-8 text, BIN for tensor bytes, ARRAY for ordered lists and MAP for
objects. MessagePack's multibyte integer/float tags retain their specified
big-endian encoding; the envelope and the contents of tensor BIN fields are
little-endian. They are different layers. Extension tags, including MessagePack
timestamp extensions, are prohibited in v1. Use the timestamp string below.

All maps, including unknown extension maps and configuration, MUST have unique
STR keys. Map order has no meaning. Integer fields MUST use MessagePack integer
tags, not BOOL or FLOAT, even for integral-valued floats. Unless narrowed below,
integer values are restricted to JavaScript's exact integer range
`[-9007199254740991, 9007199254740991]`. A **count** is an integer in
`[0, 9007199254740991]`. A **number** is a finite integer or IEEE754 MessagePack
float in that same integer range when encoded as an integer; NaN and infinities
are rejected. A **duration** is a nonnegative number, in the specified unit.
Negative zero is numerically zero. There is no coercion of strings or booleans.

Strings MUST be valid UTF-8. An **identifier** is a nonempty string; identifiers
are opaque, need not be UUIDs, and MUST NOT be resolved as paths or URLs. Plain
text may be empty and MUST be rendered as text, never HTML/Markdown or script.
Required fields cannot be absent or NIL. Recognized optional fields are omitted
when unavailable; NIL is invalid unless explicitly allowed inside `config`.
Unknown optional keys MAY be ignored, but cannot override, rename or excuse a
missing or invalid recognized field. Unknown values may contain the profile's
maps, arrays, STR, BIN, BOOL, NIL, integers and finite numbers, subject to all
structural bounds. Their presence cannot be necessary to render a v1 replay.

The top-level map requires exactly these five known fields; additional optional
extension keys are allowed:

| Key | Type | Meaning |
| --- | --- | --- |
| `metadata` | Map | Provenance and attempt identity (section 4) |
| `initial_observation` | Observation map | Start of this attempt (section 5) |
| `steps` | Array of step maps | Real post-action transitions in chronological order (section 6); may be empty |
| `termination` | Map | Boundary and final public state (section 7) |
| `summary` | Map | Observed counts and elapsed time (section 7) |

## 4. Metadata

| Key | Presence | Type and validation | Meaning |
| --- | --- | --- | --- |
| `game_id` | Required | Identifier | Full SDK-returned versioned game identifier; never shorten to a game prefix |
| `run_id` | Required | Identifier | Writer-assigned execution/run identity |
| `session_id` | Required | Identifier | Per-game session identity linking attempts |
| `attempt_index` | Required | Count | Zero-based attempt ordinal in the session |
| `seed` | Required | Integer | Actual environment seed supplied, including zero or negative seeds if used |
| `sdk_version` | Required | Identifier | Exact `arc-agi` package version used, e.g. `0.9.9` |
| `source_split` | Required | STR enum | `train`, `evaluation` or `unspecified` |
| `started_at` | Required | Timestamp STR | Time of this attempt's initial observation, with explicit UTC or numeric offset |
| `model_id` | Optional | Identifier | Public model identity, when known |
| `checkpoint_id` | Optional | Identifier | Public checkpoint identity, when known; not its bytes or a required lookup |
| `config` | Optional | JSON-compatible value | Public configuration snapshot; no BIN or extension tags |

For interoperability, `started_at` uses
`YYYY-MM-DDTHH:mm:ss[.fraction](Z|±HH:mm)`: a valid calendar date, year 0001–9999,
hours 00–23, minutes/seconds 00–59, optional 1–9 fractional digits and an explicit
offset (hours 00–23, minutes 00–59). Leap-second and timezone-free strings are
not in this profile. The writer SHOULD use UTC `Z`. This wall-clock timestamp
does not replace monotonic elapsed time. `config` permits NIL, BOOL, text, finite
numbers, arrays and string-keyed maps recursively within the common bounds.

Within a session `game_id` is unchanged and attempt indices increase by one;
a single-file reader cannot prove that companion attempts exist. IDs and seed
support provenance, not proof of determinism. Missing model/config details MUST
remain absent rather than being filled with invented defaults.

## 5. Tensors and observations

### Tensor map

Every tensor requires `dtype` (STR enum), `shape` (ARRAY of integer dimensions)
and `data` (BIN). No nested pixel arrays, base64 STR, image files or lossy encoding
are substitutes for BIN. Data is contiguous row-major, last axis fastest, with
no strides, padding or alignment assumption.

| `dtype` | Bytes per element | BIN element representation |
| --- | --- | --- |
| `u8` | 1 | Unsigned byte 0–255; observed pixels preserve palette index bytes |
| `f16` | 2 | IEEE754 binary16, little-endian; finite values only |
| `f32` | 4 | IEEE754 binary32, little-endian; finite values only |

Rank MUST be 1–8; dimensions MUST be positive integers no greater than `C`, except
for the empty-observation sentinel below. With item size `B`, the required byte
count is `B * product(shape)` and MUST be at most `C` and equal the BIN length.
Check dimensions individually and multiply with checked arithmetic, rejecting
when the next dimension exceeds `floor(C / accumulated_bytes)`; do not multiply
first and allow overflow. Zero dimensions cannot mask an absurd dimension.
Float readers MUST handle little-endian bytes explicitly, including on a host
with different endianness, and MUST NOT rely on BIN alignment for typed views.

### Observation map

| Key | Required type | Validation and interpretation |
| --- | --- | --- |
| `frames` | Tensor | `u8`, rank 3, `[frame_count, height, width]` |
| `state` | Identifier | Exact public SDK state name |
| `levels_completed` | Count | Exact public SDK counter |
| `win_levels` | Count | Exact public SDK level target; zero is allowed if returned |
| `available_actions` | ARRAY of integers | Exact SDK list/order, each ID in 0–7; an empty list is allowed |

The writer MUST copy the entire ordered SDK `frame` sequence into `frames`, with
actual dimensions and unchanged bytes. For nonempty sequences all frames must
have the same positive height/width; a ragged sequence cannot be represented by
this tensor and MUST NOT be padded, cropped or silently flattened. There is no
64×64 restriction. For an actual empty SDK sequence, the canonical tensor is
`{dtype: "u8", shape: [0, 0, 0], data: BIN of length 0}`. Other zero-dimension
shapes are invalid. A reader displays that absence and MUST NOT fabricate a frame
or borrow one from another attempt. Initialization needing RESET follows section 7.

Current SDK state names are `NOT_PLAYED`, `NOT_FINISHED`, `WIN`, `GAME_OVER`.
`state` is a string rather than a closed wire enum: readers MUST retain an
unrecognized public state name as opaque text, and MUST NOT infer game rules
from it. An implementation recording another SDK version must still obey all
v1 action and tensor constraints. Do not infer rewards, hidden level IDs or a
relationship between the level counters beyond what the public SDK supplies.

The initial observation precedes step 0. Step `i`'s observation is the complete
result of that step's real action; step `i + 1` uses it as its pre-action
observation. There is no second stored pre-action observation. Do not prepend
the previous observation to the SDK result. If the SDK itself returns repeated
frames as part of its animation, preserve those repeats. Frame count/size may
change between observations and level counters may advance without a new file.

## 6. Steps and optional diagnostics

Each step requires `index` (count), `action` (map below), `observation` (section 5).
For `S = len(steps)`, indices MUST be exactly `0, 1, ..., S - 1` in array order.
Only returned, real observations may populate steps. Imagined rollouts and
predictions MUST NOT be stored here as verified transitions.

An action requires `id` (integer) and `data` (map of STR to integer). For ordinary
steps IDs are 1–7, corresponding to SDK ACTION1–ACTION7; ID 0 is RESET and belongs
only to `termination.reset_action`. The step action ID MUST appear in its
pre-action observation's `available_actions`. No ordinary step may follow a
known `WIN`, `GAME_OVER` or initialization `NOT_PLAYED` state in the same file.
Simple actions normally have empty data. ACTION6 requires explicit integer `x`
and `y`, each 0–63 under the pinned SDK; preserve the exact submitted coordinates
and any other integer parameters. Do not rescale coordinates to a viewer canvas
or restrict them using a hard-coded frame shape. SDK action IDs and coordinate
constraints come from the [official action interface](https://docs.arcprize.org/actions).

Optional step groups are maps `decision`, `prediction`, `learning`, `timing` and
an array `notes`. Every present recognized field MUST be validated. Omit groups
that have no real diagnostic information; do not create null tensors, zero losses
or fake model predictions for a random/no-model policy.

### `decision`

Required `score_type` is `logit`, `probability` or `value`; required `candidates`
is a nonempty array of maps requiring `action` and numeric `score`. Candidate
actions use the ordinary action schema and the same pre-action availability
check. Candidate order is the writer's order; entries are alternatives, not real
actions, and do not affect `summary.real_actions`. The array MAY describe only
a scored subset; it need not contain every available action or the selected
action. For `probability`, each score MUST lie in `[0, 1]`, but a subset is not
required to sum to one. Logits and values are finite numbers with no probability
interpretation. Optional `selected_value` is a finite numeric value estimate for
the executed action; optional `action_entropy` is a nonnegative number in nats
from the decision's actual action distribution. Do not derive entropy from
unlabelled scores or pretend a candidate subset is a normalized distribution.

### `prediction`

Required `target` is `post_action_last_frame` or `post_action_sequence`.
Optional `frames` is a tensor in any of the three dtypes: rank 2 `[H, W]` for
last-frame prediction, rank 3 `[F, H, W]` for sequence prediction, with positive
dimensions. Optional `latent` is a tensor of rank 1–8 with positive dimensions.
For frame predictions the values are predicted palette indices, including
fractional estimates for float dtypes, in observation pixel coordinates. Latent
units are model-specific. An outcome with a different shape/count is a model
mismatch, not a reason to rewrite either the prediction or real observation;
readers MUST NOT require predicted and observed shapes to match.

Optional `uncertainty` is a nonnegative number, with its estimator/units described
in public `metadata.config` when recorded. Optional `errors` is a map with
optional scalar `mae` (mean absolute pixel-index error) and `mse` (mean squared
pixel-index error), each a nonnegative number. MAE units are palette-index units;
MSE units are their square. These diagnostic distances do not measure game
correctness. Record them only when prediction and target shapes align: compare
against the observation's last frame for `post_action_last_frame` or its entire
sequence for `post_action_sequence`, over all pixels. With no predicted frames,
an empty observed sequence or unequal shapes, omit these errors. Unknown optional
error metrics can be ignored, following section 3.

Target, predicted tensors and uncertainty MUST be captured before the real action
executes, without reading the resulting observation. Errors MAY be computed
afterward against the real result. A prediction group MUST contain at least one
of `frames`, `latent`, `uncertainty`; `target` alone conveys no prediction. Never
fill an unavailable prediction using the observed frame after execution.

### `learning`, `timing`, `notes`

| Group/key | Optional type | Meaning or unit |
| --- | --- | --- |
| `learning.updates` | Count | Actual optimizer updates associated with this transition |
| `learning.loss` | Number | Actual reported loss; objective/aggregation described in public config |
| `learning.replay_size` | Count | Actual replay-entry count after handling this transition |
| `timing.decision_ms` | Duration | Selecting the action, including prediction/planning if performed there |
| `timing.environment_ms` | Duration | Executing the real action and obtaining its result |
| `timing.learning_ms` | Duration | Updating from the returned real transition, if allowed/performed |
| `notes` | ARRAY of plain-text STR | Sanitized annotations; array may be empty |

Known timing durations use monotonic elapsed milliseconds. Unknown optional
timing keys MUST also hold nonnegative millisecond durations. Timings may overlap;
their sum is not required to equal wall time. Learning describes actual updates,
not imagined samples masquerading as real ones, and never grants permission to
train during evaluation. An omitted key means unavailable, not zero.

## 7. Attempt termination and summary

`termination` requires `reason` (STR enum below) and `final_state` (identifier,
public SDK state). Optional `detail` is sanitized plain text. Optional
`reset_action` is the action map `{id: 0, data: {}}`; it is permitted only for
reason `reset`, and its presence asserts that RESET was actually executed.

| `reason` | Boundary |
| --- | --- |
| `win` | Latest real state is `WIN`; no subsequent action in this attempt |
| `game_over` | Latest real state is `GAME_OVER`; no subsequent action in this attempt |
| `reset` | Attempt deliberately closes to start another via RESET |
| `action_budget` | Caller stops at its real-action allowance |
| `timeout` | Caller stops at its elapsed-time allowance |
| `error` | Execution/recording cannot continue safely; retain only verified transitions |
| `interrupted` | Caller/user/process interruption closes an otherwise partial attempt |
| `training_level_boundary` | Collection stops before persisting a transition entering a reserved level |

Stop at the first known terminal observation. A later reset of a WIN/GAME_OVER
game starts another attempt and does not change its `win`/`game_over` reason.
For a deliberately reset active attempt, `final_state` and final level counters
describe the observation **before RESET**, not its returned observation. That
post-RESET result belongs only to the next file's `initial_observation`. RESET
may restart a level rather than clear all completed levels; preserve the actual
returned state/counters instead of assuming zeros.

A reset intent stopped before execution MAY close with `reason: "reset"` without
`reset_action`; do not count or claim an unexecuted reset. Once RESET actually
executes, include `reset_action` and count it even if its result is unavailable;
do not create a next file until a verified initial observation exists. An initial
RESET needed before the first observation establishes that observation and MUST
NOT produce an empty initialization attempt. Because the file starts *after*
that bootstrap RESET, it is outside this file's action count and wall interval;
session-level accounting must count it separately. A reset closing an existing
attempt is counted only in the preceding file, never again in the next one.

For collection, do not serialize a reserved-level observation or the transition
entering it, including its action/diagnostics, anywhere in the file. Stop with
`training_level_boundary`. The crossing action still counts as a real action;
the boundary's public state and counters MAY be retained in `termination` and
`summary`, but not its frames or private game information. If initialization is
already inside a reserved level, create no training recording. This preserves
the existing collector's discard-before-record boundary and disjoint split rule.

`summary` requires:

| Key | Type and unit | Meaning |
| --- | --- | --- |
| `real_actions` | Count | Actually executed environment actions during this attempt, including its closing RESET |
| `levels_completed` | Count | Final public SDK counter, not a per-file difference |
| `win_levels` | Count | Final public SDK level target |
| `wall_seconds` | Duration, seconds | Monotonic elapsed time from initial observation to closure, including an executed closing RESET |

Normally `final_state` and summary level counters equal those in the last step
observation, or the initial observation when there are no steps. For
`training_level_boundary`, they instead describe the discarded boundary result
if available. If an action fails or is interrupted without a verified result,
use the last verified public state/counters and do not invent a post-action step.
With no known terminal state, an empty-step attempt closed by a budget, timeout,
interruption, error or reset is valid once it has a verified initial observation.

For a normal file `real_actions = len(steps) + presence(reset_action)`. For
`training_level_boundary`, `error`, `timeout` or `interrupted`, it MAY additionally
count one actually executed non-RESET action with no persisted transition (a
discarded boundary result or unavailable response). No more than one such action
is allowed: stop rather than continue across an unrecorded gap. Mere planned,
rejected-before-execution or unconfirmed calls MUST NOT be counted as executed.
An error with uncertain execution may therefore understate external action
usage; explain that uncertainty in sanitized `detail`, without inventing evidence.
A reader MUST reject counts outside these equations. Reset intentions without
execution add zero. Predictions/candidates/optimizer updates never add real actions.

For `win` and `game_over`, `final_state` MUST respectively be `WIN` and
`GAME_OVER` and match the final persisted observation. A known terminal final
persisted observation MUST use the corresponding reason. A discarded
training-boundary result is deliberately not a persisted observation.
Other reasons do not imply a new SDK state such as `TIMEOUT` or `ERROR`.

## 8. Semantic examples and independent conformance vectors

The JSON blocks below are **semantic notation**, not a JSON alternative to the
wire format. Every object of the exact form `{"$bin_hex": "..."}` stands for
MessagePack BIN with the displayed hexadecimal bytes; `$bin_hex` is never a
wire key. All other objects become MAP, arrays ARRAY, strings STR and integral
values integer tags. Fractions use MessagePack float64. A writer serializes the
whole map, computes `N` from those bytes and adds the envelope/checksummed frame.

### Baseline B: ordinary real action, no model prediction

All pixels and game identities here are synthetic fixtures, not game rules or
evaluation data. The short 2×2 frames deliberately detect a hard-coded 64×64 reader.

```json
{
  "metadata": {
    "game_id": "fixture-v1", "run_id": "run-1", "session_id": "session-1",
    "attempt_index": 0, "seed": 42, "sdk_version": "0.9.9",
    "source_split": "unspecified", "started_at": "2026-10-09T12:00:00Z"
  },
  "initial_observation": {
    "frames": {"dtype": "u8", "shape": [1, 2, 2], "data": {"$bin_hex": "00010203"}},
    "state": "NOT_FINISHED", "levels_completed": 0, "win_levels": 2,
    "available_actions": [1, 6]
  },
  "steps": [{
    "index": 0, "action": {"id": 1, "data": {}},
    "observation": {
      "frames": {"dtype": "u8", "shape": [1, 2, 2], "data": {"$bin_hex": "03020100"}},
      "state": "NOT_FINISHED", "levels_completed": 0, "win_levels": 2,
      "available_actions": [1, 6]
    }
  }],
  "termination": {"reason": "action_budget", "final_state": "NOT_FINISHED"},
  "summary": {"real_actions": 1, "levels_completed": 0, "win_levels": 2, "wall_seconds": 0.25}
}
```

For each row, independently deep-copy B and apply only the listed changes.
Unless a row says otherwise, recompute `N` and a valid Zstandard checksum so the
case reaches the intended semantic guard. Positive outcomes are literal values,
not expected results calculated using a downstream implementation.

| ID | Changes to B | Expected decoded outcome |
| --- | --- | --- |
| V01 ordinary/no prediction | None | Accept; initial pixels `[[0,1],[2,3]]`, step 0 pixels `[[3,2],[1,0]]`; one action; all five diagnostic groups absent |
| V02 multiframe | Step frames shape `[2,2,2]`, BIN hex `0405060708090a0b` | Accept; ordered frames `[[4,5],[6,7]]`, then `[[8,9],[10,11]]`; still one real action, not two; no prepended initial frame |
| V03 click | Step action `{id:6,data:{x:1,y:0}}` | Accept; exact coordinate pair `(1,0)` survives; no coordinate scaling |
| V04 level transition | Step `levels_completed=1`, summary `levels_completed=1` | Accept; one file, same attempt index, final public counter 1 |
| V05 win | Step `state="WIN"`, `levels_completed=2`, `available_actions=[]`; termination `{reason:"win",final_state:"WIN"}`; summary `levels_completed=2` | Accept; completed levels 2; no invented reward/score |
| V06 game over | Step `state="GAME_OVER"`, `available_actions=[0]`; termination `{reason:"game_over",final_state:"GAME_OVER"}` | Accept; game-over boundary, one action |
| V07 closing RESET | Termination `{reason:"reset",final_state:"NOT_FINISHED",reset_action:{id:0,data:{}}}`; summary `real_actions=2` | Accept; one ordinary step plus closing RESET; no post-RESET observation stored |
| V08 next attempt | Metadata `attempt_index=1`; initial frame BIN `09090909`; `steps=[]`; summary `real_actions=0`, `wall_seconds=0.0` | Accept independently; initial pixels all 9; same session, no duplicate reset count; this observation is the returned RESET result paired with V07 |
| V09 training boundary | `steps=[]`; termination `{reason:"training_level_boundary",final_state:"NOT_FINISHED"}`; summary `real_actions=1`, `levels_completed=1`; metadata `source_split="train"` | Accept; only initial pixels present; crossing action/result omitted; final public level counter 1 and one executed action |
| V10 empty returned sequence | Step frames `{dtype:"u8",shape:[0,0,0],data:BIN hex ""}` | Accept; step has zero returned frames; no borrowed or invented frame |
| V11 prediction | Add step `prediction` from the block below; metadata `config={"uncertainty_estimator":"synthetic fixture","uncertainty_units":"unitless"}` | Accept; predicted pixels `[[0,1],[2,3]]`, latent `[1.0,-2.0]`, MAE 2, MSE 5; real pixels remain `[[3,2],[1,0]]` |
| V12 decisions/learning/timing | Add step groups from the block below; metadata `config={"loss_objective":"synthetic fixture","loss_aggregation":"one update"}` | Accept; two alternatives, one executed action; probability subset totals 0.75; loss 0.125; timing values in ms |
| V13 shape mismatch prediction | Add `prediction:{target:"post_action_sequence",frames:{dtype:"u8",shape:[1,1,1],data:BIN hex "07"}}` | Accept; prediction size differs from truth; omit pixel errors, keep truth intact |
| V14 optional extension/minor | Header minor `1`; add top-level `public_note:"fixture"` | Accept under forward-minor policy; all v1 required values unchanged |
| V15 unavailable result | Termination `{reason:"error",final_state:"NOT_FINISHED",detail:"Action executed; response unavailable"}`; summary `real_actions=2` | Accept; one verified step and one counted action with no fabricated result |
| V16 unexecuted reset | Termination `{reason:"reset",final_state:"NOT_FINISHED"}` | Accept; one real action, no claimed RESET |

V11's optional prediction (captured before action; errors added after result):

```json
{
  "target": "post_action_last_frame",
  "frames": {"dtype": "f32", "shape": [2, 2], "data": {"$bin_hex": "000000000000803f0000004000004040"}},
  "latent": {"dtype": "f16", "shape": [2], "data": {"$bin_hex": "003c00c0"}},
  "uncertainty": 0.5,
  "errors": {"mae": 2.0, "mse": 5.0}
}
```

For V11, absolute errors are `[3,1,1,3]` and squared errors `[9,1,1,9]`.
V12's step additions (public config describes the synthetic loss):

```json
{
  "decision": {
    "score_type": "probability",
    "candidates": [
      {"action": {"id": 1, "data": {}}, "score": 0.5},
      {"action": {"id": 6, "data": {"x": 1, "y": 0}}, "score": 0.25}
    ],
    "selected_value": -0.25, "action_entropy": 1.0397207708399179
  },
  "learning": {"updates": 1, "loss": 0.125, "replay_size": 8},
  "timing": {"decision_ms": 2.5, "environment_ms": 10.0, "learning_ms": 1.0},
  "notes": ["Synthetic diagnostic example"]
}
```

The entropy in V12 comes from the complete distribution `[0.5,0.25,0.25]`,
while only two scored candidates are included. It is not recomputed from the subset.

### Independent rejection and boundary vectors

Each row starts from B, changes only the specified condition and expects rejection
at that boundary. Diagnostic wording and error codes are implementation-specific;
accept/reject and the decoded semantic values above are normative.

| ID | Input/change | Required outcome |
| --- | --- | --- |
| E01 header | Fewer than 16 bytes; separately replace magic with `ARX3` | Reject truncated header; separately reject wrong magic |
| E02 version | Major `2` | Reject unsupported major before payload decode |
| E03 size | Header `N=0`; separately `N=536870913`; separately `N=18446744073709551615` | Reject invalid/over-cap length before allocation; no uint64 truncation |
| E04 exact size | Header `N` one smaller or one larger than actual decompressed bytes, with otherwise valid compressed frame | Reject excess output or length mismatch, including when frame content size is omitted |
| E05 checksum | Generate frame with checksum disabled; separately flip a bit in an enabled checksum | Reject absent checksum; separately reject corrupt checksum |
| E06 compressed suffix | Append byte `00`; separately append another ordinary or skippable Zstandard frame | Reject every suffix, including a second frame that decodes to zero bytes |
| E07 incomplete frame | Remove final checksum byte; separately truncate a data block | Reject incomplete frame, never return a partial attempt |
| E08 dictionary/window | Use a nonzero dictionary ID; separately a declared window of 1 GiB with small `N` | Reject dictionary dependency; separately reject window before decoder allocation |
| E09 MessagePack root/suffix | Encode ARRAY root; separately valid B followed by MessagePack NIL inside the same checksummed frame | Reject non-map root; separately reject a second value despite valid checksum/size |
| E10 keys/types | Duplicate `summary` key; separately omit `termination`; separately encode invalid UTF-8 STR | Reject duplicate key before object conversion; missing field; malformed text |
| E11 strict numeric type | Encode step index as BOOL `false` or FLOAT `0.0`; separately seed integer `9007199254740992` | Reject coercion or unsafe integer, even if a language would round it |
| E12 tensors | Step shape `[-1,2,2]`; separately `[1,2]`; separately `[1,2,2]` with 3 BIN bytes; separately STR data `"03020100"` | Reject negative dimension, observation rank, length mismatch and non-BIN data independently |
| E13 shape bounds | Step shape `[0,536870913,0]`, empty BIN; separately add `prediction:{target:"post_action_last_frame",latent:{dtype:"u8",shape:[1,1,1,1,1,1,1,1,1],data:BIN hex "00"}}`; separately add the same prediction with f32 latent shape `[536870912]`, empty BIN | Reject absurd dimension/invalid empty sentinel, excess rank and over-cap byte product before reshaping |
| E14 chronology/action | Step index `1`; separately step action ID `0`; separately ID `7` not in pre-action availability; separately click `x=64` | Reject gap, regular RESET, unavailable action and illegal coordinate independently |
| E15 malformed optional | Present `prediction:null`; separately add V12's decision with first probability score `1.01`; separately add `timing:{decision_ms:-1}`; separately add `prediction:{target:"post_action_last_frame",latent:{dtype:"f16",shape:[1],data:BIN hex "007c"}}` (positive infinity) | Reject each present invalid optional value; ignoring unknown keys does not excuse it |
| E16 termination/count | Reason `"done"`; separately attach `reset_action` to `action_budget`; separately summary `real_actions=0` or `3`; separately final state `"WIN"` with baseline observation | Reject unknown reason, misplaced RESET, inconsistent count and inconsistent final state |
| E17 provenance/config | Source split `"held_out"`; separately timestamp without timezone; separately BIN inside `config` | Reject invalid enum, timestamp profile and non-JSON-compatible config |
| E18 structural allocation | Unknown top-level value ARRAY declares `2^32-1` children with no remaining bytes; separately a top-level extension value nests 64 arrays (depth 65 including root); separately use timestamp EXT | Reject count before allocation, depth excess and prohibited extension tag |
| E19 terminal continuation | Change initial state to `GAME_OVER` while retaining B's ordinary step | Reject action after known terminal state |
| E20 malformed known error | Add V13 prediction with `errors:{mae:0}` despite unequal target shape | Reject claimed comparable pixel error when shapes differ |
| L01 payload-cap equality | Header `N=536870912`, valid independent full-size payload/frame | Length gate allows equality under default cap; later checks still apply; no huge fixture is required for this issue |
| L02 lower reader cap | B with reader cap `C=N-1`; separately `C=N` and frame window at most `N` | Refuse at lower cap; equality passes the length/window gates and normal validation |
| L03 UTF-8/endian | A public note `"π"`; standalone f32 tensor shape `[1]`, BIN `0000803f` | Note decodes as one Unicode character; f32 value is exactly 1.0, not big-endian interpretation |

Envelope primitive independent vector: with declared `N=256`, the first 16 bytes
are `41 52 43 33 01 00 00 00 00 01 00 00 00 00 00 00`. This checks header encoding
only, not a valid complete file. Tensor primitive vectors use MessagePack BIN
bytes `c4 04 03 02 01 00` for four u8 values `[3,2,1,0]` and
`c4 04 00 3c 00 c0` for two f16 values `[1.0,-2.0]`. The byte after `c4` is the
BIN length; it is not an element of the tensor.

Downstream Python and TypeScript codecs SHOULD generate fixtures from these
semantic vectors and test each other's files, rather than relying only on a
writer/reader round trip that can share the same bug. Binary golden artifacts
and actual codec/browser conformance execution belong to those downstream issues.
This document's examples do not claim to be captured SDK transitions.

## 9. Provenance, policy and rendering

Real observations and actually executed actions are ground truth. Prediction
quality, completion, action efficiency, compute and wall time are separate
measurements. Optional diagnostics do not change the public state or invent
reward, correctness, internal scores or privileged game rules. A file MUST NOT
contain secrets, tokens, private/hidden evaluation data or source-code-derived
rules. `detail`, `notes` and `config` require sanitization before writing; readers
cannot establish from syntax alone that these provenance restrictions were met.

`source_split` labels the data's actual purpose; it is not permission to train.
`train` collection retains disjoint game splits and excludes reserved-level
transitions. `evaluation` recording, if separately authorized, is read-only with
respect to training data and model weights and does not change the current
evaluation entry point. `unspecified` MUST NOT be silently treated as training
data. Public/synthetic/held-out provenance remains a writer/execution-policy
responsibility, not something a viewer infers from a game name.

Frame values are palette indices. A viewer may bundle a fixed SDK display palette
and present raw indices for unrecognized values; it MUST NOT fetch a palette or
change stored bytes. Actions/state/metrics and annotations must remain viewable
without resolving model/checkpoint IDs, metadata strings, URLs or paths.
Untrusted text is displayed literally. Zstandard decoding and MessagePack parsing
need only local bundled code; no network connection is part of this format.

The SDK baseline is the repository's pinned
[`arc-agi` v0.9.9 release](https://github.com/arcprize/ARC-AGI/releases/tag/v0.9.9)
and the public `FrameData`/`FrameDataRaw`, `GameAction` and `GameState` types
inspected through that package's `arcengine` dependency. Only public fields cross
the [environment adapter](../src/arc3/envs/sdk.py); no game implementation source
is needed to record or interpret this file.

## 10. Compatibility and immutability

Version 1.0's envelope, five required groups, field meanings, units, validation
rules and attempt boundaries are immutable after adoption. Do not silently
change them in a writer or viewer. Corrections that change accepted wire values
are format changes, not editorial clarifications.

A breaking change requires a new major (v2), including a change to required
fields, enum meanings, action IDs, dtype semantics, compression framing or
RESET/counting semantics. A backward-compatible optional-field addition MAY
use a minor bump only after its normative definition and compatibility tests
are accepted. It cannot make an old optional field required or change what an
existing reader accepts as a valid known field.

V1 readers MUST reject unsupported majors. They MUST accept later minor values
within major 1 when all known required/optional fields still conform to this
v1 contract, ignoring unknown optional keys after structural validation. Writers
conforming only to this document emit `1.0`. Unknown fields are never permission
to skip bounds/checksums or resolve external assets. Compression output, map
ordering and choice among equivalent integer encodings do not require a version
bump; semantic decoded equality, not byte-for-byte compressed equality, defines
interoperability.
