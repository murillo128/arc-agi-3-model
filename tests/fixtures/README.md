`arc3-v1-baseline.hex` is a frozen, tiny synthetic wire fixture for baseline B in
`docs/arc3-format-v1.md` section 8. It was assembled independently of the
production encoder from the literal baseline values using MessagePack 1.2.3
(`use_bin_type=True`), pyzstd 0.19.1 (checksum and content size enabled), and a
little-endian `ARC3`, major 1, minor 0, payload-length envelope. Whitespace in
the hex file is only a text storage convenience. It contains no SDK game data.

Tests compare its decoded map and tensor values with the specification's literal
values. Other vectors build wire bytes directly with the installed libraries,
bypassing production encoding, and separately verify semantic round-trip.
Compression/map ordering is not canonical, so different interoperable encodings
need not reproduce this fixture's compressed bytes. Do not refresh it to match
an implementation change.

## Stored recorder acceptance samples

Three original synthetic `.arc3` files total **1,354 bytes**. Their SHA-256
hashes are frozen in [arc3-sha256.json](arc3-sha256.json). They were generated with
Python 3.12, `arc-agi==0.9.9`, `arcengine==0.9.3`, MessagePack 1.2.3 and pyzstd
0.19.1. [generate_arc3.py](generate_arc3.py) owns the deterministic input and
executes the actual `SDKEnvironment` → `RandomPolicy`/synthetic diagnostic policy
→ `run_game` → `EvaluationRecorder` → atomic stored files path. The SDK transport
uses concrete `FrameData`/`GameState` types with literal public responses. It
contains no game rules and does not execute an actual game. Only UUID, UTC clock
and recorder elapsed times are fixed to produce repeatable bytes.

Independent expected results, from frozen v1 sections 7–8 and these literal inputs:

| File | Semantic expectation |
| --- | --- |
| `recorder-baseline-0.arc3` | Initial `[[0,1],[2,3]]`; seed 42 chooses ACTION6 `(1,0)`; returned frames `[[4,5],[6,7]]`, `[[8,9],[10,11]]` in order; level counter 0→1 in the same attempt; ACTION1 returns `[[3,2],[1,0]]` and GAME_OVER; two real actions, no model groups. |
| `recorder-baseline-1.arc3` | Same session, attempt 1; terminal RESET's returned `[[9,9],[9,9]]` is the new initial observation with counter 0; one ACTION1 returns all-zero pixels; one real action, budget closure. |
| `recorder-prediction.arc3` | Synthetic pre-action prediction `[[0,1],[2,3]]` remains distinct from the returned `[[3,2],[1,0]]`; absolute errors `[3,1,1,3]` yield MAE 2 and MSE 5; four differing pixels; one executed ACTION6 `(1,0)`. This prediction is an authored example, not a trained model or real SDK prediction. |

The baseline session makes five calls: bootstrap RESET, ACTION6, ACTION1,
terminal RESET, ACTION1. Both RESETs occur outside an active attempt, so the files
account for 2+1 actions; session usage is five. The terminal attempt is finalized
before RESET and has no closing `reset_action`. The different nonterminal closing
RESET boundary remains covered by the codec vectors, recorder tests and viewer's
`closing-reset` sample. No fabricated post-reset frame is appended to the old file.

[test_arc3_fixtures.py](../test_arc3_fixtures.py) checks literal pixels, action and
attempt boundaries, absent diagnostics, prediction errors, tiny size, frozen
hashes and freshly recorded bytes. It also owns extraction/bootstrapping/reserved
level guards for the optional capture below. Compression/map ordering is not
canonical across dependency versions: fresh outputs must match frozen decoded
semantics, and repeat runs with the same dependencies must produce identical bytes.
Browser acceptance consumes the checked-in goldens. The viewer generator verifies new recorder semantics against them
before copying them into ignored `.fixtures/`; it cannot silently refresh them.
Malformed/truncated/over-cap/checksum/version vectors remain owned by the existing
codec generator/tests, with representative visible rejection in browser acceptance.

From the repository root after installing the package, regenerate into a **new**
ignored directory and compare; do not overwrite goldens to make a test pass:

```sh
PYTHONPATH=src .venv/bin/python tests/fixtures/generate_arc3.py \
  --output artifacts/recorder-acceptance-local
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests \
  -p 'test_arc3_fixtures.py' -v
```

## Optional actual public SDK structure extraction

On 2026-10-09 an offline run used the locally provisioned public
**`ls20-9607627b`**, `arc-agi==0.9.9` / `arcengine==0.9.3`, seed **42** and
**ACTION1 `{}` → ACTION2 `{}` → ACTION3 `{}`**. The example split explicitly
assigns `ls20` to training with cap 1, disjoint from `ft09`/`vc33`. `make` returned
an initialized observation; no bootstrap RESET was needed. Initialization and
all three responses each exposed one 64×64 frame, state `NOT_FINISHED`, public
counter 0/7 and available actions `[1,2,3,4]`. No level transition, terminal state,
click or multiframe behavior was observed in this actual run; those behaviors
above are synthesized from the documented SDK/format contracts, not claimed
captures. This is interface evidence, not game-completion or performance evidence.

[capture_public_sdk.py](capture_public_sdk.py) reproduces the short run using
only public adapter methods/fields, in OFFLINE mode without downloads. It checks
the training split and delegates discard-before-record to the existing runtime's
level cap, including the reserved final level. It never reads game implementation
code or saves original pixels. Its local `structure.json` contains only public
shapes/state/counters/actions, while `synthetic-public-structure.arc3` retains
sequence counts/state/counters/action IDs and replaces **every** source bitmap
with fresh 2×2 ramps `(phase + pixel_index) % 16`. It does not downscale, sample
or recolor game pixels. Dimensions, palette/pixels, fixture identity, timestamp
and zero wall time are synthesized; real source version, seed, actions and public
structure remain identified in provenance/config. No model diagnostics are added.
The output is `unspecified` provenance, not a training dataset.

The [official SDK offline documentation](https://docs.arcprize.org/toolkit/arc_agi)
supports local development. The SDK package declares MIT licensing; that alone
does not establish redistribution rights for separately downloaded game material.
No game-source or direct frame redistribution permission was established here.
Consequently **no direct capture or game-derived binary is checked in**; the
three committed files above are entirely original synthetic examples. Local
structure extraction and its synthetic output stay ignored, and may be shared
only after checking the source game's applicable terms. The command needs already
authorized local public files; when they are unavailable, the synthetic test
suite still runs without downloads or credentials.

```sh
PYTHONPATH=src .venv/bin/python tests/fixtures/capture_public_sdk.py \
  --environments-dir environment_files --split configs/splits.example.json \
  --game ls20-9607627b --seed 42 --output artifacts/public-sdk-structure-local
```

Never substitute an evaluation/held-out game or remove the level guard. Open the
generated synthetic derivative with the same [manual viewer acceptance](../../viewer/README.md#manual-recorder-to-browser-acceptance).
