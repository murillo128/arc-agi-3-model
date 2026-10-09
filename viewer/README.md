# Browser .arc3 replay console

This static Vite/TypeScript entry point loads the frozen [v1 format](../docs/arc3-format-v1.md)
through a local file dialog, drag/drop, a submitted HTTP(S) URL, or an explicit
`?file=<encoded-https-url>` deep-link. It requires no application backend, SDK,
game execution, Python process, account, database, or proxy at runtime.

The English replay console uses original magenta/purple arcade styling. A validated
file supplies the game, attempt/session, step/action counts, terminal reason, SDK
and model identifiers, source provenance and summary. The public completed-level
counter is shown as recorded; it is not a hidden level ID. Summary counters may
differ from the last persisted observation when an executed result was unavailable
or intentionally excluded at a training boundary.

`REAL OBSERVATION` renders the entire selected frame with Canvas 2D, using its
actual height and width, nearest-neighbor display and the bundled SDK display
palette. The adjacent thumbnail shows the **last frame of the pre-action
observation**, not a prepended result frame. Unknown palette values use an explicitly
labelled cyan fallback with their exact indices; raw bytes are preserved. A text
readout exposes up to 32 × 32 exact indices. Empty sequences show an unavailable
state and clear the old bitmap; they never borrow a previous image. Frames larger
than 16,384 on either axis or 16 Mi pixels are explicitly refused by the bitmap
renderer, without cropping; decode and the bounded text readout still work.
Browser allocation failure is also reported.

The timeline distinguishes **position 0** (initial observation), **position n**
(result of action step `n − 1`) and a closing RESET boundary, when present.
Action steps and frame indices are zero-based. Level/status markers use public
counters and states. Scrub, jump, select an action, or use Previous/Next/Restart.
The scrollable action history includes every recorded action with its actual
`ACTION1`–`ACTION7` name, exact coordinate parameters and observation/terminal
changes. Future recorded actions are dimmed, and the selection is highlighted.
A closing executed RESET is labelled separately: its result belongs to the next
attempt and is never fabricated in this file. A requested but unexecuted RESET
has no action row. Counted actions without persisted results are explicitly noted.

Episode playback visits each observation's ordered frames before moving to the
next action, preserving repeated animation frames. Default pacing holds each
last frame for one second; intermediate frames use the selected frame speed,
scaled by episode playback speed. These are viewer settings, not measured SDK
latencies. Play frames/Pause frames animates only the current observation and
stops at its last frame. Seeking or manually selecting a frame pauses playback.
Opening another file and hiding the browser tab also stop playback.

With focus on the observation display or outside interactive controls, use
Left/Right for previous/next, Space for play/pause, Home for restart and End for
the final position. Native inputs, buttons and other interactive controls retain
their normal keyboard behavior, with visible focus rings. The desktop layout
stacks at narrow widths. Optional diagnostic panels consume the frozen v1 fields
described below. Missing groups and samples show `Not recorded`.

## Optional agent and model diagnostics

`MODEL PREDICTION` and `DIFFERENCE MAP` refer to the same step result and selected
frame as `REAL OBSERVATION`; the pre-action thumbnail remains the preceding
observation's last frame. A `post_action_sequence` prediction aligns frame `i`
with real result frame `i`, requiring the full `[F, H, W]` shape to match. A
`post_action_last_frame` prediction aligns only with the last real result frame;
earlier frames show `Not recorded` rather than reuse that prediction. Empty real
sequences and unequal spatial or sequence shapes have no comparable prediction.
Latents cannot supply a visual prediction without a model-specific decoder.

The frozen format records predicted palette-index values in `u8`, little-endian
`f16` or `f32`. The viewer decodes all three. Float estimates are rounded to the
nearest integer **for palette display only**; unknown/out-of-range display indices
use the labelled cyan fallback. A bounded text preview preserves exact decoded
values. Pixel disagreement compares exact decoded values with actual indices,
before display rounding, and reports the fraction/count for the **selected frame**.
The red/black map means disagree/equal; it does not measure game success. Recorded
MAE/MSE retain their own target-wide semantics and units in a separate detail.
Visual diagnostic work is limited to **65,536 pixels per selected frame**;
larger valid frames show an explicit refusal without a partial metric.

`CHOSEN ACTION` uses the executed action even when absent from the candidate set.
`ACTION SCORES` preserves writer order and labels raw logits, probabilities or
values without normalization. Candidate subsets need not sum to one. Tables
preview at most 256 candidates and disclose truncation. The optional selected
value is labelled as a value estimate. `UNCERTAINTY` shows the recorded estimator
value and action entropy in nats, including zero; it does not derive entropy from
candidate scores. Public recorded model config is available in a detail for
estimator units/loss objectives. `LEVEL PROGRESS` uses the selected observation's
public `levels_completed / win_levels` counters. A zero target is labelled
unavailable; no game reward is fabricated.

`ONLINE LEARNING & TIMING` displays recorded optimizer updates, loss, replay-entry
counts and millisecond timings. Notes are plain text, with a bounded preview;
the separate `Inferred by viewer` event uses only public observation changes,
level counters and states. Neither is represented as unlogged model reasoning.
`LATENT TENSOR` shows dtype, shape, element count and an L2 norm in model-specific
units. Norm computation is limited to **65,536 elements**; no projection or t-SNE
is claimed. Both tensor calculations are bounded and run locally.

The diagnostics chart selects recorded uncertainty, entropy, loss, updates,
replay size or standard decision/environment/learning timings. An inclusive
zero-based step range filters up to **2,048 steps** per plot; larger traces start
with that first range and can be filtered further. Missing samples remain gaps,
including in the exact-value table. Zero and negative loss/value samples are
preserved. The selected step is highlighted when its sample is in the range.
Unknown timing keys remain visible in the timing panel. Charts, notes and tensor
diagnostics initiate no requests after opening a trace and need no ML library,
analytics service or backend.

The palette is the only reused SDK visual material, from
[`COLOR_MAP` at the pinned SDK revision](https://github.com/arcprize/ARC-AGI/blob/f12822c4d550121c35a275008d964afbbed47d2f/arc_agi/rendering.py).
Its MIT attribution/license is included in
[`public/third-party-notices.txt`](public/third-party-notices.txt), which ships
with the static build. CSS and icons are original; no game assets or mockup
screenshots ship as renderers.

## Run and build

Use Node.js 22.12 or newer and npm:

```sh
cd viewer
npm ci
npm run dev
```

Build and serve the assets with relative paths:

```sh
npm run build
python3 -m http.server 8080 --directory dist
```

Open `http://localhost:8080/`. Any ordinary static host can serve `dist/` at a
nested path, including GitHub Pages. Serve over HTTP(S); opening `index.html`
through `file://` does not provide the module/worker/WASM hosting boundary.
There is no deployment workflow. Build outputs are ignored.

No file is uploaded. Local reads never initiate a remote request. The worker
and WASM are served from the viewer's own bundled static assets, with no CDN or
external decoder service. Offline use requires those application assets to be
available locally; remote URL loads require a network connection. The remote
host must authorize the viewer's origin through CORS. Missing authorization,
network/offline failures, HTTP errors, timeouts and size refusals are reported;
download the recording separately and open it locally when remote access fails.
URL fetches omit cookies/credentials and referrers. URLs embedded in recording
metadata are never resolved. Text is assigned through `textContent`, never HTML.

## Validation and resource limits

`src/decoder.ts` exposes `decodeAttempt(bytes, cap?)`, returning version metadata
and the complete public `Attempt` view model in `src/types.ts`. Optional model,
prediction, learning and diagnostic fields retain their recorded absence; no
reward, model output or default loss is invented. Unknown optional values are
preserved after profile validation; future v1 minors are accepted.

The decoder checks the uint64 header as `BigInt`, the ordinary single-frame
Zstandard envelope, checksum flag, dictionary prohibition, window, exact content
size, block boundary and absence of suffixes. Decompression uses a fixed output
capacity of exactly the declared length; the native decoder verifies the checksum.
Only then does the bounded MessagePack parser validate unique STR map keys,
UTF-8, integer versus FLOAT tags, finite values, lengths, depth and the entire
required/recognized optional schema. Tensor bytes are validated before use,
including unaligned little-endian f16/f32 values. Chronology, pre-action
availability, RESET and summary/termination equations match the Python codec.

The default uncompressed and window cap is **512 MiB**. The API permits a lower
positive cap, never a higher one. Input streams and the decoder also refuse
compressed files larger than the cap plus 1 MiB plus the 16-byte envelope, matching
the Python file-reader policy. MessagePack has a **2,000,000 decoded-value budget**
(including keys and containers), as well as the format's 64-container depth limit.
The browser bounds URL reads to **30 seconds** and worker decode to **60 seconds**.
These additional limits are resource refusals: some otherwise valid files can
exceed them. Total browser memory can exceed the raw-output cap; the compressed
buffer, WASM heap and parsed objects also take space. Smaller files may be needed
on memory-constrained devices. Each load uses a fresh worker, terminated on
success, failure, cancellation, supersession or deadline, releasing its WASM heap.

The sole package runtime dependency is pinned `@bokuweb/zstd-wasm@0.0.27`, a WASM build of
upstream libzstd. Its [wrapper source](https://github.com/bokuweb/zstd-wasm/blob/87277ab93c2861d6c265a2ff43c7e7fb14d9a8c5/lib/simple/decompress.ts)
and [build recipe](https://github.com/bokuweb/zstd-wasm/blob/87277ab93c2861d6c265a2ff43c7e7fb14d9a8c5/build.sh)
were inspected for allocation, output-capacity, error handling and checksum use;
tests verify checksum corruption and output overflow rejection through the actual
dependency in Node and Chromium. This is a scoped source audit, not a claim of
independent third-party certification. The app guards properties that libzstd
alone permits, such as concatenated frames. npm's lockfile pins artifact integrity.
The [license notices](public/third-party-notices.txt) are copied into `dist/`.

## Focused local tests

Fixture generation needs **Python 3.12** with the installed repository package,
including its pinned SDK types, `msgpack` and `pyzstd`. It executes the Python
recorder against synthetic SDK transport doubles; it never loads actual games.
From the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
cd viewer
npm ci
PYTHON=../.venv/bin/python npm run fixtures
npm test
npm run build
npx playwright install chromium
npm run test:browser
```

On a Linux host missing Chromium shared libraries, use Playwright's documented
`npx playwright install --with-deps chromium` setup. An already installed compatible
Chromium can be selected with `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/absolute/path`.

Unit tests own header/framing/decompression/MessagePack/schema/tensor conformance
and bounded stream/URL policy. Fixture expectations come from frozen v1 section 8:
the independent baseline golden plus production-Python one/multi-frame samples,
literal pixel/float/action assertions and malformed vectors constructed directly
with msgpack/pyzstd. The generator verifies every expected accept/reject result
with Python before emitting ignored `.fixtures/` files. It does not regenerate
the repository's independent golden. An undersized output buffer is rejected by
libzstd; decoder error wording is not part of the wire contract.

Playwright owns the real built browser/worker/WASM and Canvas boundary: local file dialog,
drag/drop, sequence preservation, literal hostile text, user-gated URL success,
HTTPS deep-link, actual CORS refusal, mocked network/HTTP/size failures,
cancellation, supersession, empty frames and recovery. Python's ordinary static
server serves `dist/` at a nested path; a second loopback static fixture server
on port 4178 sends the actual CORS header for origin `http://127.0.0.1:4177`.
`tests/acceptance.browser.ts` opens three checked-in recorder goldens through
drag/drop and real cross-origin HTTP without route interception, checking their
literal pixel/count/action/reset/level expectations, synthetic prediction
disagreement, optional-panel clearing, HTTP 404, actual CORS refusal and visible
malformed/truncated/oversized/corrupt/version rejection. It retains two screenshot
artifacts under ignored `test-results/`. The other loader URL-success cases still
use mocks; none imply arbitrary remote hosts permit requests.
No CI or push/PR test trigger is added. Run these focused
tests when relevant format, codec, fixture or viewer files change, and for integrated
acceptance; unrelated changes do not require viewer validation.

`tests/replay.test.ts` owns replay chronology, separate action/frame indices,
pre-action selection, frame playback and RESET accounting. `tests/replay.browser.ts`
owns real Canvas RGBA assertions against literal SDK palette values, native frame
sizes, empty results, selected-action synchronization, level/terminal markers,
keyboard/focus behavior, playback timing/speed/pause and replacing a loaded file.
The fixture generator produces only test-owned synthetic observations through
the production Python v1 encoder. It never reads game implementation code.

`tests/diagnostics.test.ts` owns independent tiny-matrix comparison/alignment,
float decoding, bounded latent norms and sparse chart filtering. The pixel oracle
uses literal 2 × 2 examples with exactly 0/4 and 2/4 disagreements; fractional
estimates exercise comparison before rounding. `tests/diagnostics.browser.ts`
owns actual predicted/difference Canvas pixels, all score-type labels, entropy,
public level changes, hostile plain-text notes, partial/latent-only/mismatched
predictions, metric selection/filtering, replacing a loaded trace and absence of
network requests during diagnostic interactions.

Four compact committed Playwright PNG baselines in
`tests/replay.browser.ts-snapshots/` and `tests/diagnostics.browser.ts-snapshots/`
capture complete baseline and populated desktop (1440 × 1100 viewport) and
narrow (390 × 844 viewport) layouts. They were visually inspected
for readable hierarchy, correct current/pre-action panels, selected history row,
responsive wrapping, focus ring, timeline, populated and unavailable model panels. Screenshot
comparison runs alongside behavioral checks, with a fixed 100-pixel mismatch limit.
The Linux baselines use Playwright 1.61.1's Chromium 149 headless shell and
Liberation Mono. Other platforms/fonts may need a separately inspected baseline.
Do not update snapshots simply to make a failed test pass. For an intentional
visual change, generate candidates with `npm run test:browser -- --update-snapshots`,
inspect the changed images, then run `npm run test:browser` **without** updating.
Generated `.fixtures/`, `dist/`, browser binaries and test-result logs remain ignored
or outside Git; the four compact visual fixtures are deliberate acceptance evidence.

## Manual recorder-to-browser acceptance

The [fixture provenance and hashes](../tests/fixtures/README.md) distinguish the
original synthetic recorder goldens from the optional actual public SDK structure
extraction. No games, backend or Python process are needed by the built viewer;
the commands here serve static assets and fixtures locally for acceptance.

1. Run the install/fixture/build commands above (Node **22.12+**, Python **3.12**).
   To prove fresh recorder output, from the repository root also run:

   ```sh
   PYTHONPATH=src .venv/bin/python tests/fixtures/generate_arc3.py \
     --output artifacts/recorder-manual
   ```

2. In `viewer/`, start the static viewer on port 4177:

   ```sh
   python3 -m http.server 4177 --bind 127.0.0.1 --directory .
   ```

   Open `http://127.0.0.1:4177/dist/`. Drag the freshly recorded
   `artifacts/recorder-manual/recorder-baseline-0.arc3` onto **OPEN RECORDING**.
   Verify initial pixels `[0,1;2,3]`, two steps/two real actions, and absent model
   panels. Select **#0 ACTION6 (1, 0)**: counter becomes 1/3 in the same attempt;
   frame 0 is `[4,5;6,7]`, frame 1 `[8,9;10,11]`. Select **#1 ACTION1**: the
   attempt ends `game_over`. No RESET result should appear in that file.

3. Drag `recorder-baseline-1.arc3` onto the same viewer. Verify the same session,
   attempt 1, initial `[9,9;9,9]`, counter 0/3 and one action. The terminal RESET
   response starts this attempt; neither RESET is double-counted in the files.
   For an executed RESET that closes an active nonterminal attempt, open the
   generated `viewer/.fixtures/closing-reset.arc3`: the closing boundary retains
   the pre-reset observation, and its result belongs to a separate next attempt.

4. In a second terminal in `viewer/`, start the CORS fixture server:

   ```sh
   python3 tests/serve_fixtures.py
   ```

   Paste `http://127.0.0.1:4178/recorder-baseline-0.arc3` into **Recording URL**,
   select **Load URL**, and repeat the same history/frame/level checks. The
   server sends `Access-Control-Allow-Origin: http://127.0.0.1:4177`.
   Load `http://127.0.0.1:4178/recorder-prediction.arc3`, then select its action:
   real `[3,2;1,0]` and synthetic prediction `[0,1;2,3]` differ in 4/4 pixels;
   MAE 2, MSE 5. The note identifies the prediction as synthetic. Reopen a baseline
   and verify prediction/difference panels clear and read **Not recorded**.

5. Load `http://127.0.0.1:4178/missing.arc3`: verify HTTP 404. Load
   `http://localhost:4177/.fixtures/recorder-baseline-0.arc3`: this different
   origin supplies no CORS header, so verify the CORS/network error. Drag a valid
   local file again and verify recovery without a reload. Stop both servers with
   Ctrl+C. The port-4178 helper binds only loopback and is a static acceptance
   utility, not a viewer runtime service or deployment requirement.

For the automated equivalent, after fixture generation and build:

```sh
npm run test:browser -- acceptance.browser.ts --workers=1
```

The full `npm run test:browser` additionally compares the four existing desktop
and narrow English retro UI baselines. Inspect screenshots as well as behavioral
assertions; optional diagnostics must stay unavailable on no-model attempts.
Use the optional public SDK extraction command in the provenance README when
authorized environment files are already present. Missing files do not prevent
any default test. No CI/deploy pipeline, automatic download, upload or submission
is introduced by this acceptance path.
