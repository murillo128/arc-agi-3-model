# Synthetic catalog validation

Issue #60 validates the existing local play UI and all 62 synthetic entries
(370 game/level combinations). This is a synthetic reachability and interface
acceptance exercise, not a learned-agent benchmark or an all-seeds proof.

## Reproduce

Use Python 3.12, the project's unchanged `arc-agi==0.9.9` pin, and optional
Playwright/Chromium tooling. In an installed project environment:

```sh
python -m playwright install chromium
python tests/validate_synthetic.py --originals-dir /path/to/authorized/local/cache
```

Omit `--originals-dir` when no authorized originals are installed. That lane is
explicitly reported unavailable; every synthetic lane remains mandatory. The
aggregate command fails on missing dependencies, incorrect Python/SDK, failures
or skipped mandatory unit/integration cases. It runs the existing browser suite
plus `tests/skill_browser.py`, and writes revision/dependency metadata and logs
to ignored `artifacts/synthetic-validation/`. Screenshots stay under ignored
`artifacts/square-cross/`. This introduces no CI triggers or downloads of games.

## Test plan and evidence ownership

| Boundary | Independent contract / risk | Coverage |
| --- | --- | --- |
| Pure rules | Literal positive/negative mechanics, cost, reversible state, detached labels; witnesses alone do not establish rule semantics | Existing 22-context fixtures; 23,040 generated instances; 1,440 solution witnesses at seeds 0, 1, 7, 42 |
| Appearance / addressing | Cosmetic changes preserve transitions and goals; extreme hitbox corners remain usable without overlap/HUD collision | All 360 new-profile levels at seed 123, footprints 3 and 5, complete witnesses; existing eight-appearance property family |
| Progression / rendering | Later levels must add documented structure; board drawings cannot cover the command strip | Explicit attribute/range/stroke/slot progression checks and extreme legal board coordinates |
| Native SDK | Sequential `env.step` progression rather than direct level construction; old-level transition labels, cost and transient frame order | 60 × 6 × seeds 0, 1, 42, 123 = 1,440 level completions, 240 WIN/reset lifecycles; legacy seeded suites; literal program-prefix and flow-outlet pixels |
| Play protocol | Source separation, normal payload privacy, explicit debug gate, epoch protection, reset/switch/failed-load behavior | Existing Flask/SDK suite extended for catalog and human-help/debug routes |
| Browser | Native pixels, touch/keyboard coordinates, control legends, human opt-in, animations/cancellation and stale state | Every entry loaded/actioned/reset on desktop/mobile; 22 representative contexts solved sequentially through all six levels on both viewports; existing legacy/fixture acceptance |
| Original cache | Compatibility only, without private-rule inspection or solving claims | All discovered authorized cached originals loaded, given an available action, reset and switched back to synthetic |

The four structural regression families failed against baseline
`a9a1ec01278dcb14721ca8f4c3278f14fcc2271c`: Keys had only three structural
configurations across six levels, Linkage had five, Sigils plateaued in late
levels, and recursive programs reused the same slot count in adjacent levels.
The two board regressions also failed there: large Transform and Twins drawings
reached the control strip. The corrected tests run against an isolated baseline
archive, without changing the implementation checkout. The repaired pure suite
passes those checks and the complete generation/witness matrices.

## Repairs and limits

Keys now increases its state alphabet after level 3; Linkage increases rod range
through level 6 and scales long rods to fit; Sigils increases required strokes;
recursive programs increase slots through six. Large Transform/Twins boards fit
above controls. These changes preserve IDs and mechanical contexts, but seeded
late-level puzzles in those families intentionally differ from the baseline.
Shortest path length and empirical learning difficulty are not claimed monotonic.

Human HELP comes from the active synthetic puzzle's public command descriptions
and one explanation per mechanical context. It does not duplicate the 60-entry
catalog. `--debug-subgoals` permits a separate local human endpoint; the browser
still requires a checkbox opt-in. The default play/action payload and policy
adapter are unchanged. Originals and legacy games without this interface are
reported unsupported. No puzzle decks, witnesses or full internal state are
exported. Subgoals describe the current level; action-completion teacher labels
remain old-level snapshots in the existing explicit training interface.

Two old play-protocol assertions expected only two games; the original browser
suite also assumed an unrandomized Square Cross palette. Updated assertions
retain source separation and independently identify the visible budget role.
Presentation-only badge fixtures retain the live epoch so they do not trigger
unrelated stale-session rejection on the new human-help read.

## Execution environment and status

Validation uses CPU, Python 3.12.14, arc-agi 0.9.9, ARCEngine 0.9.3,
NumPy 2.5.4, Playwright 1.64.0 and Chromium 156.0.8078.4 (build 1248).
The launcher confirmed GPT-6 Astra / high for this execution. No training
campaign, model/teacher integration, dashboard or shared App Server was changed.
PR #58 remains an independent workstream; its overlapping legacy sources are
not modified here.

Final exact-revision results are recorded in the PR after acceptance completes.

## Persistent local refresh: pending reviewed merge

The user explicitly chose **“Wait for reviewed merge”** during execution. This
supersedes the pre-handoff deployment timing, not the requirement to eventually
verify the actual persistent URL. Test-server success is not live deployment.

Read-only host inspection on 2026-10-15 found `http://127.0.0.1:8001/` owned by
PID 2592207, started as `.venv/bin/python scripts/play_game.py --mode web --seed 42`
from the now-deleted issue-20 worktree under a Codex sandbox. No dedicated play
service unit exists. The durable repository is on PR #58's training branch and
must not be reset or reused as a clean deployment checkout. No live play process
was stopped, and no current served revision is claimed to include this PR.

After reviewed merge: re-identify the listener (do not trust the old PID alone),
prepare a durable checkout/snapshot of the reviewed merged revision and its
Python 3.12 environment, preserve port 8001 and the normal local command, replace
only the identified play process, then verify a new profile, a legacy entry,
contextual HELP and inspector off/on in the browser. Enable `--debug-subgoals`
only for the local human debug capability; its checkbox must still start off.
Record the actual served revision and startup method. Do not host from a
throwaway issue/review worktree or disturb the training checkout.


## Foundation tier (issue #62)

The catalog now adds `bf01-v1`–`bf08-v1`: eight separate contexts and 48 short
levels. The 62 previous entries remain intact, including sy08's cyclic
prerequisites and sy26's nonmonotonic rods. No model, campaign selection,
training policy, dashboard, or SDK pin changes are part of this addition.

### Progression and measured witness lengths

The table records constructive real-action witness lengths for seeds 0–63.
The witnesses use advertised click hitboxes or directional actions, without
selection or execute overhead. These are finite engineering checks, not a
human study, learned-agent result, transfer claim, or proof for every seed.
Memory's counts assume the pairs are known; blind exploration may take longer.
All opening levels take 1–2 actions; observed later witnesses take at most six.

| ID / context | Structural progression over levels 1–6 | Observed actions, L1 → L6 | Suitable later practice |
| --- | --- | --- | --- |
| bf01 / Match a color | Independent two-colour tiles: 1, 1, 2, 2, 3, 3; even levels consolidate | 1, 1, 2, 2, 3, 3 | sy01, sy02, sy06 |
| bf02 / Unlock the next control | One switch; chain of 2; two independent goals; chain of 3; two chains of 2; two branches feeding a final switch | 1, 2, 2, 3, 4, 5 | sy08, sy09, sy27 |
| bf03 / Move onto a target | One object; distances 1, 2 straight, 2 with a turn, 3, 4, 6 on an unobstructed 5×5 board | 1, 2, 2, 3, 4, 6 | sc01, sy11, sy14 |
| bf04 / Turn a shape | One piece at 1/2/3 clockwise quarter turns; second asymmetric shape at L4; two independent pieces at L5–6 | 1, 2, 3, 1, 2, 3 | sy11, sy15 |
| bf05 / Follow a reflection | One fixed vertical mirror; one controlled object and one reflection; distances 1–6, direction and placement vary | 1, 2, 3, 4, 5, 6 | sy12, sy13 |
| bf06 / Turn one loop | One loop with 2, 2, 3, 3, 4, 4 tokens; paired levels consolidate the same structure | 1, 1, 1, 1, 1–2, 1–2 | sy31, sy32 |
| bf07 / Transfer one unit | Two tanks with capacity 2, 2, 4, 4; three tanks with capacity 4 at L5–6; conserved units, both directions | 1, 2, 3, 4, 4, 6 | sy40, sy41 |
| bf08 / Remember a pair | Pairs: 1, 1, 2, 2, 3, 3; even levels consolidate; no timer or reshuffle | 2, 2, 4, 4, 6, 6 (known pairs) | sy55–sy57 |

Logical generation and appearance use separate deterministic streams. Appearance
changes palettes, sizes and offsets while preserving state, targets and rules.
Undo costs one action and restores the prior command state; it can reverse a
monotonic switch or matched pair. RESET preserves the existing native retry/full
restart contract. Budgets remain generous (96–256 actions), with no new hazards.
Memory emits eight reveal frames before hiding a mismatch; the existing player
shows the faces for about 640 ms. Under reduced motion, the foundation pair
game holds one native reveal image still for 640 ms before hiding; RESET or
switching cancels that hold through the existing presentation token. Old games
retain their existing reduced-motion presentation. No rule, SDK frame, action
cost or actor-payload field changes are needed.

### Test plan, provenance and execution

The existing on-demand command `python tests/validate_synthetic.py` now includes
all foundation lanes in addition to every applicable existing synthetic check.
Missing Python 3.12, arc-agi 0.9.9, real ARCEngine, Playwright or Chromium fails
acceptance. It adds no always-on workflow or new testing framework.

| Boundary / primary test owner | Independent assertions and defects covered | Matrix / evidence |
| --- | --- | --- |
| `test_foundations.py`: literal rules | Hand-written expected states for all eight mechanics; two colours only, prerequisite blocking including arbitrary downstream nodes, final conjunctions, clockwise rotation, mirrored directions, capacity/empty guards, mismatches, permanent forward progress and Undo | Literal small fixtures are independent of the production transitions; solving witnesses alone cannot establish semantics |
| `test_foundations.py` + `foundation_reference.py`: generation | Non-solved starts, prescribed progression, real-click witness bounds, no text glyphs, detached labels, same-seed instances; cosmetic interventions cannot alter logical state or target | All 8×6×64 = 3,072 instances, each at native appearance and two palette/placement/footprint extremes (9,216 exercised variants); non-clipping drawing assertions and disjoint inclusive hitbox corners |
| `test_foundation_sdk.py`: real adapter | Discovery, all six sequential levels, native WIN/loss/retry/full reset, one action per transition, old-level before/after versus new-level labels, simultaneous tank goals, perceivable memory frames and ordinary FrameData privacy | 192 level completions at seeds 0, 1, 42, 123; all eight native lifecycle checks |
| `legacy_fingerprint.py`: preservation | Base-derived initial frames, seeded action/state/subgoal/animation fingerprints, native frames and resets, without refreshing expectations from new behavior | All 62 old IDs at seeds 0 and 42; all six pure sy levels plus real SDK sequences; existing full legacy/skill suites retained |
| `foundation_browser.py`: real input/rendering | Every foundation solved through all six levels on desktop/mobile; arrows via keyboard/D-pad, pieces via mouse/touch, canvas/native pixels, accurate six-level badge, HELP/later practice, debug opt-in, Undo/reset, memory reveal timing/cancellation and switching back to sy08/sy26/sc01/si01 | Intro/final screenshots for all eight contexts in both viewports; existing protocol and both browser regression suites also required |

The committed legacy fingerprints were captured from an isolated archive of
`ad2b6f756d47ec02485995690e38b09cb0b35136`, using the same Python/SDK environment.
They cover `sc01-v1`, `si01-v1`, and every ID from `sy01-v1` through `sy60-v1`.
To reproduce the baseline capture, extract `git archive <base>` into an isolated
directory and run `python tests/legacy_fingerprint.py /path/to/base output.json`.
The test compares that compact fixture with a fresh digest of the current tree;
legacy source files and the shared adapter are unchanged.

Visual inspection of introductory/final examples found that the reflection's
outline originally touched its fill. A literal pixel-gap regression failed
before the fix and passed after reducing the inner reflection footprint; the
controlled solid and outlined reflection are now visually distinct at both size
extremes. Browser fixture repair focused the actual directional button before
keyboard input (the canvas is not focusable). Consolidation geometry initially coincided for some seeds (for example bf01
seed 29 at L3/L4 and bf08 seed 11 at L1/L2). A regression failed before adding
deliberate even-level placement changes and passed afterward; the pair/tile
counts and rules stay the same. Catalog assertions now identify
Originals by source instead of assuming group index 1, preserving source/privacy
checks after adding the foundation group.

Full logs, per-level structural/action ranges, environment/revision metadata and
screenshots remain in ignored `artifacts/synthetic-validation/`. The original
browser suites also retain their ignored `artifacts/square-cross/` screenshots.
Validation uses CPU, Python 3.12.14, arc-agi 0.9.9, ARCEngine 0.9.3, NumPy 2.5.4,
Playwright 1.63.0 and Chromium 153.0.8010.12. The existing runner settings confirm
GPT-6 Astra / high. Exact final results belong to the implementation PR.

### Reduced-motion audit correction

The first independent audit found that the shared player discarded the new
memory game's informative mismatch frames when reduced motion was enabled.
The real Chromium regression in `check_reduced_motion_pairs` failed on
`a0cb57df45a29be6e8b1786a571571c689810373`: neither canvas paint showed both
face colours. The test uses the actual card centres/face colours, rather than
assuming that any returned animation frame contains the required information.
It now requires a still both-face reveal for at least 500 ms on desktop/mobile,
followed by hiding, with unchanged one-action cost and no teacher fields.
Actual RESET and game-switch inputs cancel the hold without a delayed repaint.
The existing ordinary-motion, old-game reduced-motion and epoch/cancellation
regressions remain required in the on-demand acceptance command. Red/green
logs are retained under ignored `artifacts/synthetic-validation/audit-repair/`.

### Live refresh remains a separate post-merge step

Read-only inspection of the durable deployment record on 2026-10-15 identifies
release `ad2b6f756d47ec02485995690e38b09cb0b35136` under
`/home/sergio/.local/share/arc-agi-3-play/releases/`, system service
`arc3-play.service`, loopback port 8001 and the existing `/arc3-play/` proxy.
This supersedes the older pre-deployment host observations above. It is a record,
not a fresh process-liveness or public-endpoint claim.

Issue #62 preserves reviewed-merge-before-live-refresh. This implementation does
not restart or deploy the game service. After reviewed merge, re-identify the
listener and current release paths, refresh only that service through its durable
release mechanism, and verify the actual user-facing endpoint contains 70
synthetic entries (eight foundations plus the unchanged 62), along with legacy
controls, HELP and inspector privacy. Keep training, its checkout, dashboard,
shared App Server, proxy and TLS settings untouched. Never serve this issue's
ephemeral worktree as a live release. Isolated acceptance is not live deployment.

## Non-linguistic presentation revision (issue #64)

The baseline is `06e7eb3e8c54dee071a8cfbec3d03d2a9f0a25da`. Revision
`symbols-v2` changes presentation and coordinate addressing, not the logical RNG,
state machine, semantic command order, budgets, levels or teacher interface.
Historical pixel recordings and pretrained visual features still describe the
old appearance; do not rewrite them or refresh the active training campaign.

Test plan before authoring: `test_symbolic_presentation.py` owns literal pixel
fixtures for separate goals/prerequisites, same/different samples, exact empty /
missing / excess / maximum quantities, switch examples and clock phases. Literal
coordinates and independently specified unit counts detect the previous decimal
and letter presentation. Palette and footprint interventions, direct-hitbox
clicks and pure reads exercise the native renderer, not a browser overlay.
Existing rule, reachability, SDK lifecycle and browser suites remain the owners
of mechanics, action costs, transient order, privacy and frontend concurrency.
A baseline archive replay supplies independent logical and unaffected-pixel
fingerprints for every synthetic ID at seeds 0, 1, 42 and 123. The new browser
lane inspects boards with HELP and the teacher inspector closed. Generated image
matrices and detailed logs belong only in ignored artifacts.

The executor launch receipt confirms `gpt-6-astra` / `xhigh`; the launcher's
validated issue override and observed new-thread selection agree. Final audit
is a separate controller with the same issue override. Live deployment remains
pending reviewed merge. The inspected deployment record names release
`06e7eb3e8c54dee071a8cfbec3d03d2a9f0a25da`, port 8001 and
`https://murillo128.hopto.org:8443/arc3-play/`. The system service reports
active with MainPID 72465, the same durable release and the port listening.
The deployment owner must re-identify it again immediately before refresh.

### Complete source-path inventory

Catalog reconciliation: 70 registered synthetic IDs, 418 levels. Native images
flow through `SkillGame` → `Puzzle.render` → the context renderer (or the two
Square Cross renderers). All 408 skill/foundation levels were inspected through
that boundary; the remaining ten are the unchanged Square Cross levels. Every
row below covers all of its registered levels, including native feedback frames.
`Canvas.label` and `Canvas.digit` remain unused legacy helpers; no authored game
render path calls them. Abstract dictionary shapes and sigil patterns remain
objects to match, with no interpretation as letters or decimal values.

| Context / IDs | Previous intentional labels or controls | symbols-v2 representation |
| --- | --- | --- |
| Dials / sy01–sy05 | Letter buttons; conventional equality/inequality signs | Tap each dial; concrete paired same/different samples on neighbour links; outgoing coupled-effect arrows; ordered colour samples with return edge |
| Keys / sy06–sy08 | Letter buttons and attribute indices; detached prerequisite swatches | Spatially aligned goals and direct attribute controls; predecessor links carry actual enabling state; separate cycle and exchange pictogram |
| Maze / sy09, sy27–sy30 | Lettered movement/bridge panel | Direction arrows in mover colour and a bridge span; board, keys and doors unchanged |
| Rules / sy10, sy47–sy48 | No literacy requirement; genuine dictionary glyphs | Unchanged, including rotated glyph identity and direct answer cells |
| Transform / sy11–sy15 | Lettered movement/rotation/reflection/axis panel | Direction and turn arrows, reflected shapes and axis movement pictograms |
| Paint / sy16–sy18 | Lettered paint selectors plus disconnected mask key | Each control contains its actual affected mask and paint sample; paired paints per region |
| Twins / sy19–sy21 | Lettered direction panel | Two mover-coloured arrows per command show shared/mirrored motion |
| Centroid / sy22–sy23 | Lettered arm selectors | Direction-arrow pairs sit on each arm’s row with a connecting line; the body sharing the first arm’s colour stays spatially distinct |
| Boxes / sy24, sy35–sy36 | Lettered direction panel | White player-direction arrows; block targets and contact behavior unchanged |
| Linkage / sy25–sy26 | Lettered rod selectors; detached generic staging icon | Rod-coloured axis arrows; prerequisite links carry the predecessor’s minimum-length/orientation sample, separately from target crosses |
| Loops / sy31–sy32 | Lettered loop/direction selectors | Turn arrows connect to the whole-loop and overlapping-short-loop rails |
| Pegs / sy33–sy34 | Decimal one-peg goal | One outlined required peg and discrete remaining units; board selection unchanged |
| Agents / sy37–sy39 | Lettered movement/wait panel | Controlled-mover direction arrows and wait bars; autonomous mover remains uncommanded |
| Tanks / sy40–sy41 | Lettered transfer panel | Transfer arrows between adjacent tanks; capacity ticks retain exact fill/target-height distinction |
| Flow / sy42–sy44 | Lettered valve/run selectors | Direct boxed valves on actual pipes; send triangle; native propagation and outlet marks retained |
| Gravity / sy45 | Lettered step/reverse panel | Lateral arrows and a gravity-reversal arrow; full settle behavior unchanged |
| Merge / sy46 | Decimal exact counts, including zero | Every tier has required hollow units above present solid units; capacity includes all twelve starting items, zero, missing and excess; blocks remain directly selectable |
| Sigils / sy49 | Decimal target/current attribute counters | Separate hollow required and filled current units, empty slots and wrap cue; patterns and drawing cells retained |
| Program / sy50–sy54 | Lettered bit/run selectors; literal opcode example | Direct paired switch cells, run triangle, all four concrete configuration/output examples, empty no-op tray and nested call/return brackets; all 36 possible output units fit without truncation |
| Memory / sy55–sy57 | None; hidden card faces | Unchanged; identities remain hidden until native reveal |
| Ghost / sy58 | Lettered movement/record/wait panel | Mover-coloured arrows, solid/outlined copy with return arrow, wait bars; recording slots and replay unchanged |
| Timing / sy59–sy60 | Decimal hazard phases; lettered command panel | Aligned hazard/clock phase tracks, current insets, forward order and wrap edge; solid active/hollow inactive hazards; advance/wait/retreat pictograms |
| Basic colour / bf01 | None | Unchanged direct tiles and targets |
| Basic unlock / bf02 | None | Unchanged connected switches |
| Basic movement / bf03 | None | Unchanged pictured arrows |
| Basic rotation / bf04 | None | Unchanged direct shapes and turn arrows |
| Basic reflection / bf05 | None | Unchanged mover/reflection and arrows |
| Basic loop / bf06 | None | Unchanged two turn arrows |
| Basic transfer / bf07 | None | Unchanged arrows and discrete target slots |
| Basic memory / bf08 | None | Unchanged hidden faces and native reveal |
| Square Cross / sc01 | None | Unchanged native sprites and HUD |
| Square Cross Interactions / si01 | None | Unchanged native sprites, selection and HUD |

Thus 54 skill IDs change appearance, while six skill IDs, eight foundations and
two Square Cross IDs remain pixel-identical on the captured trajectories. No
original SDK game source is edited. Native payload fields and the adapter are
unchanged. Human HELP and `skill_catalog.py --game …` derive command effects and
coordinates from the renderer’s existing commands/hitboxes; JavaScript contains
no duplicated per-game map. Text and numeric coordinates are outside the canvas.

### Coordinate migration and retained semantics

Semantic command index/order is stable. The old generic grid used
`(2 + 6*(i%10), 51 + 6*(i//10), 5, 5)`; generic pictorial controls now use
`(1 + 8*i, 53, 7, 7)`. Rectangles are `(x,y,width,height)` in native pixels.
Context overrides, also returned by the human catalog, are:

| Controls | New rectangle mapping |
| --- | --- |
| Dial i | `(4+10*i,27,8,8)` |
| Attribute i; optional exchange | `(6+20*i,25,9,9)`; `(54,42,9,17)` |
| Paint command i | `(2+12*(i//2),44+8*(i%2),10,7)` |
| Centroid arm command i | `(47+8*(i%2),7+10*(i//2),7,7)` |
| Twin direction i | `(2+15*i,52,13,9)` |
| Loop command i | `(3+11*(i%2)+31*(i//2),48,9,11)` |
| Tank transfer i | `(11+14*(i//2),47+8*(i%2),11,7)` |
| Valve i; send | `(9+7*i,10+5*i,5,5)`; `(51,45,10,14)` |
| Merge block i | `(5+9*(i%6),42+10*(i//6),8,9)` (previous y: 23/36) |
| Program bit i; run | `(2+9*(i//2)+4*(i%2),29,4,7)`; `(56,28,7,10)` |
| Ghost movement; record; wait | First four generic rectangles; `(34,51,17,10)`; `(54,53,7,7)` |

All other direct board hitboxes remain unchanged. Replays must address semantic
commands using the appropriate revision’s rectangles; old raw click coordinates
are not an equivalence oracle after a documented layout change. Keyboard actions,
miss/blocked/Undo costs, resets and action availability remain unchanged.

### Reproduction and evidence limits

`tests/presentation_replay.py` runs the same test-only witnesses against a chosen
root. `tests/fixtures/symbols-v1-behavior.json` was captured against the isolated
baseline archive, using the candidate test harness and the same pinned SDK.
It hashes logical states, targets, every command alternative at each witnessed
state, costs/history, SDK action availability, old-level transition labels,
sequential progression and resets. Pixel hashes are checked only for the 16
unaffected IDs. The historical 62-game fingerprint file is retained verbatim;
its eight still-applicable legacy checks remain active. The other 54 paths are
covered by the new logical reference and literal presentation fixtures, not
silently updated historical screenshots.

```sh
mkdir -p /tmp/arc3-symbols-base
git archive 06e7eb3e8c54dee071a8cfbec3d03d2a9f0a25da | tar -x -C /tmp/arc3-symbols-base
python tests/presentation_replay.py --root /tmp/arc3-symbols-base \
  --output artifacts/symbols-v2/baseline --images \
  --capture-baseline /tmp/symbols-v1-reproduced.json
python tests/presentation_replay.py --output artifacts/symbols-v2/current --images
python tests/validate_synthetic.py --originals-dir /path/to/authorized/cache
```

The `--images` lane saves each ID’s all-level initial/progressed/completed image
matrix at four logic seeds, plus all-level size/offset/palette interventions and
compressed real SDK native-frame arrays (including transients and terminal
observations). It does not edit previous recordings. Literal representation
fixtures fail on the baseline for the intended missing cues, not setup errors.
Program feedback now uses separated 2×2 units to fit the maximum sequence; its
existing native-prefix assertion intentionally changes to 4/8/12 coloured pixels.
Switch footprints still vary with the appearance size.

Visual inspection covers before/after introductory, middle and advanced images
for every changed renderer, the mandatory sy08 level 3 predecessor/goal split,
sy26 staging, empty/excess merge trays, sigil counters, program dictionaries and
hazard phases. That inspection found and corrected overwritten enabled arrows,
loop connections, orientation-specific rod samples, tiny direction chevrons and
explicit exchange/arm connections when object colours coincide. Browser screenshots keep
HELP/debug closed for introductory and advanced context states, including every
program variant. These are engineering checks and visual inspections, not a
usability study, an OCR score, measured comprehension or a benchmark improvement.

### Executed acceptance, 2026-10-15

The complete on-demand command passed against clean implementation commit
`2f5c64aa41010becfbc7b1a3c6c1f95253f24d51`. This result record is a documentation-only
follow-up; executable code, fixtures and tests are unchanged from that target.

| Executed boundary | Result |
| --- | --- |
| Rules / SDK / protocol / literal presentation | 117 tests passed, zero skips, 66.633 seconds |
| Baseline semantic replay | All 70 IDs matched; 1,672 sequential SDK level completions across seeds 0, 1, 42, 123; 22,305 native frames observed |
| Unaffected pixel replay | All 16 unchanged IDs matched baseline hashes; precisely the 54 inventoried skill IDs changed pixels |
| Regression proof | All ten presentation test families failed on the isolated baseline for assertion-level visual defects; all passed on the implementation |
| Existing play browser suite | Passed real Chromium acceptance, including native/display pixels, source switching, keyboard/touch, action gating and epoch/atomic switching regressions |
| Expanded skill browser suite | Passed desktop and mobile: every synthetic entry loaded/actioned/reset; 29 context variants each completed levels 1–6; HELP/inspector, animations, reduced motion and cancellation passed |
| Foundation browser suite | Passed desktop and mobile: eight six-level WINs per viewport, solution bounds, controls, privacy, Undo/reset and still mismatch reveals under reduced motion |
| Authorized original-cache smoke | ft09, ls20 and vc33 loaded, took legal actions, reset and switched back to synthetic on both viewports; no code downloaded or edited |
| Visual evidence | 70 before and 70 after image matrices, all-level appearance extremes, native transient archives and actual browser screenshots with HELP/debug closed; selected images inspected as described above |

The passing browser runners assert no unexpected page, console, request or HTTP
errors. They do not measure human/agent understanding. Runtime dependencies were
Python 3.12.14, arc-agi 0.9.9, ARCEngine 0.9.3, NumPy 2.5.4, Playwright 1.63.0 and
Chromium 153.0.8010.12 on CPU. The pinned project SDK requirement is unchanged.

Detailed logs and the clean tested revision/dependency record are under
`artifacts/synthetic-validation/`; native before/after matrices, compressed frame
archives and regression red/green logs are under `artifacts/symbols-v2/`.
Browser captures are under `artifacts/square-cross/`. These remain ignored.
No live release was refreshed: independent review/merge and actual durable
endpoint verification remain separate from this successful isolated acceptance.

### Reflection-axis audit correction

The audit of `79c84f5a8374527067e3274715af8ff08073482f` found that Transform's
reflection pictogram always showed a vertical axis, although the existing
flip-before-rotation semantics reflect the visible piece up/down at odd quarter
turns. The correction preserves those semantics and orients the native cue to
the visible effect; human guidance refers to the shown axis.

Regression plan: extend the literal presentation tests with independently drawn
vertical/horizontal 7×7 control fixtures and explicit reflected coordinates for
an asymmetric three-cell piece at all four rotations, both flip states and
palette/size extremes. Extend the real SDK tests with the audit's reachable
`sy11-v1`, seed 0, level 3 sequence, asserting native pixels and coordinates
before/after Rotate and Reflect. Existing replay owns all-profile logical
equivalence; the existing real browser suites own display/input/HELP integration.
Run the new regressions on the pre-fix implementation before the correction,
then run focused coverage and the complete on-demand acceptance path.

The new literal fixture failed on the pre-fix code for all 16 odd-rotation
palette/size/flip combinations, while the even-rotation cases passed. The SDK
reproducer also failed at the horizontal-axis pixel assertion, after successful
native setup and rotation. Both now pass, along with all 11 presentation tests
and the focused HELP protocol test. Red/green logs are retained in
`artifacts/symbols-v2/reflection-*.log`. Browser coverage also exercises a rotated
and reflected level-three piece through desktop clicks/mobile touches, captures
both native/display states with HELP closed, and checks the updated control text.
