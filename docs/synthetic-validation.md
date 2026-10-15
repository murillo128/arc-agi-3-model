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
shows the faces for about 640 ms unless reduced motion suppresses animation.

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
keyboard input (the canvas is not focusable). Catalog assertions now identify
Originals by source instead of assuming group index 1, preserving source/privacy
checks after adding the foundation group.

Full logs, per-level structural/action ranges, environment/revision metadata and
screenshots remain in ignored `artifacts/synthetic-validation/`. The original
browser suites also retain their ignored `artifacts/square-cross/` screenshots.
Validation uses CPU, Python 3.12.14, arc-agi 0.9.9, ARCEngine 0.9.3, NumPy 2.5.4,
Playwright 1.63.0 and Chromium 153.0.8010.12. The existing runner settings confirm
GPT-6 Astra / high. Exact final results belong to the implementation PR.

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
