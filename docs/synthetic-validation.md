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
