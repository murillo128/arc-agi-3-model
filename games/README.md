# Original synthetic games

This directory contains our own versioned ARCEngine environments, separate from
ignored downloaded `environment_files/`. The SDK discovers 71 entries (424 levels):
eight six-level foundation games, 60 six-level skill curricula, one six-level maze,
and two independent five-level legacy games:

- [Square Cross](square_cross/README.md), `sc01-v1`: five perception/navigation levels.
- [Square Cross Interactions](square_cross_interactions/README.md), `si01-v1`:
  five levels: click one square, select two simultaneously, then three
  colour/navigation tasks transferred from former Square Cross levels 6–8.
  Selection in the later tasks remains exclusive.
- [Square Cross Maze](maze_cross/README.md), `mc01-v1`: six levels with fixed walls
  and coloured floor, progressing from straight corridors through turns to dead
  ends and loops. Available for world-model training without a subgoal teacher.

These games are not held-out evaluation games and do not change the repository's
training/evaluation splits. No learning or training is performed by the launcher.

## Synthetic subgoal teacher

After the native seeded initialization, `sc01-v1` and `si01-v1` expose the optional
`games.subgoals.SyntheticSubgoals` capability to the local training adapter:

- `training_subgoals()` returns an ordered tuple of immutable target images and
  masks for the randomized level. Navigation uses the fewest straight segments
  among shortest legal routes (one to three for Square Cross); interaction
  levels use selection or delivery subgoals. It never exports solved actions.
- `training_goal_reached(goal)` checks the real native state, including selection
  events that cannot always be represented as a new picture.
- `training_goal_distance(goal)` counts the minimum **SDK actions** at goal
  activation, including clicks even when the game does not charge its HUD budget.
  Other pieces are stationary at these activation boundaries; this is not a
  general joint rearrangement oracle for arbitrary states of two pieces.

Only the current goal's image/mask crosses into the agent. The ordered list,
private geometry, minimum distance and exact success predicate remain on the
synthetic trainer's side. They are neither fields in SDK observations nor inputs
to the path planner's search. Reset/level randomization invalidates the list.
The HUD and unconstrained other pieces are excluded from goal masks. For the first click level, which has no
persistent selection marker, the mask identifies the object to click without
inventing a visual change. Goal generation renders copies, never mutates the
live game or spends actions. This privileged synthetic assistance must not be
used to claim autonomous goal discovery or held-out SDK performance.

From the repository root, install the existing Python 3.12 dependencies:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python scripts/play_game.py --mode web --seed 42
python scripts/play_game.py --mode sdk --seed 42 --max-actions 100
python scripts/play_game.py --mode web --game si01-v1 --seed 42
python scripts/play_game.py --mode sdk --game si01-v1 --seed 42 --max-actions 100
python scripts/play_game.py --mode web --game mc01-v1 --seed 42
```

The web command prints `http://127.0.0.1:8001/`. Open it in a local browser.
The top-left game badge is a native dropdown with **Synthetic · Foundations / Basics**,
**Synthetic · Skills and legacy**, and **Original ARC-AGI-3** groups, populated by two separate SDK catalogs. Selecting
a game initializes it with the current seed in the same 64 × 64 console;
switching away discards its progress. Choosing the current game does not reset it.
The default remains `sc01-v1`. All entries can be selected without restarting the server. Both legacy entries
start at 1/5; foundation, skill and maze entries start at 1/6.
Use RESET to follow the game's SDK restart rules. For all three Square Cross games,
reset retries the current level after movement/loss and starts at level 1 after a win.

Controls follow the active observation's `available_actions`:

| SDK action | Local control |
| --- | --- |
| ACTION1–4 | D-pad and arrow keys |
| ACTION5 | SPACEBAR button or Space |
| ACTION6 | Click/tap a cell on the game screen; CLICK is an availability indicator |
| ACTION7 | UNDO button or Z |
| RESET | RESET button, including after a terminal response |

Unavailable controls are disabled. SELECT stays decorative and disabled; HELP
opens human instructions without spending an action. Keyboard shortcuts are
suppressed while using the game picker or HELP. Multi-frame SDK responses play
in order before the next queued command (normally 80 ms/frame, with a two-second
target cap per sequence); reduced-motion users see the final frame. Reset and
switch cancel pending animation. This is presentation of one SDK action, not
additional game steps.

Tabs share one play session. Browser commands are queued and server writes are
locked. Reset/switch changes a session token so a stale tab's input is rejected;
the tab reloads live state and discards pending commands. A failed game load
keeps the previous session. Errors appear below the controls; RESET can recover
after an interruption, or reload the page if the server is still unreachable.
Close the server with Ctrl+C to close both SDK scorecards.

The magenta handheld shell is locally authored HTML/CSS, visually inspired by
the [ARC Prize ls20 console](https://arcprize.org/tasks/ls20). This is a **local
play UI**, not an ARC-hosted or competition interface. It does not change public
datasets, training/evaluation splits, competition scoring or model weights.
On phones the panel reflows to keep active hit targets at least 44 × 44 pixels.
The native observation uses the complete SDK palette without tint or clipping.

### Original public games

By default, both catalogs use `arc-agi==0.9.9` in OFFLINE mode: no network,
API key, GPU or model weights are needed. Synthetic sources stay in `games/`;
authorized original SDK games belong in the ignored `environment_files/` cache.
If the cache is absent/empty, Synthetic remains playable and the selector
explains that no originals were discovered. No original entries are invented.
A local directory is trusted executable SDK game code: use only public games
that you are authorized to run, not private/held-out evaluation environments.

To use an existing public SDK cache:

```sh
python scripts/play_game.py --mode web --originals-dir /path/to/environment_files --seed 42
# Start directly in ls20 if this catalog discovers a unique ls20 version:
python scripts/play_game.py --mode web --game ls20 --seed 42
```

To provision through the official SDK, explicitly opt in to NORMAL mode:

```sh
python scripts/play_game.py --mode web --originals-mode normal --seed 42
```

Select an original discovered by the SDK; its `make` operation downloads the
SDK-authorized source into `environment_files/` and initializes it locally.
Then stop and relaunch without `--originals-mode normal` to play the cache offline.
NORMAL permits network access **only for the original provider**, including
startup discovery. Availability depends on SDK access and may require an
[authorized API key](https://docs.arcprize.org/toolkit/arc_agi) via `ARC_API_KEY`.
Do not place keys in commands committed to Git or expose them to browser code.
If `OPERATION_MODE=offline` is set in your environment, the SDK may retain offline
mode even with the CLI opt-in; remove that override to use NORMAL. Online and
competition overrides are rejected. Never commit downloaded sources or caches.
No web scraping, website screenshots, remote uploads or submissions are involved.

The SDK command runs a seeded random smoke agent over the advertised actions,
including coordinates when needed, stopping at a terminal state or `--max-actions`.
Its JSON summary counts SDK actions, excluding initialization. This cap is
separate from each game's budget. It is a plumbing check, not a solver result.
Add `--render` for terminal frames. Web mode accepts `--host` and `--port` and
binds loopback by default; keep it local. The synthetic path and default cache
path resolve relative to the launcher, including from another working directory.
Flask and NumPy arrive with the SDK/ARCEngine dependencies; there is no frontend
framework, CDN or additional runtime dependency. Compatibility was exercised with
ARCEngine 0.9.3. The SDK discovers `metadata.json` recursively; the family directory
need not match the four-character game ID, but the Python file/class must match.
`games/pixel_grid.py` shares geometry, sprite drawing and HUD rendering; each
entrypoint owns its generator and rules. The existing package installation also
installs these neutral helpers, so direct SDK loading works outside the repository
working directory. Reinstall with `python -m pip install -e .` after this split
if using an older editable installation.

## Direct SDK use

Run from the repository root in the installed environment:

```python
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='./games')
try:
    env = arcade.make('sc01-v1', seed=42)  # Or 'si01-v1', an independent game.
    if env is None or env.observation_space is None:
        raise RuntimeError('Game initialization failed')
    initial = env.observation_space  # make already initialized/reset the game
    result = env.step(GameAction.ACTION4)
    if result is None:
        raise RuntimeError('Game step failed')
    print(result.state, result.levels_completed)
finally:
    arcade.close_scorecard()
```

Reference: [official game authoring](https://docs.arcprize.org/add_game),
[Arcade](https://docs.arcprize.org/toolkit/arc_agi) and
[local server](https://docs.arcprize.org/toolkit/listen_and_serve). The installed,
pinned SDK source is the compatibility authority.

## Focused validation

Run these manually when changing the game or launcher. Existing CI routing is
unchanged; its infrastructure checks do not exercise this game.

```sh
python -m unittest discover -s tests -p 'test_square_cross.py' -v
python -m unittest discover -s tests -p 'test_square_cross_interactions.py' -v
python -m unittest discover -s tests -p 'test_play_game.py' -v
```

The engine tests cover literal boundary fixtures and 64 seeds across five basic
and five interaction levels. The new click-only scenes have no targets and keep
a full bar; 64-seed checks verify solid squares, separated hitboxes, palette, HUD
and reset. Compact pre-split scene/frame fingerprints also
verify exact seed compatibility against revision `7411e2f`.
An independent test-only pixel-footprint BFS checks every pixel along each stride;
bar fixtures check every decrement through a 128-action budget on all four edges.
Two real seeded SDK instances of each game exercise all level transitions,
terminal behavior and resets. The interaction levels add wrong-colour swept-contact
checks, free click selection with a four-pixel white marker, and two solid pieces that lock on
delivery. Both delivery orders are verified against independent safe searches
and disjoint lane footprints within the shared movement budget.
Reference solving stays in tests and is never a policy input or benchmark claim.
Launcher tests exercise real, separate SDK providers, catalog/source isolation,
atomic selection and failed loads, epoch conflicts, terminal/reset behavior,
coordinate validation, action gating, bounded CLI runs and scorecard cleanup.
The locally authored `tests/fixtures/play_game/tp01` provider exercises clicks,
Space/undo and two-frame animations; it is explicitly not an official ARC game.

For actual browser acceptance, install the optional test tooling once:

```sh
python -m pip install playwright==1.61.0
python -m playwright install chromium
python tests/play_game_browser.py
```

The browser check starts/stops its own local servers and checks 390 × 844 mobile
and 1280 × 900 desktop layouts (plus 320-pixel-wide controls). The original
Square Cross lane retains pixel-for-pixel SDK/native canvas and displayed-colour
comparisons, keyboard/touch/burst input, depletion/reset, HELP/focus, disabled
controls, and injected transport-error recovery. The alternate fixture lane
checks native dropdown opening, grouped labels, real cross-source selection,
scaled edge clicks, one tap/one action, per-observation control changes, actual
SDK animation ordering/cancellation and shared-tab conflict recovery. Level/WIN
response fixtures test presentation only; the engine tests own completion proof.
A real introductory lane starts `si01-v1` at level 1, taps the square and selects
both squares on level 2, checking the two-marker native transition and disabled
arrows. An isolated advanced lane enters level 4 through real SDK actions,
then taps squares and uses the D-pad to deliver both pairs. It checks the exact
four white
selection pixels against both native canvas data and displayed pixels, and saves
native 64×64 marker images plus mobile/desktop screenshots for levels 4–5,
ending at WIN 5/5. The picker lane checks both synthetic titles, native frames,
level counters, help and controls when switching between them.

To add genuine **offline ls20** browser smoke on a provisioned public cache:

```sh
python tests/play_game_browser.py --originals-dir /path/to/environment_files
```

This selects the SDK-discovered ls20 version, checks its real initial frame,
level and actions, performs a legal action, then switches back to `sc01-v1` on
both viewports. Without this option, it explicitly reports the official lane as
not run; fixture success is not evidence of official gameplay. For manual smoke,
launch with the same `--originals-dir`, select ls20 from the badge, use an enabled
control and select Square Cross again. Screenshots are saved under ignored
`artifacts/square-cross/` (including `fixture-*` and `ls20-*` for those lanes).
For a temporary browser cache, set `PLAYWRIGHT_BROWSERS_PATH=/tmp/arc3-playwright`
for both install and test commands. On hosts with an unwritable matplotlib
config directory, `MPLCONFIGDIR=/tmp/arc3-mpl` avoids its cache warning.

## Foundations / Basics

`bf01-v1`–`bf08-v1` add 48 short preparatory levels without changing `sc01-v1`,
`si01-v1`, or `sy01-v1`–`sy60-v1`. The existing harder games remain available.
The eight contexts teach two-colour matching, permanent switch chains, movement,
rotation, a fixed reflection, one loop, unit transfer, and matching pairs.
Pieces and pictorial arrows are directly clickable; movement also uses the
keyboard/D-pad. No letters, digits, selection command or execution opcode is
required on a foundation board. HELP explains the active controls and suggests
suitable later practice; those suggestions do not schedule or modify training.

`games/skill_core.py` owns the catalog and preparatory relationships.
`python scripts/skill_catalog.py` lists both families; `--game bf01-v1` selects
an instance. `--teacher-info` is an explicit privileged inspection path, separate
from ordinary actor frames. The existing opt-in web inspector works unchanged.
See [foundation progression and validation](../docs/synthetic-validation.md#foundation-tier-issue-62)
for structural parameters, action bounds, test commands, and evidence limits.

## Native symbolic presentation

All authored synthetic frames use object, relation, quantity and transformation
cues without intentional alphabetic or decimal labels. The `symbols-v2` revision
changes 54 skill profiles while retaining the foundation tier, genuine symbolic
puzzles, all 70 entries and their 418 levels. Human HELP remains outside the
canvas and obtains exact control locations from the native renderer. See the
[revision inventory and replay/visual evidence](../docs/synthetic-validation.md#non-linguistic-presentation-revision-issue-64)
for coordinate migration and the limits of validation. Live refresh follows
independent review and merge; an isolated test server is not deployment.

## Direct actions and object clicks

Synthetic games no longer paint detached control buttons into their observation.
Movement and operations use the SDK's native actions; clicks act on the figure,
tile, card or switch itself. Foundations `bf03`/`bf05` use arrows, `bf06` uses
Left/Right, and `bf07` uses Left/Right for its first transfer connection and
Up/Down for the second. Other foundation clicks already act on real objects.

The external web controls and HELP remain available for people. HELP and
`skill_catalog.py` list the actions enabled at the current level. See the
[micro-skill controls](skills/README.md#controls-feedback-and-subgoals) for the
simplified rules in larger command sets. Weights and training settings are not
reset by this source change; running processes need to load the updated games.

## Training palette

All synthetic games apply a full permutation of the 16 colors to the final
frame, including animations, backgrounds, controls, markers and visual targets.
The seed fixes the palette across the game and its levels without changing
rules, positions or legal actions. Sequence collection also keeps this palette
fixed across resets within a session.
