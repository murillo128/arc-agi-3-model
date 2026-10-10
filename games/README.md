# Original synthetic games

This directory contains our own versioned ARCEngine environments, separate from
ignored downloaded `environment_files/`. The first family is
[Square Cross](square_cross/README.md), one game (`sc01-v1`) with five levels.
These games are not held-out evaluation games and do not change the repository's
training/evaluation splits. No learning or training is performed by the launcher.

From the repository root, install the existing Python 3.12 dependencies:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python scripts/play_game.py --mode web --seed 42
python scripts/play_game.py --mode sdk --seed 42 --max-actions 100
```

The web command prints `http://127.0.0.1:8001/`. Open it in a local browser.
The top-left game badge is a native dropdown with **Synthetic** and
**Original ARC-AGI-3** groups, populated by two separate SDK catalogs. Selecting
a game initializes it with the current seed in the same 64 × 64 console;
switching away discards its progress. Choosing the current game does not reset it.
Use RESET to follow the game's SDK restart rules. For Square Cross, reset retries
the current level after movement/loss and starts at level 1 after a win.

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

## Direct SDK use

Run from the repository root in the installed environment:

```python
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir='./games')
try:
    env = arcade.make('sc01-v1', seed=42)
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
python -m unittest discover -s tests -p 'test_play_game.py' -v
```

The engine tests cover literal boundary fixtures and 64 seeds × five levels.
An independent test-only pixel-footprint BFS checks every pixel along each stride;
bar fixtures check every decrement through a 128-action budget on all four edges.
Two real seeded SDK environments exercise all level transitions, terminal behavior
and resets.
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
