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

The web command prints `http://127.0.0.1:8001/`. Open it in a local browser and use
arrow keys or the directional buttons (including touch). The dots inside the
image are the remaining moves. Restart follows the SDK's reset lifecycle: retry
the current level after movement/loss; start at level 1 after a win. Close the
server with Ctrl+C. This is a small **local play UI**, not the hosted competition
UI. Tabs share one play session. Inputs are queued in the browser and serialized
on the server. Errors appear below the controls; Restart can recover after a
connection interruption.

The magenta handheld shell is locally authored HTML/CSS, visually inspired by
the [ARC Prize ls20 console](https://arcprize.org/tasks/ls20). Its badges show the
live game ID and level; the compact readout includes the seed and SDK state.
HELP opens instructions without spending a move. The SPACEBAR, CLICK, UNDO and
SELECT decorations are disabled. On phones the lower panel reflows to keep
active hit targets at least 44 × 44 pixels. The shell uses CSS colour properties;
the complete native 64 × 64 observation is displayed without tint or clipping.

The SDK command runs a seeded random smoke agent, stopping at a terminal state
or `--max-actions`. Its JSON summary counts directional actions, excluding SDK
initialization. Seed 42 currently ends with `GAME_OVER`, 0/5 levels and 17 actions.
This is a plumbing check, not a solver or learned-agent result. Add `--render`
for terminal frames. Both routes accept `--game sc01-v1`; web also accepts
`--host` and `--port`. The games path is resolved relative to the launcher, so an
absolute launcher path also works from another working directory after install.

Runtime needs no network, API key, GPU or model weights. The launcher uses
`arc-agi==0.9.9` in OFFLINE mode, rejects remote/competition overrides, and disables
recordings. Flask and NumPy already arrive with the SDK/ARCEngine dependencies;
there is no frontend framework, CDN or additional runtime dependency. Compatibility
was exercised with ARCEngine 0.9.3. The SDK discovers `metadata.json` recursively,
so the human-readable family directory does not need to match the four-character
ID. The file `sc01.py` and class `Sc01` do match it.

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
An independent test-only pixel-footprint BFS checks solutions; two real seeded
SDK environments exercise all level transitions, terminal behavior and resets.
Reference solving stays in tests and is never a policy input or benchmark claim.
Launcher tests exercise real SDK/Flask behavior, arbitrary working directories,
finite smoke runs and error responses. They do not claim browser coverage.

For actual browser acceptance, install the optional test tooling once:

```sh
python -m pip install playwright==1.61.0
python -m playwright install chromium
python tests/play_game_browser.py
```

The browser check starts/stops its own local server and checks 390 × 844 mobile
and 1280 × 900 desktop layouts (plus 320-pixel-wide controls). It compares native
canvas pixels and displayed cell colours with the SDK observation, exercises
keyboard/touch/burst inputs, dot depletion, reset and injected connection-error
recovery. It also checks focus, hit targets, HELP, disabled controls, loading and
state badges. Level/WIN response fixtures test presentation only; the engine
tests above own real multi-level completion proof. Chromium screenshots of the
initial, error and terminal states go under ignored `artifacts/issue-22/`.
For a temporary browser cache, set `PLAYWRIGHT_BROWSERS_PATH=/tmp/arc3-playwright`
for both the install and test commands. On hosts with an unwritable matplotlib
config directory, `MPLCONFIGDIR=/tmp/arc3-mpl` avoids its cache warning.
