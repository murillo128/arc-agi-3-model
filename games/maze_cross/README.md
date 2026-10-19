# Square Cross Maze

`mc01-v1` is an independent, locally authored synthetic navigation game. Move the
square onto the cross through coloured floor corridors; the other background
colour marks fixed, impassable walls. The native observation is 64 × 64 pixels.

| Level | Layout |
| --- | --- |
| 1 | Straight horizontal corridor, in either direction |
| 2 | Straight vertical corridor, in either direction |
| 3 | One required turn |
| 4 | Two required turns |
| 5 | Branching maze with dead ends and a unique route to the cross |
| 6 | Branching maze with two loops and alternative routes to the cross |

The seed determines the corridors, placement, orientation and palette. Colours
remain distinct and stable throughout a game; they do not prescribe a direction
or encode a solution. Every generated layout is connected and has a verified
route. Floors and walls remain fixed during an attempt. There are no keys,
doors, movable walls, enemies or hidden tiles.

ACTION1/2/3/4 move up/down/left/right by one five-pixel corridor cell. The whole
3 × 3 square must fit throughout the movement, so it cannot cross or cut corners
through walls. Blocked moves leave it in place and still consume one action.
The move bar can appear along any edge. Its budget is
`min(128, max(24, 5 * shortest_route_actions + 8))`; every scene is solvable
within that budget. Reaching the cross on the final action still succeeds.
Six level completions yield WIN; exhausting the budget yields GAME_OVER.
RESET retries the current level, or restarts at level 1 after a win.

## Play and train

With the repository's Python 3.12 dependencies installed:

```sh
python scripts/play_game.py --mode web --game mc01-v1 --seed 42
python scripts/play_game.py --mode sdk --game mc01-v1 --seed 42 --max-actions 100
```

The existing picker discovers the game automatically. HELP describes its
controls outside the native observation. The web and SDK use the same frames,
rules and lifecycle.

Training configurations can select `source: "synthetic"`, `game: "mc01-v1"`,
any ordered subset of levels 1–6, and `objective: "world_model"`. This game has
no subgoal-teacher interface: the learner receives observations and allowed
actions, never the generation graph or verified route. Registering it does not
change an existing training plan. See the [training configuration guide](../../model/README.md).

## Native validation

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_maze_cross.py' -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_play_game.py' -v
PYTHONPATH=src python -m unittest discover -s model/tests -p 'test_training_config.py' -v
```

An independent pixel-space solver checks 64 seeds across all six levels, and
graph checks distinguish straight routes, turns, dead ends and useful loops.
Literal collision fixtures cover the full square and movement sweep, blocked
action cost and last-action success. Real offline SDK runs complete all six
levels at four seeds and check reset/loss/win behavior. These are numerical,
protocol and configuration checks; no browser or visual tests are required for
this addition.
