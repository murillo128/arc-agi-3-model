# Square Cross (`sc01-v1`)

Move the solid square's **centre** onto the centre of the plus-shaped cross.
Touching an arm is insufficient. `ACTION1/2/3/4` move up/down/left/right by one
pixel. The cross is nonblocking. The full square must stay inside the playable
area and avoid the immovable grey block; blocked commands still spend a move.

| Level | Added variation |
| --- | --- |
| 1 | Fixed 5-pixel sprites, cardinal target direction/distance, varying colours |
| 2 | Independently chosen square/cross sizes: 3, 5 or 7 pixels |
| 3 | Free placement, including offsets on both axes |
| 4 | One rectangular block forcing a footprint-aware detour |
| 5 | Sparse, static background speckles; same mechanics as level 4 |

Every scene uses a native 64×64 grid and camera. A seeded choice reserves four
pixels along the top, bottom, left or right edge for the HUD. White one-pixel dots
at offsets 2, 4, …, 60 show the exact remaining budget. Colours, noise and HUD
identity remain fixed during the attempt. The browser only scales pixels for
display; there is no separate countdown input or client-side physics.

Each generated shortest path is 1–24 moves, with 4–6 extra moves and at most 30
dots. Generation checks full-square clearance, exact centre arrival and initial
non-overlap, trying at most 128 candidates before reporting the seed/level in an
error. Level-specific seeded random streams make retries independent of previous
actions. Background speckles render beneath objects/HUD and never collide.

A last-move arrival succeeds. Otherwise zero dots means `GAME_OVER`. ARCEngine
advances levels and gives each newly entered level its full budget; level 5
completion yields `WIN`. RESET uses the official engine lifecycle and spends no
move. No terminal/reset semantics are implemented by a custom environment wrapper.

See [launch instructions, direct SDK example and focused tests](../README.md).
This original synthetic game tests elementary perception/control plumbing; it
implements no agent, model, collection policy or training procedure.
