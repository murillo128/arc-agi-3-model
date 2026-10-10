# Square Cross (`sc01-v1`)

Move the solid square's **centre** onto the centre of the plus-shaped cross.
Touching an arm is insufficient. `ACTION1/2/3/4` move up/down/left/right by the
level's fixed stride. The cross is nonblocking. Every intermediate pixel of the
full square must stay inside the playable area and avoid the immovable grey
block. An obstructed command leaves the square in place and still spends a move;
passing across the cross centre without landing there does not win.

| Level | Added variation |
| --- | --- |
| 1 | Fixed 5-pixel sprites and stride 5, cardinal target direction/distance, varying colours |
| 2 | Independently chosen square/cross sizes: 3, 5 or 7 pixels; stride equals square side |
| 3 | Free placement with offsets on both axes; independently sampled stride 1–7 |
| 4 | One rectangular block forcing a footprint-aware detour; independently sampled stride 1–7 |
| 5 | Sparse, static background speckles; same mechanics as level 4 |
| 6 | One 5×5 square, stride 5, a same-colour cross and a lethal different-colour decoy on separate cardinal routes |
| 7 | Level-1-style layout; click to select the square before moving, with a white 2×2 marker |
| 8 | Two differently coloured 5×5 squares and matching crosses in separate lanes; select and deliver both, stride 5 |

Crosses render strictly behind every square. Noise, obstacles and the HUD keep
their existing stacking and colours. In levels 6 and 8, touching **any visible
pixel** of a wrong-colour cross with any part of the moving square ends the
attempt in `GAME_OVER`, even on an intermediate pixel of a successful stride.
The whole command must first pass collision checks: a blocked move cannot cause
contact. Transparent cross corners do not count. Matching delivery still requires
an exact centre landing; passing over the centre between endpoints is insufficient.
Levels 1–5 and 7 retain colour-independent centre completion.

Only levels 7–8 advertise `ACTION6` alongside directions. Click/tap any occupied
square pixel to select it, including its marker; click it again or click elsewhere
to deselect. The marker is exactly four palette-0 pixels at centre offsets
(-1,-1), (0,-1), (-1,0), (0,0), above the square and cross. It follows movement.
Clicks never move pieces or spend budget. Unselected directional attempts spend
one move without moving anything. In level 8, clicking the other undelivered
square switches selection; only that piece moves. Squares are solid to one
another throughout the full sweep. A delivered square stays drawn and locked,
cannot be selected, and clears the active selection. Both pairs must be delivered
to win. Selection clears on loss, reset and level change. The browser's CLICK
capsule only indicates availability; it is not a selection toggle.

Every scene uses a native 64×64 grid and camera. A seeded choice reserves four
pixels along the top, bottom, left or right edge for the HUD. The continuous
remaining-moves bar is 60 pixels long and 3 pixels thick, with a 1-pixel gap to
playable space. Its white fill (palette 0) on the HUD background (palette 5) contains
`ceil(180 * remaining / budget)` pixels, filling the thickness before extending
along the edge. Every directional command visibly shrinks it, even with a budget above 64.
Colours, stride, noise and HUD identity remain fixed during the attempt and
retry. The browser only scales pixels for display; there is no separate budget,
stride or geometry input and no client-side physics.

Each generated shortest path takes at most 24 directional actions. The exploratory budget is
`max(24, 5 * distance + 8)` (at most 128), where distance is the optimal action
count: for example, 6 optimal actions give 38 moves. Generation constructs
stride-aligned, visible, non-overlapping layouts with clearance around obstacles,
then verifies levels 1–5 with full-sweep, full-footprint BFS bounded to
24 actions. A failed verification reports the seed and level. Levels 6–7 use
clear cardinal routes. Level 8 uses two separated straight corridors, so both
delivery orders remain safe after locking either square. Its distance is the
minimal sum of moves for both pieces, excluding free clicks, with one shared
budget. Independent test-only searches validate all eight levels. Level-specific
seeded random streams make retries independent of previous actions. Background
speckles render beneath objects/HUD and never collide.

For a level-1 square at (20,30) and cross at (35,30), RIGHT attempts (25,30),
(30,30), then (35,30). Later strides may differ from square size and must be
inferred by interacting. Axis offsets must be divisible by stride, but alignment
alone does not prove that a route exists around obstacles or the HUD.

A last-move arrival succeeds. Otherwise an empty bar means `GAME_OVER`. ARCEngine
advances levels and gives each newly entered level its full budget; only the second
delivery in level 8 yields `WIN`, with `levels_completed=8, win_levels=8`.
RESET uses the official engine lifecycle and spends no move: after an attempt/loss
it restores the current seeded scene, both pieces and completion state; after WIN
it restarts level 1. No terminal/reset semantics are implemented by a custom
environment wrapper.

See [launch instructions, direct SDK example and focused tests](../README.md).
This original synthetic game tests elementary perception/control plumbing; it
implements no agent, model, collection policy or training procedure.
