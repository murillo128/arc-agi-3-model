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

Every scene uses a native 64×64 grid and camera. A seeded choice reserves four
pixels along the top, bottom, left or right edge for the HUD. The continuous
remaining-moves bar is 60 pixels long and 3 pixels thick, with a 1-pixel gap to
playable space. Its white fill (palette 0) on the HUD background (palette 5) contains
`ceil(180 * remaining / budget)` pixels, filling the thickness before extending
along the edge. Every command visibly shrinks it, even with a budget above 64.
Colours, stride, noise and HUD identity remain fixed during the attempt and
retry. The browser only scales pixels for display; there is no separate budget,
stride or geometry input and no client-side physics.

Each generated shortest path takes at most 24 actions. The exploratory budget is
`max(24, 5 * distance + 8)` (at most 128), where distance is the optimal action
count: for example, 6 optimal actions give 38 moves. Generation constructs
stride-aligned, visible, non-overlapping layouts with clearance around obstacles,
then verifies the shortest path with full-sweep, full-footprint BFS bounded to
24 actions. A failed verification reports the seed and level. Level-specific
seeded random streams make retries independent of previous actions. Background
speckles render beneath objects/HUD and never collide.

For a level-1 square at (20,30) and cross at (35,30), RIGHT attempts (25,30),
(30,30), then (35,30). Later strides may differ from square size and must be
inferred by interacting. Axis offsets must be divisible by stride, but alignment
alone does not prove that a route exists around obstacles or the HUD.

A last-move arrival succeeds. Otherwise an empty bar means `GAME_OVER`. ARCEngine
advances levels and gives each newly entered level its full budget; level 5
completion yields `WIN`. RESET uses the official engine lifecycle and spends no
move. No terminal/reset semantics are implemented by a custom environment wrapper.

See [launch instructions, direct SDK example and focused tests](../README.md).
This original synthetic game tests elementary perception/control plumbing; it
implements no agent, model, collection policy or training procedure.
