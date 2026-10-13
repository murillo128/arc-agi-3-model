# Square Cross (`sc01-v1`)

An independent five-level synthetic training game for elementary perception and
control. `v1/sc01.py::Sc01` owns the original levels 1–5. The former interaction
levels now belong to [Square Cross Interactions](../square_cross_interactions/README.md)
(`si01-v1`), with its own SDK identity and local numbering.

| Level | Scene |
| --- | --- |
| 1 | One 5×5 square and cross on a cardinal route; stride 5 |
| 2 | Independently chosen square/cross sizes: 3, 5 or 7 pixels; stride equals square side |
| 3 | Free placement with offsets on both axes; independently sampled stride 1–7 |
| 4 | One rectangular block forcing a footprint-aware detour; independently sampled stride 1–7 |
| 5 | Sparse, static background speckles; same mechanics as level 4 |

Only `ACTION1–4` are advertised in every level; there is no click selection.
They move up, down, left and right respectively by the level's fixed stride.
Move the square's centre exactly onto the cross, regardless of colour. Passing
over the centre between stride endpoints does not deliver. Every directional
command spends one move, including blocked commands. The entire stride and full
square footprint must clear the HUD, frame border and solid obstacle; a blocked
stride moves nothing. Crosses render behind squares. Noise is static,
non-colliding and beneath objects/HUD.

Every scene uses the native 64×64 grid and ARC palette. A seeded choice reserves
four pixels along the top, bottom, left or right edge for the HUD. The remaining
moves bar is 60 pixels long and 3 pixels thick, with a 1-pixel gap to playable
space. Its white fill (palette 0) on background 5 contains
`ceil(180 * remaining / budget)` pixels, filling thickness before length. Every
directional command visibly shrinks it, including budgets above 64. Redraws
never consume moves; the browser scales the SDK pixels without client physics.

Generation constructs stride-aligned, visible, non-overlapping layouts, then
verifies full-sweep, full-footprint shortest paths bounded to 24 actions.
The exploratory budget is `max(24, 5 * distance + 8)` (at most 128); for example,
6 optimal actions give 38 moves. A failed verification reports seed and level.
Each level uses `square-cross-v1:{seed}:{level}` as its RNG namespace; geometry,
colours, stride, HUD and level-5 noise preserve the pre-split scenes exactly.
Resets remain independent of prior actions.

For a square at (20,30) and cross at (35,30), stride-5 RIGHT attempts (25,30),
(30,30), then (35,30). Later strides may differ from square size and must be
inferred by interacting. Axis offsets must be divisible by stride, but alignment
alone does not prove a route around obstacles or the HUD.

A last-move arrival succeeds. Otherwise an empty bar means `GAME_OVER`.
ARCEngine advances levels and gives each new level its full budget. Completing
level 5 yields `WIN`, with `levels_completed=5, win_levels=5`. RESET spends no move:
after an attempt/loss it restores the current seeded level; after WIN it restarts
level 1. The official SDK owns terminal/reset semantics.

See [launch instructions, direct SDK examples and focused tests](../README.md).
This environment implements no agent, model, collection policy or training procedure.
