# Square Cross Interactions (`si01-v1`)

An independent five-level synthetic training game, implemented by
`v1/si01.py::Si01` and its own metadata. Two introductory click-only tasks precede
the mechanics previously added as Square Cross levels 6–8 in issue #28, now
numbered locally 3–5. The separate
[five-level Square Cross game](../square_cross/README.md) remains `sc01-v1`.

| Local level | Scene and completion | Available actions |
| --- | --- | --- |
| 1 | One solid 5×5 square, no cross; click anywhere inside it to show the white selection marker and advance | 6 |
| 2 | Two separated, differently coloured solid 5×5 squares, no crosses; select both simultaneously to advance | 6 |
| 3 | One 5×5 square, a same-colour cross and a lethal different-colour decoy on separate cardinal routes | 1–4 |
| 4 | One-square navigation; click to select before moving, with a white 2×2 marker | 1–4, 6 |
| 5 | Two differently coloured 5×5 squares and matching crosses in separate lanes; select and deliver both | 1–4, 6 |

In level 1, clicking empty background or the HUD has no effect. A successful click
adds the same white 2×2 selection marker used in later levels before advancing.
The marker appears in the native completion frame, including the final `WIN`
frame when training on the isolated level. Its synthetic teacher target uses
that same marked square. Level 2 uses
cumulative selection: clicking either square toggles only that square, without
clearing the other. Empty-space clicks clear all selection. Completing the second
selection shows both markers in the native transition, then advances to level 3.
These levels have no crosses, obstacles, noise or movement. Clicks are free and
the positive 24-move budget retains its full 180-pixel white bar.

The three navigation levels use stride 5. In levels 3 and 5, touching any visible
pixel of a wrong-colour cross with any part of a moving square ends the attempt
in `GAME_OVER`, including intermediate stride pixels. The whole command first
passes collision checks: a blocked move cannot cause contact. Transparent cross
corners do not count. Matching delivery requires an exact centre landing;
passing over a matching centre between endpoints does not deliver. Level 4 uses
colour-independent centre completion. Every cross renders behind every square.

In levels 4–5, selection is exclusive: click/tap any occupied square pixel to
select it, including its marker; click it again or click elsewhere to deselect.
In level 5, clicking the other undelivered square switches selection. Only the
selected piece moves. Squares are solid to one another throughout the full
sweep, including after delivery. A delivered square stays drawn and locked,
cannot be selected and clears selection. Both pairs must be delivered to win.

Each selected square's marker is exactly four palette-0 pixels at centre offsets
(-1,-1), (0,-1), (-1,0), (0,0), above the square and cross. It follows movement.
Clicks never move pieces or spend budget. Unselected or blocked directional
commands still spend one move on navigation levels. The browser's CLICK capsule
indicates availability; it is not a selection toggle. Selection clears on loss,
reset and level change, and is represented only by the SDK frame's white pixels,
never by additional protocol data.

The 64×64 frame, ARC palette, four-pixel HUD strip and shrinking white budget
bar follow [Square Cross's geometry and HUD rules](../square_cross/README.md).
Navigation budget is `max(24, 5 * distance + 8)`; clicks are excluded from distance.
Levels 3–4 use clear cardinal routes. Level 5 has disjoint straight corridors,
so either delivery order remains safe with the other square locked. Its
distance is the minimal total movement for both pieces, with one shared budget.
Independent test-only searches verify solvability and both orders over 64 seeds.

Click scenes use `square-cross-interactions-v1:{seed}:click-{local_level}`.
The navigation RNG namespace retains the historical level numbers:
`square-cross-v1:{seed}:{local_level + 3}` for local levels 3–5. Geometry, colours,
HUD edge, budget and initial pixels reproduce former levels 6–8 exactly.
Retries remain deterministic and independent of history.

A last-budget delivery succeeds. Otherwise navigation depletion means `GAME_OVER`.
Completing level 5 yields `WIN`, with `levels_completed=5, win_levels=5`.
SDK RESET retries the current seeded level, restoring all pieces, locks and
selection; after WIN it restarts level 1 with ACTION6 only.

```sh
python scripts/play_game.py --mode web --game si01-v1 --seed 42
python scripts/play_game.py --mode sdk --game si01-v1 --seed 42 --max-actions 100
```

The finite SDK command is a random plumbing smoke, not a solver; click-only
starter levels may legitimately stop at `max-actions`.
Both games appear under Synthetic in the existing picker, without restarting
the server. See [catalog, SDK and validation instructions](../README.md).
This is a synthetic training environment, not an original ARC game or an agent.
