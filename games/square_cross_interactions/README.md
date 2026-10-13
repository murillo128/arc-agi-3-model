# Square Cross Interactions (`si01-v1`)

An independent three-level synthetic training game, implemented by
`v1/si01.py::Si01` and its own metadata. These are the mechanics previously added
as Square Cross levels 6–8 in issue #28, now numbered locally 1–3. The
[five-level Square Cross game](../square_cross/README.md) remains `sc01-v1`.

| Local level | Scene | Available actions |
| --- | --- | --- |
| 1 | One 5×5 square, a same-colour cross and a lethal different-colour decoy on separate cardinal routes | 1–4 |
| 2 | Level-1-style one-square navigation; click to select before moving, with a white 2×2 marker | 1–4, 6 |
| 3 | Two differently coloured 5×5 squares and matching crosses in separate lanes; select and deliver both | 1–4, 6 |

All three levels use stride 5. In levels 1 and 3, touching any visible pixel of
a wrong-colour cross with any part of a moving square ends the attempt in
`GAME_OVER`, including intermediate stride pixels. The whole command first
passes collision checks: a blocked move cannot cause contact. Transparent cross
corners do not count. Matching delivery requires an exact centre landing;
passing over a matching centre between endpoints does not deliver. Level 2 uses
colour-independent centre completion. Every cross renders behind every square.

In levels 2–3, click/tap any occupied square pixel to select it, including its
marker; click it again or click elsewhere to deselect. The marker is exactly
four palette-0 pixels at centre offsets (-1,-1), (0,-1), (-1,0), (0,0), above
the square and cross. It follows movement. Clicks never move pieces or spend
budget. Unselected or blocked directional commands still spend one move.
The browser's CLICK capsule indicates availability; it is not a selection toggle.

In level 3, clicking the other undelivered square switches selection. Only the
selected piece moves. Squares are solid to one another throughout the full
sweep, including after delivery. A delivered square stays drawn and locked,
cannot be selected and clears selection. Both pairs must be delivered to win.
Selection clears on loss, reset and level change, and is represented only by
the SDK frame's white marker, never by additional protocol data.

The 64×64 frame, ARC palette, four-pixel HUD strip and shrinking white budget
bar follow [Square Cross's geometry and HUD rules](../square_cross/README.md).
The budget is `max(24, 5 * distance + 8)`; clicks are excluded from distance.
Levels 1–2 use clear cardinal routes. Level 3 has disjoint straight corridors,
so either delivery order remains safe with the other square locked. Its
distance is the minimal total movement for both pieces, with one shared budget.
Independent test-only searches verify solvability and both orders over 64 seeds.

The RNG namespace alone retains the historical level numbers:
`square-cross-v1:{seed}:{local_level + 5}`. Geometry, colours, HUD edge, budget
and initial pixels reproduce former levels 6–8 exactly; public counts and all
rules use levels 1–3. Retries remain deterministic and independent of history.

A last-budget delivery succeeds. Otherwise depletion means `GAME_OVER`.
Completing level 3 yields `WIN`, with `levels_completed=3, win_levels=3`.
SDK RESET retries the current seeded level, restoring all pieces, locks and
selection; after WIN it restarts level 1 with directional actions only.

```sh
python scripts/play_game.py --mode web --game si01-v1 --seed 42
python scripts/play_game.py --mode sdk --game si01-v1 --seed 42 --max-actions 100
```

Both games appear under Synthetic in the existing picker, without restarting
the server. See [catalog, SDK and validation instructions](../README.md).
This is a synthetic training environment, not an original ARC game or an agent.
