# bf02: paired real clicks from a shared history

Experiment for [issue #124](https://github.com/murillo128/arc-agi-3-model/issues/124),
2026-10-22. **No candidate was adopted.** The balanced paired examples improved
the branch probe against equal-budget unpaired adaptation, but failed the
invention gate and did not improve ordinary trajectories over the source model.
Increasing the existing spatial adapter's LR did not resolve this.

## Setup and boundaries

- Source: batch 2995, 11,849,728 environment actions, 66,080 predictor updates.
  Checkpoint SHA-256:
  `35e5f01d74ad9b58580cbd92128a972be0f67331c8714a8be5c7d3142537317a`.
- Every arm started from the same weights **and Adam state**. Predictor LR was
  `3e-5`; the optional arm used `1e-4` only for `rotation_adapter`. Encoder and
  decoder tensors were checked for exact equality with the source after every
  arm. Architecture, Gaussian click map (fixed sigma 1.5) and loss were unchanged.
- Synthetic bf02 levels 1–6. Each example had 16 real context transitions and
  one supervised endpoint. Native sequence learning rebuilt the prefix memory;
  siblings shared observations but never consumed each other's outcomes.
- Alternative clicks: original action, two spatially separated visible-pixel
  candidates ranked by the restored exploration head, and a background control.
  Each alternative was executed in a fresh seeded environment after replaying
  and verifying every prefix observation. No simulator state copying, imagined
  targets, hidden rules or teacher were used. This is a training-data experiment,
  not an evaluation-time action selection policy.
- Training pools: 192 ordinary anchors from independent scenes; 48 scenes with
  four real alternative clicks each, also 192 paired anchors. Validation used
  four different scenes per level with four clicks each: 96 anchors, including
  32 effectful actions and 535 actually changed pixels. Training seeds were even
  (`4_400_000 + 2*(level*1000+i)`); validation seeds were odd
  (`4_600_001 + 2*(level*1000+i)`).
- Additional held-out samples: 384 ordinary bf02 transitions (64 per level,
  seed `4_800_001 + level*200`) and 192 retained-game transitions (bf01, bf03,
  sc01, levels 1 and 3, 32 per session, seed `4_900_001 + level*200`). All outcomes
  came from real interactions. These small synthetic checks are not SDK or
  tournament performance estimates.
- Eight supervised endpoints per optimizer update: 192 presentations per arm
  at 24 updates, 384 at 48. Each run made 7,920 explicit environment `step`
  calls including replay; the three runs regenerated the same data, not three
  independent datasets. Factory initialization is outside that call counter.

The first probe's paired pool had only 42/192 effectful examples (21.9%), versus
76/192 (39.6%) in the ordinary pool. It degraded recall. The bounded refinement
sampled **four effectful and four no-op examples per update in both arms**, with
the same level schedule. Paired effect/no-op examples shared their prefix;
ordinary examples came from independent scenes. Balancing used already observed
training outcomes, never validation labels. It changes the sampled population,
so these are results for this particular data recipe, not proof about pairing
alone. The 48-update run restarted from the source and extended the same schedule.

## Metrics and results

Recall is `correctly predicted changed pixels / actually changed pixels`.
Delta error is `(missed + invented + wrong-color pixels) / union of actual and
predicted changes`. An unchanged background is not a correct change. Inventions
are counted per evaluated action, including no-ops. Lower error and fewer
inventions are better; higher recall is better.

| Arm | Updates | Branch recall % | Branch delta error % | Invented px/action | Ordinary bf02 delta error % | Retained-game delta error % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untouched source | 0 | 22.43 | 89.02 | 5.81 | 86.81 | 67.79 |
| Ordinary, raw sampling | 24 | 15.51 | 90.24 | 3.28 | 84.94 | 66.17 |
| Paired, raw sampling | 24 | 8.22 | 93.35 | 1.32 | 91.97 | 65.76 |
| Paired, raw + adapter LR | 24 | 8.41 | 93.19 | 1.31 | 92.05 | 65.77 |
| Ordinary, balanced | 24 | 14.95 | 91.42 | 4.14 | 86.14 | 66.06 |
| Paired, balanced | 24 | 22.99 | 88.17 | 5.26 | 87.60 | 66.92 |
| Paired, balanced + adapter LR | 24 | 23.55 | 87.93 | 5.30 | 87.61 | 66.91 |
| Ordinary, balanced | 48 | 12.34 | 91.77 | 2.78 | 86.54 | 66.33 |
| Paired, balanced | 48 | 20.93 | 87.57 | 3.81 | 87.27 | 66.58 |
| Paired, balanced + adapter LR | 48 | 20.56 | 87.82 | 3.83 | 87.38 | 66.56 |

Final branch-probe counts (535 actual changed pixels in every row):

| Arm | Correct | Missed | Wrong color | Invented | Change union |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untouched source | 120 | 359 | 56 | 558 | 1093 |
| Ordinary, balanced, 48 updates | 66 | 439 | 30 | 267 | 802 |
| Paired, balanced, 48 updates | 112 | 381 | 42 | 366 | 901 |
| Paired, balanced + adapter LR, 48 updates | 110 | 383 | 42 | 368 | 903 |

Branch delta error by level, at 48 updates:

| Level | Actual changed pixels | Source % | Ordinary balanced % | Paired balanced % | Paired + adapter LR % |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 109 | 81.42 | 81.28 | 75.69 | 75.69 |
| 2 | 65 | 90.57 | 96.88 | 91.91 | 92.05 |
| 3 | 197 | 83.67 | 93.07 | 82.59 | 82.96 |
| 4 | 8 | 99.10 | 93.48 | 98.33 | 98.31 |
| 5 | 34 | 98.44 | 100.00 | 100.00 | 100.00 |
| 6 | 122 | 98.69 | 100.00 | 95.97 | 96.64 |

The predeclared gates compared each paired candidate with its equal-budget
ordinary control: recall +5 percentage points, delta error −3 points, improvement
in at least four of six levels, ordinary delta error within +1 point, inventions
within `max(20%, 0.5 px/action)` above control, and retained-game error within
+2 points. **Every candidate failed those original gates.** Balanced 24-update
candidates failed ordinary error and inventions. At 48 updates, inventions were
3.81 versus 2.78 per action (+37%), above the allowed 3.34.

Comparing against the untouched source also matters: final paired recall was
20.93% versus 22.43%, and ordinary error was 87.27% versus 86.81%. After seeing
the control's degradation, the reusable runner received an additional safety
guard: no recall regression against the source, ordinary error within source
+1 point and retained-game error within source +2 points. This is a **post-probe
tightening**, not a predeclared test; it changes no adoption outcome. Raw result
files retain their original six gate checks.

## Cost, limitations and deployment

On an RTX 4070 Ti at the existing 150 W limit, CUDA AMP and four CPU threads:
collection/replay took 9.87–9.92 seconds per run. The 48-update training loops
took 5.72 s (ordinary), 4.94 s (paired) and 4.89 s (paired + adapter LR); peak
allocated CUDA memory was 2.26, 2.17 and 2.17 GiB respectively. Loop timings omit
encoding and evaluation, use a fixed arm order and reuse frozen latents. They
are **not production TPS or a reliable speed comparison**.

Only four held-out scenes per level were used, and level 4 contained eight
changed pixels. The same validation set informed the 24→48 extension. The probe
used one supervised endpoint after a short context, not the full 512-action
production session. It supports rejecting this deployment, not concluding that
paired learning can never work or that extra model capacity is necessary.

Production resumed from the unchanged checkpoint and optimizer with all 30
retained games and bf08 excluded. Predictor LR remains `3e-5`; encoder/decoder
LRs remain `3e-7`. Local and public dashboard state both confirmed batch 2997,
11,857,920 actions and 66,144 updates, status `running`, 30 games and no bf08.
No candidate checkpoint was exported or loaded into production.

## Re-run and evidence

From the repository root with Python 3.12 and the model dependencies installed:

```sh
PYTHONPATH=.:src python -m model.paired_click_experiment \
  --source /path/to/source.pt --output /path/to/new-ignored-output \
  --device cuda --cpu-threads 4 --updates 48 --balanced --spatial-lr
```

Use 24 updates and omit `--balanced` to reproduce the first probe. CPU is also
supported. Each invocation requires a new output directory; it never writes the
source checkpoint. No automatic production adoption is implemented.

Local evidence is in the ignored run directory
`model/runs/sequences-all-20261020/bf02-paired-20261022/`:
`source-report.json`, `{trial-24,balanced-24,balanced-48}/results.json`, their logs,
`deployment-decision.json`, and `live-verification.json`. Weights and large traces
are deliberately not committed. The tables above retain compact numerical evidence.

Validation: seven focused tests passed, covering replay/reset fidelity, sibling
isolation, metric counts, seed separation, equal budgets and balanced schedules,
native CPU sequence updates with frozen representation, and rejection gates.
The 29 existing sequence/exploration tests completed successfully with one
optional Node-dependent dashboard check skipped because Node was unavailable.
No browser or visual tests were run. The hosted executor-routing workflow's path
filters do not include this experiment, its tests or this report.
