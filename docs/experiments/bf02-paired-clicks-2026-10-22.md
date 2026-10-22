# bf02: paired real clicks from a shared history

Experiment for [issue #124](https://github.com/murillo128/arc-agi-3-model/issues/124),
2026-10-22. **No candidate was adopted.** The balanced paired examples improved
the branch probe against equal-budget unpaired adaptation, but failed the
invention gate and did not improve ordinary trajectories over the source model.
Raising the full spatial adapter's LR did not resolve this.

## Setup and boundaries

- Source: batch 2995, 11,849,728 environment actions, 66,080 predictor updates.
  Checkpoint SHA-256:
  `35e5f01d74ad9b58580cbd92128a972be0f67331c8714a8be5c7d3142537317a`.
  Corrected experiment source: `8457711811eacae44eb989618f37eb2df4a6089a`.
- Every arm started from the same weights **and Adam state**. Predictor LR was
  `3e-5`; the optional arm used `1e-4` for every `rotation_adapter` weight and
  bias, retaining each restored group's weight decay and Adam moments. Encoder and
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

The initial implementation raised LR only in Adam's matrix/kernel group, missing
four adapter biases in its second group. The correction partitions every restored
group. **All three comparisons below were regenerated**, including their ordinary
and paired controls; the original weights-only LR rows are superseded. Initial
metrics and data composition reproduce exactly. The balanced 24-update ordinary
control changed slightly (branch error 91.42% → 91.41%; ordinary error
86.14% → 86.07%); all gates use the rerun controls, not mixed historical results.

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
| Paired, raw + adapter LR | 24 | 8.41 | 93.20 | 1.32 | 91.94 | 65.76 |
| Ordinary, balanced | 24 | 14.95 | 91.41 | 4.12 | 86.07 | 66.08 |
| Paired, balanced | 24 | 22.99 | 88.17 | 5.26 | 87.60 | 66.92 |
| Paired, balanced + adapter LR | 24 | 23.36 | 88.00 | 5.28 | 87.60 | 66.90 |
| Ordinary, balanced | 48 | 12.34 | 91.77 | 2.78 | 86.54 | 66.33 |
| Paired, balanced | 48 | 20.93 | 87.57 | 3.81 | 87.27 | 66.58 |
| Paired, balanced + adapter LR | 48 | 20.93 | 87.60 | 3.83 | 87.32 | 66.56 |

Final branch-probe counts (535 actual changed pixels in every row):

| Arm | Correct | Missed | Wrong color | Invented | Change union |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untouched source | 120 | 359 | 56 | 558 | 1093 |
| Ordinary, balanced, 48 updates | 66 | 439 | 30 | 267 | 802 |
| Paired, balanced, 48 updates | 112 | 381 | 42 | 366 | 901 |
| Paired, balanced + adapter LR, 48 updates | 112 | 381 | 42 | 368 | 903 |

Branch delta error by level, at 48 updates:

| Level | Actual changed pixels | Source % | Ordinary balanced % | Paired balanced % | Paired + adapter LR % |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 109 | 81.42 | 81.28 | 75.69 | 75.82 |
| 2 | 65 | 90.57 | 96.88 | 91.91 | 91.48 |
| 3 | 197 | 83.67 | 93.07 | 82.59 | 82.53 |
| 4 | 8 | 99.10 | 93.48 | 98.33 | 98.33 |
| 5 | 34 | 98.44 | 100.00 | 100.00 | 100.00 |
| 6 | 122 | 98.69 | 100.00 | 95.97 | 96.62 |

No-op errors at 48 updates (an erroneous action invents at least one pixel):

| Arm | Erroneous branch no-ops / 64 | Invented branch pixels | Erroneous ordinary no-ops / 188 | Invented ordinary pixels |
| --- | ---: | ---: | ---: | ---: |
| Untouched source | 38 | 385 | 89 | 670 |
| Ordinary, balanced | 25 | 169 | 56 | 179 |
| Paired, balanced | 34 | 251 | 87 | 397 |
| Paired, balanced + adapter LR | 34 | 253 | 87 | 396 |

The predeclared gates compared each paired candidate with its equal-budget
ordinary control: recall +5 percentage points, delta error −3 points, improvement
in at least four of six levels, ordinary delta error within +1 point, inventions
within `max(20%, 0.5 px/action)` above control, and retained-game error within
+2 points. **Every candidate failed those original gates.** Balanced 24-update
candidates failed ordinary error and inventions. At 48 updates, both candidates
improved branch error in four levels but invented 3.81/3.83 pixels per action
versus 2.78 in the control, above the allowed 3.34. The full-adapter LR arm had
the same recall as pairing alone and slightly worse union error.

Comparing against the untouched source also matters: final paired recall was
20.93% versus 22.43%, and ordinary error was 87.27% versus 86.81%. After seeing
the control's degradation, the reusable runner received an additional safety
guard: no recall regression against the source, ordinary error within source
+1 point and retained-game error within source +2 points. This is a **post-probe
tightening**, not a predeclared test; it changes no adoption outcome. Original
result files retain six gate checks; corrected results include all nine.

## Cost, limitations and deployment

Corrected runs used the RTX 4070 Ti, CUDA AMP and four CPU threads alongside the
live trainer, with an 18% per-process CUDA allocation cap. After encoding, the
frozen encoder moves to CPU; learning and evaluation use its cached latents.
Weights, model arithmetic, data and optimizer budgets are unchanged.
Collection/replay took 10.33–10.60 s per run. Whole-process wall times were
64.20 s (raw 24), 77.01 s (balanced 24), and 84.87 s (balanced 48), with peak
resident CPU memory 10.08–10.09 GiB. The 48-update loops took 7.70 s (ordinary),
10.41 s (paired) and 8.43 s (paired + adapter LR); peak allocated CUDA memory
after cache construction was 1.92, 1.83 and 1.84 GiB respectively. Loop timings
omit encoding/evaluation, share the GPU, use a fixed arm order and reuse frozen
latents. They are **not production TPS or a reliable speed comparison**.

Before moving the encoder, two shared-GPU attempts exhausted memory during the
first backward pass (50.61/44.77 s, including an activation-offload attempt).
A CPU fallback was interrupted during encoding after 445.02 s. These completed
no optimizer updates but each collected/replayed 7,920 actions. The corrected
comparisons add 23,760 real calls and 2,304 supervised presentations across 288
updates; failed/interrupted attempts add 23,760 calls, and the original three
trials added 23,760. These regenerate the same seeded data, not independent trials.

Only four held-out scenes per level were used, and level 4 contained eight
changed pixels. The same validation set informed the 24→48 extension. The probe
used one supervised endpoint after a short context, not the full 512-action
production session. It supports rejecting this deployment, not concluding that
paired learning can never work or that extra model capacity is necessary.

The original experiment resumed production from its unchanged checkpoint and
optimizer; local/public state then confirmed batch 2997. During correction,
production was never stopped or reconfigured. The final local report confirmed
batch 3016, 11,927,552 actions, 66,752 updates, status `running`, 30 games and no
bf08. Predictor LR remains `3e-5`, encoder/decoder LRs `3e-7`, and fixed click sigma
1.5. The source checkpoint SHA-256 was verified unchanged after the reruns.
No candidate checkpoint was exported or loaded into production. Retain production
unchanged; these results do not justify this sampling or LR change.

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
To reproduce the shared-GPU cap, replace `python -m model.paired_click_experiment`
with `python -c 'import runpy, torch; torch.cuda.set_per_process_memory_fraction(.18); runpy.run_module("model.paired_click_experiment", run_name="__main__")'`
and keep the same arguments. The encoder residency change is automatic on CUDA.

Local evidence is in the ignored run directory
`model/runs/sequences-all-20261020/bf02-paired-20261022/`:
`source-report.json`, `{trial-24,balanced-24,balanced-48}/results.json`, their logs,
`deployment-decision.json`, and `live-verification.json`. Corrected evidence is
retained beside that original run in `correction-8457711/` (outside the disposable
issue worktree):
`{trial-24,balanced-24,balanced-48}-corrected/results.json`, matching `.log`/`.time`
files, `source-revision.txt`, `verified-comparisons.json`, `tests.log`, and
`live-after-reruns.json`. Original and unsuccessful runs are retained. Weights
and large traces are not committed; these tables retain compact numerical evidence.

Validation at the corrected source: `model.tests.test_paired_click_experiment`,
`model.tests.test_sequence`, `model.tests.test_sequence_exploration`, and
`model.tests.test_sequence_training` completed **48 tests: 47 passed, one optional
Node-dependent dashboard test skipped**. Eight focused cases cover replay/reset
fidelity, sibling isolation, metrics, seed separation, equal/balanced budgets,
native CPU updates with frozen representation, optimizer restoration and gates.
The new real CPU-runner regression failed against `f0a2264` for exactly the four
missed adapter biases, then passed after repair. It checks all adapter rates,
unchanged rates elsewhere, exact nonempty Adam moments/step counters and all group
options (including weight decay), unique parameter coverage, frozen representation,
an unchanged source checkpoint, and four examples in each of three actual updates.
Expected state is captured before the runner partitions groups.

All nine corrected CUDA arms passed the frozen-representation check; independent
aggregation of scene counts reproduced every metric, and traces confirmed actual
update/example budgets and matched effect/no-op counts. `git diff --check` passed.
No browser or visual tests ran. The unchanged hosted executor-routing workflow's
path filters exclude these three files; the other workflows are issue dispatchers
or reusable launchers, so no hosted CI workflow applies to this diff.
