# Teacher-guided planning during sequence training

Sequence training can opt into a visual goal head and a goal-conditioned path
head while continuing predictor optimization. This mode requires an existing
predictor-phase checkpoint with encoder and decoder already frozen. Its initial
training scope is `sc01-v1` and `si01-v1`, whose bundled synthetic trainers supply
goal images, relevant-pixel masks and real goal-completion predicates. Other
games continue the existing sequence/exploration procedure. SDK and held-out
environments never supply teacher goals.

## Adoption and resume

Copy the active training configuration and add this section; retain every other
setting, split and game weight:

```json
"sequence_planning": {
  "width": 64,
  "learning_rate": 0.0001,
  "horizon": 2,
  "beam": 4,
  "clicks": 8,
  "lifetime": 16,
  "warmup_updates": 8,
  "ramp_updates": 64,
  "learned_fraction": 0.5,
  "exploration": 0.25,
  "samples": 32,
  "minibatch": 8,
  "passes": 4
}
```

Stop at a complete batch and retain a local backup of `latest.pt`, `report.json`
and the old configuration. Use the same output directory and existing command,
with `--resume --adopt-planning`; remove the earlier `--freeze-representation`
flag. The saved freeze remains active. Do not combine adoption with learning-rate,
curriculum, exploration or architecture changes. The loader checks that only the
new section differs. New heads start randomly; all world-model weights, optimizer
moments, phase state, counters and exploration state are restored.

After the first new checkpoint, omit `--adopt-planning`. Ordinary `--resume`
requires the saved head configuration and restores both heads, their Adam moments,
private sampling RNG and update counters. Repeating adoption or silently dropping
the heads is rejected. This is a training checkpoint extension, not an automatic
Kaggle export or evaluation-policy switch.

## Learning and collection

The goal head receives detached observed latent patches and causal game memory.
It predicts a 16-color goal image and a binary mask, using a small spatial network.
The teacher's goal, game ID and private state are never model inputs. Color cross
entropy gives changed and unchanged relevant regions equal weight; mask BCE
balances included and excluded pixels. A small moving object cannot be overwhelmed
by the background solely because it occupies fewer pixels.

The existing `PathValue` predicts success within the remaining action budget and
log-duration conditional on success. Only completed real goal attempts provide
labels: success, terminal failure or exhaustion of the specified goal budget.
Interrupted attempts are censored. Two positions and the observed endpoint are
sampled per selected attempt; later positions have smaller remaining budgets.
Failures do not train the duration output. These are outcomes of the collection
policy, not oracle labels for optimal reachability.

Path goal encoding uses goal pixels inside the relevant-pixel mask and the
current real observation outside it. Each real planning decision refreshes this
composed target once and holds it fixed throughout the imagined search. During
path-head fitting, each before/after example uses its own real frame as context.
Raw goal images and masks remain unchanged for visual goal supervision, matching,
telemetry and checkpoints. Full-mask goals retain their original encoding.

Path search starts after the warmup, uses the current predictor, respects legal
actions and the Undo exclusion, and replans after each real action. It ranks
success first, then duration within the existing success margin. Click proposals
retain the existing change prior and random coverage. At least 25% of decisions
use the exploration policy. Goals remain fixed until completion, deadline, real
teacher progress or reset. Memory survives level changes within each session.

Goal proposals enter gradually after warmup, capped at half of attempts. Empty
masks or goals already satisfied by the current image fall back to the teacher.
The head can still propose impossible goals; a deadline bounds their cost. It is
therefore essential to measure real teacher progress separately from matching a
self-proposed image. The learned-goal fraction is a schedule, not a guarantee of
proposal quality.

After predictor optimization, causal memories are rebuilt with the current
weights for a bounded sample from the current batch. Separate optimizers train the
two heads with detached world features. Goal/path gradients cannot change the
encoder, decoder or predictor. Imagined states support search only; they are never
inserted as verified world-model training examples. Head sampling is bounded;
encoding and memory reconstruction still follow the real teacher-game prefixes.

## Monitoring

The existing Goals and Paths section adds goal images/masks, head losses,
pre-update goal errors, completed-attempt success, actions per successful goal,
source-separated real subgoal/level progress and head/collection time. Missing
samples remain missing. Planner collection time is part of collection time;
head learning time is a separate batch cost. Trend windows break at adoption
because the collection policy changes. These measurements use normal training
interactions and do not claim held-out performance.

The teacher goal does not itself demonstrate an action sequence, and an imperfect
predictor can still mislead search. Continued learning must be assessed by real
goal progress and throughput alongside prediction quality.
