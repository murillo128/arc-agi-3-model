"""Fixed-replay measurements and convergence control for game-private LoRA.

Errors are fractions, losses are objectives. A plateau never counts as meeting
an error threshold. Snapshots include Adam state as well as adapter parameters.
"""
from copy import deepcopy
import math
import time

import numpy as np
import torch

from .adaptation_heads import balanced_regions


GROUPS = dict(representation=('encoder', 'decoder'), predictor=('predictor',),
              path=('path',), goal=('goal',))


def recency_weights(ends, latest, config):
    floor = config.history_weight_floor
    return [floor + (1. - floor) * 2. ** (-max(0, latest - end) / config.history_half_life)
            for end in ends]


def weighted_mean(values, weights=None):
    values = torch.stack(values)
    if weights is None:
        return values.mean()
    weights = values.new_tensor(weights)
    return (values * weights).sum() / weights.sum()


def fixed_hindsight(rows):
    """Latest different same-scene outcome per origin, in linear frame work.

    If an origin equals the scene's last frame, its latest different outcome is
    the last frame different from that final frame. Otherwise it is the final
    frame itself. Only these two targets are needed, not every historical pair.
    """
    labels = []

    def scene(records, boundary):
        if len(records) < 2 or records[0].observation.final is None:
            return
        last = records[-1]
        last_frame = last.observation.final
        different = next((r for r in reversed(records[:-1])
                          if not np.array_equal(r.observation.final, last_frame)), None)
        for before in records[:-1]:
            end = different if np.array_equal(before.observation.final, last_frame) else last
            if end is None or end.number <= before.number:
                continue
            after = end.observation
            labels.append(dict(start=before.number, end=end.number, frame=after.final,
                mask=before.observation.final != after.final, success=True,
                distance=end.number-before.number, budget=end.number-before.number,
                progress=(after.state == 'WIN' or (end is last and boundary is not None and
                    boundary.observation.levels_completed > after.levels_completed))))

    start = 0
    for i, record in enumerate(rows):
        if i > start and (record.boundary or record.observation.final is None or
                          rows[i-1].observation.final is None):
            scene(rows[start:i], record if record.boundary else None)
            start = i
    scene(rows[start:], None)
    return labels


def delta_metrics(before, actual, predicted):
    real, proposed = actual != before, predicted != before
    missed = int((real & ~proposed).sum())
    invented = int((~real & proposed).sum())
    wrong_color = int((real & proposed & (actual != predicted)).sum())
    return dict(missed=missed, invented=invented, wrong_color=wrong_color,
                wrong=missed + invented + wrong_color, union=int((real | proposed).sum()),
                changed=int(real.sum()))


def goal_metrics(logits, mask_logits, before, target, mask):
    changed = (target != before) & mask
    color = balanced_regions((logits.argmax(-1) != target).float(), changed, mask & ~changed)
    mask_error = balanced_regions(((mask_logits > 0) != mask).float(), mask, ~mask)
    return dict(color_error=float(color.mean()), mask_error=float(mask_error.mean()))


def path_metrics(logits, cost, success, distance):
    # Probability error retains confidence; classification alone would accept .51.
    probability_error = float((logits.float().sigmoid() - float(success)).abs().mean())
    relative_error = None
    if success:
        # expm1 is the inverse of the trained log1p action-distance target.
        relative_error = float(((cost.float().expm1() - distance).abs() / max(1, distance)).mean())
    return dict(reachability_error=probability_error, cost_error=relative_error)


def summarize(name, rows):
    if name == 'predictor':
        result = {key: sum(row[key] for row in rows) for key in rows[0]}
        result['delta_error'] = result['wrong'] / result['union'] if result['union'] else None
        result['latest_delta_error'] = rows[-1]['wrong'] / rows[-1]['union'] if rows[-1]['union'] else None
        return result
    result = {}
    for key in rows[0]:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = sum(values) / len(values) if values else None
    if name == 'representation':
        result['latest_reconstruction_error'] = rows[-1]['reconstruction_error']
    return result


def stopping_reason(name, metrics, config):
    if name == 'representation':
        return 'threshold' if max(metrics['reconstruction_error'], metrics['latest_reconstruction_error']) <= config.reconstruction_threshold else None
    if name == 'predictor':
        if metrics['delta_error'] is None:
            return 'no_change'
        passed = (metrics['delta_error'] <= config.delta_threshold and
                  (metrics['latest_delta_error'] is None or metrics['latest_delta_error'] <= config.delta_threshold))
        return 'threshold' if passed else None
    if name == 'goal':
        passed = (metrics['color_error'] <= config.goal_color_threshold and
                  metrics['mask_error'] <= config.goal_mask_threshold)
    else:
        passed = (metrics['reachability_error'] <= config.path_reachability_threshold and
                  (metrics['cost_error'] is None or metrics['cost_error'] <= config.path_cost_threshold))
    return 'threshold' if passed else None


def snapshot(bank, groups):
    # Only small private adapters/moments, never copy the shared model.
    return ([p.detach().clone() for p in bank.parameters(groups)],
            {group: deepcopy(bank.optimizers[group].state_dict()) for group in groups})


def restore(bank, groups, saved):
    parameters, optimizers = saved
    with torch.no_grad():
        for current, previous in zip(bank.parameters(groups), parameters):
            current.copy_(previous)
    for group in groups:
        bank.optimizers[group].load_state_dict(optimizers[group])
        bank.optimizers[group].zero_grad(set_to_none=True)


def fit(policy, name, deadline):
    """One fixed set, full-replay checks after each minibatched training sweep."""
    examples = policy._examples(name)
    started = time.monotonic()
    result = dict(reason='no_data', threshold_met=False, steps=0, checks=0,
                  samples=len(examples), initial=None, final=None, best_loss=None,
                  restored_best=False, elapsed_seconds=0.)
    policy.diagnostics['convergence'][name] = result
    if not examples:
        policy.diagnostics['skipped'][name] += 1
        return
    groups = GROUPS[name]
    weights = policy._weights(name, examples)
    total_weight = sum(weights)
    result['weights'] = dict(minimum=min(weights), maximum=max(weights), total=total_weight,
        half_life=policy.config.history_half_life, floor=policy.config.history_weight_floor)
    chunks = [(examples[i:i + policy.config.minibatch], weights[i:i + policy.config.minibatch])
              for i in range(0, len(examples), policy.config.minibatch)]
    best, best_measurement = None, None
    best_loss, progress_loss, stale = math.inf, math.inf, 0

    def interrupted():
        reason = policy._stop_reason(deadline)
        if reason:
            result['reason'] = reason
            if reason == 'budget':
                policy.diagnostics['budget_stops'] += 1
        return reason

    def measure():
        statistics, total = [], 0.
        with torch.no_grad(), policy.runtime.engine.autocast():
            states = None if name == 'representation' else policy._states(cpu=True)
            for chunk, chunk_weights in chunks:
                if interrupted():
                    return None
                loss, _ = policy._objective(name, samples=chunk, states=states,
                    measured=statistics, weights=chunk_weights)()
                total += float(loss) * sum(chunk_weights)
        metrics = summarize(name, statistics)
        loss = total / total_weight
        if not math.isfinite(loss) or any(v is not None and not math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('Nonfinite fixed-replay measurement')
        result['checks'] += 1
        return dict(loss=loss, metrics=metrics)

    def train_sweep():
        optimizers = [policy.bank.optimizers[group] for group in groups]
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        total = 0.
        try:
            # No weights change until the sweep finishes, so one CPU cache of
            # current-weight states is valid for every minibatch in this update.
            states = None if name == 'representation' else policy._states(cpu=True)
            for chunk, chunk_weights in chunks:
                if interrupted():
                    return False
                objective = policy._objective(name, samples=chunk, states=states, weights=chunk_weights)
                with policy.runtime.engine.autocast():
                    loss, _ = objective()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite weighted adaptation objective')
                fraction = sum(chunk_weights) / total_weight
                (loss * fraction).backward()
                total += float(loss.detach()) * fraction
                # Release this minibatch's graph before constructing the next.
                del objective, loss
            if interrupted():
                return False
            policy._update(name, groups)
            policy.diagnostics['losses'][name] = dict(pre=total, post=None, error_pre=None, error_post=None)
            return True
        finally:
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

    with policy.runtime.router.activate(policy.bank, groups=groups):
        try:
            while not interrupted():
                measurement = measure()
                if measurement is None:
                    break
                if result['initial'] is None:
                    result['initial'] = deepcopy(measurement)
                result['final'] = measurement
                if result['steps']:
                    policy.diagnostics['losses'][name]['post'] = measurement['loss']
                loss = measurement['loss']
                if loss < best_loss:
                    best_loss, best_measurement = loss, deepcopy(measurement)
                    best = snapshot(policy.bank, groups)
                    result['best_loss'] = loss
                reason = stopping_reason(name, measurement['metrics'], policy.config)
                if reason:
                    result.update(reason=reason, threshold_met=reason == 'threshold')
                    return
                if not math.isfinite(progress_loss) or loss < progress_loss - max(abs(progress_loss), 1e-8) * policy.config.plateau_min_relative_improvement:
                    progress_loss, stale = loss, 0
                else:
                    stale += 1
                if stale >= policy.config.plateau_patience:
                    result['reason'] = 'plateau'
                    break
                if train_sweep():
                    result['steps'] += 1
                    continue
                break
        except FloatingPointError:
            result['reason'] = 'numerical_failure'
            raise
        except (KeyboardInterrupt, InterruptedError):
            result['reason'] = 'cancelled'
            raise
        except Exception:
            result['reason'] = 'error'
            raise
        finally:
            if best is not None and result['reason'] not in ('threshold', 'no_change'):
                restore(policy.bank, groups, best)
                result.update(final=best_measurement, restored_best=True)
            result['elapsed_seconds'] = time.monotonic() - started
