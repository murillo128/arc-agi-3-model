"""Fixed-replay measurements and convergence control for game-private LoRA.

Errors are fractions, losses are objectives. A plateau never counts as meeting
an error threshold. Snapshots include Adam state as well as adapter parameters.
"""
from copy import deepcopy
import math
import time

import torch

from .adaptation_heads import balanced_regions


GROUPS = dict(representation=('encoder', 'decoder'), predictor=('predictor',),
              path=('path',), goal=('goal',))


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
    chunks = [examples[i:i + policy.config.minibatch]
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
            states = None if name == 'representation' else policy._states()
            for chunk in chunks:
                if interrupted():
                    return None
                loss, _ = policy._objective(name, samples=chunk, states=states, measured=statistics)()
                total += float(loss) * len(chunk)
        metrics = summarize(name, statistics)
        loss = total / len(examples)
        if not math.isfinite(loss) or any(v is not None and not math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('Nonfinite fixed-replay measurement')
        result['checks'] += 1
        return dict(loss=loss, metrics=metrics)

    with policy.runtime.router.activate(policy.bank, groups=groups):
        try:
            while not interrupted():
                measurement = measure()
                if measurement is None:
                    break
                if result['initial'] is None:
                    result['initial'] = deepcopy(measurement)
                result['final'] = measurement
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
                for chunk in chunks:
                    if interrupted():
                        break
                    # Rebuild current-weight latents/carry before every update.
                    objective = policy._objective(name, samples=chunk)
                    # The reconstruction stop is owned by the full replay check,
                    # not a potentially easy individual minibatch.
                    policy._step(name, groups, objective, check_threshold=False)
                    result['steps'] += 1
                else:
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
