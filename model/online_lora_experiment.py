"""Manual public held-out LoRA experiment, separate from frozen evaluation.

Only the existing Kaggle AdaptivePolicy learns. Diagnostic predictions precede
SDK.step; returned frames are measured before confirm_result admits them. No
frames, replay, adapters or checkpoints are written by this runner.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from arcengine import GameAction

from arc3.envs.sdk import open_arcade
from arc3.envs.splits import load_split
from .adaptation import AdaptationConfig, color_macro, load
from .adaptation_smoke import bank_digest, base_digest, _runtime_diagnostics, _tensor_digest
from .data import Observation
from .metrics import delta_counts


GAMES = ('ar25-0c556536', 'cn04-2fe56bfb', 'lf52-271a04aa',
         'tn36-ef4dde99', 'vc33-5430563c')
CHECKPOINT_SHA256 = '4d125575cdc126044ec9b2e773252237cb9743c36032a08d4857f4d876150db7'
CONFIG = AdaptationConfig(rank=4, alpha=4., learning_rate=1e-4, replay_size=32,
    minibatch=4, representation_steps=2, predictor_steps=1, path_steps=1,
    goal_steps=1, reconstruction_threshold=.10, per_observation_seconds=10.,
    total_seconds=300., exploration=.25, horizon=2, beam=4, clicks=8,
    goal_lifetime=16, allow_undo=False)
COUNT_KEYS = ('wrong', 'union', 'changed', 'missed', 'invented', 'wrong_color',
              'correct', 'samples', 'unchanged_samples')


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def counts(before, actual, predicted):
    wrong, union, changed = delta_counts(before, actual, predicted)
    real, proposed = actual != before, predicted != before
    return dict(wrong=wrong, union=union, changed=changed,
        missed=int((real & ~proposed).sum()), invented=int((~real & proposed).sum()),
        wrong_color=int((real & proposed & (actual != predicted)).sum()),
        correct=int((real & (actual == predicted)).sum()), samples=1,
        unchanged_samples=int(not real.any()))


def aggregate_counts(items):
    items = list(items)
    total = {key: sum(item.get(key, 0) for item in items) for key in COUNT_KEYS}
    total['error_percent'] = 100 * total['wrong'] / total['union'] if total['union'] else None
    return total


@torch.no_grad()
def reconstruction(policy, observation):
    """Read-only new-observation quality, before its admission/training."""
    if observation.final is None:
        return None
    runtime, engine = policy.runtime, policy.runtime.engine
    with runtime.router.activate(policy.bank), engine.autocast():
        z = engine.encode([observation])[0]
        logits = engine.decode(z[None], *observation.final.shape)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite diagnostic reconstruction')
        target = torch.tensor(np.array(observation.final), device=engine.device).long()[None]
        loss, error = color_macro(logits, target)
        wrong = int((logits.argmax(-1) != target).sum())
        return dict(wrong=wrong, pixels=target.numel(), color_macro_error=float(error),
                    color_macro_loss=float(loss), logits_dtype=str(logits.dtype))


@torch.no_grad()
def predict_action(policy, observation, action):
    """Use the causal state built by choose_action, without consuming its RNG."""
    if action.id == 0 or observation.final is None or policy.state is None:
        return None, {}
    runtime, engine, state = policy.runtime, policy.runtime.engine, policy.state
    if state.z is None:
        return None, {}
    with runtime.router.activate(policy.bank), engine.autocast():
        z, _, delta = engine.predict(state.z.to(engine.device)[None],
            state.h.to(engine.device), engine.actions([action]), *state.geometry)
        logits = engine.decode(z, *state.geometry)
        if not all(bool(torch.isfinite(value).all()) for value in (z, delta, logits)):
            raise FloatingPointError('Nonfinite pre-action diagnostic prediction')
        predicted = logits.argmax(-1)[0].cpu().numpy().astype(np.uint8)
        return predicted, dict(latent_dtype=str(z.dtype), logits_dtype=str(logits.dtype))


def window(rows):
    eligible = [row for row in rows if row.get('prediction_counts') is not None]
    return dict(actions=[row['action_number'] for row in eligible],
        levels=dict(Counter(str(row['level_before']) for row in eligible)),
        counts=aggregate_counts(row['prediction_counts'] for row in eligible))


def group_digests(bank):
    return {group: _tensor_digest((str(i), tensor) for i, tensor in
            enumerate(bank.parameters((group,))))[0] for group in bank.groups}


def run_game(wrapper, policy, *, started, emit, seconds=900.,
             predict=predict_action, reconstruct=reconstruction):
    """One SDK lifetime; RESET never replaces the policy. Flush every real return.

    The wall deadline includes initialization, decisions, diagnostics, SDK and
    adaptation. Checks before selection and immediately before SDK.step avoid
    beginning another action after expiry. A running operation is not preempted.
    """
    rows, requests, returned, resets = [], 0, 0, 0
    observation = None
    reason, error = 'action_budget', None
    diagnostic_failures = 0
    initial_reconstruction = None
    try:
        observation = Observation.from_sdk(wrapper.observation_space)
        initial_reconstruction = reconstruct(policy, observation)
        emit(dict(event='initial', levels_completed=observation.levels_completed,
                  total_levels=observation.win_levels, reconstruction=initial_reconstruction))
        while requests < policy.max_actions:
            if observation.state == 'WIN':
                reason = 'win'
                break
            if time.monotonic() - started >= seconds:
                reason = 'wall_timeout'
                break
            before_diagnostics = deepcopy(policy.diagnostics)
            decision_started = time.monotonic()
            action = policy.choose_action(observation)
            try:
                predicted, precision = predict(policy, observation, action)
            except FloatingPointError as exc:
                predicted, precision = None, {}
                diagnostic_failures += 1
                emit(dict(event='diagnostic_failure', phase='prediction', error=str(exc)))
            decision_seconds = time.monotonic() - decision_started
            selection_adaptation = (policy.diagnostics['adaptation_seconds'] -
                                    before_diagnostics['adaptation_seconds'])
            if time.monotonic() - started >= seconds:
                policy.cancel_pending()
                reason = 'wall_timeout'
                emit(dict(event='cancelled_at_deadline', decision_seconds=decision_seconds,
                          selection_adaptation_seconds=selection_adaptation))
                break
            requests += 1
            row = dict(action_number=requests, action_id=action.id, action_data=dict(action.data),
                level_before=observation.levels_completed + 1, state_before=observation.state,
                decision_seconds=decision_seconds, selection_adaptation_seconds=selection_adaptation,
                prediction_precision=precision, prediction_counts=None, confirmed=False,
                returned=False, selection_updates=deepcopy(policy.diagnostics['updates']))
            rows.append(row)
            emit(dict(event='request', **row))
            sdk_started = time.monotonic()
            try:
                raw = wrapper.step(GameAction.from_id(action.id), data=dict(action.data),
                                   reasoning={'policy': 'per-game LoRA adaptation'})
                result = Observation.from_sdk(raw)
            except BaseException:
                policy.cancel_pending()
                raise
            returned += 1
            resets += int(action.id == 0)
            row.update(returned=True, sdk_seconds=time.monotonic() - sdk_started,
                levels_completed=result.levels_completed, total_levels=result.win_levels,
                state_after=result.state, full_reset=bool(getattr(raw, 'full_reset', False)))
            # Durable scalar evidence even if measurement/adaptation subsequently fails.
            emit(dict(event='sdk_result', **row))
            boundary = (action.id == 0 or observation.final is None or result.final is None
                or observation.final.shape != result.final.shape
                or observation.levels_completed != result.levels_completed)
            row['boundary'] = boundary
            row['no_visual_effect'] = bool(observation.final is not None and result.final is not None
                and np.array_equal(observation.final, result.final))
            if predicted is not None and result.final is not None and predicted.shape == result.final.shape:
                row['boundary_counts' if boundary else 'prediction_counts'] = counts(
                    observation.final, result.final, predicted)
            measurement_started = time.monotonic()
            try:
                row['new_observation_reconstruction'] = reconstruct(policy, result)
            except FloatingPointError as exc:
                row['new_observation_reconstruction'] = None
                diagnostic_failures += 1
                emit(dict(event='diagnostic_failure', phase='reconstruction', error=str(exc)))
            row['measurement_seconds'] = time.monotonic() - measurement_started
            emit(dict(event='measured_before_learning', **row))
            before_adaptation = policy.diagnostics['adaptation_seconds']
            confirm_started = time.monotonic()
            # The same confirmation boundary as the Kaggle agent, including WIN,
            # GAME_OVER, level changes and the last budgeted action.
            if not policy.confirm_result(result, action=action):
                raise RuntimeError('SDK result was not confirmed exactly once')
            row.update(confirmed=True, confirm_wall_seconds=time.monotonic() - confirm_started,
                result_adaptation_seconds=policy.diagnostics['adaptation_seconds'] - before_adaptation,
                elapsed_seconds=time.monotonic() - started, diagnostics=deepcopy(policy.diagnostics))
            observation = result
            emit(dict(event='action', **row))
            if observation.state == 'WIN':
                reason = 'win'
                break
    except Exception as exc:
        reason, error = 'error', f'{type(exc).__name__}: {exc}'
        logging.exception('Game failed; retaining partial results')
        emit(dict(event='error', error=error))
    elapsed = time.monotonic() - started
    if reason == 'action_budget' and elapsed >= seconds:
        reason = 'wall_timeout'
    eligible = [row for row in rows if row.get('prediction_counts') is not None]
    level_actions = {}
    for row in rows:
        if row['returned']:
            level = level_actions.setdefault(str(row['level_before']), dict(actions=0, resets=0,
                ordinary_actions=0, no_visual_effect=0))
            level['actions'] += 1
            level['resets'] += int(row['action_id'] == 0)
            level['ordinary_actions'] += int(row['action_id'] != 0)
            level['no_visual_effect'] += int(row.get('no_visual_effect', False))
    confirmations = sum(row['confirmed'] for row in rows)
    if confirmations != policy.actions_taken:
        reason, error = 'error', (error or '') + '; Policy/runner action confirmation count mismatch'
    adaptation_calls = [r[key] for r in rows for key in
        ('selection_adaptation_seconds', 'result_adaptation_seconds') if key in r and r[key] > 0]
    updates = policy.diagnostics['updates']
    return dict(stop_reason=reason, error=error, truncated=reason in ('action_budget', 'wall_timeout', 'error'),
        requests=requests, actions=returned, confirmations=confirmations, resets=resets,
        sdk_initializations=1, levels_completed=max([0] + [r.get('levels_completed', 0) for r in rows]),
        total_levels=observation.win_levels if observation is not None else None,
        win=reason == 'win', elapsed_seconds=elapsed, wall_overrun_seconds=max(0., elapsed-seconds),
        actions_by_level=level_actions, prediction=aggregate_counts(r['prediction_counts'] for r in eligible),
        first=window(eligible[:10]), last=window(eligible[-10:]),
        window_overlap=max(0, min(10, len(eligible))*2-len(eligible)),
        boundary_prediction=aggregate_counts(r['boundary_counts'] for r in rows if 'boundary_counts' in r),
        adaptation_limits=dict(max_observation_seconds=max(adaptation_calls, default=0.),
            observations_over_10_seconds=sum(s > 10. for s in adaptation_calls),
            total_overrun_seconds=max(0., policy.diagnostics['adaptation_seconds']-300.)),
        optimizer_steps_by_module=dict(encoder=updates.get('representation', 0),
            decoder=updates.get('representation', 0), predictor=updates.get('predictor', 0),
            path=updates.get('path', 0), goal=updates.get('goal', 0)),
        diagnostic_failures=diagnostic_failures, initial_reconstruction=initial_reconstruction,
        diagnostics=deepcopy(policy.diagnostics), rows=rows)


def official_score(arcade, game_id):
    """Read scores only for reporting; never pass references to the policy."""
    card = arcade.get_scorecard()
    environment = None if card is None else card.find_environment(game_id)
    if environment is None:
        return dict(rhae_percent=None, message='No official SDK scorecard available', runs=[])
    fields = {'score', 'levels_completed', 'actions', 'resets', 'state', 'completed',
              'level_scores', 'level_actions', 'level_baseline_actions', 'message'}
    runs = [run.model_dump(mode='json', include=fields) for run in environment.runs]
    available = bool(runs) and all(run.get('level_baseline_actions') and
        all(n > 0 for n in run['level_baseline_actions']) and not run.get('message') for run in runs)
    return dict(rhae_percent=environment.score if available else None,
        message=None if available else 'Official human references unavailable or incomplete',
        actions=environment.actions, resets=environment.resets, runs=runs)


def aggregate_games(games):
    scores = [g.get('official', {}).get('rhae_percent') for g in games]
    return dict(games_recorded=len(games), requested_games=len(GAMES),
        levels_completed=sum(g.get('levels_completed', 0) for g in games),
        total_levels=sum(g.get('total_levels') or 0 for g in games),
        wins=sum(g.get('win', False) for g in games), actions=sum(g.get('actions', 0) for g in games),
        rhae_mean_percent=sum(scores)/len(GAMES) if len(scores) == len(GAMES) and all(s is not None for s in scores) else None,
        prediction=aggregate_counts(g.get('prediction', {}) for g in games),
        adaptation_seconds=sum(g.get('diagnostics', {}).get('adaptation_seconds', 0.) for g in games),
        elapsed_seconds=sum(g.get('elapsed_seconds', 0.) for g in games),
        peak_gpu_allocated_bytes=max((g.get('peak_gpu_allocated_bytes', 0) for g in games), default=0),
        numerical_failures=sum(g.get('diagnostics', {}).get('numerical_failures', 0) for g in games))


def save_report(output, report):
    report['aggregate'] = aggregate_games(report['games'])
    write_json(output / 'summary.json', report)
    fields = ('game', 'levels_completed', 'total_levels', 'win', 'actions', 'resets',
              'rhae_percent', 'predictor_error_percent', 'predictor_wrong', 'predictor_union',
              'adaptation_seconds', 'elapsed_seconds', 'peak_gpu_allocated_bytes', 'stop_reason')
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for game in report['games']:
            row = {key: game.get(key) for key in fields}
            row.update(rhae_percent=game.get('official', {}).get('rhae_percent'),
                predictor_error_percent=game.get('prediction', {}).get('error_percent'),
                predictor_wrong=game.get('prediction', {}).get('wrong'),
                predictor_union=game.get('prediction', {}).get('union'),
                adaptation_seconds=game.get('diagnostics', {}).get('adaptation_seconds'))
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--split', default='configs/splits.public-development.json')
    parser.add_argument('--environments-dir', default='environment_files')
    parser.add_argument('--mode', choices=('normal', 'offline'), default='normal')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error('--threads must be in 1..16')
    if tuple(load_split(args.split).evaluation) != GAMES:
        parser.error('Split differs from the five authorized public held-out games')
    if asdict(CONFIG) != asdict(AdaptationConfig()):
        parser.error('Adaptation defaults changed; document discrepancy before executing')
    if args.output_dir.exists():
        parser.error('Output directory already exists; refusing overwrite/restart')
    if not torch.cuda.is_available():
        parser.error('CUDA required; no CPU fallback')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir
    logging.basicConfig(filename=output / 'runner.log', level=logging.INFO, force=True)
    started = time.monotonic()
    report = dict(format='arc3-online-lora-experiment-v1', manual=True,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        command=[sys.executable, '-m', 'model.online_lora_experiment', *sys.argv[1:]],
        sdk=version('arc-agi'), seed=42, policy_seed='42:<game_id>', max_actions=80,
        max_game_seconds=900, config=asdict(CONFIG), games=[], status='preparing',
        checkpoint=dict(expected_sha256=CHECKPOINT_SHA256, dataset='sergiogarciamurillo/arc3-lora-smoke-base',
                        dataset_version=1, source_batches=8246), threads=args.threads)
    arcade = None
    try:
        digest = file_digest(args.checkpoint)
        report['checkpoint']['sha256_before'] = digest
        if digest != CHECKPOINT_SHA256:
            raise ValueError('Portable checkpoint SHA-256 does not match the authorized batch-8246 asset')
        # An unused bank retains the shared base and checks isolation, without
        # running a second policy or producing a frozen gameplay comparison.
        sentinel = load(args.checkpoint, device='cuda:0', config=CONFIG, max_actions=80, seed='isolation')
        runtime = sentinel.runtime
        if runtime.metadata['source_batches'] != 8246:
            raise ValueError('Checkpoint source batch mismatch')
        report['runtime'] = _runtime_diagnostics(runtime) | dict(
            gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
            cuda_version=torch.version.cuda, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=torch.backends.cudnn.allow_tf32)
        report['targets'] = runtime.router.report
        base_before, base_bytes = base_digest(runtime)
        inactive_before = bank_digest(sentinel.bank)
        report.update(base_sha256_before=base_before, base_bytes=base_bytes,
                      startup_seconds=time.monotonic()-started, status='running')
        save_report(output, report)
        arcade = open_arcade(args.mode, args.environments_dir)
        for game_id in GAMES:
            print(json.dumps(dict(event='game_start', game=game_id)), flush=True)
            game_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            game = dict(game=game_id, stop_reason='error')
            policy = None
            try:
                policy = load(args.checkpoint, device='cuda:0', config=CONFIG, max_actions=80, seed=f'42:{game_id}')
                pristine = (not policy.history and not policy.path_labels and policy.state is None
                    and policy.goal is None and policy.actions_taken == 0
                    and policy.observation is None and policy.pending_action is None
                    and all(not opt.state for opt in policy.bank.optimizers.values())
                    and all(torch.count_nonzero(pair[1]).item() == 0 for pair in policy.bank.tensors.values()))
                if not pristine or policy.runtime is not runtime:
                    raise RuntimeError('New-game isolation invariant failed')
                adapter_before = bank_digest(policy.bank)
                groups_before = group_digests(policy.bank)
                wrapper = arcade.make(game_id, seed=42, save_recording=False)
                if wrapper is None:
                    raise RuntimeError(f'Official SDK could not provision {game_id}')
                with (output / f'{game_id}.jsonl').open('x') as stream:
                    def emit(event):
                        stream.write(json.dumps(event, allow_nan=False) + '\n')
                        stream.flush()
                        if event['event'] == 'action':
                            print(json.dumps(dict(event='progress', game=game_id,
                                action=event['action_number'], levels=event['levels_completed'],
                                elapsed=round(event['elapsed_seconds'], 2),
                                adaptation=round(event['diagnostics']['adaptation_seconds'], 2))), flush=True)
                    game.update(run_game(wrapper, policy, started=game_started, emit=emit))
                game.update(fresh_private_state=pristine, adapter_sha256_before=adapter_before,
                            adapter_sha256_after=bank_digest(policy.bank), adapter_bytes=policy.bank.bytes,
                            adapter_groups_changed={k: v != groups_before[k]
                                for k, v in group_digests(policy.bank).items()})
                game['official'] = official_score(arcade, game_id)
            except Exception as exc:
                game.update(stop_reason='error', error=f'{type(exc).__name__}: {exc}',
                            elapsed_seconds=time.monotonic()-game_started)
                logging.exception('Game setup/reporting failed')
            torch.cuda.synchronize()
            verification_started = time.monotonic()
            game.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
            after, _ = base_digest(runtime)
            game.update(base_sha256_after=after, base_immutable=after == base_before,
                inactive_bank_unchanged=bank_digest(sentinel.bank) == inactive_before,
                base_frozen=all(not p.requires_grad and p.grad is None
                    for module in runtime.router.modules.values() for p in module.parameters()))
            game['verification_seconds'] = time.monotonic()-verification_started
            write_json(output / f'{game_id}.json', game)
            report['games'].append({key: value for key, value in game.items() if key != 'rows'})
            save_report(output, report)
            print(json.dumps(dict(event='game_end', game=game_id,
                actions=game.get('actions'), levels=game.get('levels_completed'), reason=game['stop_reason'])), flush=True)
            if not all(game[key] for key in ('base_immutable', 'inactive_bank_unchanged', 'base_frozen')):
                raise RuntimeError('Integrity failure; remaining games must not run')
            del policy
        report['status'] = 'finished_with_errors' if any(g['stop_reason'] == 'error' for g in report['games']) else 'finished'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        logging.exception('Experiment failed; retaining partial report')
        raise
    finally:
        if arcade is not None:
            try:
                arcade.close_scorecard()
            except Exception as exc:
                report['scorecard_close_error'] = f'{type(exc).__name__}: {exc}'
        if args.checkpoint.is_file():
            report['checkpoint']['sha256_after'] = file_digest(args.checkpoint)
            report['checkpoint']['file_immutable'] = report['checkpoint']['sha256_after'] == report['checkpoint'].get('sha256_before')
            if not report['checkpoint']['file_immutable']:
                report['status'] = 'failed'
        report['total_wall_seconds'] = time.monotonic()-started
        save_report(output, report)
    if report['status'] != 'finished' or not report['checkpoint']['file_immutable']:
        raise SystemExit('Experiment incomplete or integrity failed; inspect partial report')


if __name__ == '__main__':
    main()
