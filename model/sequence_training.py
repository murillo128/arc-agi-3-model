"""Train independent game sessions with cross-level recurrent context.

Run: python -m model.sequence_training --training-config CONFIG --output DIR
Use --source CHECKPOINT to warm-start weights, or --resume to retain optimizers.
SIGINT/SIGTERM finish the current batch and atomically save its completed state.
"""
import argparse
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import signal
import time

import torch

from . import checkpoint
from .config import NetworkConfig, TrainConfig, SequenceConfig
from .environment import Factory
from .game_sequences import (SequenceLearner, EncodedSessions, collect_session,
                             evaluate_sessions, schedule_games, merge_evaluations)
from .learning import Engine
from .networks import WorldModel
from .online_learning import OnlineRepresentation
from .training_config import digest, load_plan, run_owner
from .sequence_control import SequenceControlConfig, SequencePhaseControl, TrainingBatchDiagnostics


def synchronize(engine):
    if engine.device.type == 'cuda':
        torch.cuda.synchronize(engine.device)


def factories_for(plan):
    sources = {}
    for stage in plan.stages:
        item = sources.setdefault(stage['game'], dict(source=stage['source'], levels=[]))
        item['levels'].append(stage['level'])
    result = {}
    try:
        for game, item in sources.items():
            if item['source'] == 'sdk':
                if plan.payload['phase'] != 'development':
                    raise ValueError('Sequence SDK experiments require a development split')
                if item['levels'] != list(range(1, max(item['levels']) + 1)):
                    raise ValueError('SDK sequence levels must be the allowed prefix starting at 1')
                sdk = plan.payload['sdk']
                factory = Factory(game, sdk['environments_dir'], sdk['mode'], reserve_final_level=True)
            else:
                factory = Factory(game, Path(__file__).resolve().parents[1] / 'games')
            result[game] = (factory, item['levels'])
    except Exception:
        close_factories(result)
        raise
    return result


def close_factories(factories):
    for factory, _ in factories.values():
        factory.arcade.close_scorecard()


def collect_batch(factories, config, batch_index, seed, *, evaluation=False):
    games = schedule_games(list(factories), batch_index, config, seed=seed)
    sessions = []
    for i, game in enumerate(games):
        # Even training / odd evaluation seeds, including intra-session resets.
        # Reserve a full batch's seed slots even when a shuffled round ends short.
        session_seed = 4_000_000 + 2 * (seed * 10_000 + batch_index * config.batch_games + i) + int(evaluation)
        factory, levels = factories[game]
        sessions.append(collect_session(factory, levels, session_seed, config))
    if any(not any(s.supervised for s in row.transitions) for row in sessions):
        raise ValueError('A session has no eligible real transitions; check level access and budgets')
    return sessions


def load_engine(plan, source=None, *, device='cpu', compile_model=False):
    initial = plan.payload['initialization']
    saved = checkpoint.load(source) if source else None
    if saved and saved.get('format') == 'arc3-agent':
        saved = saved['world']
    network = NetworkConfig(**(saved['network'] if saved else initial['network']))
    training = TrainConfig(**(saved['training'] if saved else initial.get('training', {})))
    training = replace(training, learning_rate=plan.settings['learning_rate'],
                       prediction_delta_loss=plan.settings['prediction_delta_loss'])
    net = WorldModel(network).to(device)
    if saved:
        net.load_state_dict(saved['weights'], strict=True)
    return Engine(net, training, compile_model=compile_model)


def representation_for(engine, plan):
    learner = OnlineRepresentation(engine, plan.settings['representation_learning_rate'],
        freeze_encoder=plan.settings['freeze_encoder'],
        encoder_learning_rate_after_predictor=plan.settings['encoder_learning_rate_after_predictor'])
    if 'sequence_control' not in plan.payload:
        learner.activate_predictor()
    return learner


def predictor_learning_rate(learner, value=None):
    """Override only the predictor rate, retaining Adam state and the run identity."""
    if value is not None:
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError('Predictor learning rate must be finite and between 0 and 1')
        for group in learner.optimizer.param_groups:
            group['lr'] = value
    # Ordinary resume restores the optimizer's effective rate, including an earlier
    # explicit override. Keep checkpoint training metadata consistent with it.
    learner.engine.config = replace(learner.engine.config,
        learning_rate=learner.optimizer.param_groups[0]['lr'])


def fit_batch(learner, sessions, representation=None, *, telemetry=None,
              train_predictor=True, training_probe=None):
    """One representation phase, then causal predictor blocks on fixed latents."""
    e, c = learner.engine, learner.config
    synchronize(e)
    begin = time.perf_counter()
    reconstruction = []
    if representation is not None:
        # Rotate games evenly. Frame selection is independent of prediction error.
        rng = random.Random(sum(s.seed for s in sessions))
        for i in range(c.representation_updates):
            candidates = [s for s in sessions[i % len(sessions)].transitions if s.supervised]
            if candidates:
                step = rng.choice(candidates)
                result = representation.update([step.before, step.after], telemetry=telemetry)
                if result is not None:
                    reconstruction.append(result)
    synchronize(e)
    representation_seconds = time.perf_counter() - begin
    drift_seconds = 0.
    if training_probe is not None:
        start = time.perf_counter()
        training_probe.after_representation()
        synchronize(e)
        drift_seconds = time.perf_counter() - start
    start = time.perf_counter()
    e.model.eval()
    encoded = EncodedSessions(e, sessions) if train_predictor else None
    synchronize(e)
    encode_seconds = time.perf_counter() - start
    start = time.perf_counter()
    updates = []
    length = max(len(s.transitions) for s in sessions)
    last_supervised = max((i for s in sessions for i, step in enumerate(s.transitions) if step.supervised), default=-1)
    for position in range(0, length if train_predictor else 0, c.block_steps):
        result = learner.update(encoded, position, min(length, position + c.block_steps),
            telemetry=telemetry, last=position <= last_supervised < position + c.block_steps)
        if result is not None:
            updates.append(result)
    synchronize(e)
    predictor_seconds = time.perf_counter() - start
    probe_seconds = 0.
    if telemetry and telemetry.probe_requested and train_predictor:
        from .optimization_probe import memory_probe
        probe_start = time.perf_counter()
        telemetry.probe = dict(memory_probe(learner, encoded, telemetry.probe_session),
                               batch=telemetry.batch, event_id=telemetry.tick)
        telemetry.probe_requested = False
        synchronize(e)
        probe_seconds = time.perf_counter() - probe_start
    return dict(representation_seconds=representation_seconds, encode_seconds=encode_seconds,
        predictor_seconds=predictor_seconds, training_seconds=time.perf_counter() - begin,
        drift_seconds=drift_seconds,
        probe_seconds=probe_seconds,
        updates=len(updates), examples=sum(x['examples'] for x in updates),
        loss=sum(x['loss'] for x in updates)/len(updates) if updates else None,
        predictor_terms={key: sum(x[key] for x in updates)/len(updates)
                         for key in ('latent', 'visual')} if updates else {},
        reconstruction_loss=sum(x['total'] for x in reconstruction)/len(reconstruction) if reconstruction else None,
        representation_terms={key: sum(x['terms'][key] for x in reconstruction if key in x['terms']) /
                              sum(key in x['terms'] for x in reconstruction)
                              for key in {k for x in reconstruction for k in x['terms']}},
        examples_per_second=sum(x['examples'] for x in updates)/max(1e-9, predictor_seconds))


def save_run(path, plan, learner, representation, batches, actions, control=None, *, extensions=()):
    e = learner.engine
    checkpoint.write(path, dict(kind='game_sequences', version=1, plan=plan.payload,
        network=asdict(e.model.config), training=asdict(e.config), weights=e.model.state_dict(),
        sequence=asdict(learner.config), predictor_optimizer=learner.optimizer.state_dict(),
        representation_optimizer=representation.optimizer.state_dict(),
        progress=dict(batches=batches, actions=actions, updates=learner.updates),
        curriculum_extensions=list(extensions),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if e.device.type == 'cuda' else [],
        **(dict(sequence_control=control.snapshot(), representation_state=dict(
            sigreg_enabled=representation.sigreg_enabled, updates=representation.updates)) if control else {})))


def extended_plan_matches(previous, current):
    """Only an appended curriculum and the first SDK setup may differ."""
    count = len(previous['stages'])
    if len(current['stages']) <= count or current['stages'][:count] != previous['stages']:
        return False
    expected = copy.deepcopy(current)
    expected['stages'] = previous['stages']
    if not previous['sdk']:
        expected['sdk'] = {}
    if 'sequence_control' in previous and 'sequence_control' in current:
        old_window = previous['sequence_control']['window_batches']
        if current['sequence_control']['window_batches'] < old_window:
            return False
        expected['sequence_control']['window_batches'] = old_window
    return expected == previous


def restore_run(saved, plan, learner, representation, control=None, *, adopt=False, extend=False):
    if adopt and extend:
        raise ValueError('Curriculum extension cannot be combined with phase adoption')
    expected = copy.deepcopy(plan.payload)
    if adopt:
        if control is None or 'sequence_control' in saved or 'sequence_control' in saved['plan']:
            raise ValueError('Phase adoption requires a legacy sequence checkpoint and a controlled plan')
        expected.pop('sequence_control')
        expected['settings']['encoder_learning_rate_after_predictor'] = saved['plan']['settings']['encoder_learning_rate_after_predictor']
    changed = saved['plan'] != expected
    if (saved.get('kind') != 'game_sequences' or saved.get('version') != 1 or
            (changed and not (extend and extended_plan_matches(saved['plan'], expected))) or
            saved['sequence'] != asdict(learner.config)):
        raise ValueError('Resume requires the same sequence configuration and training plan')
    if control is not None and not adopt:
        if 'sequence_control' not in saved:
            raise ValueError('Missing checkpoint phase state; use explicit adoption for legacy runs')
        control.restore(saved['sequence_control'], extend=extend and changed)
    learner.optimizer.load_state_dict(saved['predictor_optimizer'])
    representation.optimizer.load_state_dict(saved['representation_optimizer'])
    learner.updates = saved['progress']['updates']
    if control is not None:
        state = saved.get('representation_state', {})
        representation.sigreg_enabled = state.get('sigreg_enabled',
            representation.sigreg_enabled and learner.updates == 0)
        representation.updates = state.get('updates', 0)
        control.apply(representation)
    torch.set_rng_state(saved['torch_rng'])
    if learner.engine.device.type == 'cuda' and saved['cuda_rng']:
        if len(saved['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed on resume')
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved['progress']['batches'], saved['progress']['actions']


def write_report(path, report):
    report['updated_at'] = time.time()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, allow_nan=False) + '\n')
    temporary.replace(path)


def run(args):
    rate = getattr(args, 'predictor_learning_rate', None)
    if rate is not None and (not math.isfinite(rate) or not 0 < rate < 1):
        raise ValueError('Predictor learning rate must be finite and between 0 and 1')
    plan = load_plan(args.training_config)
    if 'sequence' not in plan.payload:
        raise ValueError('Add sequence settings to the training configuration')
    config = SequenceConfig(**plan.payload['sequence'])
    output = args.output.resolve()
    if args.resume and args.source:
        raise ValueError('Choose --source for a fresh run or --resume, not both')
    adopt = getattr(args, 'adopt_training_phases', False)
    extend = getattr(args, 'extend_curriculum', False)
    if adopt and not args.resume:
        raise ValueError('--adopt-training-phases requires --resume')
    if extend and (not args.resume or adopt):
        raise ValueError('--extend-curriculum requires --resume without --adopt-training-phases')
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError('--max-batches must be positive')
    action_budget = None if getattr(args, 'unlimited', False) else plan.settings['actions']
    with run_owner(output, args.resume):
        source = output / 'latest.pt' if args.resume else args.source
        torch.manual_seed(plan.settings['seed'])
        engine = load_engine(plan, source, device=args.device, compile_model=args.compile)
        learner, representation = SequenceLearner(engine, config), representation_for(engine, plan)
        control = (SequencePhaseControl(plan.stages, plan.settings,
            SequenceControlConfig(**plan.payload['sequence_control'])) if 'sequence_control' in plan.payload else None)
        batches, actions, extensions = 0, 0, []
        if args.resume:
            saved = checkpoint.load(source)
            batches, actions = restore_run(saved, plan, learner, representation, control,
                                           adopt=adopt, extend=extend)
            extensions = list(saved.get('curriculum_extensions', []))
            if extend and saved['plan'] != plan.payload:
                extensions.append(dict(batch=batches, actions=actions,
                    previous_plan=digest(saved['plan']), plan=plan.identity,
                    games_before=list(dict.fromkeys(s['game'] for s in saved['plan']['stages'])),
                    games_after=list(dict.fromkeys(s['game'] for s in plan.stages)),
                    added_stages=list(plan.stages[len(saved['plan']['stages']):])))
            del saved
        predictor_learning_rate(learner, rate)
        if control:
            control.apply(representation)
        telemetry = None
        if getattr(args, 'optimization', False):
            from .optimization_telemetry import OptimizationTelemetry
            telemetry = OptimizationTelemetry(engine.model, output, every=args.optimization_every,
                restored_batch=batches, identity=json.dumps(plan.payload, sort_keys=True))
        stop = False
        def interrupt(signum, frame):
            nonlocal stop
            stop = True
        handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
        factories = {}
        report_path = output / 'report.json'
        report = dict(kind='game_sequences', status='running', config=asdict(config), history=[],
                      progress=dict(batches=batches, actions=actions), games=list(dict.fromkeys(s['game'] for s in plan.stages)))
        report['schedule'] = list(plan.stages)
        report['curriculum_extensions'] = extensions
        report['budget'] = action_budget
        report['measurement_source'] = 'training_pre_update' if control else 'fixed_evaluation'
        if control:
            report['phase_control'] = control.report()
        report['learning_rates'] = dict(predictor=learner.optimizer.param_groups[0]['lr'],
                                        **representation.learning_rates)
        if telemetry:
            report['optimization_identity'] = telemetry.identity
        if args.resume and report_path.exists():
            previous = json.loads(report_path.read_text())
            if adopt:
                # Keep the original report as evidence; never join different metric sources.
                archive = output / f'report-before-training-phases-{batches}.json'
                if not archive.exists():
                    archive.write_text(json.dumps(previous, allow_nan=False) + '\n')
                report['adopted_at_batch'] = batches
            else:
                report['history'] = [r for r in previous.get('history', []) if r['batch'] <= batches][-200:]
                report['prediction_samples'] = previous.get('prediction_samples', [])
                report['prediction_sample_batch'] = previous.get('prediction_sample_batch')
                if 'adopted_at_batch' in previous:
                    report['adopted_at_batch'] = previous['adopted_at_batch']
        completed_here = 0
        try:
            factories = factories_for(plan)
            # Cover the entire configured catalog in bounded GPU batches.
            evaluation = [] if control else [collect_batch(factories, replace(config, game_schedule='interleaved'), i,
                                        plan.settings['seed'], evaluation=True)
                          for i in range((len(factories) + config.batch_games - 1)//config.batch_games)]
            from .sequence_diagnostics import SequenceDiagnostics
            diagnostics = SequenceDiagnostics(engine, [s for batch in evaluation for s in batch])
            write_report(report_path, report)
            while not stop and (action_budget is None or actions < action_budget):
                begin = time.perf_counter()
                if telemetry:
                    telemetry.start_batch(batches + 1)
                sessions = collect_batch(factories, config, batches, plan.settings['seed'])
                collection_seconds = time.perf_counter() - begin
                if control:
                    phase = control.phase
                    control.apply(representation)
                    learning_rates = dict(predictor=learner.optimizer.param_groups[0]['lr'], **representation.learning_rates)
                    probe = TrainingBatchDiagnostics(engine, sessions, control.config.diagnostic_samples)
                    probe.start(batches+1)
                    evaluation_start = time.perf_counter()
                    quality = merge_evaluations([evaluate_sessions(engine, sessions,
                        early_steps=config.loss_warmup_steps, reference_loss=bool(telemetry), diagnostics=probe)])
                    synchronize(engine)
                    evaluation_seconds = time.perf_counter() - evaluation_start
                    measured = fit_batch(learner, sessions, representation, telemetry=telemetry,
                        train_predictor=phase == 'predictor', training_probe=probe)
                    measured['diagnostics'], report['prediction_samples'] = probe.finish()
                    report['prediction_sample_batch'] = batches+1
                    measured.update(evaluation_seconds=evaluation_seconds, diagnostics_seconds=probe.seconds,
                        phase=phase, learning_rates=learning_rates, measurement_source='training_pre_update')
                    control.observe(batches+1, {k: dict(v) for k, v in probe.pairs.items()})
                    control.apply(representation)
                    report['phase_control'] = control.report()
                    report['learning_rates'] = dict(predictor=learner.optimizer.param_groups[0]['lr'], **representation.learning_rates)
                    measured['next_phase'] = control.phase
                else:
                    measured = (fit_batch(learner, sessions, representation, telemetry=telemetry) if telemetry
                                else fit_batch(learner, sessions, representation))
                    evaluation_start = time.perf_counter()
                    measure_diagnostics = completed_here == 0 or (batches+1) % diagnostics.every == 0
                    if measure_diagnostics:
                        diagnostics.start(batches+1)
                    quality = merge_evaluations([evaluate_sessions(engine, batch,
                        early_steps=config.loss_warmup_steps, reference_loss=bool(telemetry),
                        diagnostics=diagnostics if measure_diagnostics else None) for batch in evaluation])
                    if measure_diagnostics:
                        measured['diagnostics'], report['prediction_samples'] = diagnostics.finish()
                        report['prediction_sample_batch'] = batches+1
                        measured['diagnostics_seconds'] = diagnostics.seconds
                    synchronize(engine)
                    measured['evaluation_seconds'] = time.perf_counter() - evaluation_start
                real_actions = sum(s.environment_actions for s in sessions)
                batches += 1
                actions += real_actions
                save_start = time.perf_counter()
                save_run(output / 'latest.pt', plan, learner, representation, batches, actions, control,
                         extensions=extensions)
                measured['save_seconds'] = time.perf_counter() - save_start
                if telemetry:
                    telemetry_start = time.perf_counter()
                    capture_seconds = telemetry.seconds
                    telemetry.reference_source = report['measurement_source']
                    telemetry.finish_batch(quality['reference']['nll'], actions)
                    measured['telemetry_seconds'] = capture_seconds + time.perf_counter() - telemetry_start
                seconds = time.perf_counter() - begin
                row = dict(batch=batches, actions=actions, recent_tps=real_actions/seconds,
                           collection_seconds=collection_seconds, total_seconds=seconds,
                           quality=quality, sessions=[dict(game=s.game, steps=len(s.transitions),
                               boundaries=sum(t.reset_before for t in s.transitions), end=s.end_reason) for s in sessions], **measured)
                game_errors = [quality[g]['delta_error'] for g in factories
                               if g in quality and quality[g]['delta_error'] is not None]
                row['macro_delta_error'] = sum(game_errors)/len(game_errors) if game_errors else None
                if engine.device.type == 'cuda':
                    row['peak_gpu_gib'] = torch.cuda.max_memory_allocated(engine.device)/1024**3
                report['history'] = (report['history'] + [row])[-200:]
                report['progress'] = dict(batches=batches, actions=actions, updates=learner.updates)
                write_report(report_path, report)
                print(json.dumps(dict(event='sequence_batch', **row), allow_nan=False), flush=True)
                completed_here += 1
                if args.max_batches is not None and completed_here >= args.max_batches:
                    break
            report['status'] = 'paused' if stop or action_budget is None or actions < action_budget else 'budget_complete'
            write_report(report_path, report)
        except BaseException:
            report['status'] = 'error'
            write_report(report_path, report)
            raise
        finally:
            close_factories(factories)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--predictor-learning-rate', type=float,
                        help='Explicitly override only the predictor LR, preserving optimizer state on resume')
    parser.add_argument('--extend-curriculum', action='store_true',
                        help='Explicitly append training games/levels on resume, preserving weights and optimizers')
    parser.add_argument('--adopt-training-phases', action='store_true',
                        help='Explicitly adopt sequence_control on a legacy checkpoint, retaining weights and optimizers')
    parser.add_argument('--unlimited', action='store_true',
                        help='Ignore the total action budget; run until interrupted or --max-batches is reached')
    parser.add_argument('--optimization', action='store_true', help='Record bounded optimizer diagnostics')
    parser.add_argument('--optimization-every', type=int, default=8, help='Predictor updates between captures')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--max-batches', type=int)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    run(args)


if __name__ == '__main__':
    main()
