"""Online synthetic curriculum and standalone frozen SDK validation."""
import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import random
import time

import torch

from . import checkpoint
from .config import NetworkConfig, TrainConfig
from .data import Episode, random_action
from .environment import Factory
from .learning import Engine, Learner
from .metrics import Metrics, VisualQuality, delta_counts
from .networks import WorldModel


ROOT = Path(__file__).resolve().parents[1]


def emit(event, **values):
    print(json.dumps({'event': event, **values}, allow_nan=False), flush=True)


def validation(engine, factory, levels, seed):
    """Frozen, no replay writes; include the very first action in every episode."""
    c, results = engine.config, {}
    start = time.monotonic()
    for level in levels:
        metrics, wins, losses, actions = Metrics(), 0, 0, 0
        reconstruction_wrong, reconstruction_union, reconstruction_transitions = 0, 0, 0
        prediction_quality, reconstruction_quality = VisualQuality(), VisualQuality()
        for index in range(c.validation_episodes):
            episode_seed = 2 * ((seed * 100003 + (level or 0) * 1009 + index) % (2 ** 30)) + 1
            rng = random.Random(episode_seed)
            env = factory.make(episode_seed, level)
            observation = env.observation
            state = engine.initial(observation)
            if state.z is not None:
                metrics.observe_latent(state.z.cpu().numpy(), observation.final.shape)
            for _ in range(c.validation_steps):
                if observation.terminal:
                    break
                action = random_action(observation, rng)
                before = observation.final
                prediction = engine.prediction(state, action, before)
                next_observation = env.step(action)
                next_state = engine.observe(state, action, next_observation)
                after = next_observation.final
                if next_state.z is not None and after is not None:
                    metrics.observe_latent(next_state.z.cpu().numpy(), after.shape)
                if prediction is not None and after is not None and before.shape == after.shape:
                    metrics.add(before, after, prediction[0], state.z.cpu().numpy(),
                                next_state.z.cpu().numpy(), prediction[1])
                    prediction_quality.add(before, after, prediction[0])
                    # Diagnostic readout from the REAL successor, evaluated only
                    # after scoring the causal prediction. Never feed it back.
                    with torch.no_grad(), engine.autocast():
                        reconstruction = engine.decode(next_state.z[None], *after.shape).argmax(-1)[0]
                    reconstructed = reconstruction.cpu().numpy()
                    reconstruction_quality.add(before, after, reconstructed)
                    wrong, union, _ = delta_counts(before, after, reconstructed)
                    reconstruction_wrong += wrong
                    reconstruction_union += union
                    reconstruction_transitions += 1
                else:
                    metrics.skipped += 1
                observation, state = next_observation, next_state
                actions += 1
            wins += observation.state == 'WIN'
            losses += observation.state == 'GAME_OVER'
        reconstruction_error = reconstruction_wrong / max(1, reconstruction_union)
        passes = (metrics.passes(c) and reconstruction_transitions >= c.min_validation_transitions
                  and reconstruction_error <= c.delta_error)
        report = metrics.report() | {'passes': passes, 'wins': wins, 'game_overs': losses,
                                      'actions': actions, 'episodes': c.validation_episodes,
                                      'decoder_delta_error': reconstruction_error,
                                      'decoder_transitions': reconstruction_transitions,
                                      'prediction_quality': prediction_quality.report(),
                                      'reconstruction_quality': reconstruction_quality.report()}
        results[str(level)] = report
        emit('validation', level=level, **report)
    emit('validation_finished', seconds=round(time.monotonic() - start, 3))
    return results


def train(engine, learner, factory, run, progress, rng, output, stop_after_level=False):
    c = engine.config
    levels = run['levels']
    start = time.monotonic()
    starting_actions = progress['actions']
    episode, env = None, None
    latest_update = None
    reason = 'action_limit'
    try:
        while (not c.max_steps or progress['actions'] < c.max_steps) and progress['level_index'] < len(levels):
            level = levels[progress['level_index']]
            if c.max_level_steps and progress['level_actions'] >= c.max_level_steps:
                reason = 'level_limit_without_convergence'
                break
            if episode is None:
                episode_seed = rng.randrange(2 ** 30) * 2  # disjoint from odd validation seeds
                env = factory.make(episode_seed, level)
                episode = Episode([env.observation])
                progress['episodes'] += 1
            action = random_action(episode.observations[-1], rng)
            observation = env.step(action)
            episode.append(action, observation)
            episode.prepare(engine.device, engine.model.config.patch,
                            torch.bfloat16 if engine.amp else torch.float32, c.replay_bytes)
            progress['actions'] += 1
            progress['level_actions'] += 1
            ended = observation.terminal or len(episode.actions) >= c.max_episode_steps
            if progress['actions'] % c.train_every == 0 or ended:
                for _ in range(c.updates_per_train):
                    result = learner.update([episode])
                    if result is not None:
                        latest_update = result
                        progress['updates'] += 1
            if progress['actions'] % min(100, c.train_every) == 0 or ended or progress['actions'] == starting_actions + 1:
                elapsed = time.monotonic() - start
                emit('progress', **progress, level=level, loss=latest_update,
                     actions_per_second=round((progress['actions'] - starting_actions) / max(elapsed, 1e-6), 2),
                     replay_episodes=1, episode_actions=len(episode.actions),
                     episode_observations=len(episode.observations),
                     device_cache_mb=round(episode.device_bytes / 2 ** 20, 2),
                     gpu_peak_mb=round(torch.cuda.max_memory_allocated(engine.device) / 2 ** 20) if engine.device.type == 'cuda' else 0)
            if ended:
                # The terminal transition has already trained. No reference to
                # this episode's prepared tensors or recurrent graph survives.
                episode = None
            if progress['actions'] % c.validate_every == 0:
                reports = validation(engine, factory, levels[:progress['level_index'] + 1], run['seed'])
                current = reports[str(level)]
                progress['healthy_latent_seen'] = progress.get('healthy_latent_seen', False) or current['latent_std'] >= c.min_latent_std
                monitor_collapse = progress['healthy_latent_seen'] or progress['updates'] >= c.collapse_after_updates
                progress['collapse_streak'] = (progress.get('collapse_streak', 0) + 1
                    if monitor_collapse and current['latent_std'] < c.collapse_std else 0)
                if progress['collapse_streak'] >= c.collapse_patience:
                    reason = 'encoder_collapsed'
                    emit('encoder_collapsed', level=level, latent_std=current['latent_std'],
                         threshold=c.collapse_std, consecutive=progress['collapse_streak'])
                    break
                if current['delta_error'] < progress['best_error']:
                    progress['best_error'] = current['delta_error']
                    checkpoint.save(output / 'best.pt', learner, progress, run, rng)
                if all(r['passes'] for r in reports.values()):
                    progress['passes'] += 1
                else:
                    progress['passes'] = 0
                if progress['passes'] >= c.consecutive_passes:
                    emit('level_converged', level=level, actions=progress['level_actions'])
                    progress.update(level_index=progress['level_index'] + 1,
                                    level_actions=0, passes=0, best_error=1., collapse_streak=0,
                                    healthy_latent_seen=False)
                    # A new level never inherits the previous episode's memory.
                    episode = None
                    if stop_after_level:
                        reason = 'level_converged'
                        break
            if progress['actions'] % c.checkpoint_every == 0:
                checkpoint.save(output / 'latest.pt', learner, progress, run, rng)
                emit('checkpoint', path=str(output / 'latest.pt'), actions=progress['actions'])
        if progress['level_index'] == len(levels):
            reason = 'curriculum_converged'
    except KeyboardInterrupt:
        reason = 'interrupted'
    finally:
        checkpoint.save(output / 'latest.pt', learner, progress, run, rng)
    emit('stopped', reason=reason, **progress, checkpoint=str(output / 'latest.pt'))
    return reason


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('train', 'validate'))
    p.add_argument('--checkpoint', type=Path, help='Resume training or validate this snapshot')
    p.add_argument('--output', type=Path, default=ROOT / 'model' / 'runs' / 'square-cross')
    p.add_argument('--game', default=None)
    p.add_argument('--environments-dir', type=Path, default=None)
    p.add_argument('--mode', choices=('offline', 'normal'), default=None)
    p.add_argument('--levels', nargs='+', type=int, help='Isolated levels (bundled synthetic games only)')
    p.add_argument('--native', action='store_true', help='Use native SDK game progression without level isolation')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--profile', choices=('base', 'small'), default=None)
    p.add_argument('--device', default='auto')
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--fp8', action=argparse.BooleanOptionalAction, default=None,
                   help='FP8 tensorwise training for large Transformer linears (CUDA)')
    p.add_argument('--optimizer-bits', type=int, choices=(8, 32), default=None)
    p.add_argument('--reset-optimizer', action='store_true', help='Explicitly discard saved optimizer moments')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--stop-after-level', action='store_true', help='Stop after the current level passes all gates')
    for field in fields(TrainConfig):
        p.add_argument('--' + field.name.replace('_', '-'), type=type(field.default), default=None)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command == 'validate' and not args.checkpoint:
        p.error('validate requires --checkpoint')
    if args.native and args.levels:
        p.error('--native and --levels are mutually exclusive')
    if args.threads < 1:
        p.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    compiled = device.type == 'cuda' if args.compile is None else args.compile
    if device.type == 'cuda' and not compiled:
        p.error('CUDA execution requires torch.compile; --no-compile is for CPU development')
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    payload = checkpoint.load(args.checkpoint) if args.checkpoint else None
    saved_runtime = payload.get('runtime', {}) if payload else {}
    fp8 = args.fp8 if args.fp8 is not None else (saved_runtime.get('fp8', False) if device.type == 'cuda' else False)
    optimizer_bits = args.optimizer_bits if args.optimizer_bits is not None else (
        saved_runtime.get('optimizer_bits', 32) if device.type == 'cuda' else 32)
    if payload and args.command == 'train' and optimizer_bits != saved_runtime.get('optimizer_bits', 32) and not args.reset_optimizer:
        p.error('Changing optimizer precision requires --reset-optimizer; weights are preserved')
    run = dict(payload['run']) if payload else {
        'game': args.game or 'sc01-v1', 'environments_dir': str(args.environments_dir or ROOT / 'games'),
        'mode': args.mode or 'offline', 'seed': args.seed if args.seed is not None else 0,
        'levels': args.levels or ([1, 2, 3, 4, 5] if (args.game or 'sc01-v1') == 'sc01-v1' and not args.native else [None])}
    if args.levels and (len(set(args.levels)) != len(args.levels) or min(args.levels) < 1):
        p.error('--levels must be distinct positive levels in curriculum order')
    if payload:
        for key in ('game', 'environments_dir', 'mode', 'seed'):
            value = getattr(args, key)
            if value is not None and (str(value) if key == 'environments_dir' else value) != run[key]:
                if args.command == 'train':
                    p.error(f'Resume cannot change {key}; use validate for separate evaluation')
                run[key] = str(value) if key == 'environments_dir' else value
        if args.command == 'train' and (args.levels is not None or args.native or args.profile is not None):
            p.error('Resume restores network and curriculum; omit --profile/--levels/--native')
        if args.command == 'validate':
            run['levels'] = args.levels or ([None] if args.native or args.game else
                run['levels'][:min(len(run['levels']), payload['progress']['level_index'] + 1)])
    network = NetworkConfig(**payload['network']) if payload else (
        NetworkConfig.small() if args.profile == 'small' else NetworkConfig())
    training = TrainConfig(**{k: v for k, v in payload['training'].items()
                              if k not in {'replay_episodes', 'batch_episodes'}}) if payload else TrainConfig()
    overrides = {f.name: getattr(args, f.name) for f in fields(TrainConfig) if getattr(args, f.name) is not None}
    training = replace(training, **overrides)
    random.seed(run['seed'])
    torch.manual_seed(run['seed'])
    rng = random.Random(run['seed'])
    emit('initializing', device=str(device), compile=compiled, fp8=fp8, optimizer_bits=optimizer_bits,
         network=asdict(network), training=asdict(training), game=run['game'], levels=run['levels'])
    model = WorldModel(network).to(device)
    engine = Engine(model, training, compiled, fp8=fp8, optimizer_bits=optimizer_bits)
    factory = Factory(run['game'], run['environments_dir'], run['mode'])
    emit('ready', parameters=sum(p.numel() for p in model.parameters()), amp=engine.amp, fp8_layers=engine.fp8_layers,
         note='First calls include compilation warmup' if compiled else 'Eager execution')
    if args.command == 'validate':
        model.load_state_dict(payload['weights'])
        validation(engine, factory, run['levels'], run['seed'])
        return
    learner = Learner(engine)
    progress = dict(payload['progress']) if payload else dict(
        actions=0, updates=0, episodes=0, level_index=0, level_actions=0, passes=0, best_error=1.)
    if payload:
        checkpoint.restore(payload, learner, rng, reset_optimizer=args.reset_optimizer)
        if args.reset_optimizer:
            emit('optimizer_reset', previous_bits=saved_runtime.get('optimizer_bits', 32), optimizer_bits=optimizer_bits)
        for opt in learner.optimizers:
            for group in opt.param_groups:
                group['lr'] = training.learning_rate
        if overrides:
            progress['passes'] = 0  # previous gates used the saved configuration
            progress['collapse_streak'] = 0
            progress['best_error'] = 1.
        emit('resumed', **progress, note='Fresh seeded episode and empty device input cache')
    train(engine, learner, factory, run, progress, rng, args.output.resolve(), args.stop_after_level)
