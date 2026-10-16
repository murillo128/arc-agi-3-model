"""Bounded fresh-data comparison followed by frozen-world value calibration.

Collect each real action once. The second recipe sees exactly the same ordered
prefixes, progressively exposed from raw CPU RAM, never future observations.
Only small reports and portable weights are saved; no replay data is persisted.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from . import agent_checkpoint
from .behavior_experiment import episode_seeds, evaluate, evaluation_summary, fit
from .campaign import calibration_coverage, emit, quotas, replay_records, write_report
from .data import Episode, Replay, random_action
from .environment import Factory, SYNTHETIC_LEVELS
from .learning import Learner
from .learning_probe import collect, score
from .planning import fingerprint, outcome


def fresh_episodes(factories, pairs, actions, steps, seed):
    """Fair pair quotas; keep boundary outcomes, censor quota interruptions."""
    budgets, counts = quotas(actions, pairs), dict.fromkeys(pairs, 0)
    number = 0
    while sum(counts.values()) < actions:
        pair = min((p for p in pairs if counts[p] < budgets[p]), key=lambda p: counts[p] / budgets[p])
        game, level = pair
        env_seed = episode_seeds('train', 1, offset=500000 + seed * 1000 + number)[0]
        env, rng = factories[game].make(env_seed, level), random.Random(env_seed)
        before = env.observation
        if before.terminal:
            raise ValueError('A collection episode must start from a nonterminal observation')
        episode = Episode([before])
        reason = 'interrupted'
        for position in range(min(steps, budgets[pair] - counts[pair])):
            action = random_action(before, rng)
            after = env.step(action)
            episode.append(action, after)
            counts[pair] += 1
            event = outcome(before, after)
            if event:
                reason = 'progress' if event == 1 else 'game_over'
                break
            if position + 1 == steps:
                reason = 'budget'
            before = after
        number += 1
        yield pair, episode, reason


def digest_episode(digest, pair, episode, reason):
    digest.update(json.dumps([pair, reason, [(a.id, a.data) for a in episode.actions]], sort_keys=True).encode())
    for obs in episode.observations:
        digest.update(json.dumps([obs.frames.shape, obs.state, obs.levels_completed,
                                  obs.available_actions, obs.win_levels]).encode())
        digest.update(obs.frames.tobytes())


def selection_error(measured):
    """Predeclared macro mean: every covered pair and horizon has equal weight.

    Full delta unions penalize invented changes even on unchanged images. The
    separate strata make zero-change coverage visible rather than calling it
    successful prediction of dynamics. No confirmation data enters selection.
    """
    values = [h['readouts']['contextual']['prediction_delta_error']
              for row in measured['per_pair'] for h in row['rollouts'].values() if h['samples']]
    if not values:
        raise ValueError('No covered development anchors for selection')
    return sum(values) / len(values)


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--actions', type=int, default=4096)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--train-every', type=int, default=8)
    parser.add_argument('--evaluate-every', type=int, default=1024)
    parser.add_argument('--world-episodes', type=int, default=2)
    parser.add_argument('--world-steps', type=int, default=32)
    parser.add_argument('--behavior-episodes', type=int, default=4)
    parser.add_argument('--head-epochs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=23)
    args = parser.parse_args(argv)
    if min(args.actions, args.steps, args.train_every, args.evaluate_every, args.world_episodes,
           args.world_steps, args.behavior_episodes, args.head_epochs) < 1 or args.steps > 128 or not 0 <= args.seed < 400:
        parser.error('Budgets must be positive, steps at most 128, and seed in 0..399')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'report.json'
    if path.exists():
        parser.error('Use a fresh streaming probe directory')
    report = {'status': 'running', 'scope': 'streaming_real_training_probe',
              'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'selection_metric': 'macro_pair_horizon_visual_delta_error',
              'progress': {}, 'results': [], 'confirmation': [], 'policies': {}}
    last_write = 0.

    def publish(force=False, **values):
        nonlocal last_write
        report['progress'].update(values)
        if not force and time.monotonic() - last_write < 2:
            return
        last_write = time.monotonic()
        write_report(path, report)
        emit('streaming_probe_progress', **report['progress'])

    pairs = [(g, l) for g, n in SYNTHETIC_LEVELS.items() for l in range(1, n + 1)]
    factories = {g: Factory(g, Path(__file__).resolve().parents[1] / 'games') for g in SYNTHETIC_LEVELS}
    archive, ordered = {pair: [] for pair in pairs}, []
    baseline = None
    try:
        publish(force=True, phase='collection', split='development')
        development = {}
        for index, (game, level) in enumerate(pairs):
            development[game, level], _ = collect(factories[game], level,
                episode_seeds('dev', args.world_episodes, offset=500000 + index * 100), args.world_steps)
        for variant, visual_weight in (('latent_2step', 0.), ('visual_2step', .5)):
            report['progress'] = {'variant': variant, 'phase': 'loading', 'actions': 0,
                                  'total_actions': args.actions, 'updates': 0}
            publish(force=True)
            torch.compiler.reset()
            torch.manual_seed(args.seed)
            rng = random.Random(args.seed)
            agent = agent_checkpoint.load(args.agent, device=args.device)
            engine = agent.engine
            engine.config = replace(engine.config, freeze_representation=True, auto_representation=False,
                representation_joint_steps=0, rollout_weight=.5, rollout_visual_weight=visual_weight,
                loss_steps=8, burn_in_steps=0, cache_frozen_latents=True, max_episode_steps=args.steps)
            engine.model.dynamics.requires_grad_(True)
            learner = Learner(engine)
            frozen = {name: fingerprint(getattr(engine.model, name)) for name in ('encoder', 'decoder')}
            row = {'variant': variant, 'initial_world': fingerprint(engine.model), 'points': []}
            report['results'].append(row)
            if baseline is None:
                publish(force=True, phase='evaluation', split='development')
                baseline = score(engine, development, publish)
            initial = {'actions': 0, 'updates': 0, 'development': deepcopy(baseline),
                       'selection_error': selection_error(baseline)}
            row['points'].append(initial)
            row['best'] = {'actions': 0, 'updates': 0, 'selection_error': initial['selection_error'],
                           'checkpoint': str(args.agent.resolve())}
            replay = Replay(engine.config.replay_episodes, engine.config.replay_bytes)
            actions = updates = 0
            next_check, started = args.evaluate_every, time.monotonic()
            digest = hashlib.sha256()
            source = fresh_episodes(factories, pairs, args.actions, args.steps, args.seed) if not ordered else iter(ordered)
            for pair, actual, reason in source:
                if variant == 'latent_2step':
                    ordered.append((pair, actual, reason))
                    archive[pair].append((actual, reason))
                    if sum(ep.nbytes for _, ep, _ in ordered) > 512 * 1024**2:
                        raise MemoryError('Raw calibration archive exceeds its 512 MiB RAM budget')
                    report['coverage'] = calibration_coverage(archive)
                digest_episode(digest, pair, actual, reason)
                live = Episode([actual.observations[0]])
                replay.add(live)
                pending = False
                for index, action in enumerate(actual.actions):
                    before, after = actual.observations[index:index + 2]
                    if not after.starts_new_level(before):
                        live.append(action, after)
                        pending = True
                    actions += 1
                    ended = index + 1 == len(actual.actions)
                    publish(force=actions == 1, phase='training', actions=actions, updates=updates, game=pair[0], level=pair[1],
                            seconds=time.monotonic() - started)
                    if pending and (actions % args.train_every == 0 or ended):
                        batch = replay.sample(rng, 2, 2, latest=live, random_prefix=True)
                        loss = learner.update(batch)
                        del batch
                        if loss is not None:
                            updates += 1
                        pending = False
                        publish(phase='training', updates=updates, loss=loss)
                if actions >= next_check or actions == args.actions:
                    # Persist last weights before evaluation, as well as the
                    # separately selected best checkpoint. No raw replay export.
                    agent_checkpoint.save(args.output / f'{variant}-last.pt', agent)
                    publish(force=True, phase='evaluation', split='development', actions=actions, updates=updates)
                    measured = score(engine, development, publish)
                    point = {'actions': actions, 'updates': updates, 'seconds': time.monotonic() - started,
                             'development': measured, 'selection_error': selection_error(measured)}
                    row['points'].append(point)
                    if point['selection_error'] < row['best']['selection_error']:
                        best_path = args.output / f'{variant}-best.pt'
                        agent_checkpoint.save(best_path, agent)
                        row['best'] = {k: point[k] for k in ('actions', 'updates', 'selection_error')}
                        row['best']['checkpoint'] = str(best_path.resolve())
                    publish(force=True)
                    next_check = (actions // args.evaluate_every + 1) * args.evaluate_every
            row.update(complete=True, actions=actions, updates=updates, data_digest=digest.hexdigest(),
                       frozen_representation_verified=all(fingerprint(getattr(engine.model, name)) == expected
                           for name, expected in frozen.items()), dynamics_changed=fingerprint(engine.model) != row['initial_world'])
            if not row['frozen_representation_verified'] or not row['dynamics_changed']:
                raise RuntimeError('Unexpected predictor or frozen representation update')
            for episode in replay.episodes:
                episode.latent_cache.clear()
            del learner, replay, engine, agent, live
            release()
        a, b = report['results']
        if (a['data_digest'], a['updates'], a['initial_world']) != (b['data_digest'], b['updates'], b['initial_world']):
            raise RuntimeError('Paired recipes did not receive identical data, updates and initial weights')
        selected = min(report['results'], key=lambda r: r['best']['selection_error'])
        report['selected'] = dict(selected['best'], variant=selected['variant'])
        publish(force=True, phase='collection', split='confirmation')
        confirmation = {}
        for index, (game, level) in enumerate(pairs):
            confirmation[game, level], _ = collect(factories[game], level,
                episode_seeds('final', args.world_episodes, offset=500000 + index * 100), args.world_steps)
        for label, checkpoint in (('initial', args.agent), ('selected', Path(selected['best']['checkpoint']))):
            torch.compiler.reset()
            agent = agent_checkpoint.load(checkpoint, device=args.device)
            before = fingerprint(agent.engine.model)
            publish(force=True, phase='confirmation', split='confirmation', variant=label)
            report['confirmation'].append({'variant': label, **score(agent.engine, confirmation, publish)})
            if fingerprint(agent.engine.model) != before:
                raise RuntimeError('Confirmation changed world weights')
            del agent
            release()
        torch.compiler.reset()
        agent = agent_checkpoint.load(Path(selected['best']['checkpoint']), device=args.device)
        agent.engine.config = replace(agent.engine.config, max_episode_steps=args.steps)
        before_world, before_heads = fingerprint(agent.engine.model), fingerprint(agent.heads)
        report['progress'] = {'phase': 'calibration_data', 'variant': selected['variant'], 'actions': args.actions}
        publish(force=True)
        records = []
        for game, level in pairs:
            publish(force=True, game=game, level=level)
            records.extend(replay_records(agent.engine, {(game, level): archive[game, level]}, agent.planner.config.discount))
        report['calibration'] = dict(report['coverage'], records=len(records),
            positive_records=sum(r.target is not None and r.target > 0 for r in records),
            censored_records=sum(r.target is None for r in records))
        publish(force=True, phase='head_calibration', epochs=args.head_epochs)
        report['head_fit'] = fit(agent.engine, agent.heads, records, args.head_epochs, 32, 3e-4, args.seed,
            compiled=args.device == 'cuda', progress_callback=lambda p: publish(force=True, **p))
        del records
        release()
        if fingerprint(agent.engine.model) != before_world or fingerprint(agent.heads) == before_heads:
            raise RuntimeError('Value calibration did not preserve world weights or update value heads')
        agent.heads.eval().requires_grad_(False)
        agent_checkpoint.save(args.output / 'calibrated-agent.pt', agent)
        report['calibration']['world_unchanged'] = True
        before_evaluation = fingerprint(agent.heads)
        for mode in ('random', 'q', 'value_mpc'):
            results, began = [], time.monotonic()
            policy = {'per_pair': []}
            report['policies'][mode] = policy
            for pair_index, (game, level) in enumerate(pairs, 1):
                report['progress'] = {'phase': 'behavior', 'policy': mode, 'game': game, 'level': level,
                                      'pair_index': pair_index, 'pair_count': len(pairs)}
                publish(force=True)
                measured = evaluate(agent.engine, agent.planner, factories[game], level,
                    episode_seeds('final', args.behavior_episodes, offset=600000), args.steps, mode,
                    split='final', role='streaming_probe', game=game, pair_index=pair_index, pair_count=len(pairs),
                    progress_callback=lambda **p: publish(**p))
                results.extend(dict(row, game=game) for row in measured['results'])
                policy['per_pair'].append({k: v for k, v in measured.items() if k != 'results'})
                policy['summary'] = {k: v for k, v in evaluation_summary(results, time.monotonic() - began).items() if k != 'results'}
                publish(force=True)
            policy['complete'] = True
        if fingerprint(agent.engine.model) != before_world or fingerprint(agent.heads) != before_evaluation:
            raise RuntimeError('Policy evaluation changed weights')
        report['evaluation_weights_unchanged'] = True
        report['status'] = 'complete'
        publish(force=True, phase='complete')
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        publish(force=True)
        raise


if __name__ == '__main__':
    main()
