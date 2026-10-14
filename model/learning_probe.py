"""Paired fixed-data learning diagnostic using the production Learner.

Real synthetic training prefixes only. Frozen representation, identical initial
weights/data/update budgets. Development episodes are scored without learning;
neither the probe nor its dashboard promotes weights into the live agent.
"""
import argparse
from dataclasses import replace
import gc
from pathlib import Path
import random
import time

import torch

from . import agent_checkpoint
from .behavior_experiment import episode_seeds
from .campaign import calibration_coverage, emit, write_report
from .data import Episode, random_action
from .environment import Factory, SYNTHETIC_LEVELS
from .learning import Learner, evaluate_rollouts
from .planning import fingerprint, outcome
from .runner import merge_rollout_reports


def collect(factory, level, seeds, steps):
    episodes, raw = [], []
    for seed in seeds:
        env, rng = factory.make(seed, level), random.Random(seed)
        observation = env.observation
        episode, actual = Episode([observation]), Episode([observation])
        reason = 'budget'
        for _ in range(steps):
            if observation.terminal:
                break
            action = random_action(observation, rng)
            after = env.step(action)
            actual.append(action, after)
            if not after.starts_new_level(observation):
                episode.append(action, after)
            event = outcome(observation, after)
            if event:
                reason = 'progress' if event == 1 else 'game_over'
                break
            observation = after
        if episode.actions:
            episodes.append(episode)
        raw.append((actual, reason))
    return episodes, raw


def score(engine, data, publish):
    rows = []
    for (game, level), episodes in data.items():
        measured = []
        for index, episode in enumerate(episodes, 1):
            measured.append(evaluate_rollouts(engine, episode, horizons=(1, 2)))
            publish(game=game, level=level, episode=index, episodes=len(episodes))
        rows.append({'game': game, 'level': level, 'rollouts': merge_rollout_reports(measured, ('1', '2'))})
    return {'per_pair': rows, 'rollouts': merge_rollout_reports([r['rollouts'] for r in rows], ('1', '2'))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--episodes', type=int, default=2, help='Fixed training and development episodes per pair')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--updates', type=int, default=120)
    parser.add_argument('--evaluate-every', type=int, default=20)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args(argv)
    if min(args.episodes, args.steps, args.updates, args.evaluate_every) < 1:
        parser.error('Budgets must be positive')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'report.json'
    if path.exists():
        parser.error('Use a fresh learning probe directory')
    report = {'status': 'running', 'scope': 'fixed_real_training_probe',
              'config': {'updates': args.updates, 'episodes_per_pair': args.episodes,
                         'steps_per_episode': args.steps, 'seed': args.seed,
                         'variants': ['latent_2step', 'visual_2step']},
              'progress': {}, 'results': []}
    last_write = 0.
    def publish(force=False, **values):
        nonlocal last_write
        report['progress'].update(values)
        if not force and time.monotonic() - last_write < 2:
            return
        last_write = time.monotonic()
        write_report(path, report)
        emit('learning_probe_progress', **report['progress'])
    try:
        publish(force=True, phase='collection')
        train, development, archive = {}, {}, {}
        pairs = [(g, l) for g, n in SYNTHETIC_LEVELS.items() for l in range(1, n + 1)]
        for index, (game, level) in enumerate(pairs):
            factory = Factory(game, Path(__file__).resolve().parents[1] / 'games')
            train[game, level], archive[game, level] = collect(factory, level,
                episode_seeds('train', args.episodes, offset=300000 + index * 100), args.steps)
            development[game, level], _ = collect(factory, level,
                episode_seeds('dev', args.episodes, offset=450000 + index * 100), args.steps)
        report['coverage'] = calibration_coverage(archive)
        samples = [ep for episodes in train.values() for ep in episodes]
        if not samples:
            raise ValueError('No usable real training prefixes')
        for variant, visual_weight in (('latent_2step', 0.), ('visual_2step', .5)):
            torch.compiler.reset()
            torch.manual_seed(args.seed)
            rng = random.Random(args.seed)
            report['progress'] = {'variant': variant, 'phase': 'loading', 'updates': 0,
                                  'total_updates': args.updates}
            publish(force=True)
            agent = agent_checkpoint.load(args.agent, device=args.device)
            engine = agent.engine
            engine.config = replace(engine.config, freeze_representation=True, auto_representation=False,
                representation_joint_steps=0, rollout_weight=.5, rollout_visual_weight=visual_weight,
                loss_steps=args.steps, cache_frozen_latents=True, burn_in_steps=0)
            engine.model.dynamics.requires_grad_(True)
            learner = Learner(engine)
            frozen = {name: fingerprint(getattr(engine.model, name)) for name in ('encoder', 'decoder')}
            result = {'variant': variant, 'initial_world': fingerprint(engine.model), 'points': []}
            report['results'].append(result)
            started = time.monotonic()
            for update in range(args.updates + 1):
                if update:
                    # Publish the phase before a potentially slow first compiled
                    # update; count only optimizer updates already completed.
                    publish(force=update == 1, phase='training', updates=update - 1,
                            seconds=time.monotonic() - started)
                    batch = rng.sample(samples, min(2, len(samples)))
                    loss = learner.update(batch)
                    publish(phase='training', updates=update, loss=loss, seconds=time.monotonic() - started)
                if update == args.updates:
                    # Recovery snapshot before the final evaluation. No optimizer
                    # or raw training episodes are exported to the agent.
                    agent_checkpoint.save(args.output / f'{variant}.pt', agent)
                    result['saved_before_evaluation'] = True
                if update == 0 or update % args.evaluate_every == 0 or update == args.updates:
                    point = {'updates': update, 'seconds': time.monotonic() - started}
                    for split, data in (('fit', train), ('development', development)):
                        publish(force=True, phase='evaluation', split=split, updates=update)
                        point[split] = score(engine, data, publish)
                    result['points'].append(point)
                    publish(force=True)
            if any(fingerprint(getattr(engine.model, name)) != expected for name, expected in frozen.items()):
                raise RuntimeError('Learning probe changed frozen representation weights')
            result['frozen_representation_verified'] = True
            result['dynamics_changed'] = fingerprint(engine.model) != result['initial_world']
            if not result['dynamics_changed']:
                raise RuntimeError('Learning probe did not update predictor weights')
            result['complete'] = True
            for episode in samples:
                episode.latent_cache.clear()
            del learner, engine, agent
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        report['status'] = 'complete'
        publish(force=True, phase='complete')
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        publish(force=True)
        raise


if __name__ == '__main__':
    main()
