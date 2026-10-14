"""Frozen synthetic-game diagnostics; real counterfactuals, never training data.

Recreate each branch through public reset/step with the same seed and prefix.
No private environment state or game rules are inspected. These are small
development probes, not final confirmation or evidence of SDK generalization.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from . import agent_checkpoint
from .behavior_experiment import episode_seeds, evaluate, evaluation_summary
from .campaign import write_report
from .data import random_action
from .environment import Factory, SYNTHETIC_LEVELS
from .metrics import delta_counts
from .planning import candidates, fingerprint, outcome


def same_observation(a, b):
    return (np.array_equal(a.frames, b.frames) and a.state == b.state
            and a.levels_completed == b.levels_completed and a.win_levels == b.win_levels
            and a.available_actions == b.available_actions and a.action_input == b.action_input)


def replay_prefix(factory, level, seed, observations, actions):
    """Fail closed if replay does not reproduce every publicly observed prefix."""
    env = factory.make(seed, level)
    if not same_observation(env.observation, observations[0]):
        raise ValueError('Counterfactual reset is not reproducible')
    for action, expected in zip(actions, observations[1:], strict=True):
        actual = env.step(action)
        if not same_observation(actual, expected):
            raise ValueError('Counterfactual prefix is not reproducible')
    return env


def visual_counts(before, actual, predicted):
    wrong, union, changed = delta_counts(before, actual, predicted)
    real, proposed = actual != before, predicted != before
    return {'wrong': wrong, 'union': union, 'changed': changed,
            'invented': int((~real & proposed).sum()), 'missed': int((real & ~proposed).sum()),
            'wrong_color': int((real & proposed & (actual != predicted)).sum()),
            'correct': int((real & (actual == predicted)).sum()), 'samples': 1}


def sum_counts(rows):
    keys = ('wrong', 'union', 'changed', 'invented', 'missed', 'wrong_color', 'correct', 'samples')
    total = {key: sum(row.get(key, 0) for row in rows) for key in keys}
    total['delta_error'] = total['wrong'] / max(1, total['union']) if total['samples'] else None
    return total


@torch.no_grad()
def action_probe(engine, factory, level, seeds, *, anchors=4, clicks=16, progress=None):
    """Score correct and wrong-action predictions against separately executed branches.

    Wrong-action comparison uses a fixed cyclic permutation, without inspecting
    outcomes. Distinguishable-only results exclude equal real successor boards:
    blocked moves and equivalent clicks cannot establish action sensitivity.
    """
    matched, identity, wrong_action, distinguishable, separations = [], [], [], [], []
    skipped, anchor_count, environment_actions = 0, 0, 0
    for seed_index, seed in enumerate(seeds):
        env, rng = factory.make(seed, level), random.Random(seed)
        observation = env.observation
        state = engine.initial(observation)
        history, prefix = [observation], []
        for anchor in range(anchors):
            if observation.terminal or state.z is None or observation.final is None:
                break
            choices = candidates(observation, random.Random(seed + anchor * 100003), clicks)
            predictions, successors = [], []
            for action_index, action in enumerate(choices):
                branch = replay_prefix(factory, level, seed, history, prefix)
                environment_actions += len(prefix)
                prediction = engine.prediction(state, action, observation.final)
                after = branch.step(action)
                environment_actions += 1
                # Re-run each branch to also check stochastic successors.
                repeated = replay_prefix(factory, level, seed, history, prefix)
                repeated_after = repeated.step(action)
                environment_actions += len(prefix) + 1
                if not same_observation(after, repeated_after):
                    raise ValueError('Counterfactual successor is not reproducible')
                valid = (prediction is not None and after.final is not None
                         and not after.starts_new_level(observation)
                         and after.final.shape == observation.final.shape)
                predictions.append(prediction[0] if valid else None)
                successors.append(after.final if valid else None)
                if progress:
                    progress(seed_index=seed_index, anchor=anchor, action_index=action_index,
                             candidates=len(choices), environment_actions=environment_actions)
            anchor_count += 1
            for i, actual in enumerate(successors):
                if actual is None:
                    skipped += 1
                    continue
                before = observation.final
                matched.append(visual_counts(before, actual, predictions[i]))
                identity.append(visual_counts(before, actual, before))
                j = (i + 1) % len(choices)
                if len(choices) > 1 and successors[j] is not None:
                    swapped = visual_counts(before, actual, predictions[j])
                    wrong_action.append(swapped)
                    if not np.array_equal(actual, successors[j]):
                        distinguishable.append((matched[-1], swapped))
                        separations.append({'real_pixels': int((actual != successors[j]).sum()),
                                            'predicted_pixels': int((predictions[i] != predictions[j]).sum())})
            # Identical random prefix across checkpoints; never select by model score.
            action = random_action(observation, rng)
            env = replay_prefix(factory, level, seed, history, prefix)
            environment_actions += len(prefix)
            after = env.step(action)
            environment_actions += 1
            if outcome(observation, after):
                break
            state = engine.observe(state, action, after)
            history.append(after)
            prefix.append(action)
            observation = after
    return {'anchors': anchor_count, 'skipped': skipped, 'environment_actions': environment_actions,
            'predictor': sum_counts(matched), 'no_change': sum_counts(identity),
            'wrong_action': sum_counts(wrong_action),
            'distinct_correct': sum_counts([a for a, _ in distinguishable]),
            'distinct_wrong': sum_counts([b for _, b in distinguishable]),
            'distinct_pairs': len(separations),
            'real_separation_pixels': sum(r['real_pixels'] for r in separations),
            'predicted_separation_pixels': sum(r['predicted_pixels'] for r in separations)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agents', nargs='+', type=Path, required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--probe-seeds', type=int, default=2)
    parser.add_argument('--anchors', type=int, default=4)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    if len(args.agents) != len(args.labels) or len(set(args.labels)) != len(args.labels):
        parser.error('Supply one distinct label per agent')
    if min(args.episodes, args.steps, args.probe_seeds, args.anchors) < 1:
        parser.error('Budgets must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'report.json'
    if path.exists():
        parser.error('Use a fresh diagnostic output directory')
    report = {'status': 'running', 'scope': 'synthetic_development_probe', 'results': [],
              'config': {'labels': args.labels, 'episodes_per_pair': args.episodes,
                         'max_actions': args.steps, 'probe_seeds': args.probe_seeds, 'anchors': args.anchors},
              'progress': {}}
    last_write = 0.

    def publish(force=False, **values):
        nonlocal last_write
        report['progress'].update(values)
        if not force and time.monotonic() - last_write < 2:
            return
        last_write = time.monotonic()
        write_report(path, report)
        print(json.dumps({'event': 'diagnostic_progress', **report['progress']}), flush=True)

    pairs = [(game, level) for game, count in SYNTHETIC_LEVELS.items() for level in range(1, count + 1)]
    factories = {g: Factory(g, Path(__file__).resolve().parents[1] / 'games') for g in SYNTHETIC_LEVELS}
    try:
        for label, agent_path in zip(args.labels, args.agents, strict=True):
            report['progress'] = {'checkpoint': label, 'phase': 'loading'}
            publish(force=True)
            torch.compiler.reset()
            agent = agent_checkpoint.load(agent_path, device=args.device, max_actions=args.steps)
            before = (fingerprint(agent.engine.model), fingerprint(agent.heads))
            result = {'checkpoint': label, 'world_fingerprint': before[0], 'probes': [], 'policies': {}}
            report['results'].append(result)
            for game, level in pairs:
                report['progress'] = {'checkpoint': label, 'phase': 'actions', 'game': game, 'level': level}
                publish(force=True)
                probe = action_probe(agent.engine, factories[game], level,
                    [7000001 + i * 2 for i in range(args.probe_seeds)], anchors=args.anchors,
                    clicks=agent.planner.config.clicks, progress=lambda **p: publish(**p))
                result['probes'].append(dict(game=game, level=level, **probe))
                publish(force=True)
            for mode in ('random', 'q', 'value_mpc'):
                episodes = []
                began = time.monotonic()
                policy = {'per_pair': []}
                result['policies'][mode] = policy
                for pair_index, (game, level) in enumerate(pairs, 1):
                    report['progress'] = {'checkpoint': label, 'phase': 'behavior', 'policy': mode,
                                          'game': game, 'level': level, 'pair_index': pair_index,
                                          'pair_count': len(pairs)}
                    publish(force=True)
                    measured = evaluate(agent.engine, agent.planner, factories[game], level,
                        episode_seeds('dev', args.episodes, offset=400000), args.steps, mode,
                        split='dev', role='diagnostic', game=game,
                        pair_index=pair_index, pair_count=len(pairs),
                        progress_callback=lambda **p: publish(**p))
                    episodes.extend(dict(row, game=game) for row in measured['results'])
                    policy['per_pair'].append({k: v for k, v in measured.items() if k != 'results'})
                    policy['summary'] = {k: v for k, v in evaluation_summary(episodes, time.monotonic() - began).items() if k != 'results'}
                    publish(force=True)
                policy['complete'] = True
            after = (fingerprint(agent.engine.model), fingerprint(agent.heads))
            if before != after:
                raise RuntimeError('Frozen diagnostic changed world or behavior weights')
            result['weights_unchanged'] = True
            del agent
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
