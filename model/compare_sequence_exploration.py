"""Compare real CPU collection; fit only an auxiliary head, never world weights."""
import argparse
from dataclasses import asdict, replace
import json
import pickle
from pathlib import Path
import time

import torch

from .config import SequenceConfig
from .game_sequences import collect_session
from .sequence_exploration import (SequenceExplorationConfig, ChangeExplorer,
                                   training_examples, interaction_counts, sum_counts)
from . import checkpoint
from .sequence_training import close_factories, factories_for, write_report
from .training_config import load_plan


def run(args):
    plan = load_plan(args.training_config)
    config = SequenceConfig(**plan.payload['sequence'])
    if args.steps is not None:
        if args.steps < 1:
            raise ValueError('steps must be positive')
        config = replace(config, session_steps=args.steps, block_steps=min(config.block_steps, args.steps))
    policies = [SequenceExplorationConfig(policy=p) for p in ('uniform', 'mixed', 'learned')]
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds) or
            any(seed < 0 for seed in args.seeds) or args.head_updates < 1):
        raise ValueError('Use distinct nonnegative seeds and positive updates')
    explorer = ChangeExplorer(policies[-1])
    bootstrap = {}
    if args.head_source:
        saved = checkpoint.load(args.head_source)
        if saved.get('kind') != 'sequence_change_head':
            raise ValueError('Expected an auxiliary change-head checkpoint')
        explorer.restore(saved['exploration'])
    elif args.bootstrap_corpus:
        # Only the explicitly supplied, trusted local training collection.
        # Probe/evaluation sessions never train this auxiliary head.
        with args.bootstrap_corpus.open('rb') as handle:
            corpus = pickle.load(handle)
        rows = list(training_examples(corpus['training']))
        order = explorer.rng.permutation(len(rows))
        explorer.replay.extend(rows[i] for i in order)
        explorer.examples = len(rows)
        started = time.perf_counter()
        bootstrap = dict(explorer.fit_replay(args.head_updates), seconds=time.perf_counter()-started)
        del rows, corpus
    else:
        raise ValueError('Supply a trusted real training corpus or auxiliary head')
    if args.output.exists():
        raise ValueError('Comparison output already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = {s['game']: s['source'] for s in plan.stages}
    report = dict(kind='sequence_exploration_comparison', plan_identity=plan.identity,
        sequence=asdict(config), seeds=args.seeds, sources=sources, bootstrap=bootstrap, trials=[])
    trials = [dict(policy=asdict(policy), sessions=[]) for policy in policies]
    factories = {}
    try:
        factories = factories_for(plan)
        if args.bootstrap_mixed:
            # Deliberately separate real training seeds from the comparison.
            examples = list(explorer.replay)
            existing_examples = len(examples)
            for i, (game, (factory, levels)) in enumerate(factories.items()):
                warmup = collect_session(factory, levels, 4_900_000 + 2*i, config,
                                         exploration=policies[1])
                examples.extend(training_examples([warmup]))
            explorer.rng.shuffle(examples)
            explorer.replay.extend(examples)
            explorer.examples += len(examples) - existing_examples
            bootstrap['mixed_training'] = explorer.fit_replay(args.head_updates)
        if args.head_output:
            if args.head_output.exists():
                raise ValueError('Head output already exists')
            args.head_output.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write(args.head_output, dict(kind='sequence_change_head', exploration=explorer.snapshot()))
        for i, (game, (factory, levels)) in enumerate(factories.items()):
            for j, seed in enumerate(args.seeds):
                session_seed = 4_000_000 + 2 * (seed * 10_000 + i)
                # Rotate policy order to reduce systematic cache/order bias.
                order = list(range(len(policies)))
                offset = (i + j) % len(order)
                for k in order[offset:] + order[:offset]:
                    start = time.perf_counter()
                    session = collect_session(factory, levels, session_seed, config,
                                              exploration=policies[k], explorer=explorer if policies[k].policy == 'learned' else None)
                    seconds = time.perf_counter() - start
                    trials[k]['sessions'].append(dict(game=game, seed=session_seed,
                        source=sources[game], seconds=seconds,
                        environment_actions=session.environment_actions, end=session.end_reason,
                        **interaction_counts(session)))
                    del session
            print(json.dumps(dict(event='collected', game=game, games=i+1, total=len(factories))), flush=True)
        for trial in trials:
            rows = trial['sessions']
            trial['games'] = {game: sum_counts(r for r in rows if r['game'] == game) for game in sources}
            trial['groups'] = {}
            for source in ('all', 'synthetic', 'sdk'):
                group = [r for r in rows if source == 'all' or r['source'] == source]
                seconds = sum(r['seconds'] for r in group)
                actions = sum(r['environment_actions'] for r in group)
                trial['groups'][source] = dict(**sum_counts(group), games=len({r['game'] for r in group}),
                    sessions=len(group), environment_actions=actions, seconds=seconds,
                    actions_per_second=actions / seconds if seconds else None)
        report['trials'] = trials
        write_report(args.output, report)
    finally:
        close_factories(factories)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    parser.add_argument('--steps', type=int)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bootstrap-corpus', type=Path, help='Trusted local corpus; training sessions only')
    source.add_argument('--head-source', type=Path)
    parser.add_argument('--head-output', type=Path)
    parser.add_argument('--head-updates', type=int, default=512)
    parser.add_argument('--bootstrap-mixed', action='store_true', help='Collect separate real visual-guided warmup sessions')
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args)


if __name__ == '__main__':
    main()
