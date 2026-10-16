"""Compare success/duration margins on paired development games with fixed weights.

This command never trains or writes checkpoints. Use a new development cohort
to confirm a selected candidate; its results are not held-out evaluation.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import time

import torch

from . import agent_checkpoint
from .behavior_experiment import episode_seeds
from .environment import Factory, SYNTHETIC_LEVELS
from .path_planning import PathConfig
from .planning import fingerprint
from .subgoal_training import play


def summarize(results, baseline):
    """Keep failures and shared wins visible when comparing action efficiency."""
    reference = {r['seed']: r for r in baseline}
    if len(reference) != len(baseline) or {r['seed'] for r in results} != set(reference) or len(results) != len(baseline):
        raise ValueError('Comparison requires unique, identical episode seeds')
    wins = [r for r in results if r['success']]
    shared = [r for r in wins if reference[r['seed']]['success']]
    return dict(episodes=len(results), successes=len(wins),
        total_actions=sum(r['actions'] for r in results),
        successful_actions=sum(r['actions'] for r in wins),
        mean_successful_actions=sum(r['actions'] for r in wins) / len(wins) if wins else None,
        common_successes=len(shared), common_actions=sum(r['actions'] for r in shared),
        baseline_common_actions=sum(reference[r['seed']]['actions'] for r in shared),
        gained=[r['seed'] for r in wins if not reference[r['seed']]['success']],
        lost=[r['seed'] for r in results if not r['success'] and reference[r['seed']]['success']])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--game', default='sc01-v1', choices=sorted(SYNTHETIC_LEVELS))
    parser.add_argument('--levels', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--margins', nargs='+', type=float, default=[.02, .05, .10])
    parser.add_argument('--horizons', nargs='+', type=int,
                        help='Inference horizons; omitted uses the checkpoint horizon')
    parser.add_argument('--epsilons', nargs='+', type=float, default=[0.],
                        help='Random exploration rates; zero is the normal frozen planner evaluation')
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--dev-offset', type=int, default=710000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(argv)
    if not args.levels or any(n < 1 or n > SYNTHETIC_LEVELS[args.game] for n in args.levels):
        parser.error('Invalid synthetic levels')
    if (len(set(args.levels)) != len(args.levels) or len(set(args.margins)) != len(args.margins)
            or (args.horizons is not None and len(set(args.horizons)) != len(args.horizons))
            or len(set(args.epsilons)) != len(args.epsilons)):
        parser.error('Duplicate comparison settings')
    if any(not math.isfinite(e) or not 0 <= e <= 1 for e in args.epsilons):
        parser.error('Exploration rates must be finite and in 0..1')
    try:
        seeds = episode_seeds('dev', args.episodes, offset=args.dev_offset)
        for margin in args.margins:
            PathConfig(success_margin=margin)
        for horizon in args.horizons or []:
            PathConfig(horizon=horizon)
    except ValueError as error:
        parser.error(str(error))
    # Pin the input before invoking this command; never follow a live latest.pt.
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    agent = agent_checkpoint.load(args.source, device=args.device, compile_model=False)
    original = agent.config.path
    horizons = args.horizons or [original.horizon]
    modules = (agent.engine.model, agent.heads, agent.path_evaluator)
    before = [fingerprint(module) for module in modules]
    factory = Factory(args.game, Path(__file__).resolve().parents[1] / 'games')
    report = dict(source_sha256=source_sha, path=asdict(original), game=args.game,
                  levels=args.levels, seeds=seeds, margins=args.margins, horizons=horizons,
                  epsilons=args.epsilons,
                  device=args.device, evaluations=[])
    began = time.monotonic()
    for level in args.levels:
        baseline = None
        for horizon, margin, epsilon in product(horizons, args.margins, args.epsilons):
            config = replace(original, success_margin=margin, horizon=horizon)
            agent.config = replace(agent.config, path=config)
            agent.planner.config = config
            results = []
            for seed in seeds:
                result = play(agent, factory, level, seed, agent.config.max_actions, epsilon=epsilon)
                results.append(dict(seed=seed, success=result['success'], actions=result['actions'],
                                    decision_ms=result['decision_ms']))
            if before != [fingerprint(module) for module in modules]:
                raise RuntimeError('Frozen evaluation changed weights')
            if baseline is None:
                baseline = results
            row = dict(level=level, horizon=horizon, margin=margin, epsilon=epsilon,
                       **summarize(results, baseline),
                       results=results, elapsed_seconds=time.monotonic()-began)
            report['evaluations'].append(row)
            (args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
