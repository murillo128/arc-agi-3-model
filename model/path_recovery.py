"""Bounded synthetic planner recovery experiment; world weights stay frozen.

Uses real training episodes and separate development seeds. No checkpoint from
this command is a curriculum resume checkpoint: its experience is accounted for
separately in report.json. Large outputs belong in ignored model/runs paths.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from . import agent_checkpoint
from .behavior_experiment import episode_seeds
from .environment import Factory
from .path_planning import PathLearner, PathValue
from .planning import fingerprint
from .subgoal_training import play


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--game', default='sc01-v1')
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--episodes', type=int, default=512)
    parser.add_argument('--evaluate-every', type=int, default=64)
    parser.add_argument('--evaluation-episodes', type=int, default=8)
    parser.add_argument('--train-offset', type=int, default=900000)
    parser.add_argument('--dev-offset', type=int, default=700000)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--migrate-path-value', action='store_true')
    parser.add_argument('--fresh-path-value', action='store_true')
    parser.add_argument('--path-objective-version', type=int, choices=(3,4), default=4)
    parser.add_argument('--hindsight', action='store_true')
    parser.add_argument('--predicted-inputs', action='store_true')
    parser.add_argument('--policy', choices=('random', 'planner'), default='random')
    parser.add_argument('--epsilon', type=float, default=.25,
                        help='Random exploration fraction when policy=planner')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--replay-capacity', type=int, default=64)
    parser.add_argument('--replay-batch', type=int, default=8)
    args = parser.parse_args(argv)
    if args.episodes < 0 or min(args.evaluate_every, args.evaluation_episodes) < 1 or not 0 <= args.epsilon <= 1:
        parser.error('Invalid episode counts')
    if not 0 < args.learning_rate < 1 or not 1 <= args.replay_batch <= args.replay_capacity:
        parser.error('Invalid learning rate or replay bounds')
    training_seeds = episode_seeds('train', args.episodes, offset=args.train_offset) if args.episodes else []
    development_seeds = episode_seeds('dev', args.evaluation_episodes, offset=args.dev_offset)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    agent = agent_checkpoint.load(args.source, device=args.device, compile_model=False,
                                  migrate_path_value=args.migrate_path_value)
    if args.fresh_path_value:
        agent.path_evaluator = PathValue(agent.engine.model.config.latent,
            agent.engine.model.config.memory, objective_version=args.path_objective_version).to(args.device)
        agent.planner.evaluator = agent.planner.readout = agent.path_evaluator
    world = fingerprint(agent.engine.model)
    factory = Factory(args.game, Path(__file__).resolve().parents[1]/'games')
    learner = PathLearner(agent.planner, learning_rate=args.learning_rate,
        capacity=args.replay_capacity, batch_segments=args.replay_batch,
        seed=args.seed, hindsight=args.hindsight, predicted_inputs=args.predicted_inputs)
    report = dict(config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  world_fingerprint=world, training=[], evaluations=[])
    began = time.monotonic()

    def evaluate(index, random_policy=False):
        weights = fingerprint(agent.path_evaluator)
        episodes = []
        for seed in development_seeds:
            result = play(agent, factory, args.level, seed, agent.config.max_actions,
                          random_policy=random_policy)
            episodes.append(dict(seed=seed, success=result['success'], actions=result['actions']))
        if fingerprint(agent.path_evaluator) != weights or fingerprint(agent.engine.model) != world:
            raise RuntimeError('Frozen evaluation changed weights')
        row = dict(episodes=index, policy='random' if random_policy else 'path_mpc',
                   successes=sum(e['success'] for e in episodes), results=episodes,
                   elapsed=time.monotonic()-began)
        report['evaluations'].append(row)
        print(json.dumps(row), flush=True)
        (args.output/'report.json').write_text(json.dumps(report))

    evaluate(0)
    evaluate(0, random_policy=True)
    for index, seed in enumerate(training_seeds, 1):
        result = play(agent, factory, args.level, seed, agent.config.max_actions,
                      learner=learner, random_policy=args.policy == 'random', epsilon=args.epsilon)
        report['training'].append(dict(seed=seed, success=result['success'], actions=result['actions']))
        if index % args.evaluate_every == 0 or index == args.episodes:
            report['path_updates'] = learner.updates
            report['training_actions'] = sum(e['actions'] for e in report['training'])
            report['replay'] = learner.status()
            evaluate(index)
            agent_checkpoint.save(args.output/'latest.pt', agent)
    if fingerprint(agent.engine.model) != world:
        raise RuntimeError('Path training changed the world model')


if __name__ == '__main__':
    main()
