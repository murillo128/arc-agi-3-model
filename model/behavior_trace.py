"""Export one frozen behavior episode for the viewer, without training/replay."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
import torch

from arc3.traces import write_attempt
from . import checkpoint
from .behavior_experiment import load_heads
from .config import NetworkConfig, TrainConfig
from .data import random_action
from .environment import Factory, SYNTHETIC_LEVELS
from .learning import Engine
from .networks import WorldModel
from .planning import GoalValue, Planner, PlanConfig, fingerprint, outcome
from .goal_planning import GoalConfig, GoalPlanner


def tensor(array):
    data = np.asarray(array, dtype=np.uint8)
    return {'dtype': 'u8', 'shape': list(data.shape), 'data': data.tobytes()}


def observation(obs):
    return {'frames': tensor(obs.frames), 'state': obs.state,
            'levels_completed': obs.levels_completed, 'win_levels': obs.win_levels,
            'available_actions': list(obs.available_actions)}


@torch.no_grad()
def record_episode(engine, planner, factory, level, seed, steps, mode, source):
    if mode == 'goal_mpc':
        planner.reset()
    env, rng = factory.make(seed, level), random.Random(seed)
    obs, started = env.observation, time.monotonic()
    state = engine.initial(obs)
    identity = uuid4().hex
    attempt = {'metadata': {'game_id': factory.game_id, 'run_id': identity, 'session_id': identity,
                           'attempt_index': 0, 'seed': seed, 'sdk_version': version('arc-agi'),
                           'source_split': 'evaluation', 'started_at': datetime.now(timezone.utc).isoformat(),
                           'model_id': 'frozen-world-goal-planner', 'checkpoint_id': source,
                           'config': {'synthetic_level': level, 'policy': mode,
                                      'goal': ('Imagined latent goal hypothesis; distance-only reaching'
                                               if mode == 'goal_mpc' else
                                               'SDK level progress/WIN; learned proxy, not a goal image'),
                                      'future_legality': 'assumed equal to current legality'}},
               'initial_observation': observation(obs), 'steps': []}
    event = 0
    for index in range(steps):
        if obs.terminal:
            break
        diagnostics = {}
        start = time.monotonic()
        action = (planner.choose(state, obs, rng, mode, diagnostics=diagnostics)
                  if state.z is not None else random_action(obs, rng))
        decision_ms = 1000 * (time.monotonic() - start)
        # This frozen visual prediction is inspection only, after action choice.
        prediction = engine.prediction(state, action, obs.final) if obs.final is not None else None
        if prediction is not None:
            diagnostics['prediction'] = {
                'target': 'post_action_last_frame', 'frames': tensor(prediction[0]),
                'visual_delta': {'reference_position': index, 'reference_frame': tensor(obs.final),
                                 'delta': tensor(prediction[2])}}
        after = env.step(action)
        if mode == 'goal_mpc':
            planner.after_real_action(action, obs, after)
        attempt['steps'].append({'index': index, 'action': {'id': action.id, 'data': dict(action.data)},
                                 'observation': observation(after), **diagnostics,
                                 'policy_mode': 'learned' if 'planning' in diagnostics else 'random',
                                 'timing': {'decision_ms': decision_ms}})
        event = outcome(obs, after)
        state = engine.observe(state, action, after) if not event else state
        obs = after
        if event:
            break
    reason = ('win' if obs.state == 'WIN' else 'game_over' if obs.state == 'GAME_OVER'
              else 'training_level_boundary' if event == 1 else 'action_budget')
    attempt['termination'] = {'reason': reason, 'final_state': obs.state}
    attempt['summary'] = {'real_actions': len(attempt['steps']), 'levels_completed': obs.levels_completed,
                          'win_levels': obs.win_levels, 'wall_seconds': time.monotonic() - started}
    return attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--heads', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--game', default='sc01-v1')
    parser.add_argument('--level', default=2, type=int)
    parser.add_argument('--steps', default=128, type=int)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--goal-planning', action='store_true')
    parser.add_argument('--goal-horizon', default=8, type=int)
    parser.add_argument('--goal-beam', default=4, type=int)
    parser.add_argument('--goal-lifetime', default=8, type=int)
    parser.add_argument('--plan-horizon', default=4, type=int)
    parser.add_argument('--plan-beam', default=4, type=int)
    args = parser.parse_args()
    try:
        goal_config = GoalConfig(proposal_horizon=args.goal_horizon, proposal_beam=args.goal_beam,
                                 lifetime=args.goal_lifetime, horizon=args.plan_horizon, beam=args.plan_beam)
    except ValueError as exc:
        parser.error(str(exc))
    if args.game not in SYNTHETIC_LEVELS or args.level not in range(1, SYNTHETIC_LEVELS.get(args.game, 0) + 1):
        parser.error('Only bundled synthetic games and valid levels are admitted')
    if args.steps < 1 or args.steps > 1024:
        parser.error('Trace budget must be between 1 and 1024 actions')
    if args.output.exists() or args.output.suffix != '.arc3':
        parser.error('Choose a new .arc3 output file')
    torch.set_num_threads(4)
    payload = checkpoint.load(args.checkpoint)
    net = WorldModel(NetworkConfig(**payload['network'])).to(args.device)
    net.load_state_dict(payload['weights'])
    engine = Engine(net, TrainConfig(**payload['training']), compile_model=args.device == 'cuda')
    del payload
    source = fingerprint(net)
    heads = GoalValue(net.config.latent, net.config.memory).to(args.device)
    saved = load_heads(args.heads, heads, source)
    planner = (GoalPlanner(engine, heads, goal_config, compile_heads=args.device == 'cuda')
               if args.goal_planning else
               Planner(engine, heads, PlanConfig(**saved['planner']), compile_heads=args.device == 'cuda'))
    heads_before = fingerprint(heads)
    factory = Factory(args.game, Path(__file__).resolve().parents[1] / 'games')
    attempt = record_episode(engine, planner, factory, args.level, args.seed, args.steps,
                             'goal_mpc' if args.goal_planning else saved['report']['selected']['policy'], source)
    if fingerprint(net) != source or fingerprint(heads) != heads_before:
        raise RuntimeError('Read-only trace changed weights')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_attempt(args.output, attempt, format_minor=1)
    print(f"{args.output}: {len(attempt['steps'])} actions, {attempt['termination']['reason']}; "
          f"{attempt['summary']['wall_seconds']:.3f}s; world={source}; heads={heads_before}; weights unchanged")


if __name__ == '__main__':
    main()
