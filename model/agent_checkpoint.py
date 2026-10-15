"""Portable whole-agent snapshots and explicit conversion of legacy checkpoints.

Only the active inference modules/configuration and optional live carry are
stored. Training optimizers and replay remain trainer-owned artifacts.
"""
import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from . import checkpoint
from .agent import Agent, AgentConfig
from .config import NetworkConfig, TrainConfig
from .data import Action, Observation
from .goal_planning import Goal, GoalConfig
from .learning import Engine, State
from .networks import WorldModel
from .planning import GoalValue, PlanConfig, fingerprint
from .path_planning import PathConfig, PathValue


FORMAT = 'arc3-agent'
VERSION = 1


def _weights(module):
    return {key: tensor.detach().cpu().clone() for key, tensor in module.state_dict().items()}


def _action(action):
    return None if action is None else {'id': action.id, 'data': dict(action.data)}


def _observation(observation):
    if observation is None:
        return None
    return {'frames': torch.from_numpy(np.array(observation.frames, copy=True)),
            'available_actions': observation.available_actions, 'state': observation.state,
            'levels_completed': observation.levels_completed, 'win_levels': observation.win_levels,
            'action_input': _action(observation.action_input)}


def _read_observation(value):
    if value is None:
        return None
    action = value['action_input']
    frames = value['frames'].cpu().numpy().copy()
    frames.flags.writeable = False
    return Observation(frames, tuple(value['available_actions']), value['state'],
                       value['levels_completed'], value['win_levels'],
                       Action(**action) if action is not None else None)


def _runtime(agent):
    planner = agent.planner
    goal = getattr(planner, 'goal', None)
    visual_goal = goal if isinstance(goal, dict) else None
    if visual_goal is not None:
        goal = None
    return {'rng': agent.rng.getstate(), 'actions_taken': agent.actions_taken,
            'pending_action': _action(agent.pending_action), 'observation': _observation(agent.observation),
            'state': None if agent.state is None else {
                'z': None if agent.state.z is None else agent.state.z.detach().cpu(),
                'h': agent.state.h.detach().cpu(), 'geometry': agent.state.geometry},
            'planner': {
                'visual_goal': None if visual_goal is None else {
                    'frame': torch.from_numpy(visual_goal['frame'].copy()),
                    'mask': torch.from_numpy(visual_goal['pixel_mask'].copy())},
                'position': getattr(planner, 'position', 0),
                'goal': None if goal is None else {
                    'z': goal.z.detach().cpu(), 'geometry': goal.geometry,
                    'proposal_score': goal.proposal_score, 'created_position': goal.created_position},
                'last_plan': [_action(a) for a in getattr(planner, 'last_plan', [])],
                'last_endpoint': (planner.last_endpoint.detach().cpu()
                                  if getattr(planner, 'last_endpoint', None) is not None else None),
                'last_states': [z.detach().cpu() for z in getattr(planner, 'last_states', [])]}}


def save(path, agent, *, include_runtime=False, training_progress=None):
    payload = {'format': FORMAT, 'version': VERSION, 'agent': asdict(agent.config),
               'seed': agent.seed, 'world': {
                   'network': asdict(agent.engine.model.config), 'training': asdict(agent.engine.config),
                   'weights': _weights(agent.engine.model)},
               'outcome_value': {'width': agent.heads.spatial[0].out_channels, 'weights': _weights(agent.heads)},
               'components': {}, 'runtime': _runtime(agent) if include_runtime else None,
               'training_progress': dict(training_progress or {})}
    for name in ('proposal_policy', 'path_evaluator'):
        module = getattr(agent, name)
        if module is not None:
            if not hasattr(module, 'config'):
                raise ValueError(f'{name} requires an explicit serializable config')
            payload['components'][name] = {'config': dict(module.config), 'weights': _weights(module)}
    checkpoint.write(path, payload)


def load(path, *, device='cpu', compile_model=None, seed=None, max_actions=None,
         resume_runtime=False):
    payload = checkpoint.load(path)
    if payload.get('format') != FORMAT or payload.get('version') != VERSION:
        raise ValueError('Expected an arc3-agent v1 checkpoint; convert legacy artifacts explicitly')
    values = dict(payload['agent'])
    values['planner'] = PlanConfig(**values['planner'])
    values['goal'] = GoalConfig(**values['goal'])
    values['path'] = PathConfig(**values.get('path', {}))
    config = AgentConfig(**values)
    if max_actions is not None:
        config = replace(config, max_actions=max_actions)
    if resume_runtime and (seed is not None or max_actions is not None):
        raise ValueError('Cannot override seed or budget while resuming live state')
    world = payload['world']
    net = WorldModel(NetworkConfig(**world['network'])).to(device)
    net.load_state_dict(world['weights'], strict=True)
    if compile_model is None:
        compile_model = torch.device(device).type == 'cuda'
    engine = Engine(net, TrainConfig(**world['training']), compile_model=compile_model)
    heads = GoalValue(net.config.latent, net.config.memory,
                      width=payload['outcome_value']['width']).to(device)
    heads.load_state_dict(payload['outcome_value']['weights'], strict=True)
    components = {}
    if payload['components']:
        from .proposals import ProposalPolicy
        for name, saved in payload['components'].items():
            if name not in {'proposal_policy', 'path_evaluator'}:
                raise ValueError(f'Unknown active component: {name}')
            cls = ProposalPolicy if name == 'proposal_policy' else PathValue
            module = cls(**saved['config']).to(device)
            module.load_state_dict(saved['weights'], strict=True)
            components[name] = module
    agent = Agent(engine, heads, config, seed=payload['seed'] if seed is None else seed,
                  compile_heads=compile_model, **components)
    runtime = payload['runtime']
    if resume_runtime:
        if runtime is None:
            raise ValueError('Checkpoint has no live state to resume')
        agent.rng.setstate(runtime['rng'])
        agent.actions_taken = runtime['actions_taken']
        action = runtime['pending_action']
        agent.pending_action = Action(**action) if action is not None else None
        agent.observation = _read_observation(runtime['observation'])
        state = runtime['state']
        if state is not None:
            agent.state = State(None if state['z'] is None else state['z'].to(device),
                                state['h'].to(device), state['geometry'])
        saved = runtime['planner']
        planner = agent.planner
        if config.mode == 'path_mpc':
            visual = saved.get('visual_goal')
            if visual is not None:
                planner.set_goal(visual['frame'].numpy(), visual['mask'].numpy())
        elif hasattr(planner, 'goal'):
            goal = saved['goal']
            planner.goal = None if goal is None else Goal(**(goal | {'z': goal['z'].to(device)}))
            planner.position = saved['position']
        planner.last_plan = [Action(**a) for a in saved['last_plan']]
        planner.last_endpoint = (None if saved['last_endpoint'] is None else saved['last_endpoint'].to(device))
        planner.last_states = [z.to(device) for z in saved['last_states']]
    return agent


def convert_legacy(world_path, heads_path, output, *, mode='value_mpc', max_actions=128, seed=0):
    """One deliberate conversion; fingerprints prevent combining unrelated weights."""
    world, saved = checkpoint.load(world_path), checkpoint.load(heads_path)
    model = WorldModel(NetworkConfig(**world['network']))
    model.load_state_dict(world['weights'], strict=True)
    if fingerprint(model) != saved['source_fingerprint']:
        raise ValueError('Behavior checkpoint belongs to different world-model weights')
    width = saved['weights']['spatial.0.weight'].shape[0]
    heads = GoalValue(model.config.latent, model.config.memory, width=width)
    # The head owns the only permitted legacy migration: zero-initialized budget
    # corrections and inactive availability. No arbitrary missing keys allowed.
    if hasattr(heads, 'load_compatible'):
        heads.load_compatible(saved['weights'])
    else:
        heads.load_state_dict(saved['weights'], strict=True)
    engine = Engine(model, TrainConfig(**world['training']))
    config = AgentConfig(mode=mode, max_actions=max_actions, planner=PlanConfig(**saved['planner']))
    agent = Agent(engine, heads, config, seed=seed)
    save(output, agent)
    return agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', required=True, type=Path)
    parser.add_argument('--heads', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', default='value_mpc', choices=['q', 'mpc', 'value_mpc', 'goal_mpc', 'goal_value_mpc'])
    parser.add_argument('--max-actions', default=128, type=int)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output checkpoint path')
    convert_legacy(args.world, args.heads, args.output, mode=args.mode,
                   max_actions=args.max_actions, seed=args.seed)
    print(args.output)


if __name__ == '__main__':
    main()
