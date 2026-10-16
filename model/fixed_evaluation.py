"""Persist real evaluation trajectories separately from all training replay.

Only observations, executed actions and outcomes survive; learned representations
must be recomputed with the current weights. Cache files are local run artifacts.
"""
from dataclasses import asdict

import numpy as np
import torch

from . import checkpoint
from .data import Action, Episode, Observation


def pack_episode(episode):
    return dict(actions=[asdict(a) for a in episode.actions], observations=[dict(
        frames=torch.from_numpy(np.array(o.frames, copy=True)),
        available_actions=list(o.available_actions), state=o.state,
        levels_completed=o.levels_completed, win_levels=o.win_levels,
        action_input=asdict(o.action_input) if o.action_input is not None else None)
        for o in episode.observations])


def unpack_episode(payload):
    observations = []
    for row in payload['observations']:
        frames = row['frames'].numpy().copy()
        frames.flags.writeable = False
        observations.append(Observation(frames, tuple(row['available_actions']), row['state'],
            row['levels_completed'], row['win_levels'],
            Action(**row['action_input']) if row['action_input'] is not None else None))
    actions = [Action(**a) for a in payload['actions']]
    if len(observations) != len(actions) + 1:
        raise ValueError('Invalid fixed evaluation trajectory length')
    return Episode(observations, actions)


def random_cohort(path, metadata, collect, training_action):
    """Collect once, atomically; resume reuses the exact cohort or fails closed."""
    if path.exists():
        payload = checkpoint.load(path)
        if payload.get('version') != 1 or payload.get('metadata') != metadata:
            raise ValueError(f'Fixed evaluation cache does not match this run: {path}')
    else:
        entries = []
        for seed in metadata['seeds']:
            episode, result = collect(seed)
            entries.append(dict(seed=seed, episode=pack_episode(episode),
                result={k: result[k] for k in ('success', 'reason', 'actions', 'segments', 'decision_ms')}))
        payload = dict(version=1, metadata=metadata, collected_at_action=training_action, entries=entries)
        checkpoint.write(path, payload)
    if [r['seed'] for r in payload['entries']] != metadata['seeds']:
        raise ValueError(f'Incomplete fixed evaluation cohort: {path}')
    return payload, [(unpack_episode(r['episode']), r['result']) for r in payload['entries']]
