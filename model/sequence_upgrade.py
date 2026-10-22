"""Narrow, explicit sequence migrations; never restart learned optimizer state."""
import copy
from dataclasses import replace

import torch

from .config import NetworkConfig
from .networks import Dynamics


ADAPTER_PREFIX = 'dynamics.rotation_adapter.'


def normalized_plan(payload):
    result = copy.deepcopy(payload)
    result['initialization']['network'].setdefault('rotation_adapter_width', 0)
    if 'sequence_exploration' in result:
        result['sequence_exploration'].setdefault('allow_undo', True)
    return result


def upgrade_plan_matches(previous, current, *, disable_undo=False, upgrade_predictor=False):
    old, new = normalized_plan(previous), normalized_plan(current)
    if disable_undo:
        # Also supports legacy uniform collection without an explicit section.
        from .sequence_exploration import SequenceExplorationConfig
        from dataclasses import asdict
        before = old.get('sequence_exploration', asdict(SequenceExplorationConfig()))
        after = new.get('sequence_exploration', asdict(SequenceExplorationConfig()))
        if after != dict(before, allow_undo=False):
            return False
        old['sequence_exploration'] = dict(before, allow_undo=False)
        new['sequence_exploration'] = after
    if upgrade_predictor:
        before, after = old['initialization']['network'], new['initialization']['network']
        width = after['rotation_adapter_width']
        if before['rotation_adapter_width'] not in (0, width) or not width:
            return False
        before['rotation_adapter_width'] = width
    return old == new


def upgraded_network(saved, initial):
    previous = NetworkConfig(**saved['network'])
    target = NetworkConfig(**initial['network'])
    if (not target.rotation_adapter_width or
            previous.rotation_adapter_width not in (0, target.rotation_adapter_width) or
            replace(previous, rotation_adapter_width=target.rotation_adapter_width) != target):
        raise ValueError('Predictor upgrade may only enable a new rotation adapter')
    return target


def load_upgraded_weights(model, saved):
    previous = NetworkConfig(**saved['network'])
    if previous.rotation_adapter_width:
        model.load_state_dict(saved['weights'], strict=True)
        return
    weights = model.state_dict()
    old_keys = {k for k in weights if not k.startswith(ADAPTER_PREFIX)}
    if old_keys != set(saved['weights']):
        raise ValueError('Predictor upgrade must preserve every existing weight')
    weights.update(saved['weights'])
    model.load_state_dict(weights, strict=True)


def upgraded_optimizer(saved, dynamics):
    """Bind old Adam moments by parameter name, adding only the new residual."""
    with torch.device('meta'):
        previous = Dynamics(NetworkConfig(**saved['network']))
    old_parameters = dict(previous.named_parameters())
    old_names = list(old_parameters)
    current = dict(dynamics.named_parameters())
    new_names = list(current)
    if (new_names[:len(old_names)] != old_names or
            any(not n.startswith('rotation_adapter.') for n in new_names[len(old_names):])):
        raise ValueError('Predictor optimizer migration found an unexpected parameter layout')
    state = saved['predictor_optimizer']
    if len(state['param_groups']) != 2:
        raise ValueError('Predictor upgrade requires the original two AdamW decay groups')
    ids = [i for group in state['param_groups'] for i in group['params']]
    if len(set(ids)) != len(ids) or set(state['state']) - set(ids):
        raise ValueError('Invalid predictor optimizer parameter IDs')
    migrated, groups, offset = {}, [], 0
    for group, decay in zip(state['param_groups'], (True, False)):
        before = [n for n, p in old_parameters.items() if (p.ndim >= 2) == decay]
        after = [n for n, p in current.items() if (p.ndim >= 2) == decay]
        if len(group['params']) != len(before):
            raise ValueError('Predictor optimizer parameter group size mismatch')
        indices = {name: offset + i for i, name in enumerate(after)}
        for name, identity in zip(before, group['params']):
            values = state['state'].get(identity)
            if values is None:
                continue
            if set(values) - {'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'}:
                raise ValueError('Predictor upgrade supports standard AdamW state only')
            if any(v.shape != current[name].shape for k, v in values.items() if k != 'step'):
                raise ValueError(f'Predictor optimizer shape mismatch for {name}')
            migrated[indices[name]] = values
        groups.append(dict(group, params=list(indices.values())))
        offset += len(after)
    return dict(state=migrated, param_groups=groups)
