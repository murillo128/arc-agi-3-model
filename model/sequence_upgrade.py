"""Narrow, explicit sequence migrations; never restart learned optimizer state."""
import copy
from dataclasses import replace

import torch

from .config import NetworkConfig
from .networks import Dynamics


ADAPTER_PREFIX = 'dynamics.rotation_adapter.'
CLICK_SIGMA_KEY = ADAPTER_PREFIX + 'click_width.log_scale'


def normalized_plan(payload):
    result = copy.deepcopy(payload)
    result['initialization']['network'].setdefault('rotation_adapter_width', 0)
    result['initialization']['network'].setdefault('learnable_click_sigma', False)
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
        if before['learnable_click_sigma'] and not after['learnable_click_sigma']:
            return False
        before['rotation_adapter_width'] = width
        before['learnable_click_sigma'] = after['learnable_click_sigma']
    return old == new


def upgraded_network(saved, initial):
    previous = NetworkConfig(**saved['network'])
    target = NetworkConfig(**initial['network'])
    if (not target.rotation_adapter_width or
            previous.rotation_adapter_width not in (0, target.rotation_adapter_width) or
            (previous.learnable_click_sigma and not target.learnable_click_sigma) or
            replace(previous, rotation_adapter_width=target.rotation_adapter_width,
                    learnable_click_sigma=target.learnable_click_sigma) != target):
        raise ValueError('Predictor upgrade may only enable a rotation adapter or learnable click sigma')
    return target


def load_upgraded_weights(model, saved):
    previous = NetworkConfig(**saved['network'])
    weights = model.state_dict()
    added = ({k for k in weights if k.startswith(ADAPTER_PREFIX)} if not previous.rotation_adapter_width
             else {CLICK_SIGMA_KEY} if not previous.learnable_click_sigma and model.config.learnable_click_sigma
             else set())
    old_keys = set(weights) - added
    if old_keys != set(saved['weights']):
        raise ValueError('Predictor upgrade must preserve every existing weight')
    weights.update(saved['weights'])
    model.load_state_dict(weights, strict=True)


def upgraded_optimizer(saved, dynamics):
    """Bind old Adam moments by name, adding only approved adapter parameters."""
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
    # Adam initializes lazily, so a never-updated predictor legitimately has no
    # state. Once training starts, every old mapping must retain its moments.
    fresh = saved['progress']['updates'] == 0 and not state['state']
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
                if fresh:
                    continue
                raise ValueError(f'Missing predictor optimizer state for {name}')
            required = {'step', 'exp_avg', 'exp_avg_sq'}
            if group.get('amsgrad', False):
                required.add('max_exp_avg_sq')
            if not required <= set(values):
                raise ValueError(f'Missing predictor AdamW fields for {name}: {sorted(required - set(values))}')
            if set(values) - {'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'}:
                raise ValueError('Predictor upgrade supports standard AdamW state only')
            step = values['step']
            if not isinstance(step, torch.Tensor) or step.numel() != 1 or not torch.isfinite(step).all():
                raise ValueError(f'Predictor optimizer step must be a finite scalar for {name}')
            if any(not isinstance(v, torch.Tensor) or v.shape != current[name].shape
                   for k, v in values.items() if k != 'step'):
                raise ValueError(f'Predictor optimizer shape mismatch for {name}')
            migrated[indices[name]] = values
        groups.append(dict(group, params=list(indices.values())))
        offset += len(after)
    return dict(state=migrated, param_groups=groups)
