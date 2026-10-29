"""Bind optional object tensors only when the exported checkpoint contains them."""
from copy import deepcopy

import torch

from .config import NetworkConfig
from .object_dynamics import ObjectDynamics


def bind_objects(document, network):
    if not network.get('object_slots'):
        return document
    document = deepcopy(document)
    with torch.device('meta'):
        state = ObjectDynamics(NetworkConfig(**network)).state_dict()
    group = next((n for n in document['nodes'] if n['id'] == 'objects.student'), None)
    if group is None:
        raise ValueError('Object-enabled export requires the updated architecture.json')
    group['attributes'] = [{'name': 'object_slots', 'value': network['object_slots']},
                           {'name': 'object_width', 'value': network['object_width']}]
    group['parameter_ids'] = []
    for name, tensor in state.items():
        identity = 'weights.dynamics.objects.'+name
        document['parameters'].append(dict(id=identity, name='dynamics.objects.'+name,
            shape=[dict(kind='constant', value=d) for d in tensor.shape]))
        group['parameter_ids'].append(identity)
    document['architecture_revision'] += '-objects'
    return document
