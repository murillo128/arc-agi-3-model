"""Versioned inference-only bases for per-game adaptation, exported explicitly.

The portable payload contains no trainer optimizer, replay, RNG, paths or game
implementations. Replacing this separately mounted file needs no source change.
"""
import argparse
from dataclasses import asdict, dataclass
from collections import OrderedDict
import hashlib
from pathlib import Path

import torch

from . import checkpoint
from .adaptation_heads import VisualGoal
from .config import NetworkConfig, TrainConfig
from .learning import Engine
from .networks import WorldModel
from .path_planning import PathValue

FORMAT = 'arc3-adaptation-base'
VERSION = 1
PURPOSE = 'Engineering smoke; not final Phase 2 competition weights'
LEARNING_FIELDS = ('decoder_readout', 'prediction_delta_loss', 'delta_overlap_weight',
                   'false_change_weight', 'frame_chunk')


@dataclass
class AdaptationBase:
    engine: Engine
    goal: VisualGoal
    path: PathValue
    metadata: dict
    change: object = None


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f'Unexpected {name} fields')


def _weights(module, values, name):
    if type(values) not in (dict, OrderedDict):
        raise ValueError(f'Invalid {name} weights')
    expected = module.state_dict()
    if set(values) != set(expected):
        raise ValueError(f'Unexpected {name} weight keys')
    for key, value in values.items():
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or
                value.shape != expected[key].shape or value.dtype != expected[key].dtype or
                not value.is_floating_point() or
                not torch.isfinite(value).all()):
            raise ValueError(f'Invalid {name} tensor: {key}')
    module.load_state_dict(values, strict=True)
    module.eval().requires_grad_(False)


def _metadata(value):
    _keys(value, ('purpose', 'source_sha256', 'source_batches'), 'metadata')
    if value['purpose'] != PURPOSE:
        raise ValueError('Unknown adaptation base purpose')
    digest = value['source_sha256']
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('Invalid source SHA256')
    if type(value['source_batches']) is not int or value['source_batches'] < 0:
        raise ValueError('Invalid source batch count')


def from_payload(payload, *, device='cpu'):
    _keys(payload, ('format', 'version', 'world', 'goal', 'path', 'metadata'), 'adaptation base')
    if payload['format'] != FORMAT or type(payload['version']) is not int or payload['version'] != VERSION:
        raise ValueError('Expected arc3-adaptation-base v1')
    _metadata(payload['metadata'])
    world, goal, path = (payload[key] for key in ('world', 'goal', 'path'))
    _keys(world, ('network', 'learning', 'weights'), 'world')
    _keys(world['network'], NetworkConfig.__dataclass_fields__, 'network config')
    _keys(world['learning'], LEARNING_FIELDS, 'learning config')
    for name, field in NetworkConfig.__dataclass_fields__.items():
        value = world['network'][name]
        allowed = (int, float) if field.type is float else (field.type,)
        if type(value) not in allowed:
            raise ValueError(f'Invalid network config type: {name}')
    for name in ('delta_overlap_weight', 'false_change_weight'):
        if type(world['learning'][name]) not in (int, float):
            raise ValueError(f'Invalid learning config type: {name}')
    network = NetworkConfig(**world['network'])
    training = TrainConfig(**world['learning'])
    if type(training.frame_chunk) is not int:
        raise ValueError('frame_chunk must be an integer')
    _keys(goal, ('config', 'weights'), 'goal')
    _keys(path, ('config', 'weights'), 'path')
    _keys(goal['config'], ('latent', 'memory', 'patch', 'width'), 'goal config')
    _keys(path['config'], ('latent', 'memory', 'width', 'objective_version'), 'path config')
    for name, config in (('goal', goal['config']), ('path', path['config'])):
        if any(type(v) is not int or v <= 0 for v in config.values()):
            raise ValueError(f'Invalid {name} config')
        if config['latent'] != network.latent or config['memory'] != network.memory:
            raise ValueError(f'{name} dimensions disagree with world')
        if not 1 <= config['width'] <= 256:
            raise ValueError(f'Invalid {name} width')
    if goal['config']['patch'] != network.patch or path['config']['objective_version'] != 4:
        raise ValueError('Unsupported goal patch / path objective')
    # Construction must not consume any game's/global torch RNG stream.
    with torch.random.fork_rng(devices=[]):
        model = WorldModel(network)
        goal_head, path_head = VisualGoal(**goal['config']), PathValue(**path['config'])
    _weights(model, world['weights'], 'world')
    _weights(goal_head, goal['weights'], 'goal')
    _weights(path_head, path['weights'], 'path')
    model.to(device)
    goal_head.to(device)
    path_head.to(device)
    return AdaptationBase(Engine(model, training, compile_model=False), goal_head,
                          path_head, dict(payload['metadata']))


def load(path, *, device='cpu'):
    # Mounted offline inputs need not support efficient memory mapping. Read the
    # storages eagerly before validation/copying; do not retain a payload/cache.
    # Peak host storage is the payload plus one base and per-tensor check scratch.
    return from_payload(torch.load(path, map_location='cpu', weights_only=True, mmap=False), device=device)


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def export(source, output):
    """Export a trusted sequence-training snapshot without modifying the source."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Export must not overwrite its training source')
    saved = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    if saved.get('kind') != 'game_sequences' or saved.get('version') != 1:
        raise ValueError('Expected a version-1 sequence-training snapshot')
    planning = saved.get('planning')
    if not isinstance(planning, dict) or planning.get('version') != 1:
        raise ValueError('Sequence snapshot must contain trained planning heads')
    # Only adopted inference configuration is copied, never source run/path data.
    network = NetworkConfig(**saved['network'])
    training = TrainConfig(**saved['training'])
    width = planning['config']['width']
    goal_config = dict(latent=network.latent, memory=network.memory, patch=network.patch, width=width)
    path_config = dict(latent=network.latent, memory=network.memory, width=width, objective_version=4)
    payload = dict(format=FORMAT, version=VERSION,
                   world=dict(network=asdict(network), learning={k: getattr(training, k) for k in LEARNING_FIELDS},
                              weights=dict(saved['weights'])),
                   goal=dict(config=goal_config, weights=dict(planning['goal'])),
                   path=dict(config=path_config, weights=dict(planning['path'])),
                   metadata=dict(purpose=PURPOSE, source_sha256=_digest(source),
                                 source_batches=saved['progress']['batches']))
    # Strict shape/config/finite validation happens before the output is written.
    validated = from_payload(payload)
    del validated
    checkpoint.write(output, payload)
    return dict(payload['metadata'], output_bytes=output.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error('--threads must be in 1..16')
    torch.set_num_threads(args.threads)
    import json
    print(json.dumps(export(args.source, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
