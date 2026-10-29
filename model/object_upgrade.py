"""Explicit copy-only sequence migration. Never replace the source run/weights."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path

import torch

from . import checkpoint
from .config import NetworkConfig
from .networks import Dynamics
from .object_dynamics import ObjectDynamics
from .sequence_upgrade import normalized_plan, upgraded_optimizer
from .training_config import load_plan


def migrate(saved, plan):
    if saved.get('kind') != 'game_sequences' or saved.get('version') != 1:
        raise ValueError('Object migration requires a sequence checkpoint')
    if 'sequence_objects' not in plan.payload:
        raise ValueError('Missing sequence_objects configuration')
    old = NetworkConfig(**saved['network'])
    target = NetworkConfig(**plan.payload['initialization']['network'])
    if old.object_slots:
        before, after = normalized_plan(saved['plan']), normalized_plan(plan.payload)
        was_requested = before['sequence_objects'].get('activate_residual', False)
        requested = after['sequence_objects'].get('activate_residual', False)
        before['sequence_objects']['activate_residual'] = True
        after['sequence_objects']['activate_residual'] = True
        if old != target or was_requested or not requested or before != after:
            raise ValueError('An object run may only request residual activation, preserving its configuration')
        # Keep learned masks and every Adam state. This requests the existing
        # warmup/IoU gate; it does not force a gain of one on unready masks.
        return dict(saved, plan=plan.payload, predictor_adaptations=list(saved.get('predictor_adaptations', [])) + [dict(
            kind='object_residual_requested', batch=saved['progress']['batches'])])
    if old.object_slots or not target.object_slots or replace(target, object_slots=0, object_width=old.object_width) != old:
        raise ValueError('Object migration may only add object slots/width')
    before, after = normalized_plan(saved['plan']), normalized_plan(plan.payload)
    after.pop('sequence_objects')
    after['initialization']['network']['object_slots'] = 0
    after['initialization']['network']['object_width'] = old.object_width
    if before != after:
        raise ValueError('Object migration may not change curriculum, rates, phases or other configuration')
    if saved.get('sequence_control', {}).get('representation_frozen_at_batch') is None:
        raise ValueError('Freeze representation before adopting object dynamics')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(161)
        head = ObjectDynamics(target)
    with torch.device('meta'):
        dynamics = Dynamics(target)
    result = dict(saved, network=asdict(target), plan=plan.payload, weights=dict(saved['weights']))
    result['weights'].update({'dynamics.objects.'+k: v for k, v in head.state_dict().items()})
    result['predictor_optimizer'] = upgraded_optimizer(saved, dynamics, add_objects=True)
    result['predictor_adaptations'] = list(saved.get('predictor_adaptations', [])) + [dict(
        kind='object_dynamics', batch=saved['progress']['batches'], object_slots=target.object_slots,
        object_width=target.object_width, initial_gain=0., source='official frozen SAM2 pseudo-labels')]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--training-config', required=True)
    parser.add_argument('--output', required=True, help='A new run directory; never the source run')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output must be a new directory to preserve the source run')
    plan = load_plan(args.training_config)
    payload = torch.load(args.source, map_location='cpu', weights_only=True, mmap=True)
    updated = migrate(payload, plan)
    checkpoint.write(output/'latest.pt', updated)
    print(f'Migrated batch {updated["progress"]["batches"]}; existing weights/Adam retained; source untouched.')


if __name__ == '__main__':
    main()
