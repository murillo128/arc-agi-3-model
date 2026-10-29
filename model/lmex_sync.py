"""Publish the repo architecture and latest saved weights to local LMEX.

The public package is an atomic symlink to a complete snapshot INSIDE the LMEX
root. The trainer's checkpoint and architecture are strictly read-only inputs.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import torch
from safetensors.torch import save_file


PRODUCER = 'arc3-lmex-sync'
EXPORT_VERSION = 3  # Optional object tensors are bound for enabled checkpoints.


def planning_export(saved, network):
    """Validate head inventories on meta; return only weights and public metadata."""
    from dataclasses import asdict
    from .sequence_planning import SequencePlanningConfig, VisualGoal
    from .path_planning import PathValue

    if not isinstance(saved, dict) or saved.get('version') != 1:
        raise ValueError('Unsupported sequence planning checkpoint version')
    try:
        settings = SequencePlanningConfig(**saved['config'])
        with torch.device('meta'):
            modules = {
                'goal': ('goal_planner', VisualGoal(network['latent'], network['memory'], network['patch'], settings.width)),
                'path': ('path_evaluator', PathValue(network['latent'], network['memory'], settings.width)),
            }
        weights = {}
        for key, (prefix, module) in modules.items():
            actual, expected = saved[key], module.state_dict()
            if not isinstance(actual, dict) or actual.keys() != expected.keys():
                raise ValueError(f'Incomplete sequence planning {key} state_dict')
            for name, tensor in actual.items():
                if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                        or tensor.shape != expected[name].shape or not tensor.is_floating_point()):
                    raise ValueError(f'Invalid sequence planning tensor: {key}.{name}')
                weights[f'{prefix}.{name}'] = tensor
        metadata = {key: saved[key] for key in ('goal_updates', 'path_updates', 'teacher_actions',
                                               'learned_probability', 'adopted_batch')}
        metadata.update(version=1, config=asdict(settings))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'Invalid sequence planning checkpoint: {error}') from error
    return weights, metadata


def stamp(path):
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def read_owned(directory, owner):
    """Only our immediate, non-symlink package directories may be replaced/deleted."""
    if directory.parent != owner or directory.is_symlink() or not directory.name.startswith('package-'):
        raise ValueError('Refusing an unowned LMEX package target')
    manifest = json.loads((directory / '.sync.json').read_text())
    if manifest.get('producer') != PRODUCER:
        raise ValueError('Refusing an unowned LMEX package target')
    return manifest


def sync(checkpoint, architecture, model_root, model_id='arc3-delta-world-model'):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', model_id):
        raise ValueError('model_id must be a single simple directory name')
    checkpoint, architecture = Path(checkpoint).resolve(), Path(architecture).resolve()
    root = Path(model_root).resolve(strict=True)
    owner = root / '.arc3-lmex-sync' / model_id
    owner.mkdir(parents=True, exist_ok=True)
    if owner.resolve() != owner:
        raise ValueError('Publication directory must not contain symlinks')
    public = root / model_id
    with (owner / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous, previous_manifest = None, {}
        if public.is_symlink():
            previous = public.resolve(strict=True)
            previous_manifest = read_owned(previous, owner)
        elif public.exists():
            raise ValueError('Move the existing LMEX package aside before first synchronization')

        source_stamp, architecture_stamp = stamp(checkpoint), stamp(architecture)
        raw = architecture.read_bytes()
        document = json.loads(raw)
        if not isinstance(document, dict) or document.get('schema_version') != 1:
            raise ValueError('Expected a model-owned architecture document')
        digest = hashlib.sha256(raw).hexdigest()
        weights_changed = (previous_manifest.get('checkpoint_stamp') != source_stamp
                           or previous_manifest.get('export_version') != EXPORT_VERSION)
        if not weights_changed and previous_manifest.get('architecture_sha256') == digest:
            return None

        stage = Path(tempfile.mkdtemp(prefix='package-', dir=owner))
        temporary_link = owner / 'publish-link'
        published = False
        try:
            if weights_changed:
                # The file is atomically replaced by the trainer. mmap avoids
                # materializing optimizer states or moving anything onto the GPU.
                payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
                planning_metadata = None
                if payload.get('format') == 'arc3-agent':
                    if payload.get('version') != 1:
                        raise ValueError('Unsupported portable agent checkpoint version')
                    weights = dict(payload['world']['weights'])
                    # The new subgoal mode uses PathValue, not the legacy outcome
                    # heads also retained by the backward-compatible checkpoint.
                    path_value = payload.get('components', {}).get('path_evaluator')
                    if payload['agent']['mode'] == 'path_mpc':
                        if path_value is None:
                            raise ValueError('path_mpc checkpoint is missing its evaluator')
                        weights.update({f'path_evaluator.{k}': t for k, t in path_value['weights'].items()})
                    network = payload['world']['network']
                    progress = dict(payload.get('training_progress', {}))
                else:
                    weights = dict(payload['weights'])
                    network = payload['network']
                    progress = {key: payload['progress'][key] for key in
                                ('batches', 'actions', 'updates', 'episodes', 'level_index') if key in payload['progress']}
                    exploration = payload.get('exploration') if payload.get('kind') == 'game_sequences' else None
                    if exploration is not None:
                        if exploration.get('version') != 1:
                            raise ValueError('Unsupported exploration checkpoint version')
                        weights.update({f'exploration.{k}': t for k, t in exploration['weights'].items()})
                    if payload.get('kind') == 'game_sequences' and payload.get('planning') is not None:
                        heads, planning_metadata = planning_export(payload['planning'], network)
                        if weights.keys() & heads.keys():
                            raise ValueError('Sequence planning tensor names collide with world weights')
                        weights.update(heads)
                if not weights or any(not isinstance(t, torch.Tensor) or t.device.type != 'cpu'
                                      for t in weights.values()):
                    raise ValueError('Expected CPU state_dict tensors')
                save_file(weights, str(stage / 'model.safetensors'), metadata={'format': 'pt'})
                config = {'model_type': 'arc3_latent_world_model', 'architectures': ['WorldModel'],
                          'name_or_path': model_id, 'hidden_size': network['width'],
                          'num_hidden_layers': network['encoder_depth'],
                          'num_attention_heads': network['heads'], 'patch_size': network['patch'],
                          'network': network, 'training_progress': progress,
                          'parameter_count': sum(t.numel() for t in weights.values()),
                          'training_status': 'training_checkpoint',
                          'export_purpose': 'latest_saved_training_weights'}
                if payload.get('format') == 'arc3-agent':
                    config['agent'] = payload['agent']
                    if payload['agent']['mode'] == 'path_mpc':
                        config['architectures'].append('PathValue')
                elif payload.get('kind') == 'game_sequences':
                    config['training_mode'] = 'game_sequences'
                    if exploration is not None:
                        config['architectures'].append('ChangeHead')
                    if planning_metadata is not None:
                        config['architectures'].extend(['VisualGoal', 'PathValue'])
                        config['sequence_planning'] = planning_metadata
                dtypes = {str(t.dtype).removeprefix('torch.') for t in weights.values()}
                if len(dtypes) == 1:
                    config['torch_dtype'] = dtypes.pop()
                del weights, payload
            else:
                # Pure architecture edits do not require another 800+ MiB export.
                os.link(previous / 'model.safetensors', stage / 'model.safetensors')
                config = json.loads((previous / 'config.json').read_text())

            (stage / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
            if config['network'].get('object_slots'):
                from .object_architecture import bind_objects
                expanded = bind_objects(json.loads(raw), config['network'])
                (stage / 'architecture.json').write_text(json.dumps(expanded, indent=2) + '\n')
            else:
                (stage / 'architecture.json').write_bytes(raw)
            manifest = {'producer': PRODUCER, 'export_version': EXPORT_VERSION, 'checkpoint_stamp': source_stamp,
                        'architecture_sha256': digest}
            (stage / '.sync.json').write_text(json.dumps(manifest) + '\n')
            if stamp(checkpoint) != source_stamp or stamp(architecture) != architecture_stamp:
                raise RuntimeError('Inputs changed during export; retry the newest snapshot')

            # The catalogue sees either complete package, never half-written weights.
            temporary_link.unlink(missing_ok=True)
            temporary_link.symlink_to(stage.relative_to(root), target_is_directory=True)
            os.replace(temporary_link, public)
            published = True
            for old in owner.glob('package-*'):
                if old != stage:
                    try:
                        read_owned(old, owner)
                    except (ValueError, OSError):
                        continue
                    shutil.rmtree(old)
            return {'event': 'lmex_synced', 'model': model_id, 'weights_changed': weights_changed,
                    **config['training_progress'], 'parameters': config['parameter_count']}
        finally:
            temporary_link.unlink(missing_ok=True)
            if not published:
                shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--architecture', type=Path, default=Path(__file__).with_name('architecture.json'))
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--model-id', default='arc3-delta-world-model')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=5.)
    args = parser.parse_args()
    if not 1 <= args.interval <= 3600:
        parser.error('--interval must be between 1 and 3600 seconds')
    torch.set_num_threads(1)
    try:
        while True:
            try:
                report = sync(args.checkpoint, args.architecture, args.model_root, args.model_id)
                if report:
                    print(json.dumps(report), flush=True)
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                if not args.watch:
                    raise
                print(json.dumps({'event': 'lmex_sync_retry', 'error': str(error)}), flush=True)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
