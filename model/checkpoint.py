"""Atomic latest/best snapshots; no replay, frames, traces or latent caches."""
from dataclasses import asdict
from pathlib import Path
import os
import random
import torch


def save(path, learner, progress, run, rng):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'network': asdict(learner.model.config), 'training': asdict(learner.engine.config),
               'runtime': {'fp8': bool(learner.engine.fp8_layers), 'optimizer_bits': learner.engine.optimizer_bits},
               'weights': learner.model.state_dict(),
               'optimizers': [opt.state_dict() for opt in learner.optimizers],
               'progress': dict(progress), 'run': dict(run), 'rng': rng.getstate(),
               'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = path.with_suffix('.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def restore(payload, learner, rng, *, reset_optimizer=False):
    if not reset_optimizer and payload.get('runtime', {}).get('optimizer_bits', 32) != learner.engine.optimizer_bits:
        raise ValueError('Optimizer precision changed; explicit reset_optimizer is required')
    learner.model.load_state_dict(payload['weights'])
    if not reset_optimizer:
        for opt, saved in zip(learner.optimizers, payload['optimizers'], strict=True):
            opt.load_state_dict(saved)
    rng.setstate(payload['rng'])
    random.setstate(payload['python_rng'])
    torch.set_rng_state(payload['torch_rng'])
    if torch.cuda.is_available() and payload['cuda_rng']:
        if len(payload['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA device count differs; use validate for portable inference')
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
