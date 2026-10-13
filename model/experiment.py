"""Finite real-observation reconstruction diagnostic; no persisted replay."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import NetworkConfig, TrainConfig
from .data import random_action
from .environment import Factory
from .learning import Engine, optimizer, visual_delta_loss, soft_delta_error
from .metrics import delta_counts
from .networks import WorldModel


def collect(factory, count, parity):
    pairs = []
    for index in range(count):
        seed = 2 * (index + 100) + parity
        env = factory.make(seed, 1)
        before = env.observation
        after = env.step(random_action(before, random.Random(seed)))
        pairs.append((before, after))
    return pairs


@torch.no_grad()
def score(engine, pairs):
    engine.model.eval()
    wrong = union = pixels = errors = 0
    with engine.autocast():
        for before, after in pairs:
            z = engine.encode([after])[0]
            image = engine.decode(z[None], *after.final.shape)[0].argmax(-1).cpu().numpy()
            w, u, _ = delta_counts(before.final, after.final, image)
            wrong += w
            union += u
            pixels += image.size
            errors += int(np.count_nonzero(image != after.final))
    return {'delta_error': wrong / max(1, union), 'pixel_error': errors / max(1, pixels)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--pairs', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=3e-5)
    p.add_argument('--fp8', action='store_true')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--residual-scale', type=float, default=1.)
    p.add_argument('--position-scale', type=float, default=1.)
    p.add_argument('--pixel-latent', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    model = WorldModel(NetworkConfig(residual_scale=args.residual_scale, position_scale=args.position_scale,
                                    pixel_latent=args.pixel_latent)).cuda()
    engine = Engine(model, TrainConfig(learning_rate=args.learning_rate), True, fp8=args.fp8, optimizer_bits=8)
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt = optimizer(params, args.learning_rate, True, 8)
    factory = Factory('sc01-v1', Path(__file__).resolve().parents[1] / 'games')
    train, valid = collect(factory, args.pairs, 0), collect(factory, args.pairs, 1)
    print(json.dumps({'event': 'diagnostic_start', **vars(args), 'parameters': sum(p.numel() for p in model.parameters())}), flush=True)
    started = None
    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.
        for start in range(0, len(train), 4):
            part = train[start:start + 4]
            before = torch.tensor(np.stack([b.final for b, a in part]), device='cuda', dtype=torch.long)
            after = torch.tensor(np.stack([a.final for b, a in part]), device='cuda', dtype=torch.long)
            with engine.autocast():
                z = torch.stack(engine.encode([a for b, a in part], training=True))
                logits = engine.decode(z, *after.shape[1:], checkpoint_blocks=True)
                local = engine.decode(z, *after.shape[1:], local=True)
                loss = sum((visual_delta_loss(x, before, after) + soft_delta_error(x, before, after)).sum()
                           for x in (logits, local)) / len(train)
            loss.backward()
            total += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        if started is None:
            started = time.monotonic()
        if step == 1 or step % 20 == 0 or step == args.steps:
            print(json.dumps({'event': 'diagnostic', 'step': step, 'loss': total, 'gradient_norm': float(norm),
                              'seconds': round(time.monotonic() - started, 3),
                              'fit': score(engine, train), 'unseen': score(engine, valid),
                              'peak_mib': torch.cuda.max_memory_allocated() / 2**20}), flush=True)


if __name__ == '__main__':
    main()
