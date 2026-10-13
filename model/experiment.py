"""Finite real-observation fit diagnostics; no persisted replay or checkpoints."""
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
from .metrics import delta_counts, Metrics
from .networks import WorldModel


def collect(factory, count, parity, level=1):
    pairs = []
    for index in range(count):
        seed = 2 * (index + 100) + parity
        env = factory.make(seed, level)
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


@torch.no_grad()
def encode_pairs(engine, pairs):
    """Only for a frozen representation: real SDK first-action pairs."""
    engine.model.eval()
    prepared = []
    with engine.autocast():
        for before, after in pairs:
            if before.final is None or after.final is None or before.final.shape != after.final.shape:
                raise ValueError('Fit diagnostic requires a confirmed same-geometry transition')
            z, target = engine.encode([before, after])
            prepared.append((z.detach(), target.detach(), engine.actions([after.action_input])[0],
                             torch.tensor(np.array(before.final), device=engine.device, dtype=torch.long),
                             torch.tensor(np.array(after.final), device=engine.device, dtype=torch.long)))
    return prepared


@torch.no_grad()
def score_dynamics(engine, pairs):
    engine.model.eval()
    metrics = Metrics()
    copy_mse = 0.
    with engine.autocast():
        for z, target, action, before, after in pairs:
            memory = torch.zeros(1, engine.model.config.memory, device=engine.device)
            predicted, _, delta = engine.predict(z[None], memory, action[None], *before.shape)
            image = engine.decode(predicted, *before.shape).argmax(-1)[0]
            metrics.add(before.cpu().numpy(), after.cpu().numpy(), image.cpu().numpy(),
                        z.cpu().numpy(), target.cpu().numpy(), delta[0].cpu().numpy())
            metrics.observe_latent(z.cpu().numpy(), tuple(before.shape))
            metrics.observe_latent(target.cpu().numpy(), tuple(after.shape))
            copy_mse += float((target - z).square().mean())
    return metrics.report() | {'zero_delta_mse': copy_mse / len(pairs)}


def fit_dynamics(engine, args, factory, level):
    """Test capacity/optimization with fixed real targets, not generalization."""
    model = engine.model
    model.requires_grad_(False)
    model.dynamics.requires_grad_(True)
    train = encode_pairs(engine, collect(factory, args.pairs, 0, level))
    valid = encode_pairs(engine, collect(factory, args.pairs, 1, level))
    params = list(model.dynamics.parameters())
    opt = optimizer(params, args.learning_rate, engine.device.type == 'cuda', engine.optimizer_bits)
    print(json.dumps({'event': 'diagnostic', 'step': 0, 'fit': score_dynamics(engine, train),
                      'unseen': score_dynamics(engine, valid)}), flush=True)
    started = None
    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.
        # Geometry buckets avoid resizing unrelated SDK canvases.
        buckets = {}
        for pair in train:
            buckets.setdefault(tuple(pair[3].shape), []).append(pair)
        for shape, pairs in buckets.items():
            for start in range(0, len(pairs), 4):
                part = pairs[start:start + 4]
                z, target, action, before, after = map(torch.stack, zip(*part))
                memory = torch.zeros(len(part), model.config.memory, device=engine.device)
                with engine.autocast():
                    predicted, _, delta = engine.predict(z, memory, action, *shape)
                    logits = engine.decode(predicted, *shape, checkpoint_blocks=True)
                    latent = (delta.float() - (target - z).float()).square().mean()
                    visual = (visual_delta_loss(logits, before, after, args.false_change_weight)
                              + soft_delta_error(logits, before, after)).mean()
                    loss = (latent + visual) * len(part) / len(train)
                loss.backward()
                total += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        opt.step()
        if engine.device.type == 'cuda':
            torch.cuda.synchronize()
        if started is None:
            started = time.monotonic()
        if step == 1 or step % 20 == 0 or step == args.steps:
            print(json.dumps({'event': 'diagnostic', 'step': step, 'loss': total,
                              'gradient_norm': float(norm), 'seconds': time.monotonic() - started,
                              'fit': score_dynamics(engine, train), 'unseen': score_dynamics(engine, valid),
                              'peak_mib': torch.cuda.max_memory_allocated() / 2**20 if engine.device.type == 'cuda' else 0}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=('reconstruction', 'dynamics'), default='reconstruction')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--false-change-weight', type=float, default=1.)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--pairs', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=3e-5)
    p.add_argument('--fp8', action='store_true')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--residual-scale', type=float, default=1.)
    p.add_argument('--position-scale', type=float, default=1.)
    p.add_argument('--pixel-latent', action='store_true')
    args = p.parse_args()
    if args.steps < 1 or args.pairs < 1:
        p.error('--steps and --pairs must be positive')
    if args.task == 'dynamics' and args.checkpoint is None:
        p.error('Frozen-representation dynamics diagnostic requires --checkpoint')
    if args.task == 'reconstruction' and args.checkpoint is not None:
        p.error('--checkpoint is for the frozen-representation dynamics diagnostic')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True) if args.checkpoint else None
    network = NetworkConfig(**payload['network']) if payload else NetworkConfig(
        residual_scale=args.residual_scale, position_scale=args.position_scale, pixel_latent=args.pixel_latent)
    model = WorldModel(network).cuda()
    if payload:
        model.load_state_dict(payload['weights'])
    TrainConfig(false_change_weight=args.false_change_weight)
    engine = Engine(model, TrainConfig(learning_rate=args.learning_rate), True, fp8=args.fp8, optimizer_bits=8)
    run = payload['run'] if payload else {'game': 'sc01-v1', 'environments_dir': Path(__file__).resolve().parents[1] / 'games', 'mode': 'offline', 'levels': [1]}
    level = run['levels'][min(payload['progress']['level_index'], len(run['levels']) - 1)] if payload else 1
    factory = Factory(run['game'], run['environments_dir'], run['mode'])
    print(json.dumps({'event': 'diagnostic_start', **(vars(args) | {'checkpoint': str(args.checkpoint) if args.checkpoint else None}),
                      'parameters': sum(p.numel() for p in model.parameters())}), flush=True)
    if args.task == 'dynamics':
        fit_dynamics(engine, args, factory, level)
        return
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt = optimizer(params, args.learning_rate, True, 8)
    train, valid = collect(factory, args.pairs, 0), collect(factory, args.pairs, 1)
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
