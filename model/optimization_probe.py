"""A single-loss temporal derivative, isolated from optimizer-owned gradients."""
from types import SimpleNamespace
import time

import numpy as np
import torch

from .game_sequences import memory_inputs


def memory_probe(learner, encoded, session_index=0):
    e, config = learner.engine, learner.config
    if not 0 <= session_index < len(encoded.sessions):
        return dict(status='unavailable', reason='La sesión solicitada no existe en este batch.')
    session = encoded.sessions[session_index]
    supervised = [i for i, step in enumerate(session.transitions) if step.supervised]
    if not supervised:
        return dict(status='unavailable', reason='La sesión no contiene una transición supervisada.')
    target_t = supervised[-1]
    start = target_t // config.block_steps * config.block_steps
    context_start = max(0, start - config.context_steps)
    view = SimpleNamespace(sessions=[session], context=[encoded.context[session_index]], z=encoded.z)
    flags = [(p, p.requires_grad) for p in e.model.parameters()]
    modes = [(m, m.training) for m in e.model.modules()]
    devices = [e.device.index if e.device.index is not None else torch.cuda.current_device()] if e.device.type == 'cuda' else []
    begin = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=devices):
            e.model.requires_grad_(False).eval()
            memories = [h.detach().requires_grad_(True) for h in learner.prefix(view, context_start)]
            states, times, boundaries, final = [], [], [], None
            with torch.enable_grad(), e.autocast():
                for t in range(context_start, target_t + 1):
                    for ids, steps, z, h, action, geometry in memory_inputs(e, view, t, memories):
                        states.append(h)
                        times.append(t)
                        if steps[0].reset_before:
                            boundaries.append(t)
                        if t == target_t:
                            step = steps[0]
                            before = torch.as_tensor(np.array(step.before.final, copy=True), device=e.device).long()[None]
                            actual = torch.as_tensor(np.array(step.after.final, copy=True), device=e.device).long()[None]
                            target = encoded.z(step.after)[None]
                            final = (z, h, action, target, before, actual)
                            loss = learner._terms(*final)[0, 1]
                        else:
                            next_h, _ = e.advance(z, h, action)
                            memories[0] = next_h.float()
                if final is None:
                    return dict(status='unavailable', reason='No hay estado visual en el paso elegido.')
                gradients = torch.autograd.grad(loss, states, allow_unused=True)
                rms = [g.float().square().mean().sqrt() if g is not None else loss.new_tensor(float('nan'))
                       for g in gradients]
                values = torch.stack(rms).detach().cpu().tolist()
            with torch.no_grad(), e.autocast():
                z, _, action, target, before, actual = final
                cleared = learner._terms(z, torch.zeros_like(final[1]), action, target, before, actual)[0, 1]
            return dict(status='ready', session=session_index, game=session.game, seed=session.seed,
                session_steps=len(session.transitions), context_start=context_start, start=start, stop=target_t+1,
                boundaries=boundaries, steps=[dict(t=t, rms=v if np.isfinite(v) else None) for t, v in zip(times, values)],
                loss=float(loss.detach()), cleared_loss=float(cleared), seconds=time.perf_counter()-begin)
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
        for module, flag in modes:
            module.training = flag
