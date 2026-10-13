"""Real-experience JEPA updates; decoder gradients never update the simulator."""
from contextlib import nullcontext
from dataclasses import dataclass
import itertools

import numpy as np
import torch
from torch.nn import functional as F

from .networks import sigreg
from .metrics import visual_delta


@dataclass
class State:
    z: torch.Tensor | None
    h: torch.Tensor


class Engine:
    def __init__(self, model, config, compile_model=False):
        self.model, self.config = model, config
        self.device = next(model.parameters()).device
        self.amp = False
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                self.amp = torch.cuda.is_bf16_supported()
        self.spatial = model.encoder.spatial
        self.predict = model.dynamics.forward
        self.advance = model.dynamics.advance
        self.decode = model.decoder.forward
        if compile_model:
            # Keep original parameter names and checkpoint objects. Compilation
            # failure is an error, never an unreported eager fallback.
            self.spatial, self.predict, self.advance, self.decode = [
                torch.compile(f, fullgraph=True, dynamic=True) for f in
                (self.spatial, self.predict, self.advance, self.decode)]

    def autocast(self):
        return torch.autocast('cuda', dtype=torch.bfloat16) if self.amp else nullcontext()

    def actions(self, actions):
        return torch.tensor([a.features() for a in actions], device=self.device)

    def encode(self, observations, training=False):
        return self.model.encode(observations, self.spatial, self.config.frame_chunk,
                                 checkpoint_blocks=training)

    @torch.no_grad()
    def initial(self, observation):
        self.model.eval()
        with self.autocast():
            z = self.encode([observation])[0]
        return State(z, torch.zeros(1, self.model.config.memory, device=self.device))

    @torch.no_grad()
    def prediction(self, state, action, reference):
        self.model.eval()
        if state.z is None or reference is None:
            return None
        with self.autocast():
            z, _, delta = self.predict(state.z[None], state.h, self.actions([action]))
            image = self.decode(z, *reference.shape).argmax(-1)[0]
        image = image.cpu().numpy()
        return image, delta[0].cpu().numpy(), visual_delta(reference, image)

    @torch.no_grad()
    def observe(self, state, action, observation):
        self.model.eval()
        with self.autocast():
            h = state.h
            if state.z is not None:
                h, _ = self.advance(state.z[None], h, self.actions([action]))
            z = self.encode([observation])[0]
        return State(z if z is not None else state.z, h.float())

    @torch.no_grad()
    def rebuild(self, episode):
        """Re-encode the live prefix after updates; recurrent memory cannot go stale."""
        self.model.eval()
        with self.autocast():
            latents = self.encode(episode.observations)
            state = State(latents[0], torch.zeros(1, self.model.config.memory, device=self.device))
            for action, z in zip(episode.actions, latents[1:]):
                if state.z is not None:
                    state.h, _ = self.advance(state.z[None], state.h, self.actions([action]))
                if z is not None:
                    state.z = z
        return state


def optimizer(parameters, learning_rate, cuda):
    parameters = list(parameters)
    return torch.optim.AdamW([
        {'params': [p for p in parameters if p.ndim >= 2], 'weight_decay': .01},
        {'params': [p for p in parameters if p.ndim < 2], 'weight_decay': 0.},
    ], lr=learning_rate, betas=(.9, .95), eps=1e-8, fused=cuda)


class Learner:
    def __init__(self, engine):
        self.engine = engine
        self.model = engine.model
        self.primary = list(itertools.chain(self.model.encoder.parameters(), self.model.dynamics.parameters()))
        self.visual = list(self.model.decoder.parameters())
        self.optimizers = [optimizer(p, engine.config.learning_rate, engine.device.type == 'cuda')
                           for p in (self.primary, self.visual)]

    def update(self, episodes):
        if not episodes:
            return None
        e, c = self.engine, self.engine.config
        self.model.train()
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
        real_z, states, memories, actions, targets, boards, references = [], [], [], [], [], [], []
        with e.autocast():
            for episode in episodes:
                zs = e.encode(episode.observations, training=True)
                real_z.extend(z for z in zs if z is not None)
                h = torch.zeros(1, self.model.config.memory, device=e.device)
                current = zs[0]
                first_loss = max(0, len(episode.actions) - c.loss_steps)
                for t, action in enumerate(episode.actions):
                    next_z = zs[t + 1]
                    before, after = episode.observations[t].final, episode.observations[t + 1].final
                    if (t >= first_loss and current is not None and next_z is not None and
                            before is not None and after is not None and before.shape == after.shape):
                        states.append(current)
                        memories.append(h[0])
                        actions.append(action)
                        targets.append(next_z - current)
                        boards.append(after)
                        references.append(before)
                    if current is not None:
                        h, _ = e.advance(current[None], h, e.actions([action]))
                    if next_z is not None:
                        current = next_z
            if not states or len(real_z) < c.min_observations:
                return None
            predicted_z, _, delta = e.predict(torch.stack(states), torch.stack(memories), e.actions(actions))
        with torch.autocast(e.device.type, enabled=False):
            prediction_loss = F.mse_loss(delta.float(), torch.stack(targets).float())
            regularization = sigreg(torch.stack(real_z), self.model.config.sigreg_slices)
            loss = .98 * prediction_loss + .02 * regularization
        loss.backward()
        # Decoder uses frozen predicted representations. Balance changed and
        # unchanged pixels so a large static background cannot dominate learning.
        buckets = {}
        for i, board in enumerate(boards):
            buckets.setdefault(board.shape, []).append(i)
        decoder_loss = 0.
        for shape, indices in buckets.items():
            for start in range(0, len(indices), c.frame_chunk):
                selected = indices[start:start + c.frame_chunk]
                actual = torch.tensor(np.stack([boards[i] for i in selected]), device=e.device, dtype=torch.long)
                before = torch.tensor(np.stack([references[i] for i in selected]), device=e.device, dtype=torch.long)
                with e.autocast():
                    logits = e.decode(predicted_z.detach()[selected], *shape, checkpoint_blocks=True)
                ce = F.cross_entropy(logits.float().permute(0, 3, 1, 2), actual, reduction='none')
                changed = actual != before
                def mean_where(mask):
                    return (ce * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)
                changed_count = changed.sum((1, 2))
                per_frame = torch.where((changed_count > 0) & (changed_count < actual.shape[1] * actual.shape[2]),
                    .5 * (mean_where(changed) + mean_where(~changed)), ce.mean((1, 2)))
                part_loss = per_frame.sum() / len(boards)
                part_loss.backward()
                decoder_loss += float(part_loss.detach())
        # Validate both gradient sets before either optimizer mutates weights.
        norms = [torch.nn.utils.clip_grad_norm_(p, 1., error_if_nonfinite=True)
                 for p in (self.primary, self.visual)]
        for opt in self.optimizers:
            opt.step()
        return {'loss': float(loss.detach()), 'latent_mse': float(prediction_loss.detach()),
                'sigreg': float(regularization.detach()), 'decoder_ce': decoder_loss,
                'gradient_norm': float(norms[0]), 'samples': len(states)}
