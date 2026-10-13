"""Joint latent dynamics and visual-delta learning from real SDK experience."""
from contextlib import nullcontext
from dataclasses import dataclass
import itertools

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .networks import sigreg, variance_floor
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


def delta_log_probabilities(color_logits, reference):
    """Canonical 17-way delta: the reference color's probability belongs to KEEP.

    A replacement by the same color is impossible. This is a bijective relabeling
    of the 16 final colors, so its NLL equals color CE, without a second head.
    Reference pixels label the output; they never enter the neural decoder.
    """
    logp = color_logits.float().log_softmax(-1)
    keep = logp.gather(-1, reference[..., None])
    replacements = logp.scatter(-1, reference[..., None], -torch.inf)
    return torch.cat((replacements, keep), -1)


def visual_delta_loss(color_logits, reference, actual, balance='color', power=.5):
    """Balance changed/static groups, then observed colors within each group.

    Color-frequency weights protect small static details without identifying
    game objects or a special color. Power .5 moderates the rare-color boost;
    power 1 gives equal color mass, while power 0 gives equal pixel mass.
    Pixel mode retains the original objective for controlled comparisons.
    """
    changed = actual != reference
    target = torch.where(changed, actual, 16)
    logp = delta_log_probabilities(color_logits, reference)
    nll = -logp.gather(-1, target[..., None]).squeeze(-1)

    def mean_where(mask):
        if balance == 'color':
            # Group by the real final color, including KEEP pixels. Grouping
            # only by delta code would put every static color back into KEEP.
            classes = F.one_hot(actual, 16).to(nll.dtype) * mask[..., None]
            counts = classes.sum((1, 2))
            sums = (classes * nll[..., None]).sum((1, 2))
            weights = counts.clamp_min(1).pow(-power)
            return (sums * weights).sum(-1) / (counts * weights).sum(-1).clamp_min(1)
        if balance != 'pixel':
            raise ValueError('Unknown visual balance')
        return (nll * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)

    changed_count = changed.sum((1, 2))
    return torch.where((changed_count > 0) & (changed_count < actual.shape[1] * actual.shape[2]),
                       .5 * (mean_where(changed) + mean_where(~changed)), mean_where(torch.ones_like(changed)))


def soft_delta_error(color_logits, reference, actual):
    """Differentiable delta-union error, including expected false changes.

    Equals the reported delta error in the one-hot limit for nonempty real
    deltas. All-static frames use mean false-change probability so their
    gradient remains useful even when the expected union is large.
    """
    probability = color_logits.float().softmax(-1)
    correct = probability.gather(-1, actual[..., None]).squeeze(-1)
    changed = actual != reference
    false_changes = ((1 - correct) * ~changed).sum((1, 2))
    count = changed.sum((1, 2))
    wrong = ((1 - correct) * changed).sum((1, 2)) + false_changes
    return torch.where(count > 0, wrong / (count + false_changes).clamp_min(1),
                       false_changes / (actual.shape[1] * actual.shape[2]))


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
        real_z, states, memories, actions, targets, boards, references, successors = {}, [], [], [], [], [], [], []
        with e.autocast():
            for episode in episodes:
                # Keep episode outputs, not all encoder intermediates across
                # every sampled prefix. Recompute one episode at a time during
                # backward; all branches still receive the same joint gradients.
                zs = activation_checkpoint(e.encode, episode.observations,
                                           training=True, use_reentrant=False)
                for obs, z in zip(episode.observations, zs):
                    if z is not None:
                        real_z.setdefault(obs.final.shape, []).append(z)
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
                        successors.append(next_z)
                        boards.append(after)
                        references.append(before)
                    if current is not None:
                        h, _ = e.advance(current[None], h, e.actions([action]))
                    if next_z is not None:
                        current = next_z
            if not states or sum(map(len, real_z.values())) < c.min_observations:
                return None
        buckets = {}
        for i, board in enumerate(boards):
            buckets.setdefault(board.shape, []).append(i)
        prediction_loss = torch.zeros((), device=e.device)
        visual_losses = [0., 0., 0.]
        overlap_losses = [0., 0., 0.]
        backward_values, backward_gradients = [], []
        for shape, indices in buckets.items():
            with e.autocast():
                predicted_z, _, delta = e.predict(torch.stack([states[i] for i in indices]),
                    torch.stack([memories[i] for i in indices]), e.actions([actions[i] for i in indices]))
            prediction_loss = prediction_loss + len(indices) / len(states) * F.mse_loss(
                delta.float(), torch.stack([targets[i] for i in indices]).float())
            real_successors = torch.stack([successors[i] for i in indices])
            # Each decoder microbatch releases its graph; accumulated latent
            # gradients are explicitly propagated through BOTH source branches.
            decoder_inputs = torch.cat((predicted_z, real_successors)).detach().requires_grad_(True)
            for branch in (0, 1):
                for start in range(0, len(indices), c.frame_chunk):
                    selected = indices[start:start + c.frame_chunk]
                    actual = torch.tensor(np.stack([boards[i] for i in selected]), device=e.device, dtype=torch.long)
                    before = torch.tensor(np.stack([references[i] for i in selected]), device=e.device, dtype=torch.long)
                    offset = branch * len(indices) + start
                    with e.autocast():
                        logits = e.decode(decoder_inputs[offset:offset + len(selected)], *shape,
                                          checkpoint_blocks=True)
                    part_loss = visual_delta_loss(logits, before, actual, c.visual_balance, c.color_balance_power).sum() / len(boards)
                    overlap = c.delta_overlap_weight * soft_delta_error(logits, before, actual).sum() / len(boards)
                    (part_loss + overlap).backward()
                    visual_losses[branch] += float(part_loss.detach())
                    overlap_losses[branch] += float(overlap.detach())
                    if branch == 1 and c.local_reconstruction_weight:
                        with e.autocast():
                            local_logits = e.decode(decoder_inputs[offset:offset + len(selected)], *shape, local=True)
                        local_loss = c.local_reconstruction_weight * visual_delta_loss(
                            local_logits, before, actual, c.visual_balance, c.color_balance_power).sum() / len(boards)
                        local_overlap = c.local_reconstruction_weight * c.delta_overlap_weight * soft_delta_error(
                            local_logits, before, actual).sum() / len(boards)
                        (local_loss + local_overlap).backward()
                        visual_losses[2] += float(local_loss.detach())
                        overlap_losses[2] += float(local_overlap.detach())
            backward_values.extend((predicted_z, real_successors))
            backward_gradients.extend(decoder_inputs.grad.chunk(2))
        # Match positions only within the same native canvas geometry. A single
        # observation cannot establish diversity, regardless of patch count.
        eligible = [zs for zs in real_z.values() if len(zs) >= 2]
        count = sum(map(len, eligible))
        regularization = sum(len(zs) * sigreg(torch.stack(zs), self.model.config.sigreg_slices)
                             for zs in eligible) / count if count else prediction_loss * 0
        variance = sum(len(zs) * variance_floor(torch.stack(zs)) for zs in eligible) / count if count else prediction_loss * 0
        loss = (1 - c.sigreg_weight) * prediction_loss + c.sigreg_weight * regularization + c.variance_weight * variance
        torch.autograd.backward((loss, *backward_values), (None, *backward_gradients))
        # Validate both gradient sets before either optimizer mutates weights.
        norms = [torch.nn.utils.clip_grad_norm_(p, 1., error_if_nonfinite=True)
                 for p in (self.primary, self.visual)]
        for opt in self.optimizers:
            opt.step()
        return {'loss': float(loss.detach()) + sum(visual_losses) + sum(overlap_losses), 'latent_mse': float(prediction_loss.detach()),
                'sigreg': float(regularization.detach()), 'decoder_ce': visual_losses[0],
                'variance_floor': float(variance.detach()),
                'reconstruction_ce': visual_losses[1],
                'local_reconstruction_ce': visual_losses[2],
                'decoder_overlap': overlap_losses[0],
                'reconstruction_overlap': overlap_losses[1],
                'local_reconstruction_overlap': overlap_losses[2],
                'gradient_norm': float(norms[0]), 'samples': len(states)}
