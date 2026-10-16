"""Joint latent dynamics and visual-delta learning from real SDK experience."""
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import partial
import itertools
import random
import weakref

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .networks import sigreg, variance_floor
from .metrics import visual_delta, delta_counts


@dataclass
class State:
    z: torch.Tensor | None
    h: torch.Tensor
    geometry: tuple[int, int] | None = None


class _CompiledDynamics:
    """Independent compiled caches for differentiable rollouts and inference.

    Dynamo specializes on grad mode, leaf requires_grad, singleton batches and
    convolution batch families. Those legitimate variants exceed its default
    per-frame limit when learning and MPC share a single compiled forward.
    Separate entry frames retain the normal limit and the same parameters.
    """
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.with_grad = torch.compile(self._with_grad, fullgraph=True, dynamic=True)
        self.without_grad = torch.compile(self._without_grad, fullgraph=True, dynamic=True)

    def _with_grad(self, z, h, action, height, width):
        return self.dynamics(z, h, action, height, width)

    def _without_grad(self, z, h, action, height, width):
        return self.dynamics(z, h, action, height, width)

    def __call__(self, z, h, action, height=None, width=None):
        # Selected, expanded and stacked states have the same semantics. Avoid
        # extra compiled variants solely for the caller's tensor layout.
        function = self.with_grad if torch.is_grad_enabled() else self.without_grad
        return function(z.contiguous(), h.contiguous(), action.contiguous(), height, width)


class _CompiledDecoder:
    """Separate legitimate readout/gradient modes without raising cache limits."""
    def __init__(self, decoder, local=False):
        self.decoder, self.local = decoder, local
        self.functions = {key: torch.compile(function, fullgraph=True, dynamic=True)
            for key, function in (( (True, False), self._context_grad),
                                  ( (False, False), self._context_eval),
                                  ( (True, True), self._local_grad),
                                  ( (False, True), self._local_eval))}

    def _context_grad(self, z, height, width, checkpoint_blocks):
        return self.decoder(z, height, width, checkpoint_blocks=checkpoint_blocks, local=False)

    def _context_eval(self, z, height, width, checkpoint_blocks):
        return self.decoder(z, height, width, checkpoint_blocks=checkpoint_blocks, local=False)

    def _local_grad(self, z, height, width, checkpoint_blocks):
        return self.decoder(z, height, width, checkpoint_blocks=checkpoint_blocks, local=True)

    def _local_eval(self, z, height, width, checkpoint_blocks):
        return self.decoder(z, height, width, checkpoint_blocks=checkpoint_blocks, local=True)

    def __call__(self, z, height, width, checkpoint_blocks=False, local=None):
        function = self.functions[torch.is_grad_enabled(), self.local if local is None else local]
        return function(z.contiguous(), height, width, checkpoint_blocks)


class Engine:
    def __init__(self, model, config, compile_model=False, *, fp8=False, optimizer_bits=32):
        self.model, self.config = model, config
        self.device = next(model.parameters()).device
        self.optimizer_bits = optimizer_bits
        self.fp8_layers = 0
        self._latent_cache_epoch = 0
        self._latent_caches = weakref.WeakSet()
        if fp8:
            if self.device.type != 'cuda' or torch.cuda.get_device_capability(self.device) < (8, 9):
                raise ValueError('FP8 requires CUDA compute capability >= 8.9')
            if not compile_model:
                raise ValueError('FP8 training requires torch.compile')
            from torchao.float8 import Float8LinearConfig, convert_to_float8_training
            def eligible(module, name):
                return (isinstance(module, torch.nn.Linear) and '.blocks.' in name
                        and min(module.in_features, module.out_features) >= 128
                        and module.in_features % 16 == 0 and module.out_features % 16 == 0)
            self.fp8_layers = sum(eligible(m, n) for n, m in model.named_modules())
            if not self.fp8_layers:
                raise ValueError('No large Transformer linears eligible for FP8')
            convert_to_float8_training(model, module_filter_fn=eligible,
                config=Float8LinearConfig(pad_inner_dim=True))
        self.amp = False
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                self.amp = torch.cuda.is_bf16_supported()
        self.spatial = model.encoder.spatial
        self.aggregate = model.encoder.aggregate
        self.predict = model.dynamics.forward
        self.advance = model.dynamics.advance
        self.decode = partial(model.decoder.forward, local=config.decoder_readout == 'local')
        if compile_model:
            # Keep original parameter names and checkpoint objects. Compilation
            # failure is an error, never an unreported eager fallback.
            self.spatial, self.aggregate, self.advance = [
                torch.compile(f, fullgraph=True, dynamic=True) for f in
                (self.spatial, self.aggregate, self.advance)]
            self.predict = _CompiledDynamics(model.dynamics)
            self.decode = _CompiledDecoder(model.decoder, config.decoder_readout == 'local')

    def autocast(self):
        return torch.autocast('cuda', dtype=torch.bfloat16) if self.amp else nullcontext()

    def actions(self, actions):
        return torch.tensor([a.features() for a in actions], device=self.device)

    def encode(self, observations, training=False, prepared=None):
        return self.model.encode(observations, self.spatial, self.config.frame_chunk,
                                 checkpoint_blocks=training, prepared=prepared, aggregate=self.aggregate)

    def invalidate_latent_cache(self):
        """Call when replacing encoder weights/configuration or changing phase."""
        self._latent_cache_epoch += 1
        for cache in self._latent_caches:
            cache.clear()

    @torch.no_grad()
    def encode_frozen_episode(self, episode):
        """Cache z only. Recurrent h is always recomputed with current dynamics."""
        if not self.config.freeze_representation:
            raise ValueError('Learned latent caches require a frozen encoder')
        return self.encode_detached_episode(episode)

    @torch.no_grad()
    def encode_detached_episode(self, episode):
        """Inference cache, invalidated by ANY encoder parameter update.

        Online training may update the encoder between calls. Cached tensors
        are detached and must never be used as a differentiable encoder pass.
        """
        if not self.config.cache_frozen_latents or self.config.burn_in_steps:
            return self.encode(episode.observations, prepared=episode.prepared)
        cache = episode.latent_cache
        self._latent_caches.add(cache)
        # Parameter versions also catch in-place checkpoint loads and external
        # weight copies between calls, even if explicit invalidation was missed.
        token = (id(self), self._latent_cache_epoch, self.amp, str(self.device),
                 tuple(p._version for p in self.model.encoder.parameters()))
        if cache.token != token:
            cache.clear()
            cache.token = token
        prefix = cache.values[:len(episode.observations)]
        missing = self.encode(episode.observations[len(prefix):],
            prepared=episode.prepared[len(prefix):]) if len(prefix) < len(episode.observations) else []
        owner = episode.replay_owner() if episode.replay_owner else None
        used = sum(ep.device_bytes for ep in owner.episodes) if owner else episode.device_bytes
        limit = owner.byte_limit if owner else self.config.replay_bytes
        extra = sum(z.numel() * z.element_size() for z in missing if z is not None)
        if used + extra <= limit:
            cache.values.extend(missing)
        return prefix + missing

    @torch.no_grad()
    def initial(self, observation):
        self.model.eval()
        with self.autocast():
            z = self.encode([observation])[0]
        return State(z, torch.zeros(1, self.model.config.memory, device=self.device),
                     observation.final.shape if observation.final is not None else None)

    @torch.no_grad()
    def prediction(self, state, action, reference):
        self.model.eval()
        if state.z is None or reference is None:
            return None
        with self.autocast():
            geometry = state.geometry or reference.shape
            z, _, delta = self.predict(state.z[None], state.h, self.actions([action]), *geometry)
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
        return State(z if z is not None else state.z, h.float(),
                     observation.final.shape if z is not None else state.geometry)

    @torch.no_grad()
    def rebuild(self, episode):
        """Re-encode the live prefix after updates; recurrent memory cannot go stale."""
        self.model.eval()
        with self.autocast():
            latents = self.encode(episode.observations)
            state = State(latents[0], torch.zeros(1, self.model.config.memory, device=self.device),
                          episode.observations[0].final.shape if latents[0] is not None else None)
            for action, z, obs in zip(episode.actions, latents[1:], episode.observations[1:]):
                if state.z is not None:
                    state.h, _ = self.advance(state.z[None], state.h, self.actions([action]))
                if z is not None:
                    state.z = z
                    state.geometry = obs.final.shape
        return state


def optimizer(parameters, learning_rate, cuda, bits=32):
    parameters = list(parameters)
    sources = parameters if parameters and isinstance(parameters[0], dict) else [{'params': parameters}]
    groups = []
    for source in sources:
        params = list(source['params'])
        for decay in (True, False):
            groups.append(dict(source, params=[p for p in params if (p.ndim >= 2) == decay],
                               weight_decay=.01 if decay else 0.))
    if bits == 8:
        if not cuda:
            raise ValueError('8-bit AdamW requires CUDA; use --optimizer-bits 32 on CPU')
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit(groups, lr=learning_rate, betas=(.9, .95), eps=1e-8)
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(.9, .95), eps=1e-8, fused=cuda)


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


def change_union_delta_loss(color_logits, reference, actual):
    """Per-image target NLL over real OR predicted changes, with no background dilution.

    The argmax mask is discrete supervision selection; gradients flow through
    target log probabilities only. Correctly retained pixels contribute zero,
    including a differentiable zero for images whose union is empty.
    """
    union = (actual != reference) | (color_logits.detach().argmax(-1) != reference)
    nll = -color_logits.float().log_softmax(-1).gather(-1, actual[..., None]).squeeze(-1)
    return torch.where(union, nll, 0.).sum((1, 2)) / union.sum((1, 2)).clamp_min(1)


def visual_delta_loss(color_logits, reference, actual, false_change_weight=1., *, mode='balanced'):
    """Per-image NLL with balanced, hard-union or smooth static focal weights."""
    if mode == 'union':
        # This objective weights every union pixel equally, including false changes.
        return change_union_delta_loss(color_logits, reference, actual)
    if mode in {'focal', 'recall_focal', 'recall_focal_capped'}:
        # T=1, gamma=2. Both the mask and denominator depend only on real
        # changes. Keep gradients through the static weights: no argmax/detach.
        logp = color_logits.float().log_softmax(-1).gather(-1, actual[..., None]).squeeze(-1)
        changed = actual != reference
        static_weight = false_change_weight * (-logp.expm1()).square()
        weights = torch.where(changed, 1. if mode == 'focal' else 2., static_weight)
        # Sparse timer-only changes must not amplify the entire image's errors
        # by an arbitrarily small denominator. Keep old objectives unchanged.
        floor = 32 if mode == 'recall_focal_capped' else 1
        return (-weights * logp).sum((1, 2)) / changed.sum((1, 2)).clamp_min(floor)
    if mode != 'balanced':
        raise ValueError(f'Invalid delta NLL mode: {mode}')
    changed = actual != reference
    target = torch.where(changed, actual, 16)
    logp = delta_log_probabilities(color_logits, reference)
    nll = -logp.gather(-1, target[..., None]).squeeze(-1)

    def mean_where(mask):
        return (nll * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)

    changed_count = changed.sum((1, 2))
    return torch.where((changed_count > 0) & (changed_count < actual.shape[1] * actual.shape[2]),
                       .5 * (mean_where(changed) + false_change_weight * mean_where(~changed)),
                       torch.where(changed_count == 0, false_change_weight, 1.) * mean_where(torch.ones_like(changed)))


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


@torch.no_grad()
def evaluate_rollouts(engine, episode, horizons=(1, 2, 4), seed=0):
    """Frozen real-action rollouts; diagnostics never replace one-step scores.

    Each anchor starts from its real posterior memory. Subsequent steps consume
    only predicted latents and memory. Identity and action permutation use the
    exact same anchors and final targets. Both readouts see those same latents.
    """
    if len(episode.observations) != len(episode.actions) + 1:
        raise ValueError('Rollout diagnostics require aligned observations and actions')
    if any(obs.terminal for obs in episode.observations[:-1]) or any(
            after.starts_new_level(before) for before, after in
            zip(episode.observations, episode.observations[1:])):
        raise ValueError('Rollout diagnostics require a single level episode')
    if any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError('Rollout horizons must be positive integers')
    was_training = engine.model.training
    engine.model.eval()
    try:
        with engine.autocast():
            zs = engine.encode(episode.observations, prepared=episode.prepared)
            h = torch.zeros(1, engine.model.config.memory, device=engine.device)
            memories, current = [], zs[0]
            action_features = engine.actions(episode.actions)
            for t in range(len(episode.actions)):
                memories.append(h)
                if current is not None:
                    h, _ = engine.advance(current[None], h, action_features[t:t + 1])
                if zs[t + 1] is not None:
                    current = zs[t + 1]
            shuffled = list(episode.actions)
            random.Random(seed).shuffle(shuffled)
            shuffled_features = engine.actions(shuffled)
            result = {}
            for horizon in horizons:
                samples, latent, identity, shuffled_mse, sensitivity = 0, 0., 0., 0., 0.
                readouts = {name: {'prediction_wrong': 0, 'prediction_union': 0,
                                   'reconstruction_wrong': 0, 'reconstruction_union': 0}
                            for name in ('contextual', 'local')}
                strata = {kind: {'samples': 0, 'readouts': {
                    name: dict(counts) for name, counts in readouts.items()}}
                    for kind in ('changed', 'unchanged')}
                for start in range(len(episode.actions) - horizon + 1):
                    window = episode.observations[start:start + horizon + 1]
                    if any(obs.final is None for obs in window):
                        continue
                    shape = window[0].final.shape
                    if any(obs.final.shape != shape for obs in window):
                        continue
                    predicted, counterfactual = zs[start][None], zs[start][None]
                    predicted_h, counterfactual_h = memories[start], memories[start]
                    for t in range(start, start + horizon):
                        predicted, predicted_h, _ = engine.predict(
                            predicted, predicted_h, action_features[t:t + 1], *shape)
                        counterfactual, counterfactual_h, _ = engine.predict(
                            counterfactual, counterfactual_h, shuffled_features[t:t + 1], *shape)
                    actual_z = zs[start + horizon][None]
                    latent += float(F.mse_loss(predicted.float(), actual_z.float()))
                    identity += float(F.mse_loss(zs[start][None].float(), actual_z.float()))
                    shuffled_mse += float(F.mse_loss(counterfactual.float(), actual_z.float()))
                    sensitivity += float(F.mse_loss(counterfactual.float(), predicted.float()))
                    reference, actual = window[0].final, window[-1].final
                    stratum = strata['changed' if np.any(reference != actual) else 'unchanged']
                    stratum['samples'] += 1
                    for name, counts in readouts.items():
                        decoded = engine.decode(torch.cat((predicted, actual_z)), *shape,
                                                local=name == 'local').argmax(-1).cpu().numpy()
                        for branch, board in zip(('prediction', 'reconstruction'), decoded):
                            wrong, union, _ = delta_counts(reference, actual, board)
                            counts[branch + '_wrong'] += wrong
                            counts[branch + '_union'] += union
                            stratum['readouts'][name][branch + '_wrong'] += wrong
                            stratum['readouts'][name][branch + '_union'] += union
                    samples += 1
                result[str(horizon)] = {'samples': samples,
                    'latent_mse': latent / samples if samples else None,
                    'identity_latent_mse': identity / samples if samples else None,
                    'shuffled_action_latent_mse': shuffled_mse / samples if samples else None,
                    'action_sensitivity_mse': sensitivity / samples if samples else None,
                    'readouts': {name: counts | {
                        branch + '_delta_error': counts[branch + '_wrong'] / max(1, counts[branch + '_union'])
                        if samples else None for branch in ('prediction', 'reconstruction')}
                        for name, counts in readouts.items()}}
                # A zero-error unchanged image is useful consistency evidence,
                # but is not evidence of predicting an actual visual change.
                result[str(horizon)]['strata'] = strata
            return result
    finally:
        engine.model.train(was_training)


class Learner:
    def __init__(self, engine):
        self.engine = engine
        self.model = engine.model
        self.set_representation_frozen(engine.config.freeze_representation)
        self.primary = list(itertools.chain(self.model.encoder.parameters(), self.model.dynamics.parameters()))
        self.visual = list(self.model.decoder.parameters())
        self.optimizers = [optimizer(p, engine.config.learning_rate, engine.device.type == 'cuda', engine.optimizer_bits)
                           for p in (self.primary, self.visual)]

    def set_representation_frozen(self, frozen):
        """Call between updates; keep parameter identity and optimizer moments."""
        self.engine.config = replace(self.engine.config, freeze_representation=frozen)
        self.engine.invalidate_latent_cache()
        for module in (self.model.encoder, self.model.decoder):
            for parameter in module.parameters():
                parameter.grad = None
            module.requires_grad_(not frozen)

    def update(self, episodes):
        if not episodes:
            return None
        if any(after.starts_new_level(before) for episode in episodes
               for before, after in zip(episode.observations, episode.observations[1:])):
            raise ValueError('Training prefixes must not cross a native level boundary')
        if any(len(ep.observations) != len(ep.actions) + 1 or
               any(obs.terminal for obs in ep.observations[:-1]) for ep in episodes):
            raise ValueError('Training episodes require aligned actions and cannot continue after a terminal')
        e, c = self.engine, self.engine.config
        self.model.train()
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
        real_z, states, memories, actions, targets, boards, references, successors = {}, [], [], [], [], [], [], []
        rollout_pairs = []
        with e.autocast():
            for episode in episodes:
                burn_in = 0
                if c.burn_in_steps:
                    # Optional truncated history, explicitly approximate: start
                    # memory at zero, warm it on real inputs without gradients,
                    # then retain the graph for the supervised anchors only.
                    anchor = max(0, len(episode.actions) - c.loss_steps - int(c.rollout_weight > 0 or c.rollout_visual_weight > 0))
                    start = max(0, anchor - c.burn_in_steps)
                    burn_in = anchor - start
                    episode = replace(episode, observations=episode.observations[start:],
                        actions=episode.actions[start:], prepared=episode.prepared[start:] if episode.prepared else [])
                # Keep episode outputs, not all encoder intermediates across
                # every sampled prefix. Recompute one episode at a time during
                # backward; all branches still receive the same joint gradients.
                if c.freeze_representation:
                    with torch.no_grad():
                        zs = e.encode_frozen_episode(episode)
                else:
                    with torch.no_grad():
                        warm_z = e.encode(episode.observations[:burn_in],
                            prepared=episode.prepared[:burn_in]) if burn_in else []
                    zs = warm_z + activation_checkpoint(e.encode, episode.observations[burn_in:],
                        training=True, prepared=episode.prepared[burn_in:], use_reentrant=False)
                for obs, z in zip(episode.observations[burn_in:], zs[burn_in:]):
                    if z is not None:
                        real_z.setdefault(obs.final.shape, []).append(z)
                # Match recurrent states' autograd metadata from the first step.
                # This constant leaf is never optimized; parameter gradients are
                # unchanged, but compilation need not specialize on its flag.
                h = torch.zeros(1, self.model.config.memory, device=e.device, requires_grad=True)
                current = zs[0]
                episode_actions = (torch.stack([item[2] for item in episode.prepared[1:]])
                                   if episode.prepared else e.actions(episode.actions))
                first_loss = max(0, len(episode.actions) - c.loss_steps)
                history_h = []
                for t, action in enumerate(episode.actions):
                    history_h.append(h)
                    next_z = zs[t + 1]
                    before, after = episode.observations[t].final, episode.observations[t + 1].final
                    if (t >= first_loss and current is not None and next_z is not None and
                            before is not None and after is not None and before.shape == after.shape):
                        states.append(current)
                        memories.append(h[0])
                        actions.append(episode_actions[t])
                        targets.append(next_z - current)
                        successors.append(next_z)
                        boards.append(episode.prepared[t + 1][0][-1] if episode.prepared else
                                      torch.tensor(np.array(after, copy=True), device=e.device, dtype=torch.long))
                        references.append(episode.prepared[t][0][-1] if episode.prepared else
                                          torch.tensor(np.array(before, copy=True), device=e.device, dtype=torch.long))
                        # loss_steps counts supervised ENDPOINTS. The last one
                        # still has a two-step rollout when loss_steps == 1.
                        if (c.rollout_weight or c.rollout_visual_weight) and t >= 1:
                            anchor = episode.observations[t - 1].final
                            if anchor is not None and anchor.shape == before.shape and zs[t - 1] is not None:
                                rollout_pairs.append((zs[t - 1], history_h[t - 1][0],
                                    episode_actions[t - 1], episode_actions[t], next_z,
                                    torch.tensor(np.array(anchor, copy=True), device=e.device, dtype=torch.long),
                                    boards[-1], before.shape))
                    if current is not None:
                        if t < burn_in:
                            with torch.no_grad():
                                h, _ = e.advance(current[None], h, episode_actions[t:t + 1])
                        else:
                            # Compiled BF16 recurrence otherwise saves a fresh
                            # cast of its large matrices for every history step.
                            # Recompute those intermediates during backward;
                            # preserve the complete recurrent gradient chain.
                            h, _ = activation_checkpoint(e.advance, current[None], h,
                                episode_actions[t:t + 1], use_reentrant=False)
                        if t + 1 == burn_in:
                            h = h.detach().requires_grad_(True)
                    if next_z is not None:
                        current = next_z
            if not states:
                return None
        buckets = {}
        for i, board in enumerate(boards):
            buckets.setdefault(board.shape, []).append(i)
        prediction_loss = torch.zeros((), device=e.device)
        rollout_loss = torch.zeros((), device=e.device)
        rollout_visual_loss = 0.
        rollout_backward_values, rollout_backward_gradients = [], []
        ordinary_rollout_visual = torch.zeros((), device=e.device)
        rollout_buckets = {}
        for pair in rollout_pairs:
            rollout_buckets.setdefault(pair[-1], []).append(pair)
        for shape, pairs in rollout_buckets.items():
            with e.autocast():
                z1, h1, _ = e.predict(torch.stack([p[0] for p in pairs]),
                    torch.stack([p[1] for p in pairs]), torch.stack([p[2] for p in pairs]), *shape)
                # Both successor state and memory come from the prediction;
                # no real intermediate latent is injected and no gradient cut.
                z2, _, _ = e.predict(z1, h1, torch.stack([p[3] for p in pairs]), *shape)
            rollout_loss = rollout_loss + len(pairs) / len(rollout_pairs) * F.mse_loss(
                z2.float(), torch.stack([p[4] for p in pairs]).float())
            if c.rollout_visual_weight:
                decoder_input = z2.detach().requires_grad_(True) if c.backward_mode == 'fragmented' else z2
                for start in range(0, len(pairs), c.frame_chunk):
                    selected = pairs[start:start + c.frame_chunk]
                    with e.autocast():
                        logits = e.decode(decoder_input[start:start + len(selected)], *shape,
                                          checkpoint_blocks=True)
                    # Net delta from the REAL rollout anchor to its REAL endpoint.
                    # No intermediate observation enters the imagined dynamics.
                    before, actual = (torch.stack([p[i] for p in selected]) for i in (5, 6))
                    part = c.rollout_visual_weight * (visual_delta_loss(
                        logits, before, actual, c.false_change_weight, mode=c.prediction_delta_loss)
                        + c.delta_overlap_weight * soft_delta_error(logits, before, actual)).sum() / len(rollout_pairs)
                    if c.backward_mode == 'fragmented':
                        part.backward()
                    else:
                        ordinary_rollout_visual = ordinary_rollout_visual + part
                    rollout_visual_loss += float(part.detach())
                if c.backward_mode == 'fragmented':
                    rollout_backward_values.append(z2)
                    rollout_backward_gradients.append(decoder_input.grad)
        visual_losses = [0., 0., 0.]
        overlap_losses = [0., 0., 0.]
        backward_values, backward_gradients = rollout_backward_values, rollout_backward_gradients
        ordinary_visual_loss = ordinary_rollout_visual
        for shape, indices in buckets.items():
            with e.autocast():
                predicted_z, _, delta = e.predict(torch.stack([states[i] for i in indices]),
                    torch.stack([memories[i] for i in indices]), torch.stack([actions[i] for i in indices]), *shape)
            prediction_loss = prediction_loss + len(indices) / len(states) * F.mse_loss(
                delta.float(), torch.stack([targets[i] for i in indices]).float())
            real_successors = torch.stack([successors[i] for i in indices])
            # Each decoder microbatch releases its graph; accumulated latent
            # gradients are explicitly propagated through BOTH source branches.
            source_values = (predicted_z,) if c.freeze_representation else (predicted_z, real_successors)
            decoder_inputs = torch.cat(source_values)
            if c.backward_mode == 'fragmented':
                decoder_inputs = decoder_inputs.detach().requires_grad_(True)
            for branch in range(len(source_values)):
                for start in range(0, len(indices), c.frame_chunk):
                    selected = indices[start:start + c.frame_chunk]
                    actual = torch.stack([boards[i] for i in selected])
                    before = torch.stack([references[i] for i in selected])
                    offset = branch * len(indices) + start
                    with e.autocast():
                        logits = e.decode(decoder_inputs[offset:offset + len(selected)], *shape,
                                          checkpoint_blocks=True)
                    part_loss = visual_delta_loss(logits, before, actual,
                        c.false_change_weight if branch == 0 else 1.,
                        mode=c.prediction_delta_loss if branch == 0 else 'balanced').sum() / len(boards)
                    overlap = c.delta_overlap_weight * soft_delta_error(logits, before, actual).sum() / len(boards)
                    if c.backward_mode == 'fragmented':
                        (part_loss + overlap).backward()
                    else:
                        ordinary_visual_loss = ordinary_visual_loss + part_loss + overlap
                    visual_losses[branch] += float(part_loss.detach())
                    overlap_losses[branch] += float(overlap.detach())
                    if branch == 1 and c.local_reconstruction_weight and c.decoder_readout != 'local':
                        with e.autocast():
                            local_logits = e.decode(decoder_inputs[offset:offset + len(selected)], *shape, local=True)
                        local_loss = c.local_reconstruction_weight * visual_delta_loss(
                            local_logits, before, actual).sum() / len(boards)
                        local_overlap = c.local_reconstruction_weight * c.delta_overlap_weight * soft_delta_error(
                            local_logits, before, actual).sum() / len(boards)
                        if c.backward_mode == 'fragmented':
                            (local_loss + local_overlap).backward()
                        else:
                            ordinary_visual_loss = ordinary_visual_loss + local_loss + local_overlap
                        visual_losses[2] += float(local_loss.detach())
                        overlap_losses[2] += float(local_overlap.detach())
            if c.backward_mode == 'fragmented':
                backward_values.extend(source_values)
                backward_gradients.extend(decoder_inputs.grad.chunk(len(source_values)))
        # Match positions only within the same native canvas geometry. A single
        # observation cannot establish diversity, regardless of patch count.
        eligible = [] if c.freeze_representation else [zs for zs in real_z.values() if len(zs) >= c.min_observations]
        count = sum(map(len, eligible))
        regularization = sum(len(zs) * sigreg(torch.stack(zs), self.model.config.sigreg_slices)
                             for zs in eligible) / count if count else prediction_loss * 0
        variance = sum(len(zs) * variance_floor(torch.stack(zs)) for zs in eligible) / count if count else prediction_loss * 0
        loss = ((1 - c.sigreg_weight) * prediction_loss + c.sigreg_weight * regularization
                + c.variance_weight * variance + c.rollout_weight * rollout_loss)
        torch.autograd.backward((loss + ordinary_visual_loss, *backward_values), (None, *backward_gradients))
        # Validate both gradient sets before either optimizer mutates weights.
        norms = [torch.nn.utils.clip_grad_norm_(p, 1., error_if_nonfinite=True)
                 for p in (self.primary, self.visual)]
        for opt in self.optimizers[:1] if c.freeze_representation else self.optimizers:
            opt.step()
        return {'loss': float(loss.detach()) + sum(visual_losses) + sum(overlap_losses) + rollout_visual_loss, 'latent_mse': float(prediction_loss.detach()),
                'sigreg': float(regularization.detach()), 'decoder_ce': visual_losses[0],
                'variance_floor': float(variance.detach()),
                'reconstruction_ce': None if c.freeze_representation else visual_losses[1],
                'local_reconstruction_ce': None if c.freeze_representation else visual_losses[2],
                'decoder_overlap': overlap_losses[0],
                'reconstruction_overlap': None if c.freeze_representation else overlap_losses[1],
                'local_reconstruction_overlap': None if c.freeze_representation else overlap_losses[2],
                'rollout_mse': float(rollout_loss.detach()), 'rollout_visual': rollout_visual_loss,
                'rollout_samples': len(rollout_pairs),
                'gradient_norm': float(norms[0]), 'samples': len(states)}
