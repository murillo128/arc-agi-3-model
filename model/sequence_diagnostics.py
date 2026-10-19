"""Bounded, read-only monitoring on fixed real sequences, with causal memory.

At most 32 selected transitions and 64 real latent observations per measurement.
The usual evaluation supplies the one-step prediction; only reconstruction and
up to two further imagined steps require extra forwards. No sampling RNG is used.
"""
import time
from types import SimpleNamespace

import numpy as np
import torch

from .latent_diagnostics import LatentProbe, summarize_latents
from .metrics import delta_counts


class SequenceDiagnostics:
    every = 4
    limit = 32

    def __init__(self, engine, sessions):
        self.engine = engine
        self.selected = {}
        self.episodes = {}
        # Bound cost independently of the configured number of games/sessions.
        candidates = []
        for session in sessions:
            episode = 0
            for t, step in enumerate(session.transitions):
                episode += int(step.reset_before)
                self.episodes[id(session), t] = episode
                if step.supervised:
                    candidates.append((id(session), t))
        indices = np.linspace(0, len(candidates)-1, min(self.limit, len(candidates)), dtype=int)
        self.selected = {candidates[i] for i in indices}
        self.references = {}
        self.reference_batch = None

    def start(self, batch):
        self.batch = batch
        self.seconds = 0.
        self.probes = {}
        self.reconstruction_wrong = self.reconstruction_pixels = 0
        self.rollout_wrong = self.rollout_union = self.rollouts = 0
        self.drift_delta = self.drift_scale = 0.
        self.drift_count = 0
        self.samples = {}
        self.measured = 0
        if self.reference_batch is None:
            self.reference_batch = batch

    @torch.no_grad()
    def observe(self, encoded, session, t, z, predicted, next_h, pixels):
        key = (id(session), t)
        if key not in self.selected:
            return
        begin = time.perf_counter()
        e, step = self.engine, session.transitions[t]
        before, actual = step.before.final, step.after.final
        geometry = tuple(before.shape)
        state = SimpleNamespace(geometry=geometry, z=z)
        after_state = SimpleNamespace(geometry=geometry, z=encoded.z(step.after))
        probe = self.probes.setdefault((id(session), self.episodes[key]), LatentProbe(e.model.config.patch))
        probe.sample(state)
        probe.sample(after_state)
        probe.transition(state, after_state, before, actual)
        current = z.detach().float().cpu()
        if key in self.references:
            reference = self.references[key]
            self.drift_delta += float((current-reference).square().sum())
            self.drift_scale += float(reference.square().sum())
            self.drift_count += current.numel()
        else:
            self.references[key] = current.clone()
        reconstruction = e.decode(z[None], *geometry).argmax(-1)[0].cpu().numpy()
        self.reconstruction_wrong += int((reconstruction != before).sum())
        self.reconstruction_pixels += before.size
        self.measured += 1
        predicted_pixels = pixels.cpu().numpy()
        # Only the latest bounded gallery is serialized, never images per history row.
        if session.game not in self.samples and len(self.samples) < 8:
            self.samples[session.game] = dict(game=session.game, seed=session.seed,
                level=step.level_number, step=t, action=step.action.id,
                previous=before.tolist(), actual=actual.tolist(), predicted=predicted_pixels.tolist(),
                reconstruction=reconstruction.tolist())
        # Actions come from the fixed real trajectory, latents from imagination.
        # Never cross resets, terminal states or level/geometry boundaries.
        chain = session.transitions[t:t+3]
        valid = len(chain) == 3 and all(s.supervised for s in chain)
        valid = valid and all(not b.reset_before and not a.after.terminal and
            np.array_equal(a.after.final, b.before.final) for a, b in zip(chain, chain[1:]))
        if valid:
            imagined, memory = predicted[None], next_h[None]
            for future in chain[1:]:
                imagined, memory, _ = e.predict(imagined, memory, e.actions([future.action]), *geometry)
            final = e.decode(imagined, *geometry).argmax(-1)[0].cpu().numpy()
            wrong, union, _ = delta_counts(before, chain[-1].after.final, final)
            self.rollout_wrong += wrong
            self.rollout_union += union
            self.rollouts += 1
        self.seconds += time.perf_counter()-begin

    def finish(self):
        begin = time.perf_counter()
        result = dict(batch=self.batch, sampled_transitions=self.measured, limit=self.limit, every=self.every,
            reconstruction_error=self.reconstruction_wrong/self.reconstruction_pixels if self.reconstruction_pixels else None,
            reconstruction_pixels=self.reconstruction_pixels,
            rollout_error=self.rollout_wrong/self.rollout_union if self.rollout_union else None,
            rollout_wrong=self.rollout_wrong, rollout_union=self.rollout_union, rollouts=self.rollouts,
            latent=summarize_latents(list(self.probes.values())),
            latent_drift=(self.drift_delta/self.drift_scale)**.5 if self.drift_count and self.drift_scale else None,
            drift_reference_batch=self.reference_batch)
        self.seconds += time.perf_counter()-begin
        return result, list(self.samples.values())
