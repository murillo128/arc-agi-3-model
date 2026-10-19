"""Causal, real game sessions and batched truncated recurrent learning.

The session owns context; a level/reset only replaces the board. Encoded frames
are disposable within a batch. Every optimizer step rebuilds its prefix memory
with current parameters, and gradients span the selected context and block,
never their future.
"""
from collections import defaultdict
from dataclasses import dataclass, field
import random

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .config import SequenceConfig
from .data import Action, Observation, random_action
from .environment import training_boundary
from .learning import optimizer, visual_delta_loss
from .online_learning import trainable


@dataclass(frozen=True)
class Transition:
    before: Observation
    action: Action
    after: Observation
    reset_before: bool = False

    @property
    def supervised(self):
        return (self.action.id != 0 and self.before.final is not None and
                self.after.final is not None and not self.before.terminal and
                not self.after.starts_new_level(self.before) and
                self.before.final.shape == self.after.final.shape)


@dataclass
class GameSession:
    game: str
    seed: int
    transitions: list[Transition] = field(default_factory=list)
    environment_actions: int = 0
    end_reason: str = 'budget'


def collect_session(factory, levels, seed, config):
    """Random public actions. Synthetic isolation is a training-only capability.

    Native SDK sessions start at level one and progress naturally; transitions
    entering excluded levels never enter the session, including their targets.
    """
    if not levels or len(set(levels)) != len(levels):
        raise ValueError('Session levels must be nonempty and unique')
    native = bool(factory.reserve_final_level)
    if native and levels != list(range(1, max(levels) + 1)):
        raise ValueError('Native sequence levels must be the contiguous allowed prefix from 1')
    session = GameSession(factory.game_id, seed)
    rng = random.Random(seed)
    env, attempts = None, 0
    boundary = False
    for index in range(config.session_steps):
        slot = (index // config.level_steps) % len(levels)
        if env is None or env.observation.terminal or (not native and index % config.level_steps == 0):
            env = factory.make(seed + attempts * 1_000_000, None if native else levels[slot])
            attempts += 1
            boundary = bool(session.transitions)
        before = env.observation
        if native and (training_boundary(factory, before) or before.levels_completed + 1 not in levels):
            session.end_reason = 'reserved_level'
            break
        if before.state == 'NOT_PLAYED':
            before = env.step(Action(0)).level_initial()
            session.environment_actions += 1
            boundary = bool(session.transitions)
        action = random_action(before, rng)
        after = env.step(action)
        session.environment_actions += 1
        if native and (training_boundary(factory, after) or
                       (not after.terminal and after.levels_completed + 1 not in levels)):
            session.end_reason = 'reserved_level'
            break
        session.transitions.append(Transition(before, action, after, boundary))
        boundary = after.starts_new_level(before)
        if boundary:
            # Only the final new-board frame may enter the next level's encoder.
            env.observation = after.level_initial()
    return session


def schedule_games(games, batch_index, config):
    if not games or len(set(games)) != len(games):
        raise ValueError('Expected distinct configured games')
    return [games[(batch_index * config.batch_games + i) % len(games)]
            if config.game_schedule == 'interleaved' else games[batch_index % len(games)]
            for i in range(config.batch_games)]


def temporal_weight(index, config):
    return config.loss_start_weight + (1 - config.loss_start_weight) * min(
        1., index / config.loss_warmup_steps)


def loss_coefficients(sessions, start, stop, config):
    """Full-session normalization: a ramp must not cancel inside each block.

    Averaging the block objectives gives the equally weighted game objectives.
    The block factor preserves the optimizer's ordinary mean-loss scale.
    """
    weights = [[temporal_weight(t, config) if step.supervised else 0.
                for t, step in enumerate(row.transitions)] for row in sessions]
    active = sum(sum(row) > 0 for row in weights)
    blocks = (max((len(row) for row in weights), default=0) + config.block_steps - 1) // config.block_steps
    return [[blocks * w / (sum(row) * active) if sum(row) else 0. for w in row[start:stop]]
            for row in weights]


class EncodedSessions:
    def __init__(self, engine, sessions):
        self.sessions = sessions
        self.latents = {}
        observations = {}
        for session in sessions:
            for step in session.transitions:
                for obs in (step.before, step.after):
                    if obs.final is not None:
                        observations[id(obs)] = obs
        items = list(observations.items())
        # Do not materialize all one-hot animation frames in VRAM at once.
        with torch.no_grad(), engine.autocast():
            for offset in range(0, len(items), engine.config.frame_chunk):
                chunk = items[offset:offset + engine.config.frame_chunk]
                zs = engine.encode([obs for _, obs in chunk])
                self.latents.update((key, z.detach()) for (key, _), z in zip(chunk, zs))
        self.encoder_versions = tuple(p._version for p in engine.model.encoder.parameters())
        self.context = []
        for session in sessions:
            carry, geometry, row = None, None, []
            for step in session.transitions:
                if step.reset_before:
                    carry, geometry = None, None
                if step.before.final is not None:
                    carry, geometry = self.z(step.before), tuple(step.before.final.shape)
                row.append((carry, geometry))
                # An empty SDK response preserves visual belief, never becomes
                # an invented reconstruction/prediction target.
                if step.after.final is not None:
                    carry, geometry = self.z(step.after), tuple(step.after.final.shape)
            self.context.append(row)

    def check(self, engine):
        if self.encoder_versions != tuple(p._version for p in engine.model.encoder.parameters()):
            raise ValueError('Encoder changed: rebuild session latents')

    def z(self, observation):
        return self.latents.get(id(observation))


def memory_inputs(engine, encoded, t, memories, *, clear_history=False):
    """Batch by native geometry; different game memories never attend each other."""
    buckets = defaultdict(list)
    for i, session in enumerate(encoded.sessions):
        if t >= len(session.transitions):
            continue
        step = session.transitions[t]
        z, geometry = encoded.context[i][t]
        if z is not None:
            buckets[geometry].append((i, step, z))
    result = []
    for geometry, rows in buckets.items():
        indices, steps, zs = zip(*rows)
        z = torch.stack(zs)
        h = torch.cat([memories[i] for i in indices])
        if clear_history:
            h = torch.zeros_like(h)
        action = engine.actions([step.action for step in steps])
        if any(s.reset_before for s in steps):
            flags = torch.tensor([s.reset_before for s in steps], device=engine.device)[:, None]
            reset, _ = engine.advance(z, h, engine.actions([Action(0)] * len(rows)))
            h = torch.where(flags, reset, h)
        result.append((indices, steps, z, h, action, geometry))
    return result


class SequenceLearner:
    def __init__(self, engine, config=SequenceConfig()):
        self.engine, self.config = engine, config
        self.optimizer = optimizer(engine.model.dynamics.parameters(), engine.config.learning_rate,
                                   engine.device.type == 'cuda', engine.optimizer_bits)
        self.updates = 0

    def zero_memories(self, sessions):
        return [torch.zeros(1, self.engine.model.config.memory, device=self.engine.device) for _ in sessions]

    @torch.no_grad()
    def prefix(self, encoded, stop):
        e = self.engine
        memories = self.zero_memories(encoded.sessions)
        with e.autocast():
            for t in range(stop):
                for ids, _, z, h, action, _ in memory_inputs(e, encoded, t, memories):
                    next_h, _ = e.advance(z, h, action)
                    for i, value in zip(ids, next_h.split(1)):
                        memories[i] = value.float()
        return memories

    def _terms(self, z, h, action, target, before, actual):
        e = self.engine
        predicted, _, delta = e.predict(z, h, action, *before.shape[-2:])
        logits = e.decode(predicted, *before.shape[-2:], checkpoint_blocks=False)
        latent = (delta.float() - (target - z).float()).square().mean((1, 2))
        visual = visual_delta_loss(logits, before, actual, e.config.false_change_weight,
                                   mode=e.config.prediction_delta_loss)
        return torch.stack((latent, visual), -1)

    def block_loss(self, encoded, start, stop):
        """Differentiable real recurrence. No optimizer mutation: also a test seam."""
        e = self.engine
        encoded.check(e)
        coefficients = loss_coefficients(encoded.sessions, start, stop, self.config)
        context_start = max(0, start - self.config.context_steps)
        memories = self.prefix(encoded, context_start)
        terms = []
        with e.autocast():
            for t in range(context_start, stop):
                for ids, steps, z, h, action, geometry in memory_inputs(e, encoded, t, memories):
                    # The real result becomes z at the next step. No imagined
                    # result updates the observed carry; current loss is causal.
                    next_h, _ = e.advance(z, h, action)
                    for i, value in zip(ids, next_h.split(1)):
                        memories[i] = value.float()
                    if t < start:
                        continue  # Context receives gradients, not duplicate loss.
                    keep = [k for k, step in enumerate(steps) if step.supervised]
                    if not keep:
                        continue
                    selected = [steps[k] for k in keep]
                    target = torch.stack([encoded.z(s.after) for s in selected])
                    before = torch.as_tensor(np.stack([s.before.final for s in selected]), device=e.device).long()
                    actual = torch.as_tensor(np.stack([s.after.final for s in selected]), device=e.device).long()
                    # Whole readout checkpoint bounds activation memory across
                    # long sequences. Recompute remains differentiable through h.
                    values = checkpoint(self._terms, z[keep], h[keep], action[keep], target,
                                        before, actual, use_reentrant=False)
                    weight = values.new_tensor([coefficients[ids[k]][t-start] for k in keep])
                    terms.append((values * weight[:, None]).sum(0))
        return torch.stack(terms).sum(0) if terms else None

    def update(self, encoded, start, stop, *, telemetry=None, last=False):
        e = self.engine
        trainable(e, 'dynamics')
        self.optimizer.zero_grad(set_to_none=True)
        terms = self.block_loss(encoded, start, stop)
        if terms is None:
            trainable(e)
            return None
        terms.sum().backward()
        capture = telemetry.before_step('predictor', self.optimizer, start=start, stop=stop, last=last,
            context_start=max(0, start-self.config.context_steps)) if telemetry else None
        norm = torch.nn.utils.clip_grad_norm_(e.model.dynamics.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        if telemetry:
            telemetry.after_step(capture, norm, terms.detach().sum())
        self.updates += 1
        latent, visual = terms.detach().cpu().tolist()
        trainable(e)
        return dict(loss=latent + visual, latent=latent, visual=visual, grad_norm=float(norm),
                    examples=sum(s.supervised for row in encoded.sessions for s in row.transitions[start:stop]))


@torch.no_grad()
def evaluate_sessions(engine, sessions, *, clear_history=False, early_steps=32, reference_loss=False, diagnostics=None):
    """Fixed real trajectories; unweighted pixel-union errors, no optimizer."""
    engine.model.eval()
    encoded = EncodedSessions(engine, sessions)
    memories = [torch.zeros(1, engine.model.config.memory, device=engine.device) for _ in sessions]
    totals = defaultdict(lambda: [0] * 7)
    nll_sum, nll_count = 0., 0
    with engine.autocast():
        for t in range(max((len(s.transitions) for s in sessions), default=0)):
            for ids, steps, z, h, action, geometry in memory_inputs(
                    engine, encoded, t, memories, clear_history=clear_history):
                predicted, next_h, _ = engine.predict(z, h, action, *geometry)
                logits = engine.decode(predicted, *geometry)
                pixels = logits.argmax(-1)
                for k, (i, step, nh) in enumerate(zip(ids, steps, next_h.split(1))):
                    memories[i] = nh.float()
                    if not step.supervised:
                        continue
                    before = torch.as_tensor(np.array(step.before.final, copy=True), device=engine.device)
                    actual = torch.as_tensor(np.array(step.after.final, copy=True), device=engine.device)
                    if reference_loss:
                        # Fixed real targets and ordinary unweighted color NLL. Reuse the forward.
                        nll_sum += float(torch.nn.functional.cross_entropy(logits[k].float().movedim(-1, 0)[None], actual.long()[None]))
                        nll_count += 1
                    real, guessed = actual != before, pixels[k] != before
                    equal = pixels[k] == actual
                    union = real | guessed
                    # One transfer for counts; no extra model forward for the breakdown.
                    counts = torch.stack(((~equal & union).sum(), union.sum(),
                        (~real & guessed).sum(), (real & ~guessed).sum(),
                        (real & guessed & ~equal).sum(), (real & equal).sum())).cpu().tolist()
                    for key in ('all', 'early' if t < early_steps else 'late', sessions[i].game):
                        for j, value in enumerate((counts[0], counts[1], 1, *counts[2:])):
                            totals[key][j] += value
                    if diagnostics is not None:
                        diagnostics.observe(encoded, sessions[i], t, z[k], predicted[k], next_h[k], pixels[k])
    result = {}
    for key, values in totals.items():
        row = dict(zip(('wrong', 'union', 'transitions', 'invented', 'missed', 'color', 'correct'), values))
        row['delta_error'] = row['wrong']/row['union'] if row['union'] else None
        result[key] = row
    if reference_loss:
        result['reference'] = dict(nll_sum=nll_sum, transitions=nll_count)
    return result


def merge_evaluations(reports):
    counts = defaultdict(lambda: [0] * 7)
    nll_sum, nll_count = 0., 0
    for report in reports:
        for key, row in report.items():
            if key == 'reference':
                nll_sum += row['nll_sum']
                nll_count += row['transitions']
                continue
            for i, name in enumerate(('wrong', 'union', 'transitions', 'invented', 'missed', 'color', 'correct')):
                counts[key][i] += row[name]
    result = {key: dict(wrong=w, union=u, transitions=n, delta_error=w/u if u else None,
                        invented=iv, missed=mi, color=co, correct=ok)
              for key, (w, u, n, iv, mi, co, ok) in counts.items()}
    if nll_count:
        result['reference'] = dict(nll=nll_sum/nll_count, transitions=nll_count)
    return result
