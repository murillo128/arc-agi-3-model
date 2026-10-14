"""Frozen goal hypotheses proposed before a separate distance-only search.

Only causal real z/h and current legal actions enter either search. Decoding is
an optional inspection of retained tensors, never an input to action ranking.
"""
from dataclasses import dataclass
import math

import torch

from .planning import Planner, candidates


@dataclass(frozen=True)
class GoalConfig:
    proposal_horizon: int = 8
    proposal_beam: int = 4
    lifetime: int = 8
    horizon: int = 4
    beam: int = 4
    clicks: int = 8
    discount: float = .95

    def __post_init__(self):
        for name in ('proposal_horizon', 'proposal_beam', 'horizon', 'beam', 'clicks', 'lifetime'):
            value = getattr(self, name)
            maximum = 1024 if name == 'lifetime' else 64
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f'{name} must be an integer in [1, {maximum}]')
        if not math.isfinite(self.discount) or not 0 < self.discount <= 1:
            raise ValueError('discount must lie in (0, 1]')


@dataclass(frozen=True)
class Goal:
    z: torch.Tensor
    geometry: tuple[int, int]
    proposal_score: float
    created_position: int


def distance(z, target):
    if z.shape[1:] != target.shape or z.device != target.device:
        raise ValueError('Goal and state latent geometry/device must match')
    if not torch.isfinite(z).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite goal/state latent')
    result = (z.float() - target.float()).square().mean(dim=(1, 2))
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite goal distance')
    return result


class GoalPlanner(Planner):
    def __init__(self, engine, heads, config=GoalConfig(), compile_heads=False):
        super().__init__(engine, heads, config, compile_heads)
        self.heads.requires_grad_(False)
        for p in self.heads.parameters():
            p.grad = None
        self.reset()

    def reset(self):
        """Required at episode start / RESET; no cached goal crosses episodes."""
        self.goal = None
        self.position = 0

    def after_real_action(self, action, before, after):
        """Advance TTL only for executed real actions; SDK boundaries clear goals."""
        if action.id == 0:
            self.reset()
            return
        self.position += 1
        if after.terminal or after.levels_completed != before.levels_completed:
            self.goal = None
        elif self.goal is not None and self.position - self.goal.created_position >= self.config.lifetime:
            self.goal = None

    def _state(self, state):
        if state.z is None or state.geometry is None:
            raise ValueError('Goal planning requires a confirmed visual state')
        if (len(state.geometry) != 2 or any(type(s) is not int or s < 1 for s in state.geometry)
                or state.z.ndim != 2 or state.z.shape[0] != math.prod(self.shape(state.geometry))
                or state.z.shape[1] < 1 or state.h.ndim != 2 or state.h.shape[0] != 1
                or not torch.isfinite(state.z).all() or not torch.isfinite(state.h).all()):
            raise ValueError('Nonfinite or mismatched planning state geometry')

    def set_goal(self, z, state, proposal_score=0.):
        """Install an explicit target for controlled callers; copy exact tensor."""
        self._state(state)
        distance(state.z[None], z)
        if not math.isfinite(proposal_score) or not 0 <= proposal_score <= 1:
            raise ValueError('Invalid proposal score')
        self.goal = Goal(z.detach().clone(), state.geometry, proposal_score, self.position)

    @torch.no_grad()
    def _propose(self, state, actions):
        z, h = state.z[None], state.h
        count = len(actions)
        score = torch.zeros(1, device=z.device)
        survival = torch.ones_like(score)
        best_score, best_z = -1., None
        for _ in range(self.config.proposal_horizon):
            branches = len(z)
            z, h = z.repeat_interleave(count, 0), h.repeat_interleave(count, 0)
            a = actions.repeat(branches, 1)
            _, output = self.readout(z, h, a, *self.shape(state.geometry))
            p = output[:, :3].float().softmax(-1)
            score = score.repeat_interleave(count) + survival.repeat_interleave(count) * p[:, 1]
            survival = survival.repeat_interleave(count) * self.config.discount * p[:, 0]
            z, h, _ = self.engine.predict(z, h, a, *state.geometry)
            self._branches(z, h, state)
            value, _ = self.readout(z, h, torch.zeros_like(a), *self.shape(state.geometry))
            ranked = score + survival * value.float().sigmoid()
            if not torch.isfinite(ranked).all():
                raise ValueError('Nonfinite goal proposal score')
            keep = torch.argsort(ranked, descending=True, stable=True)[:self.config.proposal_beam]
            index = int(keep[0])
            if float(ranked[index]) > best_score:
                best_score, best_z = float(ranked[index]), z[index].detach().clone()
            z, h, score, survival = z[keep], h[keep], score[keep], survival[keep]
        self.set_goal(best_z, state, best_score)

    def _branches(self, z, h, state):
        if (z.shape[1:] != state.z.shape or h.shape[1:] != state.h.shape[1:]
                or not torch.isfinite(z).all() or not torch.isfinite(h).all()):
            raise ValueError('Nonfinite or mismatched imagined state')

    @torch.no_grad()
    def choose(self, state, observation, rng, mode='goal_mpc', diagnostics=None):
        if mode != 'goal_mpc':
            raise ValueError('GoalPlanner requires goal_mpc')
        self._state(state)
        if observation.terminal:
            self.goal = None
            raise ValueError('Cannot plan after SDK termination')
        if self.goal is not None and (self.goal.geometry != state.geometry or
                self.position - self.goal.created_position >= self.config.lifetime):
            self.goal = None
        choices = candidates(observation, rng, self.config.clicks)
        actions, count = self.engine.actions(choices), len(choices)
        with self.engine.autocast():
            if self.goal is None:
                self._propose(state, actions)
            goal = self.goal
            # Fixed target chosen before reaching search; never selected afterwards.
            current_distance = distance(state.z[None], goal.z)
            z, h, paths, history = state.z[None], state.h, None, None
            estimates = None
            for depth in range(self.config.horizon):
                branches = len(z)
                a = actions.repeat(branches, 1)
                z, h = z.repeat_interleave(count, 0), h.repeat_interleave(count, 0)
                if diagnostics is not None:
                    v, output = self.readout(z, h, a, *self.shape(state.geometry))
                    if depth == 0:
                        initial_value = float(v[0].float().sigmoid())
                    p = output[:, :3].float().softmax(-1)
                z, h, _ = self.engine.predict(z, h, a, *state.geometry)
                self._branches(z, h, state)
                distances = distance(z, goal.z)
                ranked = (-distances).exp()
                ids = torch.arange(count, device=z.device).repeat(branches)[:, None]
                paths = ids if paths is None else torch.cat((paths.repeat_interleave(count, 0), ids), 1)
                if diagnostics is not None:
                    history = z[:, None] if history is None else torch.cat((history.repeat_interleave(count, 0), z[:, None]), 1)
                    v, _ = self.readout(z, h, torch.zeros_like(a), *self.shape(state.geometry))
                    values = torch.cat((p, v.float().sigmoid()[:, None]), -1)[:, None]
                    estimates = values if estimates is None else torch.cat((estimates.repeat_interleave(count, 0), values), 1)
                # Distance orders even scores that underflow to zero; ties retain
                # candidate / parent order. Final-depth branch alone chooses action.
                keep = torch.argsort(distances, stable=True)[:self.config.beam]
                if depth == self.config.horizon - 1:
                    index = int(keep[0])
                    chosen = choices[int(paths[index, 0])]
                    if diagnostics is not None:
                        self._diagnostics(diagnostics, choices, 'mpc', initial_value, ranked, index,
                                          paths, estimates[:, :, :3], estimates[:, :, 3], paths[:, 0])
                        plan = diagnostics['planning']
                        plan.update(objective='latent_goal', mode='goal_mpc')
                        tensors = torch.cat((goal.z[None], history[index]), 0)
                        logits = self.engine.decode(tensors, *state.geometry)
                        if logits.shape != (len(tensors), *state.geometry, 16) or not torch.isfinite(logits).all():
                            raise ValueError('Nonfinite or mismatched diagnostic decoder output')
                        images = logits.argmax(-1).cpu().numpy()
                        from .behavior_trace import tensor
                        plan['goal'] = {'source': 'imagined_proposal', 'created_position': goal.created_position,
                                        'proposal_score': goal.proposal_score, 'distance': float(current_distance[0]),
                                        'frame': tensor(images[0])}
                        for step, latent, image in zip(plan['steps'], history[index], images[1:]):
                            step.update(frame=tensor(image), goal_distance=float(distance(latent[None], goal.z)[0]))
                    return chosen
                z, h, paths = z[keep], h[keep], paths[keep]
                if diagnostics is not None:
                    history, estimates = history[keep], estimates[keep]
