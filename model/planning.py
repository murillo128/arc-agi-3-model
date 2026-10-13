"""Goal/value readouts and receding-horizon planning over a frozen world model.

Public SDK outcomes supervise new heads. No game rules or visual decoding enter
planning; candidate futures remain predictions and are never replay targets.
"""
from dataclasses import dataclass
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .data import Action


def fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze(engine):
    engine.model.eval().requires_grad_(False)
    for parameter in engine.model.parameters():
        parameter.grad = None


def outcome(before, after):
    """0 ongoing, 1 observed progress, 2 GAME_OVER; budget exhaustion is not 2."""
    if after.state == 'WIN' or after.levels_completed > before.levels_completed:
        return 1
    return 2 if after.state == 'GAME_OVER' else 0


def returns(events, discount):
    """Finite-budget discounted real success returns, zero beyond the budget."""
    result, carry = [], 0.
    for event in reversed(events):
        carry = 1. if event == 1 else 0. if event == 2 else discount * carry
        result.append(carry)
    return list(reversed(result))


def candidates(observation, rng, clicks=8):
    """Current SDK legality only. Future legality is unknown during rollouts."""
    if clicks < 1:
        raise ValueError('clicks must be positive')
    result = []
    for identity in sorted(set(observation.available_actions)):
        if not 1 <= identity <= 7:
            continue
        if identity == 6:
            coords = rng.sample(range(64 * 64), min(clicks, 64 * 64))
            result.extend(Action(6, {'x': i % 64, 'y': i // 64}) for i in coords)
        else:
            result.append(Action(identity))
    if not result:
        raise ValueError('No legal ordinary actions')
    return result


class GoalValue(nn.Module):
    """Generic spatial latent readout; all trainable weights are new.

    V is a discounted success estimate under collected behavior. Event/Q heads
    condition on the candidate action, not the resulting SDK outcome metadata.
    """
    def __init__(self, latent, memory, width=64):
        super().__init__()
        self.norm = nn.LayerNorm(latent)
        self.spatial = nn.Sequential(
            nn.Conv2d(latent, width, 1), nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.memory = nn.Sequential(nn.Linear(memory, width), nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(17 * width, 256), nn.SiLU(),
                                    nn.Linear(256, 128), nn.SiLU())
        self.value = nn.Linear(128, 1)
        self.action = nn.Sequential(nn.Linear(140, 128), nn.SiLU(), nn.Linear(128, 4))

    def features(self, z, h, rows, columns):
        spatial = self.norm(z.float()).transpose(1, 2).reshape(z.shape[0], -1, rows, columns)
        return self.fusion(torch.cat((self.spatial(spatial), self.memory(h.float())), -1))

    def forward(self, z, h, action, rows, columns):
        feature = self.features(z, h, rows, columns)
        return self.value(feature).squeeze(-1), self.action(torch.cat((feature, action.float()), -1))


@dataclass(frozen=True)
class PlanConfig:
    horizon: int = 2
    beam: int = 4
    clicks: int = 8
    discount: float = .95

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (self.horizon, self.beam, self.clicks)):
            raise ValueError('Planner sizes must be positive integers')
        if not 0 < self.discount <= 1:
            raise ValueError('discount must lie in (0, 1]')


class Planner:
    def __init__(self, engine, heads, config=PlanConfig(), compile_heads=False):
        self.engine, self.heads, self.config = engine, heads.eval(), config
        freeze(engine)
        self.readout = torch.compile(heads, fullgraph=True, dynamic=True) if compile_heads else heads

    def shape(self, geometry):
        p = self.engine.model.config.patch
        return tuple((side + p - 1) // p for side in geometry)

    @torch.no_grad()
    def choose(self, state, observation, rng, mode='mpc'):
        if state.z is None or state.geometry is None:
            raise ValueError('Planning requires a confirmed visual state')
        if mode not in {'q', 'mpc'}:
            raise ValueError('Unknown learned policy')
        choices = candidates(observation, rng, self.config.clicks)
        action = self.engine.actions(choices)
        count = len(choices)
        z, h = state.z[None], state.h
        score = torch.zeros(1, device=z.device)
        survival = torch.ones_like(score)
        first = torch.zeros(1, dtype=torch.long, device=z.device)
        best = None
        with self.engine.autocast():
            for depth in range(1 if mode == 'q' else self.config.horizon):
                branches = z.shape[0]
                z = z.repeat_interleave(count, 0)
                h = h.repeat_interleave(count, 0)
                actions = action.repeat(branches, 1)
                _, outputs = self.readout(z, h, actions, *self.shape(state.geometry))
                if mode == 'q':
                    return choices[int(outputs[:, 3].argmax())]
                probabilities = outputs[:, :3].float().softmax(-1)
                score = score.repeat_interleave(count) + survival.repeat_interleave(count) * probabilities[:, 1]
                survival = survival.repeat_interleave(count) * self.config.discount * probabilities[:, 0]
                first = (torch.arange(count, device=z.device) if depth == 0 else first.repeat_interleave(count))
                z, h, _ = self.engine.predict(z, h, actions, *state.geometry)
                zeros = torch.zeros((len(z), 12), device=z.device)
                value, _ = self.readout(z, h, zeros, *self.shape(state.geometry))
                ranked = score + survival * value.float().sigmoid()
                best = choices[int(first[ranked.argmax()])]
                keep = ranked.topk(min(self.config.beam, len(ranked))).indices
                z, h, score, survival, first = z[keep], h[keep], score[keep], survival[keep], first[keep]
        return best


def head_loss(heads, z, h, actions, events, targets, rows, columns):
    value, output = heads(z, h, actions, rows, columns)
    # Unweighted event likelihood preserves its probability interpretation.
    event = F.cross_entropy(output[:, :3].float(), events)
    critic = F.binary_cross_entropy_with_logits(value.float(), targets)
    q = F.binary_cross_entropy_with_logits(output[:, 3].float(), targets)
    return event + critic + q
