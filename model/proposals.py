"""Optional MPC proposals: teacher imitation, then short frozen-world imagination.

These weights never replace the MPC evaluator. Teacher examples are explicitly
labelled; imagined states are ephemeral and cannot enter the real replay here.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .data import Action
from .planning import freeze


@dataclass
class TeacherRecord:
    z: torch.Tensor
    h: torch.Tensor
    geometry: tuple[int, int]
    available_actions: tuple[int, ...]
    remaining_budget: int
    action: Action
    source: str = 'mpc_teacher'


class ProposalPolicy(nn.Module):
    """Factorized action ID / pixel distribution plus a separate policy critic."""

    def __init__(self, latent, memory, width=64, patch=4):
        super().__init__()
        self.config = dict(latent=latent, memory=memory, width=width, patch=patch)
        self.patch = patch
        self.norm = nn.LayerNorm(latent)
        self.spatial = nn.Sequential(nn.Conv2d(latent, width, 1), nn.SiLU(),
                                     nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.memory = nn.Sequential(nn.Linear(memory, width), nn.Tanh())
        self.context = nn.Sequential(nn.Linear(width * 2 + 1, 128), nn.SiLU())
        self.actor = nn.Linear(128, 7)
        self.critic = nn.Linear(128, 1)
        self.click_context = nn.Linear(128, width)
        # Each latent location retains distinct pixel logits, not one click per patch.
        self.click = nn.Conv2d(width, patch * patch, 1)
        self._proposal_forward_cache = None

    def forward(self, z, h, height, width, remaining_budget):
        rows, columns = (height + self.patch - 1) // self.patch, (width + self.patch - 1) // self.patch
        spatial = self.spatial(self.norm(z.float()).transpose(1, 2).reshape(len(z), -1, rows, columns))
        budget = remaining_budget.float().reshape(-1, 1).clamp_min(0).log1p() / math.log(129)
        context = self.context(torch.cat((spatial.mean((-2, -1)), self.memory(h.float()), budget), -1))
        clicks = F.pixel_shuffle(self.click(spatial + self.click_context(context)[:, :, None, None]), self.patch)
        # SDK ACTION6 is defined in the 64 x 64 coordinate domain, independently of frame geometry.
        clicks = F.interpolate(clicks[:, :, :height, :width], size=(64, 64), mode='bilinear', align_corners=False)
        return self.actor(context), clicks.flatten(1), self.critic(context).squeeze(-1)

    @staticmethod
    def distributions(output, available_actions):
        identities, clicks, _ = output
        legal = torch.tensor([[i in actions for i in range(1, 8)] for actions in available_actions],
                             device=identities.device, dtype=torch.bool)
        if not bool(legal.any(-1).all()):
            raise ValueError('Actor needs at least one SDK-legal ordinary action')
        return (torch.distributions.Categorical(logits=identities.float().masked_fill(~legal, -1e9)),
                torch.distributions.Categorical(logits=clicks.float()))

    @staticmethod
    def action_log_prob(distributions, actions):
        identities, clicks = distributions
        ids = torch.tensor([a.id - 1 for a in actions], device=identities.logits.device)
        coords = torch.tensor([a.data.get('y', 0) * 64 + a.data.get('x', 0) for a in actions], device=ids.device)
        return identities.log_prob(ids) + torch.where(ids == 5, clicks.log_prob(coords), 0.)

    @torch.no_grad()
    def propose(self, z, h, geometry, available_actions, rng, count=8, remaining_budget=128):
        if count < 1:
            return []
        if z.ndim == 2:
            z = z[None]
        if h.ndim == 1:
            h = h[None]
        forward = self.forward
        if z.device.type == 'cuda':
            cached = self._proposal_forward_cache
            # A function, not a registered OptimizedModule: checkpoint keys stay
            # stable. A copied policy must never call the original policy's cache.
            if cached is None or cached[0] != id(self):
                cached = (id(self), torch.compile(self.forward, fullgraph=True, dynamic=True))
                self._proposal_forward_cache = cached
            forward = cached[1]
        output = forward(z, h, *geometry, torch.tensor([remaining_budget], device=z.device))
        ids, clicks = self.distributions(output, [available_actions])
        id_weights, click_weights = ids.probs[0].float().cpu().tolist(), clicks.probs[0].float().cpu().tolist()
        result = []
        for _ in range(count):
            identity = rng.choices(range(1, 8), weights=id_weights)[0]
            coordinate = rng.choices(range(4096), weights=click_weights)[0] if identity == 6 else 0
            result.append(Action(identity, {'x': coordinate % 64, 'y': coordinate // 64}) if identity == 6 else Action(identity))
        return result

    @torch.no_grad()
    def propose_route(self, engine, state, available_actions, rng, horizon=2, remaining_budget=128):
        """The next action sees the successor of the action actually proposed."""
        z, h, route = state.z[None], state.h, []
        for step in range(min(horizon, remaining_budget)):
            with engine.autocast():
                action = self.propose(z, h, state.geometry, available_actions, rng, count=1,
                                      remaining_budget=remaining_budget - step)[0]
                z, h, _ = engine.predict(z, h, engine.actions([action]), *state.geometry)
            route.append(action)
        return route


def imitation_loss(policy, records, device, forward=None):
    if not records or any(r.source != 'mpc_teacher' for r in records):
        raise ValueError('Imitation requires labelled MPC decisions')
    geometry = records[0].geometry
    if any(r.geometry != geometry for r in records):
        raise ValueError('Group teacher batches by geometry')
    z, h = (torch.stack([getattr(r, field) for r in records]).to(device) for field in ('z', 'h'))
    budget = torch.tensor([r.remaining_budget for r in records], device=device)
    output = (forward or policy)(z, h, *geometry, budget)
    distributions = policy.distributions(output, [r.available_actions for r in records])
    return -policy.action_log_prob(distributions, [r.action for r in records]).mean()


def imagined_loss(engine, heads, policy, records, *, horizon=2, discount=.95, lambda_=.95, forward=None,
                  target_policy=None, heads_forward=None):
    """REINFORCE lambda returns, with stop-gradient targets and a frozen world.

    Probability of neither progress nor failure carries future reward. The
    critic here evaluates this policy, not the MPC leaf value. Availability in
    imagined states is unknown: the public root action mask is retained.
    """
    if not records or horizon < 1 or any(r.remaining_budget < 1 for r in records):
        raise ValueError('Imagination needs observed seeds with a positive budget')
    geometry = records[0].geometry
    if any(r.geometry != geometry for r in records):
        raise ValueError('Group imagination seeds by geometry')
    freeze(engine)
    z, h = (torch.stack([getattr(r, field) for r in records]).to(engine.device, dtype=torch.float32).detach()
            for field in ('z', 'h'))
    budget = torch.tensor([r.remaining_budget for r in records], device=engine.device)
    probabilities, values, rewards, continuations, entropies, masks, target_values = [], [], [], [], [], [], []
    forward = forward or policy
    heads_forward = heads_forward or heads
    p = engine.model.config.patch
    shape = tuple((s + p - 1) // p for s in geometry)
    for step in range(min(horizon, int(budget.max()))):
        output = forward(z, h, *geometry, budget - step)
        ids, clicks = policy.distributions(output, [r.available_actions for r in records])
        id_sample, click_sample = ids.sample(), clicks.sample()
        actions = [Action(i + 1, {'x': xy % 64, 'y': xy // 64}) if i == 5 else Action(i + 1)
                   for i, xy in zip(id_sample.tolist(), click_sample.tolist())]
        masks.append((budget > step).float())
        probabilities.append(policy.action_log_prob((ids, clicks), actions))
        values.append(output[2])
        entropies.append(ids.entropy() + ids.probs[:, 5] * clicks.entropy())
        with torch.no_grad(), engine.autocast():
            a = engine.actions(actions)
            _, outcomes = heads_forward(z, h, a, *shape, remaining_budget=budget - step)
            events = outcomes[:, :3].float().softmax(-1)
            rewards.append(events[:, 1])
            continuations.append(events[:, 0] * (budget > step + 1))
            z, h, _ = engine.predict(z, h, a, *geometry)
            target = (target_policy or forward)(z, h, *geometry, (budget - step - 1).clamp_min(0))[2].float().sigmoid()
            target_values.append(target * (budget > step + 1))
    carry = target_values[-1]
    returns = []
    for index in reversed(range(len(values))):
        carry = rewards[index] + discount * continuations[index] * ((1 - lambda_) * target_values[index] + lambda_ * carry)
        returns.append(carry.detach())
    returns.reverse()
    survival = torch.ones_like(budget, dtype=torch.float32)
    actor_loss, critic_loss, entropy = (torch.zeros((), device=engine.device) for _ in range(3))
    denominator = torch.zeros((), device=engine.device)
    for index, target in enumerate(returns):
        weight = survival * masks[index]
        advantage = target - values[index].detach().sigmoid()
        actor_loss += (-weight * probabilities[index] * advantage.detach()).sum()
        critic_loss += (weight * F.binary_cross_entropy_with_logits(values[index].float(), target, reduction='none')).sum()
        entropy += (weight * entropies[index]).sum()
        denominator += weight.sum()
        survival = survival * discount * continuations[index].detach()
    denominator = denominator.clamp_min(1)
    actor_loss, critic_loss, entropy = (x / denominator for x in (actor_loss, critic_loss, entropy))
    return actor_loss + .5 * critic_loss - .001 * entropy, {
        'actor_loss': float(actor_loss.detach()), 'critic_loss': float(critic_loss.detach()),
        'entropy': float(entropy.detach()), 'imagined_horizon': len(values)}
