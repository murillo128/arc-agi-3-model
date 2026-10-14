"""Real outcome records and budget-aware value targets over frozen states."""
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .planning import candidates, read_outcome


@dataclass(frozen=True)
class RealRecord:
    z: torch.Tensor
    h: torch.Tensor
    action: torch.Tensor
    event: int
    target: float | None
    geometry: tuple[int, int]
    remaining_budget: int
    next_z: torch.Tensor | None
    next_h: torch.Tensor | None
    next_geometry: tuple[int, int] | None
    available_actions: tuple[int, ...]
    next_available_actions: tuple[int, ...]
    end_reason: str = 'ongoing'

    def __getitem__(self, index):
        # Existing MC consumers keep their six-column view. New learning uses
        # named attributes so geometry never silently moves to another slot.
        return (self.z, self.h, self.action, self.event, self.target, self.geometry)[index]


@torch.no_grad()
def double_q_targets(online, target, records, device, patch, discount, rng, clicks=16, *,
                     online_readout=None, target_readout=None):
    """Online chooses; EMA target evaluates the same *real-successor* action.

    First progress is absorbing even on the final allowed action. An interrupted
    collection can bootstrap; a consumed action budget cannot. Unusable visual
    successors are censored, never converted into a negative outcome.
    """
    result = torch.full((len(records),), float('nan'), device=device, dtype=torch.float32)
    groups = {}
    known_indices, known_targets = [], []
    for index, row in enumerate(records):
        if not isinstance(row, RealRecord):
            raise ValueError('Double-DQN requires explicit real successor records')
        if row.event == 1:
            known_indices.append(index)
            known_targets.append(1.)
            continue
        if row.event == 2 or row.remaining_budget <= 1:
            known_indices.append(index)
            known_targets.append(0.)
            continue
        if row.next_z is None or row.next_h is None or row.next_geometry is None:
            continue
        choices = candidates(SimpleNamespace(available_actions=row.next_available_actions), rng, clicks)
        groups.setdefault((row.next_geometry, len(choices)), []).append((index, row, choices))
    if known_indices:
        result[torch.tensor(known_indices, device=device)] = torch.tensor(known_targets, device=device)
    for (geometry, count), group in groups.items():
        # Batch independent real successors. Keep each state's sampled candidates
        # paired through online selection and target evaluation, entirely on GPU.
        z = torch.stack([row.next_z for _, row, _ in group]).to(device)
        h = torch.stack([row.next_h for _, row, _ in group]).to(device)
        actions = torch.tensor([[a.features() for a in choices] for _, _, choices in group], device=device)
        budget = torch.tensor([row.remaining_budget - 1 for _, row, _ in group], device=device)
        shape = tuple((s + patch - 1) // patch for s in geometry)
        _, online_q = read_outcome(online_readout or online, online,
            z[:, None].expand(-1, count, -1, -1).reshape(-1, *z.shape[1:]),
            h[:, None].expand(-1, count, -1).reshape(-1, h.shape[-1]), actions.flatten(0, 1),
            *shape, budget.repeat_interleave(count))
        selected = online_q[:, 3].reshape(len(group), count).argmax(-1)
        selected_actions = actions[torch.arange(len(group), device=device), selected]
        _, target_q = read_outcome(target_readout or target, target, z, h, selected_actions, *shape, budget)
        indices = torch.tensor([index for index, _, _ in group], device=device)
        result[indices] = discount * target_q[:, 3].float().sigmoid()
    return result


@torch.no_grad()
def update_target(target, online, tau=.005):
    """Called once *after* each optimizer update, not per environment action."""
    if not 0 < tau <= 1:
        raise ValueError('EMA coefficient must lie in (0, 1]')
    for old, current in zip(target.parameters(), online.parameters(), strict=True):
        old.lerp_(current, tau)
    for old, current in zip(target.buffers(), online.buffers(), strict=True):
        old.copy_(current)
