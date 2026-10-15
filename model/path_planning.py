"""Goal-conditioned latent search and a learned remaining-cost evaluator.

The agent accepts only the current visual goal. The synthetic oracle, distance
and completion predicate are deliberately absent from this module.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import Observation
from .learning import State
from .planning import candidates


@dataclass(frozen=True)
class PathConfig:
    horizon: int = 3
    beam: int = 4
    clicks: int = 16

    def __post_init__(self):
        if any(type(n) is not int or not 1 <= n <= 64 for n in (self.horizon, self.beam, self.clicks)):
            raise ValueError('Path search budgets must be integers in 1..64')


class PathValue(nn.Module):
    """Predict log(1 + remaining actions + failure penalty) for a visual goal."""
    def __init__(self, latent, memory, width=64):
        super().__init__()
        self.config = dict(latent=latent, memory=memory, width=width)
        self.project = nn.Linear(latent, width)
        self.spatial = nn.Sequential(nn.Conv2d(width * 3 + 3, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.memory = nn.Linear(memory, width)
        self.readout = nn.Sequential(nn.Linear(width * 17, 128), nn.SiLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.constant_(self.readout[-1].bias, math.log(math.expm1(math.log1p(16))))

    def forward(self, z, h, goal, mask, rows, columns):
        z, h, goal, mask = z.float(), h.float(), goal.float(), mask.float()
        current = self.project(z)
        target = self.project(goal).expand_as(current)
        mask = mask.expand(z.shape[0], -1, -1)
        x = torch.linspace(-1, 1, columns, device=z.device).repeat(rows)
        y = torch.linspace(-1, 1, rows, device=z.device).repeat_interleave(columns)
        xy = torch.stack((x, y), -1)[None].expand(z.shape[0], -1, -1)
        features = torch.cat((current, target, target - current, mask, xy), -1)
        image = features.transpose(1, 2).reshape(z.shape[0], -1, rows, columns)
        return F.softplus(self.readout(torch.cat((self.spatial(image).flatten(1), self.memory(h)), -1))).flatten()


class PathPlanner:
    def __init__(self, engine, evaluator, config=PathConfig(), compile_model=False):
        self.engine, self.evaluator, self.config = engine, evaluator, config
        self.readout = torch.compile(evaluator.forward, fullgraph=True, dynamic=True) if compile_model else evaluator
        self.reset()

    def reset(self):
        self.goal = None
        self.last_plan, self.last_states, self.last_endpoint = [], [], None

    @torch.no_grad()
    def set_goal(self, frame, mask):
        frame, mask = np.asarray(frame), np.asarray(mask)
        if (frame.ndim != 2 or frame.shape != mask.shape or not mask.any()
                or not np.isin(frame, np.arange(16)).all()):
            raise ValueError('A goal requires a palette frame and a nonempty same-size mask')
        observation = Observation(np.array(frame[None], dtype=np.uint8), (), 'NOT_FINISHED', 0, 0)
        with self.engine.autocast():
            z = self.engine.encode([observation])[0]
        patch = self.engine.model.config.patch
        tensor = torch.tensor(mask.astype(np.float32), device=self.engine.device)[None, None]
        tensor = F.avg_pool2d(tensor, patch, patch, ceil_mode=True).flatten(2).transpose(1, 2)
        self.goal = dict(z=z.detach(), mask=tensor, frame=np.array(frame, copy=True),
                         pixel_mask=np.array(mask, dtype=bool, copy=True))
        self.last_plan, self.last_states = [], []

    @torch.no_grad()
    def choose(self, state, observation, rng, mode='path_mpc', diagnostics=None, *, remaining_budget=None):
        if mode != 'path_mpc' or self.goal is None or observation.terminal or state.z is None:
            raise ValueError('Path planning requires a current goal and nonterminal observed state')
        if state.geometry != self.goal['frame'].shape:
            raise ValueError('Goal and real state must share canvas geometry')
        if remaining_budget is not None and remaining_budget < 1:
            raise ValueError('Path planning requires a positive action budget')
        self.engine.model.eval()
        self.evaluator.eval()
        choices = candidates(observation, rng, self.config.clicks)
        actions = self.engine.actions(choices)
        count = len(choices)
        z, h, paths = state.z[None], state.h, [[]]
        history = [[]]
        best, best_path, best_history = float('inf'), None, None
        height, width = state.geometry
        patch = self.engine.model.config.patch
        rows, columns = (height + patch - 1) // patch, (width + patch - 1) // patch
        horizon = min(self.config.horizon, remaining_budget or self.config.horizon)
        with self.engine.autocast():
            for depth in range(horizon):
                n = len(z)
                z, h, _ = self.engine.predict(z.repeat_interleave(count, 0), h.repeat_interleave(count, 0),
                                             actions.repeat(n, 1), height, width)
                costs = self.readout(z, h, self.goal['z'][None], self.goal['mask'], rows, columns).float()
                if not torch.isfinite(costs).all() or not torch.isfinite(z).all():
                    raise ValueError('Nonfinite latent path/value')
                scores = depth + 1 + costs.clamp(max=math.log1p(1024)).expm1()
                # Random tie order avoids a fixed preference for ACTION1 before
                # the fresh evaluator has received any terminal trajectories.
                order = list(range(len(z)))
                rng.shuffle(order)
                order = torch.tensor(order, device=z.device)
                keep = order[torch.argsort(scores[order], stable=True)[:self.config.beam]]
                selected_paths, selected_history = [], []
                for index in keep.tolist():
                    path = paths[index // count] + [choices[index % count]]
                    states = history[index // count] + [z[index].detach().clone()]
                    selected_paths.append(path)
                    selected_history.append(states)
                    if float(scores[index]) < best:
                        best, best_path, best_history = float(scores[index]), path, states
                z, h, paths, history = z[keep], h[keep], selected_paths, selected_history
        self.last_plan, self.last_states = best_path, best_history
        self.last_endpoint = best_history[-1]
        if diagnostics is not None:
            diagnostics['path'] = {'predicted_cost': best, 'steps': [a.id for a in best_path],
                                   'horizon': horizon, 'source': 'latent_search'}
        return best_path[0]


@torch.no_grad()
def real_states(engine, episode):
    """Refresh recurrent memory using CURRENT predictor weights and real frames."""
    with engine.autocast():
        zs = engine.encode_detached_episode(episode)
        h = torch.zeros(1, engine.model.config.memory, device=engine.device)
        result = [State(zs[0], h, episode.observations[0].final.shape)]
        for action, z, obs in zip(episode.actions, zs[1:], episode.observations[1:]):
            if result[-1].z is not None:
                h, _ = engine.advance(result[-1].z[None], h, engine.actions([action]))
            result.append(State(z, h.float(), obs.final.shape if obs.final is not None else result[-1].geometry))
    return result


class PathLearner:
    def __init__(self, planner, learning_rate=3e-4, compile_model=False):
        self.planner = planner
        self.optimizer = torch.optim.AdamW(planner.evaluator.parameters(), lr=learning_rate)
        # Independent train/inference compile frames avoid requires_grad guards.
        self.readout = torch.compile(self.forward, fullgraph=True, dynamic=True) if compile_model else self.forward
        self.updates = 0

    def forward(self, z, h, goal, mask, rows, columns):
        return self.planner.evaluator(z, h, goal, mask, rows, columns)

    def update(self, episode, start, success, budget, *, passes=4):
        """Monte Carlo targets from a FINISHED real subgoal segment, no inventions.

Success costs the real remaining actions. Failure adds one full action budget.
The final success state has zero remaining cost. Cut-off runs are not failures.
"""
        planner, engine = self.planner, self.planner.engine
        states = real_states(engine, episode)
        end = len(episode.actions)
        indices = list(range(start, end + int(success)))
        indices = [i for i in indices if states[i].z is not None]
        if not indices:
            return None
        planner.evaluator.train().requires_grad_(True)
        patch = engine.model.config.patch
        height, width = states[0].geometry
        rows, columns = (height + patch - 1) // patch, (width + patch - 1) // patch
        losses = []
        for _ in range(passes):
            for offset in range(0, len(indices), 16):
                selected = indices[offset:offset + 16]
                z = torch.stack([states[i].z for i in selected]).detach()
                h = torch.cat([states[i].h for i in selected]).detach()
                target = torch.tensor([math.log1p(segment_cost(end - i, success, budget)) for i in selected], device=engine.device)
                self.optimizer.zero_grad(set_to_none=True)
                with engine.autocast():
                    prediction = self.readout(z, h, planner.goal['z'][None], planner.goal['mask'], rows, columns)
                loss = F.smooth_l1_loss(prediction.float(), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(planner.evaluator.parameters(), 1., error_if_nonfinite=True)
                self.optimizer.step()
                self.updates += 1
                losses.append(float(loss.detach()))
        return sum(losses) / len(losses)


def segment_cost(remaining_actions, success, budget):
    if remaining_actions < 0 or budget < 1:
        raise ValueError('Invalid completed segment cost')
    return remaining_actions + (0 if success else budget)
