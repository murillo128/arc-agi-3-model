"""Goal-conditioned latent search with success-first route selection.

The agent accepts only the current visual goal. The synthetic oracle, distance
and completion predicate are deliberately absent from this module.
"""
from dataclasses import dataclass, replace
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import Observation, Episode
from .learning import State
from .planning import candidates


@dataclass(frozen=True)
class PathConfig:
    horizon: int = 3
    beam: int = 4
    clicks: int = 16
    success_margin: float = .02

    def __post_init__(self):
        if any(type(n) is not int or not 1 <= n <= 64 for n in (self.horizon, self.beam, self.clicks)):
            raise ValueError('Path search budgets must be integers in 1..64')
        if not math.isfinite(self.success_margin) or not 0 <= self.success_margin <= 1:
            raise ValueError('Success margin must be finite and in 0..1')


class PathValue(nn.Module):
    """Estimate success within budget and log1p duration conditional on success."""
    def __init__(self, latent, memory, width=64, objective_version=4):
        super().__init__()
        if objective_version not in {2, 3, 4}:
            raise ValueError('Unsupported PathValue objective')
        self.config = dict(latent=latent, memory=memory, width=width, objective_version=objective_version)
        self.project = nn.Linear(latent, width)
        self.spatial = nn.Sequential(nn.Conv2d(width * 3 + 3, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.memory = nn.Linear(memory, width)
        self.readout = nn.Sequential(nn.Linear(width * 17, 128), nn.SiLU(), nn.Linear(128, 1))
        self.normalization = nn.LayerNorm(128) if objective_version >= 3 else nn.Identity()
        self.budget = nn.Linear(1, 128, bias=False)
        self.success = nn.Linear(128, 1)
        nn.init.zeros_(self.budget.weight)
        nn.init.zeros_(self.success.weight)
        nn.init.zeros_(self.success.bias)
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.constant_(self.readout[-1].bias, math.log(math.expm1(math.log1p(16))))

    def forward(self, z, h, goal, mask, rows, columns, remaining_budget):
        z, h, goal, mask = z.float(), h.float(), goal.float(), mask.float()
        mask = mask.expand(z.shape[0], -1, -1)
        if self.config['objective_version'] >= 4:
            weight = mask.sum(1, keepdim=True).clamp_min(1e-6)
            center = ((z + goal) * mask).sum(1, keepdim=True) / (2 * weight)
            z, goal = (z - center) * mask, (goal - center) * mask
            scale = ((z.square() + goal.square()).mean(-1, keepdim=True).sum(1, keepdim=True)
                     / (2 * weight)).clamp_min(.0001).sqrt()
            z, goal = z / scale, goal / scale
        current = self.project(z)
        target = self.project(goal).expand_as(current)
        x = torch.linspace(-1, 1, columns, device=z.device).repeat(rows)
        y = torch.linspace(-1, 1, rows, device=z.device).repeat_interleave(columns)
        xy = torch.stack((x, y), -1)[None].expand(z.shape[0], -1, -1)
        features = torch.cat((current, target, target - current, mask, xy), -1)
        image = features.transpose(1, 2).reshape(z.shape[0], -1, rows, columns)
        memory = self.memory(h)
        if self.config['objective_version'] >= 4:
            memory = memory.tanh()
        hidden = self.readout[0](torch.cat((self.spatial(image).flatten(1), memory), -1))
        hidden = F.silu(self.normalization(hidden + self.budget(remaining_budget.float().reshape(-1, 1).log1p())))
        return self.success(hidden).flatten(), F.softplus(self.readout[-1](hidden)).flatten()

    def load_legacy_features(self, weights):
        """Explicit migration: old failure-penalty output is not a success duration."""
        current = self.state_dict()
        new_keys = {'budget.weight', 'success.weight', 'success.bias',
                    'normalization.weight', 'normalization.bias'}
        shared = set(current) - new_keys
        if set(weights) not in (shared, shared | {'budget.weight', 'success.weight', 'success.bias'}):
            raise ValueError('Unexpected legacy PathValue weights')
        for key, value in weights.items():
            if key not in current or value.shape != current[key].shape:
                raise ValueError(f'Unexpected legacy PathValue shape: {key}')
            if key in shared - {'readout.2.weight', 'readout.2.bias'}:
                current[key] = value
        self.load_state_dict(current, strict=True)


def rank_routes(probabilities, costs, rng, margin):
    """Success bands first, duration second; seeded random numerical ties.

Each band is anchored to its own maximum, avoiding a non-transitive pairwise
comparison which could admit a route far below the actual best probability.
"""
    remaining = list(range(len(costs)))
    ranked = []
    while remaining:
        best = max(probabilities[i] for i in remaining)
        band = [i for i in remaining if probabilities[i] >= best - margin]
        band.sort(key=lambda i: costs[i])
        while band:
            cheapest = costs[band[0]]
            tied = [i for i in band if costs[i] <= cheapest + 1e-6]
            rng.shuffle(tied)
            ranked.extend(tied)
            band = [i for i in band if i not in tied]
        remaining = [i for i in remaining if i not in ranked]
    return ranked


class PathPlanner:
    def __init__(self, engine, evaluator, config=PathConfig(), compile_model=False):
        self.engine, self.evaluator, self.config = engine, evaluator, config
        self.readout = torch.compile(evaluator.forward, fullgraph=True, dynamic=True) if compile_model else evaluator
        self.reset()

    def reset(self):
        self.goal = None
        self.clear_paths()

    def clear_paths(self):
        self.last_plan, self.last_states, self.last_endpoint = [], [], None
        self.last_forecast_plan, self.last_forecast_states, self.last_forecast_source = [], [], None
        self.last_estimate, self.last_forecast_estimate = None, None

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
        self.clear_paths()

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
        routes = []
        budget = remaining_budget if remaining_budget is not None else self.config.horizon
        height, width = state.geometry
        patch = self.engine.model.config.patch
        rows, columns = (height + patch - 1) // patch, (width + patch - 1) // patch
        horizon = min(self.config.horizon, remaining_budget or self.config.horizon)
        with self.engine.autocast():
            for depth in range(horizon):
                n = len(z)
                z, h, _ = self.engine.predict(z.repeat_interleave(count, 0), h.repeat_interleave(count, 0),
                                             actions.repeat(n, 1), height, width)
                budgets = torch.full((len(z),), budget - depth - 1, device=z.device)
                logits, log_costs = self.readout(z, h, self.goal['z'][None], self.goal['mask'],
                                                 rows, columns, budgets)
                if (not torch.isfinite(logits).all() or not torch.isfinite(log_costs).all()
                        or not torch.isfinite(z).all()):
                    raise ValueError('Nonfinite latent path/value')
                probabilities = logits.float().sigmoid().tolist()
                scores = (depth + 1 + log_costs.float().clamp(max=math.log1p(1024)).expm1()).tolist()
                keep = rank_routes(probabilities, scores, rng, self.config.success_margin)[:self.config.beam]
                selected_paths, selected_history, estimates = [], [], []
                for index in keep:
                    path = paths[index // count] + [choices[index % count]]
                    states = history[index // count] + [z[index].detach().clone()]
                    estimate = dict(success_probability=probabilities[index], successful_actions=scores[index],
                                    remaining_budget=budget, depth=depth + 1)
                    selected_paths.append(path)
                    selected_history.append(states)
                    estimates.append(estimate)
                    routes.append((path, states, estimate))
                z, h, paths, history = z[keep], h[keep], selected_paths, selected_history
        winner = rank_routes([r[2]['success_probability'] for r in routes],
                             [r[2]['successful_actions'] for r in routes], rng, self.config.success_margin)[0]
        best_path, best_history, self.last_estimate = routes[winner]
        self.last_plan, self.last_states = best_path, best_history
        self.last_endpoint = best_history[-1]
        # Keep a full explored horizon for inspection of the selected action.
        # Its success-first decision may use a shorter prefix. Prefer
        # its continuation; pruning may leave only another explored branch.
        matching_prefix = [i for i, path in enumerate(paths) if path[:len(best_path)] == best_path]
        matching_action = [i for i, path in enumerate(paths) if path[0] == best_path[0]]
        index = (matching_prefix or matching_action or [0])[0]
        self.last_forecast_plan, self.last_forecast_states = paths[index], history[index]
        self.last_forecast_estimate = estimates[index]
        self.last_forecast_source = ('selected_route' if paths[index] == best_path else
                                    'searched_continuation' if matching_prefix else 'searched_alternative')
        if diagnostics is not None:
            diagnostics['path'] = {**self.last_estimate, 'success_margin': self.config.success_margin,
                                   'steps': [a.id for a in best_path],
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
    """Fit a bounded mixture of completed real goals, never the last outcome alone."""
    def __init__(self, planner, learning_rate=1e-4, compile_model=False, *, capacity=64, batch_segments=8, seed=0,
                 hindsight=False, predicted_inputs=False):
        if not 1 <= batch_segments <= capacity:
            raise ValueError('Require 1 <= batch_segments <= capacity')
        self.planner, self.capacity, self.batch_segments = planner, capacity, batch_segments
        self.hindsight = hindsight
        self.predicted_inputs = predicted_inputs
        self.last_hindsight = 0
        self.byte_limit = getattr(planner.engine.config, 'replay_bytes', 128 * 1024 * 1024)
        self.rng, self.segments = random.Random(seed), []
        self.optimizer = torch.optim.AdamW(planner.evaluator.parameters(), lr=learning_rate)
        self.readout = torch.compile(self.forward, fullgraph=True, dynamic=True) if compile_model else self.forward
        self.updates, self.last_terms = 0, None

    def forward(self, z, h, goal, mask, rows, columns, remaining_budget):
        return self.planner.evaluator(z, h, goal, mask, rows, columns, remaining_budget)

    def status(self):
        return dict(segments=len(self.segments), successes=sum(s['success'] for s in self.segments),
                    capacity=self.capacity, batch_segments=self.batch_segments, terms=self.last_terms,
                    hindsight=self.hindsight, hindsight_examples=self.last_hindsight,
                    predicted_inputs=self.predicted_inputs)

    def update(self, episode, start, success, budget, *, passes=4):
        """Only observed, finished segments enter replay; callers exclude evaluation.

Raw observations/goals are retained. Every sampled segment gets current encodings
and recurrent memories; learned tensors are temporary and never replayed stale.
Each segment has equal BCE weight, regardless of its length. Duration has a
separate conditional mean over successful segments only.
"""
        if type(success) is not bool:
            raise ValueError('Path targets require a completed real outcome')
        planner, engine = self.planner, self.planner.engine
        end = len(episode.actions)
        if not 0 <= start <= end <= budget or passes < 1:
            raise ValueError('Invalid completed goal segment')
        if any(b.starts_new_level(a) for a,b in zip(episode.observations,episode.observations[1:])):
            raise ValueError('Cannot replay a segment crossing levels')
        observations = [replace(o, frames=np.array(o.frames, copy=True),
            action_input=replace(o.action_input, data=dict(o.action_input.data)) if o.action_input else None)
            for o in episode.observations]
        snapshot = Episode(observations, [replace(a, data=dict(a.data)) for a in episode.actions])
        frame, mask = np.array(planner.goal['frame'],copy=True), np.array(planner.goal['pixel_mask'],copy=True)
        record = dict(episode=snapshot,start=start,end=end,success=success,budget=budget,frame=frame,mask=mask,
                      nbytes=snapshot.nbytes+frame.nbytes+mask.nbytes)
        if record['nbytes'] > self.byte_limit:
            raise MemoryError('Completed goal exceeds planner replay byte limit')
        self.segments.append(record)
        while len(self.segments)>self.capacity or sum(s['nbytes'] for s in self.segments)>self.byte_limit:
            self.segments.pop(0)
        # Mixing outcomes prevents a run of failures overwriting every old success.
        pool=[s for s in self.segments[:-1] if s['frame'].shape==frame.shape]
        selected=self.rng.sample(pool,min(len(pool),self.batch_segments-1))+[record]
        engine.model.eval()
        encoded=[]
        patch=engine.model.config.patch
        height,width=frame.shape
        rows,columns=(height+patch-1)//patch,(width+patch-1)//patch
        with torch.no_grad(),engine.autocast():
            self.last_hindsight = 0
            for item in selected:
                states=real_states(engine,item['episode'])
                first_encoded = len(encoded)
                goal_mask=torch.tensor(item['mask'],device=engine.device,dtype=torch.float32)[None,None]
                goal_mask=F.avg_pool2d(goal_mask,patch,patch,ceil_mode=True).flatten(2).transpose(1,2)
                goals = [item]
                if self.hindsight:
                    future = [i for i in range(item['start']+1,item['end']+1)
                              if states[i].z is not None and item['episode'].observations[i].state != 'GAME_OVER']
                    if future:
                        end = self.rng.choice(future)
                        frame = item['episode'].observations[end].final
                        first = next(i for i in range(item['start'],end+1)
                                     if item['episode'].observations[i].final is not None and
                                     np.array_equal(item['episode'].observations[i].final[item['mask']],frame[item['mask']]))
                        if first > item['start']:
                            goals.append(dict(item,frame=frame,end=first,success=True))
                            self.last_hindsight += 1
                for target in goals:
                    indices=[i for i in range(target['start'],target['end']+1) if states[i].z is not None]
                    if not indices:
                        continue
                    if self.hindsight:
                        indices = [indices[-1]] + self.rng.sample(indices[:-1],min(3,len(indices)-1))
                    else:
                        indices=self.rng.sample(indices,min(4,len(indices)))
                    target_obs=Observation(target['frame'][None],(),'NOT_FINISHED',0,0)
                    goal=engine.encode([target_obs])[0].detach()
                    encoded.append(dict(z=torch.stack([states[i].z for i in indices]).detach(),
                        h=torch.cat([states[i].h for i in indices]).detach(),goal=goal,mask=goal_mask,
                        indices=indices,record=target))
                if self.predicted_inputs:
                    needed = sorted({i for entry in encoded[first_encoded:] for i in entry['indices']})
                    groups = {}
                    predicted = {0: (states[0].z, states[0].h)}
                    for i in needed:
                        if i:
                            horizons = [n for n in range(1,min(3,i)+1) if states[i-n].z is not None]
                            if horizons:
                                groups.setdefault(self.rng.choice(horizons), []).append(i)
                            else:
                                predicted[i] = (states[i].z,states[i].h)
                    for steps, destinations in groups.items():
                        origins = [i-steps for i in destinations]
                        z = torch.stack([states[i].z for i in origins])
                        h = torch.cat([states[i].h for i in origins])
                        for offset in range(steps):
                            actions = engine.actions([item['episode'].actions[i+offset] for i in origins])
                            z,h,_ = engine.predict(z,h,actions,height,width)
                        for j,i in enumerate(destinations):
                            predicted[i] = (z[j].detach(),h[j:j+1].detach())
                    for entry in encoded[first_encoded:]:
                        indices = entry['indices']
                        entry['z'] = torch.cat((entry['z'],torch.stack([predicted[i][0] for i in indices])))
                        entry['h'] = torch.cat((entry['h'],torch.cat([predicted[i][1] for i in indices])))
                        entry['indices'] = indices * 2
                # A real raw replay entry must not retain weight-dependent tensors.
                item['episode'].latent_cache.clear()
                item['episode'].prepared.clear()
        if not encoded:
            return None
        z=torch.cat([v['z'] for v in encoded]);h=torch.cat([v['h'] for v in encoded])
        goals=torch.cat([v['goal'][None].expand(len(v['indices']),-1,-1) for v in encoded])
        masks=torch.cat([v['mask'].expand(len(v['indices']),-1,-1) for v in encoded])
        targets=torch.tensor([float(v['record']['success']) for v in encoded for _ in v['indices']],device=engine.device)
        durations=torch.tensor([math.log1p(v['record']['end']-i) for v in encoded for i in v['indices']],device=engine.device)
        budgets=torch.tensor([v['record']['budget']-i for v in encoded for i in v['indices']],device=engine.device)
        weights=torch.tensor([1/len(v['indices'])/len(encoded) for v in encoded for _ in v['indices']],device=engine.device)
        positive=targets.bool()
        planner.evaluator.train().requires_grad_(True)
        losses=[];terms=[]
        for _ in range(passes):
            self.optimizer.zero_grad(set_to_none=True)
            with engine.autocast():
                logits,duration=self.readout(z,h,goals,masks,rows,columns,budgets)
            classification=(F.binary_cross_entropy_with_logits(logits.float(),targets,reduction='none')*weights).sum()
            loss=classification
            duration_loss=None
            if positive.any():
                duration_loss=(F.smooth_l1_loss(duration.float()[positive],durations[positive],reduction='none')
                               *weights[positive]).sum()/weights[positive].sum()
                loss=loss+duration_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(planner.evaluator.parameters(),1.,error_if_nonfinite=True)
            self.optimizer.step();self.updates+=1
            losses.append(float(loss.detach()))
            terms.append((float(classification.detach()),float(duration_loss.detach()) if duration_loss is not None else None))
        self.last_terms=dict(success_bce=sum(t[0] for t in terms)/passes,
                             successful_duration=sum(t[1] for t in terms)/passes if terms[0][1] is not None else None)
        return sum(losses)/len(losses)
