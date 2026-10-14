"""Goal/value readouts and receding-horizon planning over a frozen world model.

Public SDK outcomes supervise new heads. No game rules or visual decoding enter
planning; candidate futures remain predictions and are never replay targets.
"""
from dataclasses import dataclass
import hashlib
import math

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


def returns(events, discount, *, end_reason='budget'):
    """First-progress MC targets; interrupted continuing collection is censored."""
    if end_reason not in {'budget', 'progress', 'game_over', 'interrupted'}:
        raise ValueError('Unknown collection end reason')
    result, carry = [], None if end_reason == 'interrupted' else 0.
    for event in reversed(events):
        carry = 1. if event == 1 else 0. if event == 2 else None if carry is None else discount * carry
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


class OutcomeValue(nn.Module):
    """Generic spatial latent readout; all trainable weights are new.

    V is a discounted success estimate under collected behavior. Event/Q heads
    condition on the candidate action, not the resulting SDK outcome metadata.
    """
    supports_budget = True

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
        # Budget alters V/Q only, never the physical event probabilities.
        self.budget = nn.Sequential(nn.Linear(129, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.budget[-1].weight)
        nn.init.zeros_(self.budget[-1].bias)
        self.budget_action = nn.Sequential(nn.Linear(141, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.budget_action[-1].weight)
        nn.init.zeros_(self.budget_action[-1].bias)
        self.availability = nn.Linear(128, 7)
        nn.init.zeros_(self.availability.weight)
        nn.init.zeros_(self.availability.bias)
        self.register_buffer('availability_trained', torch.tensor(False))

    def features(self, z, h, rows, columns):
        spatial = self.norm(z.float()).transpose(1, 2).reshape(z.shape[0], -1, rows, columns)
        return self.fusion(torch.cat((self.spatial(spatial), self.memory(h.float())), -1))

    def forward(self, z, h, action, rows, columns, remaining_budget=None):
        feature = self.features(z, h, rows, columns)
        return self.from_features(feature, action, remaining_budget)

    def from_features(self, feature, action, remaining_budget=None):
        value = self.value(feature).squeeze(-1)
        output = self.action(torch.cat((feature, action.float()), -1))
        if remaining_budget is not None:
            budget = torch.as_tensor(remaining_budget, device=feature.device, dtype=torch.float32)
            budget = budget.expand(len(feature)).reshape(-1, 1)
            scaled_budget = budget.clamp_min(0).log1p() / math.log(129)
            correction = self.budget(torch.cat((feature, scaled_budget), -1))
            q_correction = self.budget_action(torch.cat((feature, action.float(), scaled_budget), -1))
            value = value + correction[:, 0]
            output = torch.cat((output[:, :3], (output[:, 3] + q_correction[:, 0])[:, None]), -1)
        return value, output

    def click_scores(self, z, h, rows, columns, remaining_budget=None):
        """Q(s,b,ACTION6,x,y), same executed-action critic, no second target.

        Encode the state once; evaluating all native coordinates must not repeat
        the convolutional readout 4096 times.
        """
        feature = self.features(z, h, rows, columns)
        positions = torch.arange(4096, device=z.device)
        actions = torch.zeros((4096, 12), device=z.device)
        actions[:, 6] = 1
        actions[:, 8] = positions.remainder(64).float() * (2 / 63) - 1
        actions[:, 9] = positions.div(64, rounding_mode='floor').float() * (2 / 63) - 1
        actions[:, 10:] = 1
        if remaining_budget is not None:
            remaining_budget = torch.as_tensor(remaining_budget, device=z.device).expand(len(z)).repeat_interleave(4096)
        _, output = self.from_features(feature.repeat_interleave(4096, 0), actions.repeat(len(z), 1), remaining_budget)
        return output[:, 3].reshape(len(z), 64, 64)

    def availability_logits(self, z, h, rows, columns):
        return self.availability(self.features(z, h, rows, columns))

    def load_compatible(self, weights):
        """Convert only the exact legacy schema, then load strictly."""
        current = self.state_dict()
        additions = {k for k in current if k.startswith(('budget.', 'budget_action.', 'availability.')) or k == 'availability_trained'}
        if set(weights) == set(current) - additions:
            converted = dict(weights)
            converted.update({k: torch.zeros_like(current[k]) if k.startswith('availability.') or
                              k == 'availability_trained' or k.startswith(('budget.2.', 'budget_action.2.')) else current[k]
                              for k in additions})
            self.load_state_dict(converted, strict=True)
            return 'legacy_outcome_value_to_budget_v1'
        self.load_state_dict(weights, strict=True)
        return None


GoalValue = OutcomeValue


def read_outcome(readout, heads, z, h, action, rows, columns, remaining_budget=None):
    if remaining_budget is not None and getattr(heads, 'supports_budget', False):
        budget = torch.as_tensor(remaining_budget, device=z.device, dtype=torch.float32).expand(len(z))
        return readout(z, h, action, rows, columns, remaining_budget=budget)
    return readout(z, h, action, rows, columns)


@dataclass(frozen=True)
class PlanConfig:
    horizon: int = 2
    beam: int = 4
    clicks: int = 8
    discount: float = .95
    click_proposals: str = 'uniform'
    future_availability: bool = False
    leaf_value: str = 'v'

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (self.horizon, self.beam, self.clicks)):
            raise ValueError('Planner sizes must be positive integers')
        if self.click_proposals not in {'uniform', 'mixed', 'actor'}:
            raise ValueError('Unknown click proposal mode')
        if self.click_proposals != 'uniform' and self.clicks < 16:
            raise ValueError('Mixed click proposals require at least sixteen candidates')
        if self.leaf_value not in {'v', 'q'}:
            raise ValueError('Leaf value must be v or q')
        if not 0 < self.discount <= 1:
            raise ValueError('discount must lie in (0, 1]')


class Planner:
    def __init__(self, engine, heads, config=PlanConfig(), compile_heads=False):
        self.engine, self.heads, self.config = engine, heads.eval(), config
        freeze(engine)
        self.readout = torch.compile(heads, fullgraph=True, dynamic=True) if compile_heads else heads
        self.proposal_policy = None
        self.reset()

    def reset(self):
        self.last_plan = []
        self.last_endpoint = None
        self.last_states = []

    def after_real_action(self, action, before, after):
        if action.id == 0 or after.terminal or after.levels_completed != before.levels_completed:
            self.reset()
        elif self.last_plan and self.last_plan[0] == action:
            self.last_plan = self.last_plan[1:]
        else:
            self.last_plan = []

    def _readout(self, z, h, action, geometry, remaining_budget=None):
        return read_outcome(self.readout, self.heads, z, h, action,
                            *self.shape(geometry), remaining_budget)

    @torch.no_grad()
    def action_candidates(self, state, observation, rng, remaining_budget=None):
        proposal_mode = getattr(self.config, 'click_proposals', 'uniform')
        if proposal_mode == 'uniform' or 6 not in observation.available_actions:
            return candidates(observation, rng, self.config.clicks)
        retained = [a for a in self.last_plan if a.id == 6][:4]
        pixels = [Action(6, {'x': i % 64, 'y': i // 64}) for i in range(4096)]
        if proposal_mode == 'actor':
            if self.proposal_policy is None:
                raise ValueError('Actor candidate mode requires a proposal policy')
            learned = self.proposal_policy.propose(state.z, state.h, state.geometry,
                                                   observation.available_actions, rng, count=8,
                                                   remaining_budget=128 if remaining_budget is None else remaining_budget)
            learned = [a for a in learned if a.id == 6][:8]
        else:
            with self.engine.autocast():
                if hasattr(self.heads, 'click_scores'):
                    scores = self.heads.click_scores(state.z[None], state.h, *self.shape(state.geometry), remaining_budget).flatten()
                else:
                    chunks = []
                    for start in range(0, 4096, 256):
                        chunk = pixels[start:start + 256]
                        _, output = self._readout(state.z[None].expand(len(chunk), -1, -1),
                                                   state.h.expand(len(chunk), -1), self.engine.actions(chunk),
                                                   state.geometry, remaining_budget)
                        chunks.append(output[:, 3].float())
                    scores = torch.cat(chunks)
            learned = [pixels[i] for i in scores.argsort(descending=True, stable=True).tolist()[:8]]
        ordinary = [Action(i) for i in sorted(set(observation.available_actions)) if 1 <= i <= 7 and i != 6]
        selected, used = [], set()
        for a in retained + learned:
            key = a.data['y'] * 64 + a.data['x']
            if key not in used:
                selected.append(a)
                used.add(key)
        remaining = [i for i in range(4096) if i not in used]
        selected += [pixels[i] for i in rng.sample(remaining, min(self.config.clicks, 4096) - len(selected))]
        return ordinary + selected

    def shape(self, geometry):
        p = self.engine.model.config.patch
        return tuple((side + p - 1) // p for side in geometry)

    def _leaf(self, z, h, geometry, remaining_budget, choices):
        if remaining_budget == 0:
            return torch.zeros(len(z), device=z.device)
        if getattr(self.config, 'leaf_value', 'v') == 'q':
            actions = self.engine.actions(choices)
            _, output = self._readout(z.repeat_interleave(len(choices), 0), h.repeat_interleave(len(choices), 0),
                                       actions.repeat(len(z), 1), geometry, remaining_budget)
            return output[:, 3].float().sigmoid().reshape(len(z), len(choices)).max(-1).values
        value, _ = self._readout(z, h, torch.zeros((len(z), 12), device=z.device), geometry, remaining_budget)
        return value.float().sigmoid()

    @torch.no_grad()
    def evaluate_route(self, state, route, remaining_budget=None, *, leaf_choices=None):
        """Re-simulate a retained/proposed route from confirmed state, never cache h."""
        horizon = len(route) if remaining_budget is None else min(len(route), remaining_budget)
        if horizon < 1:
            raise ValueError('A candidate route needs at least one action')
        z, h = state.z[None], state.h
        total, survival, states, probabilities, values = 0., 1., [], [], []
        initial_value = None
        with self.engine.autocast():
            for depth, choice in enumerate(route[:horizon]):
                action = self.engine.actions([choice])
                budget = None if remaining_budget is None else remaining_budget - depth
                value, output = self._readout(z, h, action, state.geometry, budget)
                if initial_value is None:
                    initial_value = float(value[0].float().sigmoid())
                p = output[0, :3].float().softmax(-1)
                if depth and getattr(self.config, 'future_availability', False):
                    if not bool(getattr(self.heads, 'availability_trained', False)):
                        raise ValueError('Future availability requires a trained, validated readout')
                    legal = self.heads.availability_logits(z, h, *self.shape(state.geometry)).float().sigmoid()
                    survival *= float(legal[0, choice.id - 1])
                total += survival * float(p[1])
                survival *= self.config.discount * float(p[0])
                z, h, _ = self.engine.predict(z, h, action, *state.geometry)
                leaf = self._leaf(z, h, state.geometry, None if budget is None else budget - 1,
                                  route if leaf_choices is None else leaf_choices)
                values.append(float(leaf[0]))
                probabilities.append(p)
                states.append(z[0].detach().clone())
        return {'score': total + survival * values[-1], 'route': route[:horizon],
                'states': states, 'probabilities': torch.stack(probabilities),
                'values': torch.tensor(values, device=z.device), 'state_value': initial_value}

    @torch.no_grad()
    def choose(self, state, observation, rng, mode='mpc', diagnostics=None, *, remaining_budget=None):
        if state.z is None or state.geometry is None:
            raise ValueError('Planning requires a confirmed visual state')
        if remaining_budget is not None and (type(remaining_budget) is not int or remaining_budget < 1):
            raise ValueError('Planning requires a positive remaining action budget')
        if mode not in {'q', 'mpc', 'value_mpc'}:
            raise ValueError('Unknown learned policy')
        if diagnostics is not None and self.config.horizon > 64:
            raise ValueError('Recorded plans support at most 64 steps')
        retained = list(self.last_plan) if mode == 'value_mpc' else []
        choices = self.action_candidates(state, observation, rng, remaining_budget)
        action = self.engine.actions(choices)
        count = len(choices)
        z, h = state.z[None], state.h
        score = torch.zeros(1, device=z.device)
        survival = torch.ones_like(score)
        first = torch.zeros(1, dtype=torch.long, device=z.device)
        best = None
        paths, estimates, history = None, None, None
        horizon = min(self.config.horizon, remaining_budget) if remaining_budget is not None else self.config.horizon
        with self.engine.autocast():
            for depth in range(1 if mode == 'q' else horizon):
                branches = z.shape[0]
                z = z.repeat_interleave(count, 0)
                h = h.repeat_interleave(count, 0)
                actions = action.repeat(branches, 1)
                budget = None if remaining_budget is None else remaining_budget - depth
                current_value, outputs = self._readout(z, h, actions, state.geometry, budget)
                probabilities = outputs[:, :3].float().softmax(-1) if mode != 'q' or diagnostics is not None else None
                ids = torch.arange(count, device=z.device).repeat(branches)[:, None]
                paths = ids if depth == 0 else torch.cat((paths.repeat_interleave(count, 0), ids), 1)
                if depth == 0:
                    initial_value = float(current_value[0].float().sigmoid())
                if mode == 'q':
                    index = int(outputs[:, 3].argmax())
                    if diagnostics is not None:
                        ranked = outputs[:, 3].float().sigmoid()
                        self._diagnostics(diagnostics, choices, mode, initial_value, ranked, index,
                                          paths, probabilities[:, None, :], None, torch.arange(count))
                    self.last_plan = [choices[index]]
                    return choices[index]
                if depth and getattr(self.config, 'future_availability', False):
                    if not bool(getattr(self.heads, 'availability_trained', False)):
                        raise ValueError('Future availability requires a trained, validated readout')
                    availability = self.heads.availability_logits(z, h, *self.shape(state.geometry)).float().sigmoid()
                    identity = actions[:, :8].argmax(-1).sub(1).clamp(0, 6)
                    factor = availability.gather(1, identity[:, None]).squeeze(1)
                else:
                    factor = torch.ones(len(z), device=z.device)
                branch_survival = survival.repeat_interleave(count) * factor
                score = score.repeat_interleave(count) + branch_survival * probabilities[:, 1]
                # Ongoing probability excludes BOTH first progress and failure.
                survival = branch_survival * self.config.discount * probabilities[:, 0]
                first = (torch.arange(count, device=z.device) if depth == 0 else first.repeat_interleave(count))
                z, h, _ = self.engine.predict(z, h, actions, *state.geometry)
                next_budget = None if budget is None else budget - 1
                leaf = self._leaf(z, h, state.geometry, next_budget, choices)
                ranked = score + survival * leaf
                history = z[:, None] if history is None else torch.cat((history.repeat_interleave(count, 0), z[:, None]), 1)
                index = int(ranked.argmax())
                best = choices[int(first[index])]
                if diagnostics is not None:
                    step_values = torch.cat((probabilities, leaf[:, None]), -1)[:, None, :]
                    estimates = step_values if depth == 0 else torch.cat((estimates.repeat_interleave(count, 0), step_values), 1)
                    if depth == horizon - 1:
                        self._diagnostics(diagnostics, choices, 'mpc' if mode == 'value_mpc' else mode, initial_value, ranked, index,
                                          paths, estimates[:, :, :3], estimates[:, :, 3], first)
                if depth == horizon - 1:
                    self.last_plan = [choices[i] for i in paths[index].tolist()]
                    self.last_endpoint = z[index].detach().clone()
                    self.last_states = list(history[index].detach().clone().unbind())
                    if diagnostics is not None and mode == 'value_mpc':
                        diagnostics['planning_v2'] = {'score_semantics': 'discounted_first_progress_return',
                            'remaining_budget': remaining_budget, 'endpoint_role': 'predicted_plan_result'}
                keep = torch.argsort(ranked, descending=True, stable=True)[:self.config.beam]
                z, h, score, survival, first = z[keep], h[keep], score[keep], survival[keep], first[keep]
                paths, history = paths[keep], history[keep]
                if diagnostics is not None:
                    estimates = estimates[keep]
        extra_routes = [('retained', retained)] if retained else []
        if mode == 'value_mpc' and self.proposal_policy is not None:
            extra_routes.append(('actor', self.proposal_policy.propose_route(
                self.engine, state, observation.available_actions, rng, horizon=horizon,
                remaining_budget=128 if remaining_budget is None else remaining_budget)))
        best_score = float(ranked[index])
        for source, route in extra_routes:
            if not route or any(a.id not in observation.available_actions for a in route):
                continue
            # A retained click need not survive the fresh random candidate pool.
            # Its whole suffix is still evaluated as an additional candidate.
            warm = self.evaluate_route(state, route[:horizon], remaining_budget, leaf_choices=choices)
            if warm['score'] > best_score:
                best_score = warm['score']
                best = warm['route'][0]
                self.last_plan, self.last_states = warm['route'], warm['states']
                self.last_endpoint = self.last_states[-1].detach().clone()
                if diagnostics is not None:
                    n = len(self.last_plan)
                    self._diagnostics(diagnostics, self.last_plan, 'mpc', warm['state_value'],
                                      torch.tensor([warm['score']]), 0,
                                      torch.arange(n)[None], warm['probabilities'][None],
                                      warm['values'][None], torch.zeros(1, dtype=torch.long))
                    diagnostics['planning_v2']['selected_route_source'] = source
        return best

    def _diagnostics(self, dest, choices, mode, initial_value, ranked, index, paths, probabilities, values, first):
        """Snapshot the chosen branch from this search, before executing anything."""
        def wire(action):
            return {'id': action.id, 'data': dict(action.data)}
        steps = []
        for depth, choice in enumerate(paths[index].tolist()):
            p = probabilities[index, depth].tolist()
            step = {'action': wire(choices[choice]), 'continue': p[0], 'progress': p[1], 'game_over': p[2]}
            if values is not None:
                step['successor_value'] = float(values[index, depth])
            steps.append(step)
        scores = {}
        for root, score in zip(first.tolist(), ranked.tolist()):
            scores[root] = max(scores.get(root, -float('inf')), score)
        dest.update(planning={'objective': 'sdk_level_progress', 'mode': mode,
                              'horizon': len(steps), 'beam': self.config.beam, 'discount': self.config.discount,
                              'state_value': initial_value, 'selected_score': float(ranked[index]), 'steps': steps},
                    decision={'score_type': 'value', 'selected_value': float(ranked[index]),
                              'candidates': [{'action': wire(choices[i]), 'score': s} for i, s in scores.items()]})


def head_loss(heads, z, h, actions, events, targets, rows, columns, *, components=False, remaining_budget=None):
    value, output = (heads(z, h, actions, rows, columns) if remaining_budget is None else
                     heads(z, h, actions, rows, columns, remaining_budget=remaining_budget))
    # Unweighted event likelihood preserves its probability interpretation.
    event = F.cross_entropy(output[:, :3].float(), events)
    valid = torch.isfinite(targets)
    safe_targets = torch.nan_to_num(targets)
    count = valid.float().sum().clamp_min(1)
    critic = (F.binary_cross_entropy_with_logits(value.float(), safe_targets, reduction='none') * valid).sum() / count
    q = (F.binary_cross_entropy_with_logits(output[:, 3].float(), safe_targets, reduction='none') * valid).sum() / count
    total = event + critic + q
    return (total, torch.stack((event.detach(), critic.detach(), q.detach()))) if components else total
