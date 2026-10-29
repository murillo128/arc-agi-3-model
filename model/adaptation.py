"""Transient per-game learning at the explicit online/Kaggle boundary.

Only ``confirm_result`` admits a pending real action. Replay contains raw public
observations, never imagined frames or cached latents. Every reconstruction of
memory starts at zero at the retained window's left edge, applies RESET tokens
at scene boundaries, and uses the current game's current adapters throughout.
"""
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import math
from pathlib import Path
import random
import threading
import time
import weakref

import numpy as np
import torch
from torch.nn import functional as F

from .adaptation_heads import goal_loss, path_loss, matches
from .adaptation_convergence import (GROUPS, delta_metrics, goal_metrics, path_metrics,
                                    fit, fixed_hindsight, successful_hindsight, predictive_pair,
                                    recency_weights, weighted_mean)
from .data import Action, Observation
from .goal_observation import compose_goal_observation
from .learning import State, visual_delta_loss
from .lora import AdapterRouter
from .path_planning import PathConfig, PathPlanner
from .sequence_exploration import visible_clicks


@dataclass(frozen=True)
class AdaptationConfig:
    rank: int = 4
    alpha: float = 4.
    learning_rate: float = 1e-4
    replay_size: int | None = None
    history_half_life: float = 32.
    history_weight_floor: float = .25
    minibatch: int = 4
    stopping: str = 'convergence'
    plateau_patience: int = 8
    plateau_min_relative_improvement: float = .001
    representation_steps: int = 2
    predictor_steps: int = 1
    path_steps: int = 1
    goal_steps: int = 1
    reconstruction_threshold: float = .10
    delta_threshold: float = .05
    rollout_delta_threshold: float = .10
    goal_color_threshold: float = .10
    goal_mask_threshold: float = .10
    path_reachability_threshold: float = .10
    path_cost_threshold: float = .10
    per_observation_seconds: float | None = None
    total_seconds: float | None = None
    exploration: float = .25
    goal_lifetime: int = 16
    horizon: int = 2
    beam: int = 4
    clicks: int = 8
    allow_undo: bool = False
    numerical_failure_limit: int = 2

    @property
    def mode(self):
        return 'adaptive'

    def __post_init__(self):
        for key, low, high in (
            ('rank', 1, 64), ('replay_size', 1, 256), ('minibatch', 1, 4),
            ('representation_steps', 0, 64), ('predictor_steps', 0, 64),
            ('path_steps', 0, 64), ('goal_steps', 0, 64),
            ('plateau_patience', 1, 1000),
            ('goal_lifetime', 1, 256), ('numerical_failure_limit', 1, 8)):
            value = getattr(self, key)
            if key == 'replay_size' and value is None:
                continue
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'Invalid adaptation {key}')
        for key in ('alpha', 'learning_rate', 'history_half_life', 'per_observation_seconds', 'total_seconds'):
            value = getattr(self, key)
            if key.endswith('seconds') and value is None:
                continue
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid adaptation {key}')
        for key, low in (('exploration', .000001), ('reconstruction_threshold', 0.),
                         ('history_weight_floor', .000001),
                         ('delta_threshold', 0.), ('rollout_delta_threshold', 0.), ('goal_color_threshold', 0.),
                         ('goal_mask_threshold', 0.), ('path_reachability_threshold', 0.),
                         ('path_cost_threshold', 0.), ('plateau_min_relative_improvement', .000001)):
            value = getattr(self, key)
            if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= 1:
                raise ValueError(f'Invalid adaptation {key}')
        if type(self.allow_undo) is not bool:
            raise ValueError('allow_undo must be boolean')
        if self.stopping not in ('convergence', 'steps'):
            raise ValueError('stopping must be convergence or steps')
        PathConfig(horizon=self.horizon, beam=self.beam, clicks=self.clicks)


def public_observation(value):
    if isinstance(value, Observation):
        frames = value.frames
        obs = value
    else:
        frames = np.asarray(value.frame)
        obs = Observation(frames, tuple(value.available_actions), value.state,
                          value.levels_completed, value.win_levels)
    if frames.size == 0:
        frames = np.empty((0, 0, 0), dtype=np.uint8)
    elif (frames.ndim != 3 or min(frames.shape) < 1 or max(frames.shape[-2:]) > 64
          or not np.isin(frames, np.arange(16)).all()):
        raise ValueError('Expected native public palette frames [F,H,W], at most 64x64')
    frames = np.array(frames, dtype=np.uint8, copy=True)
    frames.flags.writeable = False
    return replace(obs, frames=frames)


def same_observation(a, b):
    return (a.state == b.state and a.levels_completed == b.levels_completed
            and a.available_actions == b.available_actions and np.array_equal(a.frames, b.frames))


def color_macro(logits, target):
    """Equal weight per observed color, independently for each real frame."""
    nll = F.cross_entropy(logits.float().movedim(-1, 1), target, reduction='none')
    error = (logits.detach().argmax(-1) != target).float()
    losses, errors = [], []
    for row in range(len(target)):
        colors = target[row].unique()
        losses.append(torch.stack([nll[row][target[row] == c].mean() for c in colors]).mean())
        errors.append(torch.stack([error[row][target[row] == c].mean() for c in colors]).mean())
    return torch.stack(losses).mean(), torch.stack(errors).mean()


class Runtime:
    def __init__(self, bundle):
        self.engine, self.goal, self.path = bundle.engine, bundle.goal, bundle.path
        self.metadata = bundle.metadata
        # Adaptation must not opt into BF16 emulation on T4/P100. Keep this
        # policy at the online boundary; the training Engine policy is unchanged.
        self.engine.amp = False
        if self.engine.device.type == 'cuda':
            with torch.cuda.device(self.engine.device):
                self.engine.amp = torch.cuda.is_bf16_supported(including_emulation=False)
        self.router = AdapterRouter(dict(encoder=self.engine.model.encoder,
            decoder=self.engine.model.decoder, predictor=self.engine.model.dynamics,
            goal=self.goal, path=self.path))


_runtimes = weakref.WeakValueDictionary()
_runtime_lock = threading.RLock()


def load(path, *, device='cpu', config=None, max_actions=None, seed=0):
    """Load once per file identity/device, only when an adaptive game is made."""
    from .adaptation_checkpoint import load as load_base
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    key = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, str(torch.device(device)))
    with _runtime_lock:
        runtime = _runtimes.get(key)
        if runtime is None:
            runtime = Runtime(load_base(path, device=device))
            _runtimes[key] = runtime
    return AdaptivePolicy(runtime, config or AdaptationConfig(), max_actions=max_actions, seed=seed)


@dataclass
class Record:
    number: int
    observation: Observation
    action: Action | None = None  # incoming confirmed action
    boundary: bool = False
    scene_observation: Observation = field(init=False, repr=False)

    def __post_init__(self):
        # Retain every SDK frame, but do not encode an old level's animation
        # as context for the new scene. This view shares the raw CPU pixels.
        self.scene_observation = self.observation.level_initial() if self.boundary else self.observation


class _VisibleCandidates:
    def candidates(self, observation, rng, count):
        result = [Action(i) for i in sorted(set(observation.available_actions)) if 1 <= i <= 7 and i != 6]
        if 6 in observation.available_actions and observation.final is not None:
            clicks = visible_clicks(observation, rng, count)
            if not clicks:
                height, width = observation.final.shape
                clicks = [Action(6, dict(x=rng.randrange(width), y=rng.randrange(height))) for _ in range(count)]
            result.extend(clicks)
        if not result:
            raise ValueError('No permitted ordinary actions')
        return result


class AdaptivePolicy:
    def __init__(self, runtime, config=AdaptationConfig(), *, max_actions=None, seed=0):
        if max_actions is not None and (type(max_actions) is not int or max_actions <= 0):
            raise ValueError('max_actions must be positive or None')
        self.runtime, self.config, self.max_actions = runtime, config, max_actions
        self.rng = random.Random(seed)
        bank_seed = int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], 'little') % (2**63)
        self._bank_seed = bank_seed
        self.bank = runtime.router.new_bank(rank=config.rank, alpha=config.alpha, seed=bank_seed,
                                            learning_rate=config.learning_rate)
        self.history = deque(maxlen=None if config.replay_size is None else config.replay_size + 1)
        self.path_labels = deque(maxlen=config.replay_size)
        self.pending_action = self.observation = self.state = self.goal = None
        self.actions_taken = 0
        self._predictor_adapted_through = -1
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self._candidates = _VisibleCandidates()
        self.diagnostics = dict(observations=0, confirmed_transitions=0, predictive_pairs=0,
            boundary_skips=0, updates=dict.fromkeys(('representation', 'predictor', 'path', 'goal'), 0),
            losses={}, convergence={}, adaptation_seconds=0., numerical_failures=0, budget_stops=0,
            skipped=Counter(), goal_outcomes=Counter(), native_outcomes=Counter(), masks=0,
            goal_adapter_resets=0)

    def reset_goal_adaptation(self):
        """Deliberately discard only this game's Goal LoRA, between SDK actions."""
        with self._lock:
            if self.pending_action is not None:
                raise ValueError('Cannot reset Goal with a pending SDK action')
            fresh = self.runtime.router.new_bank(rank=self.config.rank, alpha=self.config.alpha,
                seed=self._bank_seed, learning_rate=self.config.learning_rate)
            with self.runtime.router.activate(self.bank), torch.no_grad():
                for current, initial in zip(self.bank.parameters(('goal',)), fresh.parameters(('goal',))):
                    current.copy_(initial)
                optimizer = self.bank.optimizers['goal']
                optimizer.state.clear()
                optimizer.zero_grad(set_to_none=True)
            if self.goal is not None:
                self.diagnostics['goal_outcomes']['censored'] += 1
            self.goal = None
            self.diagnostics['losses'].pop('goal', None)
            self.diagnostics['convergence'].pop('goal', None)
            self.diagnostics['goal_adapter_resets'] += 1

    def _filtered(self, obs):
        return replace(obs, available_actions=tuple(i for i in obs.available_actions
                       if 1 <= i <= 7 and (i != 7 or self.config.allow_undo)))

    def choose_action(self, observation):
        obs = public_observation(observation)
        with self._lock:
            if self._cancelled.is_set():
                raise InterruptedError('Game adaptation cancelled')
            if self.max_actions is not None and self.actions_taken >= self.max_actions:
                raise RuntimeError('Action budget exhausted')
            if self.pending_action is not None:
                if not same_observation(obs, self.observation):
                    raise ValueError('Pending action needs an explicit SDK result')
                return Action(self.pending_action.id, dict(self.pending_action.data))
            if self.observation is None:
                self._admit(obs)
            elif not same_observation(obs, self.observation):
                raise ValueError('New observations require confirm_result, not choose_action')
            if self._cancelled.is_set():
                raise InterruptedError('Game adaptation cancelled')
            if obs.state == 'WIN':
                raise RuntimeError('No action after WIN')
            if obs.state in ('NOT_PLAYED', 'GAME_OVER'):
                action = Action(0)
            else:
                legal = self._filtered(obs)
                with self.runtime.router.activate(self.bank):
                    try:
                        states = self._states(numbers={self.history[-1].number})
                        state = states[self.history[-1].number]
                        self.state = State(None if state.z is None else state.z.cpu(), state.h.cpu(), state.geometry)
                        action = self._plan(state, legal)
                    except (FloatingPointError, ArithmeticError):
                        self.diagnostics['skipped']['nonfinite_inference'] += 1
                        action = self.rng.choice(self._candidates.candidates(legal, self.rng, self.config.clicks))
            self.pending_action = Action(action.id, dict(action.data))
            return Action(action.id, dict(action.data))

    def cancel_pending(self):
        with self._lock:
            self.pending_action = None

    def request_stop(self):
        """Cooperative cancellation, including from another thread during SFT."""
        self._cancelled.set()

    def confirm_result(self, observation, action=None):
        obs = public_observation(observation)
        with self._lock:
            if self.pending_action is None:
                return False
            selected = self.pending_action
            if action is not None and (action.id != selected.id or action.data != selected.data):
                raise ValueError('Result action does not match the pending SDK request')
            if obs.action_input is not None and obs.action_input != selected:
                raise ValueError('SDK action echo does not match the pending request')
            before = self.observation
            boundary = (selected.id == 0 or before.final is None or obs.final is None
                or before.final.shape != obs.final.shape or before.levels_completed != obs.levels_completed)
            self.pending_action = None
            self.actions_taken += 1
            self.diagnostics['confirmed_transitions'] += 1
            self.diagnostics['boundary_skips' if boundary else 'predictive_pairs'] += 1
            if obs.terminal:
                self.diagnostics['native_outcomes'][obs.state] += 1
            if obs.levels_completed > before.levels_completed:
                self.diagnostics['native_outcomes']['level_progress'] += 1
            self._finish_goal(obs, boundary)
            self._admit(obs, selected, boundary)
            return True

    def _admit(self, obs, action=None, boundary=False):
        number = self.diagnostics['observations']
        self.history.append(Record(number, obs, action, boundary))
        self.observation = obs
        self.diagnostics['observations'] += 1
        if self.config.replay_size is not None:
            first = self.history[0].number
            self.path_labels = deque((row for row in self.path_labels if row['start'] >= first),
                                     maxlen=self.config.replay_size)
            if self.goal is not None and self.goal['start'] < first:
                self.diagnostics['goal_outcomes']['censored'] += 1
                self.goal = None
        with self.runtime.router.activate(self.bank):
            self._adapt()

    @torch.no_grad()
    def _states(self, *, cpu=False, numbers=None):
        engine = self.runtime.engine
        result = {}
        last = max(numbers) if numbers else None
        h = torch.zeros(1, engine.model.config.memory, device=engine.device)
        previous = None
        with engine.autocast():
            for record in self.history:
                if self._cancelled.is_set():
                    raise InterruptedError('Game adaptation cancelled')
                z = engine.encode([record.scene_observation])[0]
                if record.boundary:
                    if z is not None:
                        h, _ = engine.advance(z[None], h, engine.actions([Action(0)]))
                elif previous is not None and previous.z is not None and record.action is not None:
                    h, _ = engine.advance(previous.z[None], h, engine.actions([record.action]))
                state = State(z, h.float(), None if z is None else record.observation.final.shape)
                if numbers is None or record.number in numbers:
                    result[record.number] = (State(None if z is None else z.cpu(), state.h.cpu(), state.geometry)
                                             if cpu else state)
                previous = state
                if last is not None and record.number >= last:
                    break
        return result

    def _hindsight(self, *, fixed=False):
        rows = list(self.history)
        if fixed:
            return fixed_hindsight(rows)
        labels = []
        for end in range(1, len(rows)):
            after = rows[end].observation
            if after.final is None or rows[end].boundary:
                continue
            next_boundary = next((r for r in rows[end+1:] if r.boundary), None)
            progress = (after.state == 'WIN' or (next_boundary is not None and
                        next_boundary.observation.levels_completed > after.levels_completed))
            for start in range(end - 1, -1, -1):
                before = rows[start].observation
                if before.final is None or before.final.shape != after.final.shape:
                    break
                mask = before.final != after.final
                if mask.any():
                    labels.append(dict(start=rows[start].number, end=rows[end].number,
                        frame=after.final, mask=mask, success=True, distance=end-start,
                        budget=end-start, progress=progress))
                if rows[start].boundary:
                    break
        # Public observed success takes priority; never inspect future game state.
        self.rng.shuffle(labels)
        labels.sort(key=lambda row: not row['progress'])
        return labels[:self.config.minibatch]

    def _finish_goal(self, obs, boundary):
        if self.goal is None:
            return
        goal = self.goal
        distance = self.actions_taken - goal['created_action']
        success = not boundary and matches(obs.final, goal['frame'], goal['mask'])
        reason = ('reached' if success else 'censored' if boundary else
                  'terminal' if obs.terminal else 'expired' if distance >= goal['budget'] else None)
        if reason is None:
            return
        self.diagnostics['goal_outcomes'][reason] += 1
        if reason != 'censored':
            self.path_labels.append(dict(start=goal['start'], frame=goal['frame'], mask=goal['mask'],
                end=self.diagnostics['observations'], success=success,
                distance=distance, budget=goal['budget']))
        self.goal = None

    def _plan(self, state, obs):
        if state.z is not None and (not torch.isfinite(state.z).all() or not torch.isfinite(state.h).all()):
            raise FloatingPointError('Nonfinite current state')
        if state.z is not None and self.goal is None:
            with torch.no_grad(), self.runtime.engine.autocast():
                features = self.runtime.engine.planning_features(state.z[None], *state.geometry)
                logits, mask_logits = self.runtime.goal(features, state.h, *state.geometry)
            if not torch.isfinite(logits).all() or not torch.isfinite(mask_logits).all():
                raise FloatingPointError('Nonfinite visual goal')
            frame = logits.argmax(-1)[0].cpu().numpy().astype(np.uint8)
            mask = (mask_logits[0] > 0).cpu().numpy()
            if mask.any() and not matches(obs.final, frame, mask):
                self.goal = dict(frame=frame, mask=mask, start=self.history[-1].number,
                    created_action=self.actions_taken,
                    budget=(self.config.goal_lifetime if self.max_actions is None else
                            min(self.config.goal_lifetime, self.max_actions-self.actions_taken)))
        if self.goal is not None and self.rng.random() >= self.config.exploration:
            planner = PathPlanner(self.runtime.engine, self.runtime.path,
                PathConfig(horizon=self.config.horizon, beam=self.config.beam, clicks=self.config.clicks),
                change_prior=self._candidates)
            planner.set_goal(self.goal['frame'], self.goal['mask'])
            remaining = self.goal['budget'] - (self.actions_taken-self.goal['created_action'])
            try:
                return planner.choose(state, obs, self.rng, remaining_budget=remaining)
            except ValueError as exc:
                if 'Nonfinite' in str(exc):
                    raise FloatingPointError(str(exc)) from exc
                raise
        return self.rng.choice(self._candidates.candidates(obs, self.rng, self.config.clicks))

    def _adapt(self):
        start = time.monotonic()
        elapsed = self.diagnostics['adaptation_seconds']
        limits = [v for v in (self.config.per_observation_seconds,
                  None if self.config.total_seconds is None else self.config.total_seconds - elapsed)
                  if v is not None]
        deadline = start + min(limits) if limits else None
        self.diagnostics['convergence'] = {}
        if self.diagnostics['numerical_failures'] >= self.config.numerical_failure_limit:
            self.diagnostics['skipped']['learning_disabled'] += 1
            if self.config.stopping == 'convergence':
                self.diagnostics['convergence'] = {name: dict(reason='learning_disabled',
                    steps=0, threshold_met=False) for name in GROUPS}
            return
        try:
            for name, groups in GROUPS.items():
                if self.config.stopping == 'convergence':
                    if getattr(self.config, name + '_steps') == 0:
                        self.diagnostics['convergence'][name] = dict(reason='disabled', steps=0, threshold_met=False)
                        continue
                    fit(self, name, deadline)
                    if self.diagnostics['convergence'][name]['reason'] in ('budget', 'cancelled'):
                        return
                    continue
                for _ in range(getattr(self.config, name + '_steps')):
                    if reason := self._stop_reason(deadline):
                        self.diagnostics['budget_stops'] += int(reason == 'budget')
                        return
                    with self.runtime.router.activate(self.bank, groups=groups):
                        objective = self._objective(name)
                        if objective is None:
                            self.diagnostics['skipped'][name] += 1
                            break
                        if not self._step(name, groups, objective):
                            break
        except FloatingPointError:
            self.diagnostics['numerical_failures'] += 1
        finally:
            self.diagnostics['adaptation_seconds'] += time.monotonic() - start

    def _stop_reason(self, deadline):
        if self._cancelled.is_set():
            return 'cancelled'
        if deadline is not None and time.monotonic() >= deadline:
            return 'budget'
        return None

    def _examples(self, name):
        rows = list(self.history)
        if name == 'representation':
            return [r.scene_observation for r in rows if r.observation.final is not None]
        if name == 'predictor':
            return [(before, after) for before, after in zip(rows, rows[1:])
                    if predictive_pair(before, after)]
        if name == 'goal':
            return successful_hindsight(rows)
        labels = self._hindsight(fixed=True)
        return list(self.path_labels) + labels

    def _weights(self, name, samples):
        if name == 'representation':
            numbers = {id(row.scene_observation): row.number for row in self.history}
            ends = [numbers[id(obs)] for obs in samples]
        elif name == 'predictor':
            ends = [after.number for _, after in samples]
        else:
            ends = [row['end'] for row in samples]
        return recency_weights(ends, self.history[-1].number, self.config)

    def _objective(self, name, *, samples=None, states=None, measured=None, weights=None):
        engine = self.runtime.engine
        rows = list(self.history)
        if name == 'representation':
            if samples is None:
                samples = [r.scene_observation for r in rows if r.observation.final is not None]
                if samples:
                    samples = [samples[-1]] + self.rng.sample(samples[:-1], min(len(samples)-1, self.config.minibatch-1))
            if not samples:
                return None

            def objective():
                losses, errors = [], []
                for obs in samples:
                    z = engine.encode([obs], training=True)[0]
                    logits = engine.decode(z[None], *obs.final.shape, checkpoint_blocks=True)
                    target = torch.tensor(np.array(obs.final), device=engine.device).long()[None]
                    loss, error = color_macro(logits, target)
                    losses.append(loss)
                    errors.append(error)
                    if measured is not None:
                        measured.append(dict(reconstruction_error=float(error)))
                return weighted_mean(losses, weights), torch.stack(errors).mean()
            return objective
        # Re-encode real frames and causal carries after representation and
        # every predictor update. CPU caches never survive an optimizer update.
        states = self._states(cpu=True) if states is None else states
        def current(number):
            state = states[number]
            return State(None if state.z is None else state.z.to(engine.device),
                         state.h.to(engine.device), state.geometry)
        records = {r.number: r for r in rows}
        if name == 'predictor':
            pairs = samples
            if pairs is None:
                pairs = self._examples(name)
                pairs = self.rng.sample(pairs, min(len(pairs), self.config.minibatch))
            if not pairs:
                return None

            def prediction_loss(origin, after, steps):
                state, target_state = current(origin.number), current(after.number)
                z, h = state.z[None], state.h
                for record in steps:
                    z, h, _ = engine.predict(z, h, engine.actions([record.action]), *state.geometry)
                latent = F.mse_loss(z.float(), target_state.z.detach().float()[None])
                logits = engine.decode(z, *state.geometry)
                reference = torch.tensor(np.array(origin.observation.final), device=engine.device).long()[None]
                actual = torch.tensor(np.array(after.observation.final), device=engine.device).long()[None]
                visual = visual_delta_loss(logits, reference, actual, engine.config.false_change_weight,
                                           mode=engine.config.prediction_delta_loss).mean()
                metrics = delta_metrics(reference, actual, logits.argmax(-1)) if measured is not None else None
                return latent + visual, metrics

            def objective():
                losses = []
                for before, after in pairs:
                    loss, metrics = prediction_loss(before, after, [after])
                    window = [records.get(i) for i in range(after.number-3, after.number+1)]
                    eligible = all(r is not None for r in window) and all(
                        predictive_pair(a, b) for a, b in zip(window, window[1:]))
                    if measured is not None:
                        metrics.update(rollout_samples=0, rollout_wrong=0, rollout_union=0)
                    if eligible:
                        rollout_loss, rollout_metrics = prediction_loss(window[0], after, window[1:])
                        loss = (loss + rollout_loss) / 2
                        if measured is not None:
                            metrics.update(rollout_samples=1, rollout_wrong=rollout_metrics['wrong'],
                                           rollout_union=rollout_metrics['union'])
                    losses.append(loss)
                    if measured is not None:
                        measured.append(metrics)
                return weighted_mean(losses, weights), None
            return objective
        labels = samples
        if labels is None:
            if name == 'goal':
                labels = self._examples('goal')
                labels = self.rng.sample(labels, min(len(labels), self.config.minibatch))
            else:
                labels = self._hindsight()
                labels = (list(self.path_labels)[-self.config.minibatch:] + labels)[:self.config.minibatch]
        if not labels:
            return None
        self.diagnostics['masks'] += sum(int(row['mask'].sum()) for row in labels)
        if name == 'goal':
            def objective():
                losses = []
                for row in labels:
                    state = current(row['start'])
                    features = engine.planning_features(state.z[None], *state.geometry)
                    logits, mask_logits = self.runtime.goal(features, state.h, *state.geometry)
                    tensors = [torch.tensor(np.array(v), device=engine.device)[None] for v in
                               (records[row['start']].observation.final, row['frame'], row['mask'])]
                    losses.append(goal_loss(logits, mask_logits, tensors[0].long(), tensors[1].long(), tensors[2].bool()))
                    if measured is not None:
                        measured.append(goal_metrics(logits, mask_logits, tensors[0].long(), tensors[1].long(), tensors[2].bool()))
                return weighted_mean(losses, weights), None
            return objective
        encoded = []
        with torch.no_grad(), engine.autocast():
            for row in labels:
                before = records[row['start']].observation
                frame = compose_goal_observation(row['frame'], row['mask'], before.final)
                z = engine.encode([replace(before, frames=frame[None])])[0].detach()
                patch = engine.model.config.patch
                mask = torch.tensor(row['mask'].astype(np.float32), device=engine.device)[None, None]
                mask = F.avg_pool2d(mask, patch, patch, ceil_mode=True).flatten(2).transpose(1, 2)
                encoded.append((z, mask))

        def objective():
            losses = []
            for row, (z, mask) in zip(labels, encoded):
                state = current(row['start'])
                patch = engine.model.config.patch
                height, width = state.geometry
                features = engine.planning_features(state.z[None], height, width)
                goal_features = engine.planning_features(z[None], height, width)
                logits, cost = self.runtime.path(features, state.h, goal_features, mask,
                    (height+patch-1)//patch, (width+patch-1)//patch,
                    torch.tensor([row['budget']], device=engine.device))
                targets = torch.tensor([float(row['success'])], device=engine.device)
                duration = torch.tensor([math.log1p(row['distance'])], device=engine.device)
                losses.append(path_loss(logits, cost, targets, duration))
                if measured is not None:
                    measured.append(path_metrics(logits, cost, row['success'], row['distance']))
            return weighted_mean(losses, weights), None
        return objective

    def _step(self, name, groups, objective, *, check_threshold=True):
        optimizers = [self.bank.optimizers[g] for g in groups]
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        engine = self.runtime.engine
        with engine.autocast():
            loss, error = objective()
        if not torch.isfinite(loss) or (error is not None and not torch.isfinite(error)):
            raise FloatingPointError('Nonfinite adaptation objective')
        pre = float(loss.detach())
        metrics = dict(pre=pre, post=pre, error_pre=None if error is None else float(error),
                       error_post=None if error is None else float(error))
        self.diagnostics['losses'][name] = metrics
        if check_threshold and name == 'representation' and float(error) <= self.config.reconstruction_threshold:
            self.diagnostics['skipped']['reconstruction_threshold'] += 1
            return False
        loss.backward()

        def check():
            with torch.no_grad(), engine.autocast():
                post, post_error = objective()
            if not torch.isfinite(post) or (post_error is not None and not torch.isfinite(post_error)):
                raise FloatingPointError('Nonfinite updated adapter objective')
            metrics.update(post=float(post), error_post=None if post_error is None else float(post_error))

        self._update(name, groups, check=check)
        return True

    def _update(self, name, groups, *, check=None):
        """Apply accumulated gradients atomically, including Adam moments."""
        parameters = list(self.bank.parameters(groups))
        optimizers = [self.bank.optimizers[g] for g in groups]
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            raise FloatingPointError('Nonfinite adapter gradient')
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        # Small game-owned adapter/Adam rollback, never a world-model copy.
        previous = [p.detach().clone() for p in parameters]
        optimizer_states = [deepcopy(opt.state_dict()) for opt in optimizers]
        try:
            for opt in optimizers:
                opt.step()
            if (any(not torch.isfinite(p).all() for p in parameters)
                    or any(torch.is_tensor(v) and not torch.isfinite(v).all()
                           for opt in optimizers for state in opt.state.values() for v in state.values())):
                raise FloatingPointError('Nonfinite updated adapter')
            if check is not None:
                check()
        except BaseException:
            with torch.no_grad():
                for parameter, saved in zip(parameters, previous):
                    parameter.copy_(saved)
            for opt, saved in zip(optimizers, optimizer_states):
                opt.load_state_dict(saved)
            raise
        finally:
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
        self.diagnostics['updates'][name] += 1
