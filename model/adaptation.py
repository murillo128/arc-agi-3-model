"""Transient per-game learning at the explicit online/Kaggle boundary.

Only ``confirm_result`` admits a pending real action. Replay contains raw public
observations, never imagined frames or cached latents. Every reconstruction of
memory starts at zero at the retained window's left edge, applies RESET tokens
at scene boundaries, and uses the current game's current adapters throughout.
"""
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass, replace
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
    replay_size: int = 32
    minibatch: int = 4
    representation_steps: int = 2
    predictor_steps: int = 1
    path_steps: int = 1
    goal_steps: int = 1
    reconstruction_threshold: float = .10
    per_observation_seconds: float = 10.
    total_seconds: float = 300.
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
            ('goal_lifetime', 1, 256), ('numerical_failure_limit', 1, 8)):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'Invalid adaptation {key}')
        for key in ('alpha', 'learning_rate', 'per_observation_seconds', 'total_seconds'):
            value = getattr(self, key)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid adaptation {key}')
        for key, low in (('exploration', .000001), ('reconstruction_threshold', 0.)):
            value = getattr(self, key)
            if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= 1:
                raise ValueError(f'Invalid adaptation {key}')
        if type(self.allow_undo) is not bool:
            raise ValueError('allow_undo must be boolean')
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


def load(path, *, device='cpu', config=None, max_actions=80, seed=0):
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
    def __init__(self, runtime, config=AdaptationConfig(), *, max_actions=80, seed=0):
        if type(max_actions) is not int or max_actions <= 0:
            raise ValueError('max_actions must be positive')
        self.runtime, self.config, self.max_actions = runtime, config, max_actions
        self.rng = random.Random(seed)
        bank_seed = int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], 'little') % (2**63)
        self.bank = runtime.router.new_bank(rank=config.rank, alpha=config.alpha, seed=bank_seed,
                                            learning_rate=config.learning_rate)
        self.history = deque(maxlen=config.replay_size + 1)
        self.path_labels = deque(maxlen=config.replay_size)
        self.pending_action = self.observation = self.state = self.goal = None
        self.actions_taken = 0
        self._lock = threading.RLock()
        self._candidates = _VisibleCandidates()
        self.diagnostics = dict(observations=0, confirmed_transitions=0, predictive_pairs=0,
            boundary_skips=0, updates=dict.fromkeys(('representation', 'predictor', 'path', 'goal'), 0),
            losses={}, adaptation_seconds=0., numerical_failures=0, budget_stops=0,
            skipped=Counter(), goal_outcomes=Counter(), native_outcomes=Counter(), masks=0)

    def _filtered(self, obs):
        return replace(obs, available_actions=tuple(i for i in obs.available_actions
                       if 1 <= i <= 7 and (i != 7 or self.config.allow_undo)))

    def choose_action(self, observation):
        obs = public_observation(observation)
        with self._lock:
            if self.actions_taken >= self.max_actions:
                raise RuntimeError('Action budget exhausted')
            if self.pending_action is not None:
                if not same_observation(obs, self.observation):
                    raise ValueError('Pending action needs an explicit SDK result')
                return Action(self.pending_action.id, dict(self.pending_action.data))
            if self.observation is None:
                self._admit(obs)
            elif not same_observation(obs, self.observation):
                raise ValueError('New observations require confirm_result, not choose_action')
            if obs.state == 'WIN':
                raise RuntimeError('No action after WIN')
            if obs.state in ('NOT_PLAYED', 'GAME_OVER'):
                action = Action(0)
            else:
                legal = self._filtered(obs)
                with self.runtime.router.activate(self.bank):
                    try:
                        states = self._states()
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
        self.history.append(Record(number, obs.level_initial() if boundary else obs, action, boundary))
        self.observation = obs
        self.diagnostics['observations'] += 1
        first = self.history[0].number
        self.path_labels = deque((row for row in self.path_labels if row['start'] >= first),
                                 maxlen=self.config.replay_size)
        if self.goal is not None and self.goal['start'] < first:
            self.diagnostics['goal_outcomes']['censored'] += 1
            self.goal = None
        with self.runtime.router.activate(self.bank):
            self._adapt()

    @torch.no_grad()
    def _states(self):
        engine = self.runtime.engine
        result = {}
        h = torch.zeros(1, engine.model.config.memory, device=engine.device)
        previous = None
        with engine.autocast():
            for record in self.history:
                z = engine.encode([record.observation])[0]
                if record.boundary:
                    if z is not None:
                        h, _ = engine.advance(z[None], h, engine.actions([Action(0)]))
                elif previous is not None and previous.z is not None and record.action is not None:
                    h, _ = engine.advance(previous.z[None], h, engine.actions([record.action]))
                state = State(z, h.float(), None if z is None else record.observation.final.shape)
                result[record.number] = state
                previous = state
        return result

    def _hindsight(self):
        rows = list(self.history)
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
                success=success, distance=distance, budget=goal['budget']))
        self.goal = None

    def _plan(self, state, obs):
        if state.z is not None and (not torch.isfinite(state.z).all() or not torch.isfinite(state.h).all()):
            raise FloatingPointError('Nonfinite current state')
        if state.z is not None and self.goal is None:
            with torch.no_grad(), self.runtime.engine.autocast():
                logits, mask_logits = self.runtime.goal(state.z[None], state.h, *state.geometry)
            if not torch.isfinite(logits).all() or not torch.isfinite(mask_logits).all():
                raise FloatingPointError('Nonfinite visual goal')
            frame = logits.argmax(-1)[0].cpu().numpy().astype(np.uint8)
            mask = (mask_logits[0] > 0).cpu().numpy()
            if mask.any() and not matches(obs.final, frame, mask):
                self.goal = dict(frame=frame, mask=mask, start=self.history[-1].number,
                    created_action=self.actions_taken,
                    budget=min(self.config.goal_lifetime, self.max_actions-self.actions_taken))
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
        deadline = start + min(self.config.per_observation_seconds, self.config.total_seconds - elapsed)
        if self.diagnostics['numerical_failures'] >= self.config.numerical_failure_limit:
            self.diagnostics['skipped']['learning_disabled'] += 1
            return
        try:
            for name, groups in (('representation', ('encoder', 'decoder')), ('predictor', ('predictor',)),
                                 ('path', ('path',)), ('goal', ('goal',))):
                for _ in range(getattr(self.config, name + '_steps')):
                    if time.monotonic() >= deadline:
                        self.diagnostics['budget_stops'] += 1
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

    def _objective(self, name):
        engine = self.runtime.engine
        rows = list(self.history)
        if name == 'representation':
            samples = [r.observation for r in rows if r.observation.final is not None]
            if not samples:
                return None
            newest = samples[-1]
            samples = [newest] + self.rng.sample(samples[:-1], min(len(samples)-1, self.config.minibatch-1))

            def objective():
                losses, errors = [], []
                for obs in samples:
                    z = engine.encode([obs], training=True)[0]
                    logits = engine.decode(z[None], *obs.final.shape, checkpoint_blocks=True)
                    target = torch.tensor(np.array(obs.final), device=engine.device).long()[None]
                    loss, error = color_macro(logits, target)
                    losses.append(loss)
                    errors.append(error)
                return torch.stack(losses).mean(), torch.stack(errors).mean()
            return objective
        # Re-encode all real frames and causal carries after representation and
        # every predictor update. These tensors never survive this bounded step.
        states = self._states()
        records = {r.number: r for r in rows}
        if name == 'predictor':
            pairs = [(before, after) for before, after in zip(rows, rows[1:])
                     if not after.boundary and before.observation.final is not None
                     and after.observation.final is not None]
            pairs = self.rng.sample(pairs, min(len(pairs), self.config.minibatch))
            if not pairs:
                return None

            def objective():
                losses = []
                for before, after in pairs:
                    state, target_state = states[before.number], states[after.number]
                    predicted, _, delta = engine.predict(state.z[None], state.h,
                        engine.actions([after.action]), *state.geometry)
                    target = (target_state.z - state.z).detach()[None]
                    latent = F.mse_loss(delta.float(), target.float())
                    logits = engine.decode(predicted, *state.geometry)
                    reference = torch.tensor(np.array(before.observation.final), device=engine.device).long()[None]
                    actual = torch.tensor(np.array(after.observation.final), device=engine.device).long()[None]
                    visual = visual_delta_loss(logits, reference, actual, engine.config.false_change_weight,
                                               mode=engine.config.prediction_delta_loss).mean()
                    losses.append(latent + visual)
                return torch.stack(losses).mean(), None
            return objective
        labels = self._hindsight()
        if name == 'path':
            labels = list(self.path_labels)[-self.config.minibatch:] + labels
            labels = labels[:self.config.minibatch]
        if not labels:
            return None
        self.diagnostics['masks'] += sum(int(row['mask'].sum()) for row in labels)
        if name == 'goal':
            def objective():
                losses = []
                for row in labels:
                    state = states[row['start']]
                    logits, mask_logits = self.runtime.goal(state.z[None], state.h, *state.geometry)
                    tensors = [torch.tensor(np.array(v), device=engine.device)[None] for v in
                               (records[row['start']].observation.final, row['frame'], row['mask'])]
                    losses.append(goal_loss(logits, mask_logits, tensors[0].long(), tensors[1].long(), tensors[2].bool()))
                return torch.stack(losses).mean(), None
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
                state = states[row['start']]
                patch = engine.model.config.patch
                height, width = state.geometry
                logits, cost = self.runtime.path(state.z[None], state.h, z[None], mask,
                    (height+patch-1)//patch, (width+patch-1)//patch,
                    torch.tensor([row['budget']], device=engine.device))
                targets = torch.tensor([float(row['success'])], device=engine.device)
                duration = torch.tensor([math.log1p(row['distance'])], device=engine.device)
                losses.append(path_loss(logits, cost, targets, duration))
            return torch.stack(losses).mean(), None
        return objective

    def _step(self, name, groups, objective):
        parameters = list(self.bank.parameters(groups))
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
        if name == 'representation' and float(error) <= self.config.reconstruction_threshold:
            self.diagnostics['skipped']['reconstruction_threshold'] += 1
            return False
        loss.backward()
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
            with torch.no_grad(), engine.autocast():
                post, post_error = objective()
            if (not torch.isfinite(post) or any(not torch.isfinite(p).all() for p in parameters)
                    or any(torch.is_tensor(v) and not torch.isfinite(v).all()
                           for opt in optimizers for state in opt.state.values() for v in state.values())):
                raise FloatingPointError('Nonfinite updated adapter')
        except Exception:
            with torch.no_grad():
                for parameter, saved in zip(parameters, previous):
                    parameter.copy_(saved)
            for opt, saved in zip(optimizers, optimizer_states):
                opt.load_state_dict(saved)
            raise
        finally:
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
        metrics.update(post=float(post), error_post=None if post_error is None else float(post_error))
        self.diagnostics['updates'][name] += 1
        return True
