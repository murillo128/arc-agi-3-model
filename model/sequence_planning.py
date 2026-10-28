"""Teacher-supervised goal proposals and real-outcome path learning.

Only the collection coordinator sees the bundled teacher. Heads consume causal
z/h and visual targets; their optimizers never own world-model parameters.
"""
from dataclasses import asdict, dataclass, replace
import math
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import Episode, Observation
from .game_sequences import EncodedSessions, memory_inputs
from .goal_observation import compose_goal_observation
from .path_planning import PathConfig, PathPlanner, PathValue
from .adaptation_heads import VisualGoal, balanced_regions, goal_loss, matches, path_loss

TEACHER_GAMES = ('sc01-v1', 'si01-v1')


@dataclass(frozen=True)
class SequencePlanningConfig:
    width: int = 64
    learning_rate: float = 1e-4
    horizon: int = 2
    beam: int = 4
    clicks: int = 8
    lifetime: int = 16
    warmup_updates: int = 8
    ramp_updates: int = 64
    learned_fraction: float = .5
    exploration: float = .25
    samples: int = 32
    minibatch: int = 8
    passes: int = 4

    def __post_init__(self):
        for key in ('width', 'horizon', 'beam', 'clicks', 'lifetime', 'warmup_updates',
                    'ramp_updates', 'samples', 'minibatch', 'passes'):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= 256:
                raise ValueError(f'Invalid sequence_planning.{key}')
        for key, lo, hi in (('learning_rate', 0., 1.), ('learned_fraction', 0., .5),
                            ('exploration', .25, 1.)):
            value = getattr(self, key)
            if type(value) not in (float, int) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'Invalid sequence_planning.{key}')
        if self.learning_rate == 0 or self.minibatch > self.samples or self.samples < 2:
            raise ValueError('Positive learning rate, samples >= 2 and minibatch <= samples required')
        PathConfig(horizon=self.horizon, beam=self.beam, clicks=self.clicks)


class SequencePlanning:
    def __init__(self, engine, config=SequencePlanningConfig(), *, explorer=None, seed=0):
        self.engine, self.config = engine, config
        c = engine.model.config
        self.goal = VisualGoal(c.latent, c.memory, c.patch, config.width).to(engine.device)
        self.path = PathValue(c.latent, c.memory, config.width).to(engine.device)
        self.goal_optimizer = torch.optim.AdamW(self.goal.parameters(), lr=config.learning_rate)
        self.path_optimizer = torch.optim.AdamW(self.path.parameters(), lr=config.learning_rate)
        self.explorer = explorer
        self.rng = random.Random(seed)
        self.goal_updates = self.path_updates = self.teacher_actions = 0
        self.adopted_batch = None
        self.trajectory = self.capture_context = None

    def begin_capture(self, trajectory, *, batch):
        """Reserve at most one supported episode in this batch, without sampling RNG."""
        self.trajectory = trajectory
        self.capture_context = dict(batch=batch)

    @property
    def learned_probability(self):
        return self.config.learned_fraction * min(1., max(0.,
            (self.goal_updates-self.config.warmup_updates)/self.config.ramp_updates))

    def status(self):
        return dict(goal_updates=self.goal_updates, path_updates=self.path_updates,
                    teacher_actions=self.teacher_actions, learned_probability=self.learned_probability,
                    adopted_batch=self.adopted_batch)

    def snapshot(self):
        return dict(version=1, config=asdict(self.config), goal=self.goal.state_dict(), path=self.path.state_dict(),
                    goal_optimizer=self.goal_optimizer.state_dict(), path_optimizer=self.path_optimizer.state_dict(),
                    rng=self.rng.getstate(), **self.status())

    def restore(self, saved):
        if saved.get('version') != 1 or saved.get('config') != asdict(self.config):
            raise ValueError('Sequence planner checkpoint/config mismatch')
        self.goal.load_state_dict(saved['goal'], strict=True)
        self.path.load_state_dict(saved['path'], strict=True)
        self.goal_optimizer.load_state_dict(saved['goal_optimizer'])
        self.path_optimizer.load_state_dict(saved['path_optimizer'])
        self.rng.setstate(saved['rng'])
        for key in ('goal_updates', 'path_updates', 'teacher_actions', 'adopted_batch'):
            setattr(self, key, saved[key])

    def collector(self, factory, session):
        if factory.game_id not in TEACHER_GAMES or factory.reserve_final_level:
            return None
        return TeacherCollection(self, factory, session)

    @torch.no_grad()
    def propose(self, state):
        self.goal.eval()
        with self.engine.autocast():
            logits, mask = self.goal(state.z[None], state.h, *state.geometry)
        return logits.argmax(-1)[0].cpu().numpy().astype(np.uint8), (mask[0] > 0).cpu().numpy()

    def fit(self, sessions, *, telemetry=None):
        begin = time.perf_counter()
        selected = [s for s in sessions if s.planning is not None]
        metrics = {key: sum(s.planning['counts'].get(key, 0) for s in selected) for key in (
            'teacher_attempts', 'teacher_reached', 'teacher_steps', 'learned_attempts', 'learned_reached',
            'learned_steps', 'teacher_progress', 'level_successes', 'episodes', 'fallbacks',
            'planner_actions', 'exploration_actions', 'censored',
            'teacher_task_progress', 'learned_task_progress', 'teacher_level_successes', 'learned_level_successes')}
        metrics['collection_seconds'] = sum(s.planning['seconds'] for s in selected)
        metrics.update(goal_loss=None, path_loss=None, goal_color_error=None, goal_mask_error=None)
        if not selected:
            return {**metrics, **self.status(), 'seconds': 0.}, None
        self.teacher_actions += sum(len(s.transitions) for s in selected)
        examples = [(i, row) for i, s in enumerate(selected) for row in s.planning['labels']]
        examples = self.rng.sample(examples, min(len(examples), self.config.samples))
        segments = [(i, row) for i, s in enumerate(selected) for row in s.planning['segments'] if row['success'] is not None]
        segments = self.rng.sample(segments, min(len(segments), self.config.samples // 2))
        path_rows = []
        for i, segment in segments:
            indices = self.rng.sample(range(segment['start'], segment['end']),
                                      min(2, segment['end']-segment['start']))
            for index in indices:
                path_rows.append((i, index, False, segment))
            path_rows.append((i, segment['end']-1, True, segment))
        needed = {(i, row['index'], False) for i, row in examples}
        needed.update((i, t, after) for i, t, after, _ in path_rows)
        if not needed:
            return {**metrics, **self.status(), 'seconds': time.perf_counter()-begin}, None
        # Prefix memory is rebuilt AFTER predictor optimization, preserving resets
        # and cross-level context exactly as the sequence learner does.
        encoded = EncodedSessions(self.engine, selected)
        memories = [torch.zeros(1, self.engine.model.config.memory, device=self.engine.device) for _ in selected]
        states = {}
        with torch.no_grad(), self.engine.autocast():
            for t in range(max(k[1] for k in needed)+1):
                for ids, steps, z, h, action, _ in memory_inputs(self.engine, encoded, t, memories):
                    next_h, _ = self.engine.advance(z, h, action)
                    for k, (i, step) in enumerate(zip(ids, steps)):
                        memories[i] = next_h[k:k+1].float()
                        if (i, t, False) in needed:
                            states[i, t, False] = (z[k].detach(), h[k].detach())
                        if (i, t, True) in needed and encoded.z(step.after) is not None:
                            states[i, t, True] = (encoded.z(step.after).detach(), next_h[k].detach())
        e, device = self.engine, self.engine.device
        losses, colors, masks, sample = [], [], [], None
        goal_batches = []
        self.goal.eval()
        for offset in range(0, len(examples), self.config.minibatch):
            rows = examples[offset:offset+self.config.minibatch]
            z, h = (torch.stack([states[i, row['index'], False][k] for i, row in rows]) for k in (0, 1))
            before = torch.tensor(np.stack([selected[i].transitions[row['index']].before.final for i,row in rows]),device=device).long()
            target = torch.tensor(np.stack([row['frame'] for _,row in rows]),device=device).long()
            mask = torch.tensor(np.stack([row['mask'] for _,row in rows]),device=device).bool()
            goal_batches.append((z, h, before, target, mask))
            # Measure the entire sample before any head update, so later
            # minibatches cannot benefit from earlier examples in this batch.
            with torch.no_grad(), e.autocast():
                logits, mask_logits = self.goal(z, h, *before.shape[-2:])
                predicted = logits.argmax(-1)
                union = ((before != target) | (before != predicted)) & mask
                colors.extend((((predicted != target) & union).sum((1,2))/union.sum((1,2)).clamp_min(1)).cpu().tolist())
                masks.extend(balanced_regions(((mask_logits>0) != mask).float(),mask,~mask).cpu().tolist())
                if sample is None:
                    sample = dict(game=selected[rows[0][0]].game, before=before[0].cpu().tolist(),
                        target=target[0].cpu().tolist(), mask=mask[0].cpu().tolist(),
                        predicted=predicted[0].cpu().tolist(), predicted_mask=(mask_logits[0]>0).cpu().tolist())
        self.goal.train()
        for z, h, before, target, mask in goal_batches:
            for _ in range(self.config.passes):
                self.goal_optimizer.zero_grad(set_to_none=True)
                with e.autocast():
                    logits, mask_logits = self.goal(z, h, *before.shape[-2:])
                    loss = goal_loss(logits, mask_logits, before, target, mask)
                loss.backward()
                capture = telemetry.before_step('goal_planner', self.goal_optimizer) if telemetry else None
                norm = torch.nn.utils.clip_grad_norm_(self.goal.parameters(), 1., error_if_nonfinite=True)
                self.goal_optimizer.step(); self.goal_updates += 1
                if telemetry:
                    telemetry.after_step(capture, norm, loss.detach())
                losses.append(float(loss.detach()))
        metrics.update(goal_loss=sum(losses)/len(losses) if losses else None,
            goal_color_error=sum(colors)/len(colors) if colors else None,
            goal_mask_error=sum(masks)/len(masks) if masks else None)
        path_rows = [row for row in path_rows if row[:3] in states]
        losses = []
        for offset in range(0, len(path_rows), self.config.minibatch):
            rows = path_rows[offset:offset+self.config.minibatch]
            z,h = (torch.stack([states[i,t,after][k] for i,t,after,_ in rows]) for k in (0,1))
            observations = []
            for i, t, after, segment in rows:
                step = selected[i].transitions[t]
                context = step.after.final if after else step.before.final
                frame = compose_goal_observation(segment['frame'], segment['mask'], context)
                observations.append(Observation(frame[None], (), 'NOT_FINISHED', 0, 0))
            with torch.no_grad(), e.autocast():
                goals = torch.stack(e.encode(observations))
            height, width = observations[0].final.shape
            patch = e.model.config.patch
            mask = torch.tensor(np.stack([s['mask'] for *_,s in rows]), device=device).float()[:,None]
            mask = F.avg_pool2d(mask, patch, patch, ceil_mode=True).flatten(2).transpose(1,2)
            positions = [t+int(after) for _,t,after,_ in rows]
            budget = torch.tensor([s['deadline']-pos for pos,(*_,s) in zip(positions,rows)],device=device)
            targets = torch.tensor([s['success'] for *_,s in rows],device=device).float()
            duration = torch.tensor([math.log1p(s['end']-pos) for pos,(*_,s) in zip(positions,rows)],device=device)
            self.path.train()
            for _ in range(self.config.passes):
                self.path_optimizer.zero_grad(set_to_none=True)
                with e.autocast():
                    logits,cost = self.path(z,h,goals,mask,(height+patch-1)//patch,(width+patch-1)//patch,budget)
                    loss = path_loss(logits, cost, targets, duration)
                loss.backward()
                capture = telemetry.before_step('path_planner', self.path_optimizer) if telemetry else None
                norm = torch.nn.utils.clip_grad_norm_(self.path.parameters(),1.,error_if_nonfinite=True)
                self.path_optimizer.step(); self.path_updates += 1
                if telemetry:
                    telemetry.after_step(capture, norm, loss.detach())
                losses.append(float(loss.detach()))
        metrics['path_loss'] = sum(losses)/len(losses) if losses else None
        metrics.update(self.status(), seconds=time.perf_counter()-begin, goal_examples=len(examples), path_examples=len(path_rows))
        return metrics, sample


class TeacherCollection:
    """Trainer-only coordinator. The teacher never enters either learned head."""
    def __init__(self, owner, factory, session):
        self.owner, self.factory, self.session = owner, factory, session
        self.engine = owner.engine
        self.last_diagnostics = {}
        self.capture = None
        self.state = self.active = self.env = None
        self.teacher = None
        self.goals, self.goal_index = (), 0
        self.pending_reset = True
        self.planner = PathPlanner(owner.engine, owner.path, PathConfig(horizon=owner.config.horizon,
            beam=owner.config.beam, clicks=owner.config.clicks), change_prior=owner.explorer)
        session.planning = dict(labels=[], segments=[], counts={}, seconds=0.)

    def count(self, key, amount=1):
        counts = self.session.planning['counts']
        counts[key] = counts.get(key, 0)+amount

    def start(self, env):
        self.close_trajectory('level_budget')
        self.finish(None)
        self.env, self.teacher = env, self.factory.teacher(env)
        self.goals, self.goal_index = self.teacher.training_subgoals(), 0
        self.pending_reset = True
        self.count('episodes')
        if self.owner.capture_context is not None:
            self.capture_context = dict(self.owner.capture_context, game=self.session.game,
                                        episode_seed=self.session.seed, phase='training')
            self.owner.capture_context = None
            self.capture = self.owner.trajectory.begin()
            self.trace_observations, self.trace_actions, self.trace_records = [], [], []

    def close_trajectory(self, reason, success=False):
        if self.capture is None:
            return
        if self.trace_actions:
            self.owner.trajectory.add(Episode(self.trace_observations, self.trace_actions), self.trace_records,
                dict(success=success, reason=reason), **self.capture_context,
                truncated=reason in ('level_budget', 'session_budget', 'sample_limit'))
        self.capture = None
        self.trace_observations, self.trace_actions, self.trace_records = [], [], []

    def finish(self, success):
        if self.active is None:
            return
        end = len(self.session.transitions)
        segment = dict(self.active, end=end, success=success)
        if end > segment['start']:
            self.session.planning['segments'].append(segment)
            if success is None:
                self.count('censored')
            else:
                self.count(segment['source']+'_attempts')
                if success:
                    self.count(segment['source']+'_reached')
                    self.count(segment['source']+'_steps', end-segment['start'])
        self.active = None

    def action(self, before, remaining, rng, explore, *, allow_undo=True):
        begin = time.perf_counter()
        e, config = self.owner.engine, self.owner.config
        if self.pending_reset:
            self.state = e.initial(before, memory=None if self.state is None else self.state.h)
            self.pending_reset = False
        if self.goal_index == len(self.goals):
            self.close_trajectory('goals_reached')
            self.count('exploration_actions')
            action = explore()
            self.session.planning['seconds'] += time.perf_counter()-begin
            return action
        teacher_goal = self.goals[self.goal_index]
        self.session.planning['labels'].append(dict(index=len(self.session.transitions),
            frame=teacher_goal.frame, mask=teacher_goal.mask))
        if self.active is None:
            frame, mask, source = teacher_goal.frame, teacher_goal.mask, 'teacher'
            if rng.random() < self.owner.learned_probability:
                frame, mask = self.owner.propose(self.state)
                source = 'learned'
                if not mask.any() or matches(before.final, frame, mask):
                    frame, mask, source = teacher_goal.frame, teacher_goal.mask, 'teacher'
                    self.count('fallbacks')
            start = len(self.session.transitions)
            self.active = dict(start=start, deadline=start+min(config.lifetime, remaining),
                               frame=frame, mask=mask, source=source)
            self.planner.set_goal(frame, mask)
        exploring = self.owner.path_updates < config.warmup_updates or rng.random() < config.exploration
        self.last_diagnostics = {}
        if exploring:
            action = explore()
            self.count('exploration_actions')
        else:
            permitted = before if allow_undo else replace(before,
                available_actions=tuple(a for a in before.available_actions if a != 7))
            action = self.planner.choose(self.state, permitted, rng, diagnostics=self.last_diagnostics,
                remaining_budget=self.active['deadline']-len(self.session.transitions))
            self.count('planner_actions')
        if self.capture is not None:
            if not self.trace_observations:
                self.trace_observations.append(before)
            self.capture(self, action, exploring)
            self.trace_actions.append(action)
            self.trace_records.append(dict(action_source='exploration' if exploring else 'planner',
                goal='Learned goal' if self.active['source'] == 'learned' else 'Teacher goal',
                goal_source=self.active['source']))
        self.session.planning['seconds'] += time.perf_counter()-begin
        return action

    def observe(self, before, action, after):
        begin = time.perf_counter()
        if self.capture is not None:
            self.trace_observations.append(after)
            self.capture_context['level'] = self.session.transitions[-1].level_number
        self.state = self.owner.engine.observe(self.state, action, after)
        self.planner.after_real_action(action, before, after)
        old = self.goal_index
        while self.goal_index < len(self.goals) and self.teacher.training_goal_reached(self.goals[self.goal_index]):
            self.goal_index += 1
        self.count('teacher_progress', self.goal_index-old)
        if self.active:
            self.count(self.active['source']+'_task_progress', self.goal_index-old)
        if after.state == 'WIN' or after.starts_new_level(before):
            self.count('level_successes')
            if self.active:
                self.count(self.active['source']+'_level_successes')
        if self.active:
            success = (self.goal_index > old if self.active['source'] == 'teacher'
                       else matches(after.final, self.active['frame'], self.active['mask']))
            if success:
                self.finish(True)
            elif after.terminal or len(self.session.transitions) >= self.active['deadline']:
                self.finish(False)
            elif self.goal_index > old:
                self.finish(None)
        if self.capture is not None:
            if after.state == 'WIN' or after.starts_new_level(before):
                self.close_trajectory('success', True)
            elif after.terminal:
                self.close_trajectory('game_over')
            elif self.goal_index == len(self.goals):
                self.close_trajectory('goals_reached')
            elif len(self.trace_actions) >= 128:
                self.close_trajectory('sample_limit')
        self.session.planning['seconds'] += time.perf_counter()-begin
