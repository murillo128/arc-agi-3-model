"""Sparse teacher supervision on normal training sessions; never held-out data."""
from dataclasses import asdict, dataclass
import math
import random
import time

import numpy as np
import torch

from .game_sequences import EncodedSessions, memory_inputs
from .learning import visual_delta_loss
from .object_dynamics import mask_loss, probabilities
from .object_preview import instance_preview
from .object_observation import ObservationReplay


@dataclass(frozen=True)
class ObjectTrainingConfig:
    checkpoint: str
    device: str = 'cpu'
    samples: int = 8
    passes: int = 2
    learning_rate: float = 1e-3
    warmup_updates: int = 64
    activation_iou: float = .5
    teacher_frames: int = 24
    click_frames: int = 4
    cache_frames: int = 256
    points_per_side: int = 8
    activate_residual: bool = False
    observer_steps: int = 32
    observer_replay_frames: int = 64

    def __post_init__(self):
        if not isinstance(self.checkpoint, str) or not self.checkpoint:
            raise ValueError('Object training requires an offline SAM2 checkpoint')
        if not isinstance(self.device, str) or (self.device != 'cpu' and not self.device.startswith('cuda')):
            raise ValueError('Unsupported object teacher device')
        if type(self.activate_residual) is not bool:
            raise ValueError('activate_residual must be boolean')
        for key in ('samples', 'passes', 'warmup_updates', 'teacher_frames', 'cache_frames', 'points_per_side'):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= 4096:
                raise ValueError(f'Invalid objects.{key}')
        if type(self.observer_steps) is not int or not 0 <= self.observer_steps <= 256:
            raise ValueError('observer_steps must be 0..256')
        if type(self.observer_replay_frames) is not int or not 1 <= self.observer_replay_frames <= 256:
            raise ValueError('observer_replay_frames must be 1..256')
        if type(self.click_frames) is not int or not 0 <= self.click_frames <= 64:
            raise ValueError('click_frames must be 0..64')
        if not 2 <= self.points_per_side <= 32 or self.samples > 64 or self.passes > 16:
            raise ValueError('Object teacher sampling/passes exceed bounded limits')
        for key in ('learning_rate', 'activation_iou'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f'Invalid objects.{key}')


def rollout_window(session, start, length=3):
    rows = session.transitions[start:start+length]
    if len(rows) != length or any(not s.supervised or s.after.terminal for s in rows):
        return []
    if any(s.reset_before for s in rows[1:]):
        return []
    if any(a.after.final.shape != b.before.final.shape or a.level_number != b.level_number
           or not np.array_equal(a.after.final, b.before.final) for a, b in zip(rows, rows[1:])):
        return []
    return rows


def target_tensors(labels, slots, device):
    masks = torch.zeros(1, slots, *labels.masks.shape[1:], device=device)
    confidence = torch.zeros(1, slots, device=device)
    n = min(slots, len(labels.masks))
    masks[0, :n] = torch.as_tensor(np.array(labels.masks[:n]), device=device)
    confidence[0, :n] = torch.as_tensor(np.array(labels.confidence[:n]), device=device)
    return masks, confidence


def object_rollout(engine, z, h, actions, geometry, masks, *, warmup=False):
    """Action sequence only: neither real intermediate frames nor masks are inputs."""
    result = []
    for action in actions:
        z, h, _, prediction = engine.predict_objects(z, h, action, *geometry, masks)
        masks = probabilities(prediction)
        if warmup:
            # With gain=0 these are frozen base forecasts. Preserve the mask
            # graph across steps, without retaining a useless base backward.
            z, h = z.detach(), h.detach()
        result.append((z, h, prediction))
    return result


def label_image(masks, confidence=None):
    result = np.zeros(masks.shape[-2:], dtype=np.uint8)
    for i, mask in enumerate(masks):
        if confidence is None or confidence[i] >= .5:
            result[mask > .5] = (i % 15)+1
    return result.tolist()


class ObjectTrainer:
    def __init__(self, engine, config, segmenter):
        self.engine, self.config, self.segmenter = engine, config, segmenter
        if not engine.model.config.object_slots:
            raise ValueError('Object training requires object_slots > 0')
        engine.segmenter = segmenter
        self.observations = ObservationReplay(config.observer_replay_frames)

    def begin_collection(self):
        self.segmenter.remaining = self.config.click_frames

    def fit(self, learner, sessions):
        e, c = self.engine, self.config
        head = e.model.dynamics.objects
        start = time.perf_counter()
        self.segmenter.remaining = c.teacher_frames
        self.observations.check_encoder(e)
        # Rotate coverage over games instead of spending every label on one game.
        rng = random.Random(sum(s.seed for s in sessions) + learner.updates)
        candidates = []
        order = list(range(len(sessions))); rng.shuffle(order)
        for i in order:
            valid = [t for t in range(len(sessions[i].transitions)) if rollout_window(sessions[i], t, 1)]
            rng.shuffle(valid)
            candidates.append([(i, t) for t in valid])
        selected = []
        while len(selected) < c.samples and any(candidates):
            for rows in candidates:
                if rows and len(selected) < c.samples:
                    selected.append(rows.pop())
        examples, labeled_observations = [], []
        for i, t in selected:
            window = rollout_window(sessions[i], t, 3) or rollout_window(sessions[i], t, 1)
            before = self.segmenter.segment(window[0].before.final)
            after = self.segmenter.segment(window[0].after.final)
            endpoint = self.segmenter.segment(window[-1].after.final) if len(window) == 3 else None
            for observation, labels in ((window[0].before, before), (window[0].after, after),
                                         (window[-1].after, endpoint)):
                if labels is not None and len(labels.masks):
                    labeled_observations.append((sessions[i].game, observation, labels))
            if before is not None and after is not None and len(before.masks) and len(after.masks):
                examples.append((i, t, window, before, after, endpoint))
        metrics = dict(samples=len(examples), updates=0, loss=None, mean_iou=None, rollout_iou=None,
                       matched=0, rollout_matched=0, teacher=self.segmenter.report(),
                       metric_version=2, measurement='before auxiliary fitting',
                       observer_updates=0, observer_loss=None, observer_frames=0,
                       replay_frames=len(self.observations.rows), observed_post_iou=None,
                       observed_fit_iou=None)
        quality = {key: [] for key in ('observed_iou', 'sam_iou', 'student_iou',
                   'sam_rollout_iou', 'student_rollout_iou', 'copy_iou', 'copy_rollout_iou')}
        metrics.update({key: None for key in quality})
        sample = None
        if not examples and not labeled_observations:
            return dict(metrics, seconds=time.perf_counter()-start, active=bool(head.gain)), sample
        encoded = EncodedSessions(e, sessions)
        memories = learner.zero_memories(sessions)
        states = {}
        wanted = {(i, t) for i, t, *_ in examples}
        with torch.no_grad(), e.autocast():
            for t in range(max((t for _, t in wanted), default=-1)+1):
                for ids, _, z, h, action, _ in memory_inputs(e, encoded, t, memories):
                    next_h, _ = e.advance(z, h, action)
                    for k, i in enumerate(ids):
                        if (i, t) in wanted:
                            states[i, t] = (z[k:k+1].detach(), h[k:k+1].detach())
                        memories[i] = next_h[k:k+1]
        current_rows = [self.observations.add(game, observation, encoded.z(observation)[None], labels)
                        for game, observation, labels in labeled_observations]
        prepared = []
        with torch.no_grad(), e.autocast():
            for i, t, window, before, after, endpoint in examples:
                z, h = states[i, t]
                geometry = window[0].before.final.shape
                actions = [e.actions([step.action]) for step in window]
                targets = [target_tensors(after, head.slots, e.device)]
                if len(window) == 3 and endpoint is not None and len(endpoint.masks):
                    targets.append(target_tensors(endpoint, head.slots, e.device))
                masks, confidence = target_tensors(before, head.slots, e.device)
                # Gain-zero base forecasts are independent of all object weights.
                # Cache those expensive forwards once; all mask graphs remain live.
                frozen = None
                if not bool(head.gain):
                    frozen, zz, hh = [], z, h
                    for action in actions:
                        next_z, next_h, _ = e.predict(zz, hh, action, *geometry)
                        frozen.append((zz.detach(), next_h.detach(), action, next_z.detach()))
                        zz, hh = next_z, next_h
                item = dict(i=i, t=t, window=window, before=before, after=after,
                            z=z, h=h, geometry=geometry, actions=actions,
                            masks=masks, confidence=confidence, targets=targets, frozen=frozen)
                prepared.append(item)
                current = head.observe(z, *geometry)
                add_quality(quality, 'observed_iou', current, (masks, confidence))
                for conditioning, source in (('sam', masks), ('student', probabilities(current))):
                    future = self._future(item, source)
                    add_quality(quality, conditioning+'_iou', future[0][2], targets[0])
                    if len(targets) == 2:
                        add_quality(quality, conditioning+'_rollout_iou', future[-1][2], targets[1])
                    if sample is None and conditioning == 'student':
                        sample = self._preview(sessions[i].game, item, current, future[0][2])
                copy = unchanged_prediction(masks)
                add_quality(quality, 'copy_iou', copy, targets[0])
                if len(targets) == 2:
                    add_quality(quality, 'copy_rollout_iou', copy, targets[1])
                if frozen is not None:
                    item['visual'] = self._visual_losses(item, self._future(item, masks))
        metrics.update({key: average(values) for key, values in quality.items()})
        metrics.update(observer_frames=len({row.key for row in current_rows}),
                       replay_frames=len(self.observations.rows))
        flags = [(p, p.requires_grad) for p in e.model.parameters()]
        rates = [g['lr'] for g in learner.optimizer.param_groups]
        losses, ious, rollouts, observed, observer_losses = [], [], [], [], []
        e.model.eval().requires_grad_(False); head.train().requires_grad_(True)
        try:
            # Same named Adam states; no old parameter receives an auxiliary step.
            for group in learner.optimizer.param_groups: group['lr'] = c.learning_rate
            for _ in range(c.passes):
                for item in prepared:
                    learner.optimizer.zero_grad(set_to_none=True)
                    with e.autocast():
                        current = head.observe(item['z'], *item['geometry'])
                        observed_loss, current_measure = mask_loss(current, item['masks'], item['confidence'])
                        conditioning = 'sam' if rng.random() < .5 else 'student'
                        source = item['masks'] if conditioning == 'sam' else probabilities(current)
                        future = self._future(item, source)
                        auxiliary, measured = mask_loss(future[0][2], *item['targets'][0])
                        visual = item.get('visual')
                        if visual is None:
                            visual = self._visual_losses(item, future)
                        loss = observed_loss + auxiliary + visual[0]
                        rollout_measure = None
                        if len(item['targets']) == 2:
                            term, rollout_measure = mask_loss(future[-1][2], *item['targets'][1])
                            loss = observed_loss + (auxiliary + term + visual[0] + visual[1])/2
                    self._step(learner, loss); head.updates.add_(1)
                    losses.append(float(loss.detach()))
                    if current_measure['mean_iou'] is not None: observed.append(current_measure['mean_iou'])
                    if measured['mean_iou'] is not None: ious.append(measured['mean_iou'])
                    metrics['matched'] += measured['matched']
                    if rollout_measure:
                        metrics['rollout_matched'] += rollout_measure['matched']
                        if rollout_measure['mean_iou'] is not None: rollouts.append(rollout_measure['mean_iou'])
            for row in self.observations.schedule(current_rows, c.observer_steps, rng):
                learner.optimizer.zero_grad(set_to_none=True)
                with e.autocast():
                    prediction = head.observe(row.latent.to(e.device), *row.geometry)
                    loss, _ = mask_loss(prediction, *target_tensors(row.labels, head.slots, e.device))
                self._step(learner, loss)
                observer_losses.append(float(loss.detach()))
            post = []
            with torch.no_grad(), e.autocast():
                for item in prepared:
                    _, measured = mask_loss(head.observe(item['z'], *item['geometry']),
                                            item['masks'], item['confidence'])
                    if measured['mean_iou'] is not None: post.append(measured['mean_iou'])
            # Admission is based on the unaided pre-fit forecasts, never the
            # easier teacher-conditioned or repeatedly fitted measurements.
            ready = all(metrics[key] is not None and metrics[key] >= c.activation_iou
                        for key in ('observed_iou', 'student_iou', 'student_rollout_iou'))
            if c.activate_residual and int(head.updates) >= c.warmup_updates and ready:
                head.gain.fill_(1.)
            metrics.update(updates=len(losses), total_updates=int(head.updates), loss=average(losses),
                mean_iou=average(ious), rollout_iou=average(rollouts), observed_fit_iou=average(observed),
                observed_post_iou=average(post), observer_updates=len(observer_losses),
                observer_loss=average(observer_losses), ready=ready)
        finally:
            learner.optimizer.zero_grad(set_to_none=True)
            for group, rate in zip(learner.optimizer.param_groups, rates): group['lr'] = rate
            for parameter, flag in flags: parameter.requires_grad_(flag)
            e.model.eval()
        return dict(metrics, seconds=time.perf_counter()-start, active=bool(head.gain)), sample

    def _future(self, item, masks):
        if item['frozen'] is None:
            return object_rollout(self.engine, item['z'], item['h'], item['actions'], item['geometry'], masks)
        result = []
        for z, h_next, action, predicted in item['frozen']:
            _, prediction = self.engine.model.dynamics.objects(z, h_next, action, *item['geometry'], masks)
            masks = probabilities(prediction)
            result.append((predicted, h_next, prediction))
        return result

    def _visual_losses(self, item, future):
        e = self.engine
        window, geometry = item['window'], item['geometry']
        a = torch.as_tensor(np.array(window[0].before.final), device=e.device).long()[None]
        result = []
        for index in (0, -1) if len(item['targets']) == 2 else (0,):
            b = torch.as_tensor(np.array(window[index].after.final), device=e.device).long()[None]
            result.append(visual_delta_loss(e.decode(future[index][0], *geometry), a, b,
                e.config.false_change_weight, mode=e.config.prediction_delta_loss).mean())
        return result

    def _step(self, learner, loss):
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite object dynamics loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.engine.model.dynamics.objects.parameters(), 1., error_if_nonfinite=True)
        learner.optimizer.step()

    def _preview(self, game, item, current, predicted):
        masks, presence = [value[0].detach().sigmoid().float().cpu().numpy() for value in predicted]
        learned, learned_presence = [value[0].detach().sigmoid().float().cpu().numpy() for value in current]
        return dict(game=game, step=item['t'], source='frozen SAM2 pseudo-labels',
            action=asdict(item['window'][0].action), conditioning='student', measurement='before auxiliary fitting',
            before=item['window'][0].before.final.tolist(), actual=item['window'][0].after.final.tolist(),
            observed_masks=label_image(item['before'].masks), actual_masks=label_image(item['after'].masks),
            learned_current_masks=label_image(learned, learned_presence), predicted_masks=label_image(masks, presence),
            instances=instance_preview(item['before'], item['after'], masks, presence,
                                       learned_current=learned, learned_presence=learned_presence))


def average(values):
    return sum(values)/len(values) if values else None


def add_quality(quality, key, prediction, target):
    _, measured = mask_loss(prediction, *target)
    if measured['mean_iou'] is not None:
        quality[key].append(measured['mean_iou'])


def unchanged_prediction(masks):
    """A literal no-action SAM-copy baseline, expressed in the metric's logits."""
    exists = masks.flatten(2).any(2)
    return torch.where(masks > .5, 20., -20.), torch.where(exists, 20., -20.)
