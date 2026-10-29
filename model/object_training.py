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

    def begin_collection(self):
        self.segmenter.remaining = self.config.click_frames

    def fit(self, learner, sessions):
        e, c = self.engine, self.config
        head = e.model.dynamics.objects
        start = time.perf_counter()
        self.segmenter.remaining = c.teacher_frames
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
        examples = []
        for i, t in selected:
            window = rollout_window(sessions[i], t, 3) or rollout_window(sessions[i], t, 1)
            before = self.segmenter.segment(window[0].before.final)
            after = self.segmenter.segment(window[0].after.final)
            endpoint = self.segmenter.segment(window[-1].after.final) if len(window) == 3 else None
            if before is not None and after is not None and len(before.masks) and len(after.masks):
                examples.append((i, t, window, before, after, endpoint))
        metrics = dict(samples=len(examples), updates=0, loss=None, mean_iou=None, rollout_iou=None,
                       matched=0, rollout_matched=0, teacher=self.segmenter.report())
        sample = None
        if not examples:
            return dict(metrics, seconds=time.perf_counter()-start, active=bool(head.gain)), sample
        encoded = EncodedSessions(e, sessions)
        memories = learner.zero_memories(sessions)
        states = {}
        wanted = {(i, t) for i, t, *_ in examples}
        with torch.no_grad(), e.autocast():
            for t in range(max(t for _, t in wanted)+1):
                for ids, _, z, h, action, _ in memory_inputs(e, encoded, t, memories):
                    next_h, _ = e.advance(z, h, action)
                    for k, i in enumerate(ids):
                        if (i, t) in wanted:
                            states[i, t] = (z[k:k+1].detach(), h[k:k+1].detach())
                        memories[i] = next_h[k:k+1]
        flags = [(p, p.requires_grad) for p in e.model.parameters()]
        rates = [g['lr'] for g in learner.optimizer.param_groups]
        losses, ious, rollouts, observed = [], [], [], []
        e.model.eval().requires_grad_(False); head.train().requires_grad_(True)
        try:
            # Same named Adam states; no old parameter receives an auxiliary step.
            for group in learner.optimizer.param_groups: group['lr'] = c.learning_rate
            for _ in range(c.passes):
                for i, t, window, before, after, endpoint in examples:
                    learner.optimizer.zero_grad(set_to_none=True)
                    z, h = states[i, t]; geometry = window[0].before.final.shape
                    masks, confidence = target_tensors(before, head.slots, e.device)
                    target, target_confidence = target_tensors(after, head.slots, e.device)
                    with e.autocast():
                        current = head.observe(z, *geometry)
                        observed_loss, current_measure = mask_loss(current, masks, confidence)
                        # Both teacher-conditioned and student-conditioned dynamics
                        # are trained. During imagination only predictions survive.
                        conditioning = 'sam' if rng.random() < .5 else 'student'
                        source = masks if conditioning == 'sam' else probabilities(current)
                        actions = [e.actions([s.action]) for s in window]
                        future = object_rollout(e, z, h, actions, geometry, source, warmup=not bool(head.gain))
                        predicted, _, prediction = future[0]
                        auxiliary, measured = mask_loss(prediction, target, target_confidence)
                        a = torch.as_tensor(np.array(window[0].before.final), device=e.device).long()[None]
                        b = torch.as_tensor(np.array(window[0].after.final), device=e.device).long()[None]
                        visual = visual_delta_loss(e.decode(predicted, *geometry), a, b,
                            e.config.false_change_weight, mode=e.config.prediction_delta_loss).mean()
                        loss = observed_loss + auxiliary + visual
                        rollout_measure = None
                        if len(future) == 3 and endpoint is not None and len(endpoint.masks):
                            target3, confidence3 = target_tensors(endpoint, head.slots, e.device)
                            term, rollout_measure = mask_loss(future[-1][2], target3, confidence3)
                            b3 = torch.as_tensor(np.array(window[-1].after.final), device=e.device).long()[None]
                            visual3 = visual_delta_loss(e.decode(future[-1][0], *geometry), a, b3,
                                e.config.false_change_weight, mode=e.config.prediction_delta_loss).mean()
                            loss = observed_loss + (auxiliary + term + visual + visual3)/2
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite object dynamics loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                    learner.optimizer.step(); head.updates.add_(1)
                    losses.append(float(loss.detach()))
                    if current_measure['mean_iou'] is not None: observed.append(current_measure['mean_iou'])
                    if measured['mean_iou'] is not None: ious.append(measured['mean_iou'])
                    metrics['matched'] += measured['matched']
                    if rollout_measure:
                        metrics['rollout_matched'] += rollout_measure['matched']
                        if rollout_measure['mean_iou'] is not None: rollouts.append(rollout_measure['mean_iou'])
                    if sample is None:
                        predicted_masks = prediction[0][0].detach().sigmoid().float().cpu().numpy()
                        predicted_presence = prediction[1][0].detach().sigmoid().float().cpu().numpy()
                        sample = dict(game=sessions[i].game, step=t, source='frozen SAM2 pseudo-labels',
                            action=asdict(window[0].action), conditioning=conditioning,
                            before=window[0].before.final.tolist(), actual=window[0].after.final.tolist(),
                            observed_masks=label_image(before.masks), actual_masks=label_image(after.masks),
                            predicted_masks=label_image(predicted_masks, predicted_presence),
                            instances=instance_preview(before, after, predicted_masks, predicted_presence))
            mean_iou = sum(ious)/len(ious) if ious else None
            observed_iou = sum(observed)/len(observed) if observed else None
            rollout_iou = sum(rollouts)/len(rollouts) if rollouts else None
            ready = all(value is not None and value >= c.activation_iou
                        for value in (observed_iou, mean_iou, rollout_iou))
            if c.activate_residual and int(head.updates) >= c.warmup_updates and ready:
                head.gain.fill_(1.)
            metrics.update(updates=len(losses), total_updates=int(head.updates), loss=sum(losses)/len(losses),
                mean_iou=mean_iou, rollout_iou=rollout_iou, observed_iou=observed_iou, ready=ready)
        finally:
            learner.optimizer.zero_grad(set_to_none=True)
            for group, rate in zip(learner.optimizer.param_groups, rates): group['lr'] = rate
            for parameter, flag in flags: parameter.requires_grad_(flag)
            e.model.eval()
        return dict(metrics, seconds=time.perf_counter()-start, active=bool(head.gain)), sample
