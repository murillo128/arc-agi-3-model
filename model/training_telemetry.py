"""Disposable dashboard telemetry; selected latents are decoded, never trained."""
import base64
from collections import deque
import time
import uuid

import numpy as np
import torch

from .campaign import write_report


class RecentThroughput:
    """Real training completions over a trailing wall-clock window."""
    def __init__(self, window=60., *, clock=time.monotonic):
        self.window, self.clock = window, clock
        self.started = None
        self.completions = deque()

    def record(self):
        now = self.clock()
        if self.started is None:
            # Initial compile/first action is warmup, not a timed interval.
            self.started = now
        else:
            self.completions.append(now)
        return self.value(now)

    def value(self, now=None):
        now = self.clock() if now is None else now
        if self.started is None or now <= self.started:
            return None
        while self.completions and self.completions[0] <= now - self.window:
            self.completions.popleft()
        return len(self.completions) / min(self.window, now - self.started)


class PlannerTrajectory:
    """One executed episode or explicitly partial window, atomically replaced at most every 2s."""
    def __init__(self, path, *, interval=2., max_bytes=4 * 1024 * 1024, clock=time.monotonic):
        self.path, self.interval, self.max_bytes, self.clock = path, interval, max_bytes, clock
        self.last_capture = -float('inf')
        self.metadata = None
        self.error = None
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.rows = None
        self.capture_error = None
        self.trace_bytes = 0

    def begin(self):
        """Reserve one eligible episode before paying any decoder/readout cost."""
        self.rows, self.capture_error, self.trace_bytes = None, None, 0
        if self.clock() - self.last_capture < self.interval:
            return None
        self.rows = []
        return self.capture_step

    @torch.no_grad()
    def capture_step(self, agent, action, exploring):
        """Capture an explored horizon BEFORE the real result or any weight update."""
        if self.capture_error:
            return
        try:
            planner = agent.planner
            goal = planner.goal
            image = np.where(goal['pixel_mask'], goal['frame'], 16).astype(np.uint8)[None] if goal else None
            row = dict(goal=image, imagination=[], status='exploration' if exploring else 'unavailable', source=None)
            selected = planner.last_plan
            plan = getattr(planner, 'last_forecast_plan', None) or selected
            states = getattr(planner, 'last_forecast_states', None) or planner.last_states
            # Exploration leaves old planner diagnostics untouched; never reuse them.
            if (not exploring and agent.last_diagnostics.get('path') and selected and selected[0] == action
                    and len(plan) == len(states)):
                horizon = min(3, len(states))
                with agent.engine.autocast():
                    images = agent.engine.decode(torch.stack(states[:horizon]).detach(), *agent.state.geometry)
                images = images.argmax(-1).to(torch.uint8).cpu().numpy()
                row['imagination'] = [dict(action=dict(id=a.id, data=dict(a.data)), frame=f[None])
                                      for a, f in zip(plan[:horizon], images)]
                row['selected_estimate'] = getattr(planner, 'last_estimate', None)
                row['forecast_estimate'] = getattr(planner, 'last_forecast_estimate', None)
                row['status'] = 'planned'
                row['source'] = getattr(planner, 'last_forecast_source', None) or 'selected_route'
            self.trace_bytes += (image.nbytes if image is not None else 0) + sum(
                p['frame'].nbytes for p in row['imagination'])
            if self.trace_bytes > self.max_bytes or len(self.rows) >= 128:
                raise ValueError('La trayectoria supera el límite de tamaño de la muestra')
            self.rows.append(row)
        except (ValueError, RuntimeError) as error:
            self.capture_error = str(error)
            self.rows = []

    def add(self, episode, records, result, **context):
        now = self.clock()
        if now - self.last_capture < self.interval:
            return
        self.last_capture = now
        try:
            if self.capture_error:
                raise ValueError(self.capture_error)
            if not 0 < len(episode.actions) <= 128 or len(records) != len(episode.actions):
                raise ValueError('Trayectoria sin acciones o incompleta')
            if len(episode.observations) != len(episode.actions) + 1:
                raise ValueError('Faltan observaciones de la trayectoria')
            arrays = [episode.observations[0].frames[-1:]] + [o.frames for o in episode.observations[1:]]
            if sum(a.nbytes for a in arrays) + self.trace_bytes > self.max_bytes:
                raise ValueError('La trayectoria supera el límite de tamaño de la muestra')
            if any(a.dtype != np.uint8 or a.ndim != 3 or min(a.shape) < 1 or np.any(a > 15) for a in arrays):
                raise ValueError('La trayectoria contiene frames ausentes o inválidos')
            def pack(array):
                return dict(shape=list(array.shape), pixels=base64.b64encode(array.tobytes()).decode('ascii'))
            steps = [dict(action=dict(id=action.id, data=dict(action.data)), source=record['action_source'],
                          goal=record['goal'], goal_source=record.get('goal_source'), frames=pack(frames))
                     for action, record, frames in zip(episode.actions, records, arrays[1:])]
            if self.rows is not None:
                if len(self.rows) != len(steps):
                    raise ValueError('Faltan predicciones de la trayectoria')
                for step, row in zip(steps, self.rows):
                    step.update(goal_frame=pack(row['goal']) if row['goal'] is not None else None,
                                selected_estimate=row.get('selected_estimate'), forecast_estimate=row.get('forecast_estimate'),
                                imagination_status=row['status'], imagination_source=row['source'], imagination=[
                                    dict(action=p['action'], frames=pack(p['frame'])) for p in row['imagination']])
            self.sequence += 1
            metadata = dict(id=f'{self.session}:{self.sequence}', captured_at=time.time(), **context,
                            actions=len(steps), frame_count=sum(len(a) for a in arrays),
                            success=result['success'], reason=result['reason'])
            write_report(self.path, dict(kind='planner_trajectory', format_version=2 if self.rows is not None else 1,
                                        **metadata, initial=pack(arrays[0]), steps=steps))
            self.metadata, self.error = metadata, None
        except (OSError, ValueError) as error:
            # The viewer must never terminate the optimizer or replace a good
            # sample with a partial one. Training's checkpoint IO is separate.
            self.error = str(error)
