"""Disposable dashboard telemetry; no model calls or training random state."""
import base64
from collections import deque
import time
import uuid

import numpy as np

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
    """One complete executed episode, atomically replaced at most every 2s."""
    def __init__(self, path, *, interval=2., max_bytes=4 * 1024 * 1024, clock=time.monotonic):
        self.path, self.interval, self.max_bytes, self.clock = path, interval, max_bytes, clock
        self.last_capture = -float('inf')
        self.metadata = None
        self.error = None
        self.session = uuid.uuid4().hex
        self.sequence = 0

    def add(self, episode, records, result, **context):
        now = self.clock()
        if now - self.last_capture < self.interval:
            return
        self.last_capture = now
        try:
            if not 0 < len(episode.actions) <= 128 or len(records) != len(episode.actions):
                raise ValueError('Trayectoria sin acciones o incompleta')
            if len(episode.observations) != len(episode.actions) + 1:
                raise ValueError('Faltan observaciones de la trayectoria')
            arrays = [episode.observations[0].frames[-1:]] + [o.frames for o in episode.observations[1:]]
            if sum(a.nbytes for a in arrays) > self.max_bytes:
                raise ValueError('La trayectoria supera el límite de tamaño de la muestra')
            if any(a.dtype != np.uint8 or a.ndim != 3 or min(a.shape) < 1 or np.any(a > 15) for a in arrays):
                raise ValueError('La trayectoria contiene frames ausentes o inválidos')
            def pack(array):
                return dict(shape=list(array.shape), pixels=base64.b64encode(array.tobytes()).decode('ascii'))
            steps = [dict(action=dict(id=action.id, data=dict(action.data)), source=record['action_source'],
                          goal=record['goal'], frames=pack(frames))
                     for action, record, frames in zip(episode.actions, records, arrays[1:])]
            self.sequence += 1
            metadata = dict(id=f'{self.session}:{self.sequence}', captured_at=time.time(), **context,
                            actions=len(steps), frame_count=sum(len(a) for a in arrays),
                            success=result['success'], reason=result['reason'])
            write_report(self.path, dict(kind='planner_trajectory', **metadata, initial=pack(arrays[0]), steps=steps))
            self.metadata, self.error = metadata, None
        except (OSError, ValueError) as error:
            # The viewer must never terminate the optimizer or replace a good
            # sample with a partial one. Training's checkpoint IO is separate.
            self.error = str(error)
