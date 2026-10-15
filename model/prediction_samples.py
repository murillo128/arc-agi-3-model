"""Bounded, disposable previews of predictions made before seeing real results."""
from collections import deque
import random
import time

from .metrics import delta_counts, visual_delta


class PredictionSamples:
    def __init__(self, limit=64, *, seed=None):
        self.samples = deque(maxlen=limit)
        # Display sampling must not consume the environment/training RNG.
        self.rng = random.Random(seed)
        self.sequence = 0
        self.previous = None

    def add(self, reference, actual, predicted, *, reconstruction=None, **context):
        wrong, union, _ = delta_counts(reference, actual, predicted)
        self.sequence += 1
        self.samples.append(dict(id=self.sequence, captured_at=time.time(), **context,
            previous=reference.copy(), reconstruction=None if reconstruction is None else reconstruction.copy(),
            real_delta=visual_delta(reference, actual),
            predicted_delta=visual_delta(reference, predicted), wrong=wrong, union=union))

    def draw(self):
        if not self.samples:
            return None
        choices = [s for s in self.samples if s['id'] != self.previous] or list(self.samples)
        selected = self.rng.choice(choices)
        self.previous = selected['id']
        return {key: value.tolist() if value is not None and key in
                ('previous', 'reconstruction', 'real_delta', 'predicted_delta') else value
                for key, value in selected.items()}
