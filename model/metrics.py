"""Prediction quality, independent of game completion or background size."""
from dataclasses import dataclass, field
import numpy as np


def visual_delta(reference, final):
    if reference.shape != final.shape:
        raise ValueError('A visual delta requires identical native canvas geometry')
    return np.where(final == reference, 16, final).astype(np.uint8)


def delta_counts(before, actual, predicted):
    if before.shape != actual.shape or actual.shape != predicted.shape:
        raise ValueError('A visual delta requires identical native canvas geometry')
    actual_delta, predicted_delta = visual_delta(before, actual), visual_delta(before, predicted)
    changed = actual_delta != 16
    predicted_changed = predicted_delta != 16
    union = changed | predicted_changed
    # Missing changes, spurious changes and wrong new colors all count.
    wrong = union & (actual_delta != predicted_delta)
    return int(wrong.sum()), int(union.sum()), int(changed.sum())


@dataclass
class Metrics:
    transitions: int = 0
    changed_transitions: int = 0
    wrong: int = 0
    union: int = 0
    latent_squared: float = 0.
    latent_elements: int = 0
    latent_n: int = 0
    latent_sum: np.ndarray | None = field(default=None, repr=False)
    latent_squares: np.ndarray | None = field(default=None, repr=False)
    skipped: int = 0

    def observe_latent(self, z):
        z = np.asarray(z, dtype=np.float64)
        if self.latent_sum is None:
            self.latent_sum = np.zeros_like(z)
            self.latent_squares = np.zeros_like(z)
        self.latent_n += 1
        self.latent_sum += z
        self.latent_squares += z * z

    def add(self, before, actual, predicted, z_before, z_after, predicted_delta):
        wrong, union, changed = delta_counts(before, actual, predicted)
        self.transitions += 1
        self.changed_transitions += bool(changed)
        self.wrong += wrong
        self.union += union
        error = np.asarray(predicted_delta, dtype=np.float64) - (z_after - z_before)
        self.latent_squared += float(np.square(error).sum())
        self.latent_elements += error.size

    def report(self):
        variance = (self.latent_squares / self.latent_n - (self.latent_sum / self.latent_n) ** 2
                    if self.latent_n else np.array([0.]))
        return {'transitions': self.transitions, 'changed_transitions': self.changed_transitions,
                'delta_error': self.wrong / max(1, self.union),
                'latent_delta_mse': self.latent_squared / max(1, self.latent_elements),
                'latent_std': float(np.sqrt(np.maximum(variance, 0)).mean()),
                'skipped': self.skipped}

    def passes(self, config):
        r = self.report()
        return (r['transitions'] >= config.min_validation_transitions and
                r['changed_transitions'] >= config.min_changed_transitions and
                r['delta_error'] <= config.delta_error and
                r['latent_delta_mse'] <= config.latent_delta_mse and
                r['latent_std'] >= config.min_latent_std and r['skipped'] == 0)
