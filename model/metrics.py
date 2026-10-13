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


class VisualQuality:
    """Color-macro pixel errors expose rare/static detail loss without game rules."""
    def __init__(self):
        self.counts = np.zeros((3, 16), dtype=np.int64)
        self.wrong = np.zeros_like(self.counts)

    def add(self, before, actual, predicted):
        if before.shape != actual.shape or actual.shape != predicted.shape:
            raise ValueError('Visual quality requires matching geometry')
        for i, mask in enumerate((np.ones_like(actual, dtype=bool), actual == before, actual != before)):
            self.counts[i] += np.bincount(actual[mask], minlength=16)
            self.wrong[i] += np.bincount(actual[mask & (actual != predicted)], minlength=16)

    def report(self):
        result = {}
        for i, name in enumerate(('color_error', 'static_color_error', 'changed_color_error')):
            present = self.counts[i] > 0
            result[name] = float((self.wrong[i, present] / self.counts[i, present]).mean()) if present.any() else None
        result['by_color'] = {str(i): {'pixels': int(n), 'error': float(self.wrong[0, i] / n)}
                              for i, n in enumerate(self.counts[0]) if n}
        return result


@dataclass
class Metrics:
    transitions: int = 0
    changed_transitions: int = 0
    wrong: int = 0
    union: int = 0
    missed_changes: int = 0
    extra_changes: int = 0
    wrong_replacement_colors: int = 0
    correct_changes: int = 0
    latent_squared: float = 0.
    latent_elements: int = 0
    latent_n: dict = field(default_factory=dict, repr=False)
    latent_sum: dict = field(default_factory=dict, repr=False)
    latent_squares: dict = field(default_factory=dict, repr=False)
    skipped: int = 0

    def observe_latent(self, z, geometry=None):
        z = np.asarray(z, dtype=np.float64)
        key = (geometry, z.shape)
        if key not in self.latent_n:
            self.latent_n[key] = 0
            self.latent_sum[key] = np.zeros_like(z)
            self.latent_squares[key] = np.zeros_like(z)
        self.latent_n[key] += 1
        self.latent_sum[key] += z
        self.latent_squares[key] += z * z

    def add(self, before, actual, predicted, z_before, z_after, predicted_delta):
        wrong, union, changed = delta_counts(before, actual, predicted)
        self.transitions += 1
        self.changed_transitions += bool(changed)
        self.wrong += wrong
        self.union += union
        real_change, predicted_change = actual != before, predicted != before
        self.missed_changes += int((real_change & ~predicted_change).sum())
        self.extra_changes += int((~real_change & predicted_change).sum())
        self.wrong_replacement_colors += int((real_change & predicted_change & (actual != predicted)).sum())
        self.correct_changes += int((real_change & (actual == predicted)).sum())
        error = np.asarray(predicted_delta, dtype=np.float64) - (z_after - z_before)
        self.latent_squared += float(np.square(error).sum())
        self.latent_elements += error.size

    def report(self):
        spread = 0.
        for key, count in self.latent_n.items():
            variance = self.latent_squares[key] / count - (self.latent_sum[key] / count) ** 2
            spread += count * float(np.sqrt(np.maximum(variance, 0)).mean())
        return {'transitions': self.transitions, 'changed_transitions': self.changed_transitions,
                'delta_error': self.wrong / max(1, self.union),
                'missed_changes': self.missed_changes, 'extra_changes': self.extra_changes,
                'wrong_replacement_colors': self.wrong_replacement_colors, 'correct_changes': self.correct_changes,
                'latent_delta_mse': self.latent_squared / max(1, self.latent_elements),
                'latent_std': spread / max(1, sum(self.latent_n.values())),
                'skipped': self.skipped}

    def passes(self, config):
        r = self.report()
        return (r['transitions'] >= config.min_validation_transitions and
                r['changed_transitions'] >= config.min_changed_transitions and
                r['delta_error'] <= config.delta_error and
                r['latent_delta_mse'] <= config.latent_delta_mse and
                r['latent_std'] >= config.min_latent_std and r['skipped'] == 0)
