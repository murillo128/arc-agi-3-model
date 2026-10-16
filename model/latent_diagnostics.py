"""Read-only latent diagnostics on fixed real evaluation trajectories.

Keep the first eight observed states per episode on CPU. Compare coordinates at
matching positions/geometries; positional codes alone must not count as diversity.
No sampling RNG, optimizer, whitening or extra encoder invocation is involved.
"""
from collections import defaultdict
import math

import numpy as np
import torch


OBSERVATIONS_PER_EPISODE = 8
RANK_OBSERVATIONS = 64
STD_THRESHOLD = .01


class LatentProbe:
    def __init__(self, patch):
        self.patch = patch
        self.samples = []
        self.squared = dict(changed=0., unchanged=0.)
        self.elements = dict(changed=0, unchanged=0)
        self.regions = dict(changed=0, unchanged=0)
        self.transitions = self.skipped = 0

    def sample(self, state):
        if state.z is not None and len(self.samples) < OBSERVATIONS_PER_EPISODE:
            self.samples.append((tuple(state.geometry), state.z.detach().float().cpu().clone()))

    @torch.no_grad()
    def transition(self, previous, current, before, after):
        """RMS of delta-z in patches with/without actual final-frame changes."""
        if (previous.z is None or current.z is None or before is None or after is None
                or before.shape != after.shape or previous.geometry != current.geometry
                or previous.z.shape != current.z.shape):
            self.skipped += 1
            return
        h, w = before.shape
        p = self.patch
        rows, cols = (h + p - 1) // p, (w + p - 1) // p
        if previous.z.ndim != 2 or previous.z.shape[0] != rows * cols:
            self.skipped += 1
            return
        changed = np.pad(before != after, ((0, rows * p - h), (0, cols * p - w)))
        changed = changed.reshape(rows, p, cols, p).any(axis=(1, 3)).reshape(-1)
        mask = torch.as_tensor(changed, device=current.z.device)
        delta = (current.z.detach().float() - previous.z.detach().float()).square().sum(-1)
        for name, selected in (('changed', mask), ('unchanged', ~mask)):
            count = int(selected.sum())
            self.squared[name] += float(delta[selected].sum())
            self.elements[name] += count * current.z.shape[-1]
            self.regions[name] += count
        self.transitions += 1


def _spread(samples):
    # Welford on CPU avoids cancellation and a full double-precision batch copy.
    mean = torch.zeros_like(samples[0], dtype=torch.float64)
    m2 = torch.zeros_like(mean)
    for n, sample in enumerate(samples, 1):
        value = sample.double()
        delta = value - mean
        mean += delta / n
        m2 += delta * (value - mean)
    return (m2 / len(samples)).clamp_min(0).sqrt()


def _rank(samples):
    # Evenly spaced deterministic subsampling bounds the Gram matrix and CPU cost.
    indices = torch.linspace(0, len(samples) - 1, min(len(samples), RANK_OBSERVATIONS)).long()
    matrix = torch.stack([samples[int(i)].flatten() for i in indices]).double()
    matrix -= matrix.mean(0)
    eigenvalues = torch.linalg.eigvalsh(matrix @ matrix.T / matrix.shape[1]).clamp_min(0)
    total = float(eigenvalues.sum())
    maximum = min(len(matrix) - 1, matrix.shape[1])
    if total == 0:
        return 0., maximum, len(matrix)  # Convention for an exactly constant latent.
    probabilities = eigenvalues[eigenvalues > 0] / total
    effective = float(torch.exp(-(probabilities * probabilities.log()).sum()))
    return min(effective, maximum), maximum, len(matrix)


@torch.no_grad()
def summarize_latents(probes):
    """Geometry groups are never mixed; reported means weight observations."""
    groups = defaultdict(list)
    within_sum = within_count = 0
    for probe in probes:
        episode_groups = defaultdict(list)
        for geometry, z in probe.samples:
            key = (geometry, tuple(z.shape))
            groups[key].append(z)
            episode_groups[key].append(z)
        for samples in episode_groups.values():
            if len(samples) >= 2:
                within_sum += len(samples) * float(_spread(samples).mean())
                within_count += len(samples)
    std_sum = inactive_sum = rank_sum = maximum_sum = 0.
    count = rank_count = 0
    for samples in groups.values():
        if len(samples) < 2:
            continue
        std = _spread(samples)
        n = len(samples)
        std_sum += n * float(std.mean())
        inactive_sum += n * float((std < STD_THRESHOLD).double().mean())
        rank, maximum, used = _rank(samples)
        rank_sum += used * rank
        maximum_sum += used * maximum
        count += n
        rank_count += used
    result = dict(
        std_mean=std_sum / count if count else None,
        within_episode_std=within_sum / within_count if within_count else None,
        inactive_fraction=inactive_sum / count if count else None,
        effective_rank=rank_sum / rank_count if rank_count else None,
        max_rank=maximum_sum / rank_count if rank_count else None,
        sample_count=sum(len(p.samples) for p in probes),
        episode_count=sum(bool(p.samples) for p in probes), geometry_count=len(groups),
        rank_sample_count=rank_count, std_threshold=STD_THRESHOLD,
        observations_per_episode=OBSERVATIONS_PER_EPISODE,
        rank_observations_per_geometry=RANK_OBSERVATIONS,
        transitions=sum(p.transitions for p in probes), skipped_transitions=sum(p.skipped for p in probes),
        patch_sizes=sorted({p.patch for p in probes}))
    for name in ('changed', 'unchanged'):
        elements = sum(p.elements[name] for p in probes)
        result[name + '_rms'] = math.sqrt(sum(p.squared[name] for p in probes) / elements) if elements else None
        result[name + '_regions'] = sum(p.regions[name] for p in probes)
    return result
