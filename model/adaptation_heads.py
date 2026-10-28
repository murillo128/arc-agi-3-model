"""Portable learned goal head and real-outcome objectives, without collectors."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class VisualGoal(nn.Module):
    """Spatial palette/mask prediction from observation and rule memory only."""
    def __init__(self, latent, memory, patch, width):
        super().__init__()
        self.patch = patch
        self.norm = nn.LayerNorm(latent)
        self.project = nn.Linear(latent, width)
        self.memory = nn.Linear(memory, width)
        self.spatial = nn.Sequential(nn.Conv2d(width + 2, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, 17 * patch * patch, 1), nn.PixelShuffle(patch))

    def forward(self, z, h, height, width):
        p = self.patch
        rows, columns = (height+p-1)//p, (width+p-1)//p
        x = self.project(self.norm(z.float())) + self.memory(h.float()).tanh()[:, None]
        x = x.transpose(1, 2).reshape(len(z), -1, rows, columns)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, rows, device=z.device),
                                torch.linspace(-1, 1, columns, device=z.device), indexing='ij')
        xy = torch.stack((xx, yy))[None].expand(len(z), -1, -1, -1)
        result = self.spatial(torch.cat((x, xy), 1))[:, :, :height, :width]
        return result[:, :16].permute(0, 2, 3, 1), result[:, 16]


def balanced_regions(values, first, second):
    dims = tuple(range(1, values.ndim))
    a, b = first.sum(dims), second.sum(dims)
    means = (values * first).sum(dims) / a.clamp_min(1)
    means += (values * second).sum(dims) / b.clamp_min(1)
    return means / ((a > 0).float() + (b > 0).float()).clamp_min(1)


def goal_loss(logits, mask_logits, before, target, mask):
    changed = (target != before) & mask
    color = F.cross_entropy(logits.float().movedim(-1, 1), target, reduction='none')
    color = balanced_regions(color, changed, mask & ~changed)
    loss = F.binary_cross_entropy_with_logits(mask_logits.float(), mask.float(), reduction='none')
    return color.mean() + balanced_regions(loss, mask, ~mask).mean()


def matches(frame, goal, mask):
    return frame is not None and frame.shape == goal.shape and bool(mask.any()) and np.array_equal(frame[mask], goal[mask])


def path_loss(logits, cost, targets, duration):
    loss = F.binary_cross_entropy_with_logits(logits.float(), targets)
    succeeded = targets.bool()
    if succeeded.any():
        loss = loss + F.smooth_l1_loss(cost.float()[succeeded], duration[succeeded])
    return loss
