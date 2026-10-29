"""Small causal instance dynamics; no SAM dependency in learned inference."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def assignment(cost):
    """Minimum-cost injection of target columns into prediction rows (Hungarian)."""
    if cost.ndim != 2 or not torch.isfinite(cost).all():
        raise ValueError('Object assignment requires a finite cost matrix')
    a = cost.detach().double().cpu().T.tolist()
    n, m = len(a), len(a[0]) if len(a) else 0
    if n > m:
        raise ValueError('More object targets than prediction slots')
    u, v, p, way = [0.]*(n+1), [0.]*(m+1), [0]*(m+1), [0]*(m+1)
    for i in range(1, n+1):
        p[0] = i; j0 = 0
        minimum, used = [float('inf')]*(m+1), [False]*(m+1)
        while True:
            used[j0] = True; i0 = p[j0]; delta = float('inf'); j1 = 0
            for j in range(1, m+1):
                if not used[j]:
                    value = a[i0-1][j-1] - u[i0] - v[j]
                    if value < minimum[j]: minimum[j], way[j] = value, j0
                    if minimum[j] < delta: delta, j1 = minimum[j], j
            for j in range(m+1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else: minimum[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while j0:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
    return [(j-1, p[j]-1) for j in range(1, m+1) if p[j]]


def mask_loss(prediction, target, confidence, *, missing_weight=.05):
    """Per-instance Dice + balanced BCE; teacher omissions have weak weight.

    Order-free targets, real foreground and background averaged separately.
    An entirely empty teacher result is unknown and contributes no supervision.
    """
    logits, presence = prediction
    losses, ious = [], []
    for masks, exists, real, scores in zip(logits.float(), presence.float(), target, confidence):
        real, scores = real.float(), scores.float()
        valid = (scores > 0) & real.flatten(1).any(1)
        real, scores = real[valid], scores[valid]
        if not len(real):
            continue
        prob = masks.sigmoid().flatten(1)
        truth = real.flatten(1)
        overlap = prob @ truth.T
        dice = 1 - (2*overlap + 1e-6) / (prob.sum(1)[:, None] + truth.sum(1)[None] + 1e-6)
        pairs = assignment(dice - exists.sigmoid()[:, None] * .1)
        pi, ti = zip(*pairs)
        selected, expected = masks[list(pi)], real[list(ti)]
        ce = F.binary_cross_entropy_with_logits(selected, expected, reduction='none')
        positive, negative = expected.sum((1, 2)), (1-expected).sum((1, 2))
        balanced = (ce*expected).sum((1, 2))/positive.clamp_min(1)
        balanced += (ce*(1-expected)).sum((1, 2))/negative.clamp_min(1)
        balanced /= (positive > 0).float() + (negative > 0).float()
        weights = scores[list(ti)]
        objects = ((balanced + dice[list(pi), list(ti)]) * weights).sum()/weights.sum().clamp_min(1e-6)
        labels = torch.zeros_like(exists); labels[list(pi)] = 1
        existence_weight = torch.full_like(exists, missing_weight)
        existence_weight[list(pi)] = weights
        existence = (F.binary_cross_entropy_with_logits(exists, labels, reduction='none') * existence_weight).sum()
        losses.append(objects + existence/existence_weight.sum().clamp_min(1e-6))
        hard = (selected > 0) & (exists[list(pi)] > 0)[:, None, None]
        true = expected.bool()
        union = (hard | true).sum((1, 2)).clamp_min(1)
        ious.extend(((hard & true).sum((1, 2))/union).detach().cpu().tolist())
    zero = logits.sum()*0 + presence.sum()*0
    return (torch.stack(losses).mean() if losses else zero), dict(
        matched=len(ious), mean_iou=sum(ious)/len(ious) if ious else None)


def probabilities(prediction):
    masks, presence = prediction
    return masks.sigmoid() * presence.sigmoid()[:, :, None, None]


class ObjectDynamics(nn.Module):
    """Read current instances, transport/deform them, and add a zero-start residual.

    Slots with little occupancy use learned discovery queries, so new instances
    can appear. Masks are soft through rollouts; no argmax or identity reset.
    """
    def __init__(self, config):
        super().__init__()
        self.patch, self.slots, self.width = config.patch, config.object_slots, config.object_width
        channels = config.latent // config.patch**2
        w = self.width
        self.features = nn.Sequential(nn.Conv2d(channels+2, w, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(w, w, 3, padding=1), nn.SiLU())
        self.segment = nn.Conv2d(w, self.slots, 1)
        self.existence = nn.Linear(w, self.slots)
        self.memory = nn.Linear(config.memory, w)
        self.queries = nn.Parameter(torch.randn(self.slots, w)*.02)
        self.transition = nn.Sequential(nn.Linear(2*w+12, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU())
        self.motion = nn.Linear(w, 4)  # dx/dy, angle, log scale
        self.refine = nn.Linear(w, w)
        self.presence = nn.Linear(w, 1)
        self.output = nn.Conv2d(w, channels, 1)
        self.register_buffer('gain', torch.tensor(0.))
        self.register_buffer('updates', torch.tensor(0, dtype=torch.long))
        self.reset_outputs()

    def reset_outputs(self):
        nn.init.zeros_(self.motion.weight); nn.init.zeros_(self.motion.bias)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        nn.init.constant_(self.existence.bias, -2.)

    def grid(self, z, height, width):
        p = self.patch; rows, cols = (height+p-1)//p, (width+p-1)//p
        grid = z.reshape(len(z), rows, cols, p, p, -1).permute(0, 5, 1, 3, 2, 4)
        return grid.reshape(len(z), -1, rows*p, cols*p)[:, :, :height, :width].float()

    def pack(self, grid, z, height, width):
        p = self.patch; rows, cols = (height+p-1)//p, (width+p-1)//p
        grid = F.pad(grid, (0, cols*p-width, 0, rows*p-height))
        return grid.reshape(len(z), -1, rows, p, cols, p).permute(0, 2, 4, 3, 5, 1).reshape_as(z)

    def scene(self, z, height, width):
        grid = self.grid(z, height, width)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=z.device),
                               torch.linspace(-1, 1, width, device=z.device), indexing='ij')
        xy = torch.stack((xx, yy))[None].expand(len(z), -1, -1, -1)
        return self.features(torch.cat((grid, xy), 1))

    def observe(self, z, height, width):
        features = self.scene(z, height, width)
        return self.segment(features), self.existence(features.mean((2, 3)))

    def pooled(self, features, masks):
        mass = masks.sum((2, 3))
        tokens = torch.einsum('bkhw,bdhw->bkd', masks, features) / mass.clamp_min(1)[:, :, None]
        return tokens + self.queries[None] * (1-mass.clamp(0, 1))[:, :, None]

    def context(self, z, height, width, masks=None):
        features = self.scene(z, height, width)
        if masks is None:
            masks = probabilities((self.segment(features), self.existence(features.mean((2, 3)))))
        tokens = self.pooled(features, masks)
        field = torch.einsum('bkhw,bkd->bdhw', masks, tokens) / masks.sum(1, keepdim=True).clamp_min(1)
        return self.gain * self.pack(self.output(field), z, height, width)

    def forward(self, z, h, action, height, width, masks=None):
        features = self.scene(z, height, width)
        if masks is None:
            masks = probabilities((self.segment(features), self.existence(features.mean((2, 3)))))
        tokens = self.pooled(features, masks)
        condition = torch.cat((self.memory(h.float()).tanh(), action.float()), -1)
        query = self.transition(torch.cat((tokens, condition[:, None].expand(-1, self.slots, -1)), -1))
        motion = self.motion(query).float()
        angle, scale = motion[..., 2].tanh()*math.pi, motion[..., 3].tanh().exp()
        transform = torch.zeros(len(z), self.slots, 2, 3, device=z.device)
        transform[..., 0, 0] = angle.cos()/scale; transform[..., 1, 1] = angle.cos()/scale
        transform[..., 0, 1] = angle.sin()/scale * (height/width)
        transform[..., 1, 0] = -angle.sin()/scale * (width/height)
        transform[..., :, 2] = motion[..., :2].tanh()
        grid = F.affine_grid(transform.flatten(0, 1), (len(z)*self.slots, 1, height, width), align_corners=False)
        transported = F.grid_sample(masks.flatten(0, 1)[:, None].float(), grid,
                                     align_corners=False).reshape_as(masks)
        residual = torch.einsum('bkd,bdhw->bkhw', self.refine(query), features) / math.sqrt(self.width)
        logits = torch.logit(transported.clamp(.01, .99)) + residual
        exists = self.presence(query).squeeze(-1)
        predicted = probabilities((logits, exists))
        field = torch.einsum('bkhw,bkd->bdhw', predicted, query) / predicted.sum(1, keepdim=True).clamp_min(1)
        correction = self.gain * self.pack(self.output(field), z, height, width)
        return correction, (logits, exists)
