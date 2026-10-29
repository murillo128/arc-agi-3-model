"""Frozen, offline SAM2 pseudo-labels in native SDK coordinates.

SAM is optional and imported only when explicitly configured. No game rules,
future frames, downloads or trainable teacher state enter this boundary.
"""
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np

SAM2_REVISION = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
TINY_SHA256 = '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'
PALETTE = np.array([list(bytes.fromhex(c)) for c in (
    'ffffff', 'cccccc', '999999', '666666', '333333', '000000', 'e53aa3', 'ff7bcc',
    'f93c31', '1e93ff', '88d8f1', 'ffdc00', 'ff851b', '921231', '4fcc30', 'a356d6')], dtype=np.uint8)


@dataclass(frozen=True)
class Segmentation:
    masks: np.ndarray
    confidence: np.ndarray
    complete: bool = False  # Capacity/filtering is never proof of full scene coverage.


def native_masks(proposals, shape, capacity):
    """Project probabilities to native pixels, deduplicate, keep small proposals.

    A low-confidence/missing proposal is unknown, not a negative object label.
    Masks may overlap: SAM can return both an object and a meaningful part.
    """
    height, width = shape
    kept = []
    for row in proposals:
        mask = np.asarray(row['segmentation'])
        if mask.ndim != 2 or not mask.size:
            continue
        ys = np.minimum(((np.arange(height) + .5) * mask.shape[0] / height).astype(int), mask.shape[0]-1)
        xs = np.minimum(((np.arange(width) + .5) * mask.shape[1] / width).astype(int), mask.shape[1]-1)
        mask = np.array(mask[np.ix_(ys, xs)], dtype=bool)
        score = min(float(row.get('predicted_iou', 0)), float(row.get('stability_score', 0)))
        if not np.isfinite(score) or score < .7 or not mask.any() or mask.mean() > .9:
            continue
        if any(np.logical_and(mask, other).sum() / np.logical_or(mask, other).sum() > .95
               for other, _ in kept):
            continue
        kept.append((mask, min(1., score)))
    # Keep multiple scales when prompts produce more parts than available slots.
    kept.sort(key=lambda row: (int(row[0].sum()), -row[1], row[0].tobytes()))
    if len(kept) > capacity:
        buckets = {}
        for row in kept:
            buckets.setdefault(int(row[0].sum()).bit_length(), []).append(row)
        selected = []
        while len(selected) < capacity:
            for rows in buckets.values():
                if rows and len(selected) < capacity:
                    selected.append(rows.pop(0))
        kept = sorted(selected, key=lambda row: (int(row[0].sum()), -row[1], row[0].tobytes()))
    masks = np.stack([m for m, _ in kept]) if kept else np.zeros((0, height, width), dtype=bool)
    confidence = np.array([c for _, c in kept], dtype=np.float32)
    masks.setflags(write=False); confidence.setflags(write=False)
    return Segmentation(masks, confidence)


def interior_clicks(segmentation, rng, count):
    """Round-robin object coverage; every proposed coordinate lies in its mask."""
    if count <= 0 or segmentation is None or not len(segmentation.masks):
        return []
    from .data import Action
    objects = list(range(len(segmentation.masks))); rng.shuffle(objects)
    result, used = [], set()
    for repeat in range(max(1, count)):
        for i in objects:
            points = np.argwhere(segmentation.masks[i])
            if not len(points):
                continue
            if repeat == 0:
                # The centroid of a ring may be empty. Snap to an actual pixel.
                point = points[np.square(points - points.mean(0)).sum(1).argmin()]
            else:
                point = points[rng.randrange(len(points))]
            y, x = map(int, point)
            if x >= 64 or y >= 64 or (x, y) in used:
                continue
            result.append(Action(6, dict(x=x, y=y))); used.add((x, y))
            if len(result) == count:
                return result
    return result


def imagined_candidates(choices, masks, rng):
    """Replace some future clicks using predicted occupancy; preserve legal IDs.

    Never invoke the teacher on an imagined frame. A quarter of the click pool
    remains uniform, even when a predicted mask is wrong or incomplete.
    """
    from .data import Action
    clicks = sum(a.id == 6 for a in choices)
    if not clicks:
        return choices
    occupied = masks.detach().float().cpu().numpy() > .5
    occupied = occupied[(occupied.sum((1, 2)) > 0) & (occupied.mean((1, 2)) < .9)]
    labels = Segmentation(occupied, np.ones(len(occupied), dtype=np.float32))
    reserve = max(1, int(np.ceil(clicks*.25)))
    proposals = interior_clicks(labels, rng, clicks-reserve)
    if not proposals:
        return choices
    used = {(a.data['x'], a.data['y']) for a in proposals}
    for xy in rng.sample(range(4096), 4096):
        if len(proposals) == clicks:
            break
        if (xy % 64, xy // 64) not in used:
            proposals.append(Action(6, dict(x=xy % 64, y=xy // 64)))
            used.add((xy % 64, xy // 64))
    return [a for a in choices if a.id != 6] + proposals


def detail_prompts(frame, limit=32):
    """Point prompts inside small visible regions; SAM still determines masks.

    A uniform grid often never touches a tiny ARC sprite. These prompts use only
    equal-color connectivity in the current image, with no game-specific rules.
    They are not object labels, and multi-color objects may span several regions.
    """
    height, width = frame.shape
    seen = np.zeros(frame.shape, bool)
    regions = []
    for y in range(height):
        for x in range(width):
            if seen[y, x]:
                continue
            todo = [(y, x)]; seen[y, x] = True; pixels = []
            while todo:
                yy, xx = todo.pop(); pixels.append((yy, xx))
                for ny, nx in ((yy-1, xx), (yy+1, xx), (yy, xx-1), (yy, xx+1)):
                    if (0 <= ny < height and 0 <= nx < width and not seen[ny, nx]
                            and frame[ny, nx] == frame[y, x]):
                        seen[ny, nx] = True; todo.append((ny, nx))
            if len(pixels) < frame.size/4:
                points = np.array(pixels)
                cy, cx = points[np.square(points-points.mean(0)).sum(1).argmin()]
                regions.append((len(pixels), (cx+.5)/width, (cy+.5)/height))
    regions.sort()
    return np.array([(x, y) for _, x, y in regions[:limit]], dtype=np.float32).reshape(-1, 2)


class SAM2Segmenter:
    def __init__(self, checkpoint, *, device='cpu', capacity=32, cache_frames=256, points_per_side=8):
        if not 1 <= capacity <= 64 or not 1 <= cache_frames <= 4096 or not 2 <= points_per_side <= 32:
            raise ValueError('Invalid segmentation capacity/cache/prompt grid')
        checkpoint = Path(checkpoint)
        with checkpoint.open('rb') as stream:
            identity = hashlib.file_digest(stream, 'sha256').hexdigest()
        if identity != TINY_SHA256:
            raise ValueError('Expected the pinned official SAM 2.1 Tiny checkpoint')
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        self.device = torch.device(device)
        # Share the trainer's GPU only while an uncached label is being computed.
        load_device = 'cpu' if self.device.type == 'cuda' else device
        self.model = build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml', str(checkpoint),
                               device=load_device, apply_postprocessing=False).eval().requires_grad_(False)
        self.generator = SAM2AutomaticMaskGenerator(self.model, points_per_side=points_per_side,
            points_per_batch=16, pred_iou_thresh=.7, stability_score_thresh=.8, crop_n_layers=0)
        self.uniform_points = self.generator.point_grids[0].copy()
        self.capacity, self.cache_frames = capacity, cache_frames
        self.cache = OrderedDict()
        self.calls = self.hits = 0
        self.seconds = 0.
        self.remaining = None
        self.deferred = 0

    def segment(self, frame):
        if frame is None:
            return None
        frame = np.asarray(frame)
        if frame.ndim != 2 or not frame.size or max(frame.shape) > 64 or not np.isin(frame, np.arange(16)).all():
            raise ValueError('Segmentation requires a native palette frame')
        key = (frame.shape, hashlib.sha256(frame.astype(np.uint8).tobytes()).digest())
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        if self.remaining == 0:
            self.deferred += 1
            return None
        if self.remaining is not None:
            self.remaining -= 1
        import torch
        from PIL import Image
        start = time.perf_counter()
        # Nearest-neighbour preserves palette edges; masks return to native geometry.
        scale = max(1, 512 // max(frame.shape))
        rgb = np.array(Image.fromarray(PALETTE[frame.astype(np.uint8)]).resize(
            (frame.shape[1]*scale, frame.shape[0]*scale), Image.Resampling.NEAREST))
        cuda = self.device.type == 'cuda'
        with torch.cuda.device(self.device) if cuda else nullcontext():
            if cuda and not torch.cuda.is_bf16_supported(including_emulation=False):
                raise RuntimeError(f'SAM2 on {self.device} requires native bfloat16 support')
            try:
                if cuda:
                    torch.cuda.synchronize(self.device)
                    torch.cuda.empty_cache()
                    self.model.to(self.device)
                precision = torch.autocast('cuda', dtype=torch.bfloat16) if cuda else nullcontext()
                with torch.inference_mode(), precision:
                    self.generator.point_grids = [np.unique(np.concatenate((self.uniform_points, detail_prompts(frame))), axis=0)]
                    proposals = self.generator.generate(rgb)
                result = native_masks(proposals, frame.shape, self.capacity)
            finally:
                # SAM can retain image embeddings when generation raises. Offload
                # even if predictor cleanup fails, and retain trainer peak counters.
                try:
                    self.generator.predictor.reset_predictor()
                finally:
                    if cuda:
                        try:
                            self.model.to('cpu')
                        finally:
                            try:
                                torch.cuda.synchronize(self.device)
                            finally:
                                torch.cuda.empty_cache()
        self.seconds += time.perf_counter() - start; self.calls += 1
        self.cache[key] = result
        while len(self.cache) > self.cache_frames:
            self.cache.popitem(last=False)
        return result

    def report(self):
        return dict(provider='sam2.1_hiera_tiny', revision=SAM2_REVISION, checkpoint_sha256=TINY_SHA256,
                    calls=self.calls, cache_hits=self.hits, seconds=self.seconds, cached_frames=len(self.cache),
                    deferred=self.deferred, detail_prompts=32, device=str(self.device),
                    precision='bfloat16' if self.device.type == 'cuda' else 'float32',
                    resident_device=str(next(self.model.parameters()).device))
