"""One-frame teacher device/memory check using a normal training report.

Run with the trainer stopped at a saved batch boundary. This is not a model
quality benchmark and never changes weights or opens an evaluation game.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .segmentation import SAM2Segmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1:
        parser.error('Use a new output path and positive CPU thread count')
    torch.set_num_threads(args.threads)
    report = json.loads(args.report.read_text())
    if report.get('kind') != 'game_sequences' or not report.get('object_sample'):
        parser.error('A normal sequence report with an object preview is required')
    config = report['objects']['config']
    sample = report['object_sample']
    frame = np.asarray(sample['before'], dtype=np.uint8)
    device = torch.device(args.device)
    baseline = 0
    if device.type == 'cuda':
        # cuBLAS retains its own workspace after the first GEMM. Warm it before
        # the baseline so shared runtime memory is not mistaken for a SAM leak.
        with torch.inference_mode():
            warm = torch.ones(16, 16, device=device)
            product = warm @ warm
            del product, warm
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)  # This dedicated probe only.
    teacher = SAM2Segmenter(config['checkpoint'], device=str(device),
        capacity=report['network']['object_slots'], points_per_side=config['points_per_side'])
    if device.type == 'cuda' and torch.cuda.memory_allocated(device) != baseline:
        raise ValueError('SAM retained live GPU tensors after construction')
    start = time.perf_counter()
    labels = teacher.segment(frame)
    seconds = time.perf_counter()-start
    if (labels is None or not len(labels.masks) or labels.masks.shape[1:] != frame.shape
            or labels.masks.dtype != np.bool_ or not np.isfinite(labels.confidence).all()):
        raise ValueError('Teacher did not return usable native masks for this training sample')
    if any(p.device.type != 'cpu' or p.requires_grad for p in teacher.model.parameters()):
        raise ValueError('Teacher must be frozen and back on CPU after the call')
    evidence = dict(game=sample['game'], batch=sample.get('batch'), device=str(device),
        seconds=seconds, frame_shape=list(frame.shape), masks=len(labels.masks),
        mask_pixels=labels.masks.sum((1, 2)).tolist(), teacher=teacher.report())
    if device.type == 'cuda':
        evidence.update(baseline_allocated_bytes=baseline,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            idle_allocated_bytes=torch.cuda.memory_allocated(device),
            idle_reserved_bytes=torch.cuda.memory_reserved(device))
        if evidence['idle_allocated_bytes'] != baseline:
            raise ValueError('SAM retained live GPU tensors after inference')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    print(json.dumps(evidence, allow_nan=False))


if __name__ == '__main__':
    main()
