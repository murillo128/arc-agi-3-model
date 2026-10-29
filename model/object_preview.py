"""Bounded, lossless instance-mask telemetry for the current dashboard sample."""
import base64

import numpy as np


def mask_instances(masks, confidence):
    """Keep overlaps and original slot indices; encode native binary masks."""
    masks, confidence = np.asarray(masks), np.asarray(confidence)
    if (masks.ndim != 3 or not 0 <= masks.shape[0] <= 64
            or not all(1 <= n <= 64 for n in masks.shape[1:])
            or confidence.shape != (masks.shape[0],)):
        raise ValueError('Invalid native object preview dimensions')
    if not np.isfinite(masks).all() or not np.isfinite(confidence).all():
        raise ValueError('Object preview must be finite')
    result = []
    for index, (mask, score) in enumerate(zip(masks, confidence)):
        occupied = mask > .5
        area = int(occupied.sum())
        if score < .5 or not area:
            continue
        packed = np.packbits(occupied.reshape(-1), bitorder='little').tobytes()
        result.append(dict(id=index, bits=base64.b64encode(packed).decode('ascii'),
                           area=area, confidence=float(score)))
    return result


def instance_preview(before, after, predicted, presence, *, learned_current=None, learned_presence=None):
    """One sample only, not the batch history; no soft probabilities retained."""
    shape = before.masks.shape[1:]
    if after.masks.shape[1:] != shape or predicted.shape[1:] != shape:
        raise ValueError('Object preview sets must use the same native geometry')
    sets = dict(
        current=mask_instances(before.masks, before.confidence),
        next=mask_instances(after.masks, after.confidence),
        predicted=mask_instances(predicted, presence))
    if learned_current is not None:
        if learned_current.shape[1:] != shape or learned_presence is None:
            raise ValueError('Learned current preview requires matching geometry and presence')
        sets['learned_current'] = mask_instances(learned_current, learned_presence)
    elif learned_presence is not None:
        raise ValueError('Learned presence requires current masks')
    return dict(version=1, encoding='packbits-little-base64', shape=list(shape), sets=sets)
