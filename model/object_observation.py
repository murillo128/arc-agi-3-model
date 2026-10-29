"""Bounded CPU replay of teacher-labeled training observations."""
from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib

import numpy as np
import torch


@dataclass(frozen=True)
class LabeledObservation:
    game: str
    key: tuple
    latent: torch.Tensor
    labels: object

    @property
    def geometry(self):
        return self.labels.masks.shape[1:]


class ObservationReplay:
    """Keep native labels and detached encodings, never recurrent/model graphs.

    Remove the oldest frame from the most represented game when full. Encoder
    updates invalidate all cached encodings, including animation aggregation.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.rows = OrderedDict()
        self.encoder_token = None

    def check_encoder(self, engine):
        token = (getattr(engine, '_latent_cache_epoch', None),
                 tuple((id(p), p._version) for p in engine.model.encoder.parameters()))
        if token != self.encoder_token:
            self.rows.clear()
            self.encoder_token = token

    def add(self, game, observation, latent, labels):
        frames = np.ascontiguousarray(observation.frames)
        digest = hashlib.blake2b(frames.tobytes(), digest_size=16).digest()
        key = (game, frames.shape, digest)
        # Clone prevents retaining the full encoded-session GPU/CPU storage.
        row = LabeledObservation(game, key, latent.detach().to('cpu').clone(), labels)
        self.rows.pop(key, None)
        self.rows[key] = row
        while len(self.rows) > self.capacity:
            counts = Counter(item.game for item in self.rows.values())
            largest = max(counts.values())
            evict = next(k for k, item in self.rows.items() if counts[item.game] == largest)
            del self.rows[evict]
        return row

    def schedule(self, current, steps, rng):
        """Visit each current labeled frame first, then alternate new and replay.

        Game-round-robin replay avoids making frequent games dominate the small
        observer budget. A bounded caller budget may truncate the first pass.
        """
        current = list({row.key: row for row in current}.values())
        rng.shuffle(current)
        replay = {}
        for row in self.rows.values():
            replay.setdefault(row.game, []).append(row)
        for rows in replay.values():
            rng.shuffle(rows)
        games = list(replay); rng.shuffle(games)
        replay_order = []
        while any(replay.values()):
            for game in games:
                if replay[game]:
                    replay_order.append(replay[game].pop())
        if not current:
            current = replay_order
        if not current:
            return []
        rows = current[:steps]
        offset = 0
        while len(rows) < steps:
            pool = replay_order if offset % 2 == 0 and replay_order else current
            rows.append(pool[(offset // 2) % len(pool)])
            offset += 1
        return rows
