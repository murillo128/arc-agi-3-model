"""Phase admission from recent real training batches, never held-out trajectories."""
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import math
import time

import numpy as np
import torch

from .sequence_diagnostics import SequenceDiagnostics


@dataclass(frozen=True)
class SequenceControlConfig:
    window_batches: int = 4
    diagnostic_samples: int = 128
    decoder_learning_rate_after_predictor: float = 3e-7

    def __post_init__(self):
        for name in ('window_batches', 'diagnostic_samples'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        rate = self.decoder_learning_rate_after_predictor
        if type(rate) not in (float, int) or not math.isfinite(rate) or not 0 < rate < 1:
            raise ValueError('Invalid post-predictor decoder LR')


def pair_key(game, level):
    return f'{game}/{level}'


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


class SequencePhaseControl:
    def __init__(self, stages, settings, config):
        self.config, self.settings = config, settings
        self.stages = {pair_key(s['game'], s['level']): s for s in stages}
        self.window = deque(maxlen=config.window_batches)
        self.streaks = dict.fromkeys(self.stages, 0)
        self.phase, self.admitted_batch = 'representation', None

    def apply(self, representation):
        """Only change optimizer rates, retaining weights and optimizer moments."""
        representation.frozen = False
        predictive = self.phase == 'predictor'
        for group in representation.optimizer.param_groups:
            rate = representation.base_learning_rate
            if predictive:
                rate = (self.settings['encoder_learning_rate_after_predictor'] if group['name'] == 'encoder'
                        else self.config.decoder_learning_rate_after_predictor)
            group['lr'] = rate
        if predictive:
            representation.sigreg_enabled = False

    def rows(self):
        totals = defaultdict(lambda: defaultdict(float))
        for entry in self.window:
            for key, values in entry['pairs'].items():
                for name, value in values.items():
                    totals[key][name] += value
        rows = []
        for key, stage in self.stages.items():
            counts, gates = totals[key], stage['thresholds']
            reconstruction = ratio(counts['reconstruction_wrong'], counts['reconstruction_union'])
            drift_square = ratio(counts['drift_delta'], counts['drift_scale'])
            drift = math.sqrt(drift_square) if drift_square is not None else None
            enough = (counts['samples'] >= gates['minimum_samples'] and
                      counts['changed'] >= gates['minimum_changed'] and
                      counts['drift_samples'] >= gates['minimum_samples'])
            passing = (enough and reconstruction is not None and drift is not None and
                       reconstruction <= gates['representation_open'] and drift <= gates['latent_drift'])
            rows.append(dict(game=stage['game'], level=stage['level'], key=key,
                samples=int(counts['samples']), changed=int(counts['changed']),
                reconstruction_error=reconstruction, latent_drift=drift,
                prediction_error=ratio(counts['prediction_wrong'], counts['prediction_union']),
                enough=enough, passing=bool(passing), passes=self.streaks[key], thresholds=gates))
        return rows

    def observe(self, batch, pairs):
        if self.window and batch <= self.window[-1]['batch']:
            raise ValueError('Phase evidence must come from a new batch')
        if set(pairs) - self.stages.keys():
            raise ValueError('Unexpected game/level in phase evidence')
        if any(not math.isfinite(v) or v < 0 for row in pairs.values() for v in row.values()):
            raise ValueError('Nonfinite or negative phase evidence')
        self.window.append(dict(batch=batch, pairs=pairs))
        for row in self.rows():
            key = row['key']
            if not row['passing']:
                self.streaks[key] = 0
            elif pairs.get(key, {}).get('samples', 0):
                self.streaks[key] += 1
        if self.phase == 'representation' and all(
                self.streaks[k] >= s['thresholds']['patience'] for k, s in self.stages.items()):
            self.phase, self.admitted_batch = 'predictor', batch

    def snapshot(self):
        return dict(version=1, config=asdict(self.config), phase=self.phase,
                    admitted_batch=self.admitted_batch, window=list(self.window), streaks=dict(self.streaks))

    def restore(self, saved):
        if (saved.get('version') != 1 or saved['config'] != asdict(self.config) or
                saved['phase'] not in ('representation', 'predictor') or
                set(saved['streaks']) != set(self.stages)):
            raise ValueError('Incompatible sequence phase state')
        self.phase, self.admitted_batch = saved['phase'], saved['admitted_batch']
        self.window.extend(saved['window'])
        self.streaks = dict(saved['streaks'])

    def report(self):
        return dict(phase=self.phase, admitted_batch=self.admitted_batch,
                    window_batches=self.config.window_batches, batches=len(self.window),
                    source='training_pre_update', rows=self.rows())


class TrainingBatchDiagnostics(SequenceDiagnostics):
    """Stratified reconstruction before learning; same-image encoder drift after it."""
    every = 1

    def __init__(self, engine, sessions, limit=128):
        super().__init__(engine, sessions)
        self.limit = limit
        groups = defaultdict(list)
        for session in sessions:
            for t, step in enumerate(session.transitions):
                if step.supervised:
                    groups[pair_key(session.game, step.level_number)].append(
                        ((id(session), t), bool(np.any(step.before.final != step.after.final))))
        # Reserve half of each stratum for observed changes. No generated data or
        # action-policy changes; this only prevents rare real clicks being lost.
        candidates = {}
        quota = max(1, limit // max(1, len(groups)))
        for key, group in groups.items():
            changed = [index for index, change in group if change]
            n = min(len(changed), max(1, quota // 2))
            selected = [changed[i] for i in np.linspace(0, len(changed)-1, n, dtype=int)] if n else []
            remaining = [index for index, _ in group if index not in selected]
            n = min(len(remaining), quota - len(selected))
            if n:
                selected.extend(remaining[i] for i in np.linspace(0, len(remaining)-1, n, dtype=int))
            candidates[key] = selected
        self.selected = set(index for group in candidates.values() for index in group[:quota])
        if len(self.selected) > limit:
            # A larger configured catalog needs sufficient diagnostic capacity.
            raise ValueError('diagnostic_samples must cover every game/level in a batch')
        self.pairs = defaultdict(lambda: defaultdict(float))
        self.anchors = []

    @torch.no_grad()
    def observe(self, encoded, session, t, z, predicted, next_h, pixels):
        super().observe(encoded, session, t, z, predicted, next_h, pixels)
        if (id(session), t) not in self.selected:
            return
        begin = time.perf_counter()
        step, e = session.transitions[t], self.engine
        before, actual = step.before.final, step.after.final
        key = pair_key(session.game, step.level_number)
        row = self.pairs[key]
        reconstructed = e.decode(encoded.z(step.after)[None], *actual.shape).argmax(-1)[0].cpu().numpy()
        real = actual != before
        union = real | (reconstructed != before)
        row['reconstruction_wrong'] += int(((reconstructed != actual) & union).sum())
        row['reconstruction_union'] += int(union.sum())
        row['samples'] += 1
        row['changed'] += int(real.any())
        guessed = pixels.cpu().numpy()
        union = real | (guessed != before)
        row['prediction_wrong'] += int(((guessed != actual) & union).sum())
        row['prediction_union'] += int(union.sum())
        self.anchors.append((key, step.before, z.detach().float().cpu().clone()))
        self.seconds += time.perf_counter() - begin

    @torch.no_grad()
    def after_representation(self):
        begin = time.perf_counter()
        e = self.engine
        e.model.eval()
        with e.autocast():
            for key, observation, before in self.anchors:
                after = e.encode([observation])[0].detach().float().cpu()
                row = self.pairs[key]
                row['drift_delta'] += float((after-before).square().sum())
                row['drift_scale'] += float(before.square().sum())
                row['drift_samples'] += 1
        self.seconds += time.perf_counter() - begin

    def finish(self):
        result, samples = super().finish()
        delta = sum(row['drift_delta'] for row in self.pairs.values())
        scale = sum(row['drift_scale'] for row in self.pairs.values())
        result.update(latent_drift=math.sqrt(delta/scale) if scale else None,
                      drift_reference_batch=None, drift_source='same_batch_before_after',
                      reconstruction_delta_error=ratio(
                          sum(row['reconstruction_wrong'] for row in self.pairs.values()),
                          sum(row['reconstruction_union'] for row in self.pairs.values())))
        return result, samples
