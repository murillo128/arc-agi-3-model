"""Live dashboard for the trainer's existing logs/reports.

Run with python -m model.dashboard --log model/runs/exploration/run.log.
Read-only by default; explicit curriculum service enables two signal controls.
No torch imports or persistent dashboard storage.
"""
import argparse
from collections import deque
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import urlsplit

from .curriculum_control import CurriculumControl
from .optimization_control import OptimizationControl


MAX_READ = 8 * 1024 * 1024
MAX_LINE = 64 * 1024
ASSETS = Path(__file__).with_name('dashboard_web')
PROGRESS_KEYS = ('actions', 'updates', 'episodes', 'level_index', 'level_actions',
                 'passes', 'collapse_streak', 'actions_per_second', 'replay_episodes', 'gpu_peak_mb',
                 'episode_actions', 'episode_observations', 'device_cache_mb', 'trained_transitions', 'trained_transitions_per_second')
LOSS_KEYS = ('loss', 'latent_mse', 'sigreg', 'variance_floor', 'decoder_ce',
             'reconstruction_ce', 'local_reconstruction_ce', 'gradient_norm', 'samples', 'seconds',
             'rollout_mse', 'rollout_visual', 'rollout_samples')
VALIDATION_KEYS = ('delta_error', 'decoder_delta_error', 'latent_delta_mse', 'latent_std',
                   'transitions', 'changed_transitions', 'skipped', 'wins', 'game_overs',
                   'extra_changes', 'missed_changes', 'wrong_replacement_colors', 'correct_changes')
CONFIG_KEYS = ('learning_rate', 'validate_every', 'consecutive_passes', 'delta_error',
               'latent_delta_mse', 'min_latent_std', 'collapse_std', 'false_change_weight',
               'representation_unfreeze_error', 'representation_freeze_error', 'representation_gate_patience')
STATISTICS_KEYS = ('actions', 'updates', 'episodes', 'level_actions', 'elapsed_seconds', 'actions_per_second',
                   'trained_transitions', 'trained_transitions_per_second')


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def numbers(source, keys):
    if not isinstance(source, dict):
        return {}
    return {k: number(source[k]) for k in keys if k in source}


def compact_values(value, depth=0):
    """Bound small campaign metadata, preserving finite numbers and plain text."""
    if type(value) is bool or value is None:
        return value
    if type(value) in (int, float):
        return number(value)
    if isinstance(value, str):
        return value[:240]
    if depth >= 5:
        return None
    if isinstance(value, list):
        return [compact_values(v, depth + 1) for v in value[:100]]
    if isinstance(value, dict):
        return {str(k)[:80]: compact_values(v, depth + 1) for k, v in list(value.items())[:100]}
    return None


class GpuMonitor:
    """Current server telemetry, shared across clients without touching the trainer."""

    def __init__(self):
        self.lock = threading.Lock()
        self.next_sample = 0.
        self.sample = {'devices': [], 'sampled_at': None}

    def snapshot(self):
        with self.lock:
            if time.monotonic() < self.next_sample:
                return self.sample
            devices = []
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,temperature.gpu',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, check=True, timeout=1)
                for row in csv.reader(result.stdout.splitlines()):
                    if len(row) != 4:
                        continue
                    try:
                        index = int(row[0])
                    except ValueError:
                        continue
                    if index < 0:
                        continue
                    values = []
                    for raw in row[2:]:
                        try:
                            value = number(float(raw))
                        except ValueError:
                            value = None
                        values.append(value if value is not None and value >= 0 else None)
                    usage, temperature = values
                    devices.append(dict(index=index, name=row[1].strip()[:100],
                        utilization_percent=usage if usage is None or usage <= 100 else None,
                        temperature_c=temperature))
            except (OSError, subprocess.SubprocessError, UnicodeError, csv.Error):
                devices = []
            # Failed queries replace old readings too: missing telemetry is not 0%.
            self.sample = {'devices': devices, 'sampled_at': time.time()}
            self.next_sample = time.monotonic() + 2
            return self.sample


class LogMonitor:
    """Bounded incremental reader shared by browser clients; never writes the log."""

    max_line = MAX_LINE

    def __init__(self, path, max_points=2000):
        self.path = Path(path)
        self.max_points = max_points
        self.lock = threading.Lock()
        self.identity = None
        self.offset = 0
        self.anchor = b''
        self.pending = b''
        self.skip_line = False
        self.generation = 0
        self.revision = 0
        self._reset()

    def _reset(self):
        self.generation += 1
        self.revision += 1
        self.run = {}
        self.progress = {}
        self.total_actions = None
        self.progress_points = deque(maxlen=self.max_points)
        self.validations = deque(maxlen=self.max_points)
        self.events = deque(maxlen=12)
        self.phase = 'waiting'
        self.reason = None
        self.limited = False

    def _event(self, row):
        if not isinstance(row, dict):
            return
        event = row.get('event')
        if event == 'initializing':
            self._reset()
            self.phase = 'initializing'
            self.run = {k: str(row[k])[:80] for k in ('game', 'device') if k in row}
            self.run['levels'] = [number(x) for x in row.get('levels', [])[:100]] if isinstance(row.get('levels'), list) else []
            self.run['training'] = numbers(row.get('training'), CONFIG_KEYS)
            training = row.get('training')
            if isinstance(training, dict):
                for key in ('freeze_representation', 'auto_representation'):
                    if type(training.get(key)) is bool:
                        self.run['training'][key] = training[key]
            self.run['compile'] = row.get('compile') is True
            if row.get('statistics_origin') == 'after_first_action':
                self.run['statistics_origin'] = 'after_first_action'
        elif event == 'ready':
            self.run['parameters'] = number(row.get('parameters'))
            self.phase = 'ready'
        elif event in ('progress', 'resumed', 'stopped'):
            actions = number(row.get('actions'))
            previous = self.total_actions
            if actions is not None and previous is not None and actions < previous:
                self._reset()
            self.total_actions = actions
            self.progress.update(numbers(row, PROGRESS_KEYS))
            statistics = numbers(row.get('statistics'), STATISTICS_KEYS)
            if statistics:
                # Self-contained records also survive tail-only reads and a
                # dashboard restart. All consumers share this action origin.
                self.run['statistics_origin'] = 'after_first_action'
                self.progress.update(statistics)
            elif self.run.get('statistics_origin') == 'after_first_action':
                self.progress.update(dict.fromkeys(STATISTICS_KEYS))
            if 'level' in row:
                self.progress['level'] = number(row['level'])
            if event == 'progress':
                self.progress['loss'] = numbers(row.get('loss'), LOSS_KEYS)
                self.progress_points.append(dict(self.progress))
            self.phase = event
            if event == 'stopped':
                self.reason = str(row.get('reason', 'unknown'))[:80]
        elif event == 'validation':
            # row.actions/episodes count ONLY this frozen evaluation, not training.
            point = numbers(row, VALIDATION_KEYS)
            point.update(level=number(row.get('level')), actions=self.progress.get('actions'),
                         updates=self.progress.get('updates'), passes=row.get('passes') is True,
                         validation_actions=number(row.get('actions')),
                         validation_episodes=number(row.get('episodes')))
            self.validations.append(point)
            self.phase = 'validation'
        elif event == 'validation_finished':
            self.phase = 'validation_finished'
        elif event == 'representation_gate':
            pass  # Evidence updates mode without disguising validation as training.
        elif event not in ('checkpoint', 'level_converged', 'encoder_collapsed'):
            return
        training = self.run.setdefault('training', {})
        if type(row.get('freeze_representation')) is bool:
            training['freeze_representation'] = row['freeze_representation']
        gate = row if event == 'representation_gate' else row.get('representation_gate')
        if isinstance(gate, dict):
            status = numbers(gate, ('reconstruction_error', 'open_streak', 'close_streak',
                                    'unfreeze_error', 'freeze_error', 'patience'))
            for key in ('auto_representation', 'freeze_representation', 'eligible', 'changed'):
                if type(gate.get(key)) is bool:
                    status[key] = gate[key]
                    if key in ('auto_representation', 'freeze_representation'):
                        training[key] = gate[key]
            status['reason'] = str(gate.get('reason', 'unknown'))[:80]
            self.run['representation_gate'] = status
        if event != 'progress':
            self.events.append({'event': event, 'actions': self.progress.get('actions'),
                                'level': number(row.get('level')), 'reason': self.reason})
        self.revision += 1

    def _read(self):
        try:
            with self.path.open('rb') as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                stream.seek(max(0, self.offset - len(self.anchor)))
                replaced = (identity != self.identity or stat.st_size < self.offset
                            or stream.read(len(self.anchor)) != self.anchor)
                if replaced:
                    self._reset()
                    self.identity = identity
                    self.offset = max(0, stat.st_size - MAX_READ)
                    self.pending = b''
                    self.skip_line = self.offset > 0
                    self.limited = self.offset > 0
                stream.seek(self.offset)
                chunk = stream.read(MAX_READ)
                self.offset = stream.tell()
                stream.seek(max(0, self.offset - 256))
                self.anchor = stream.read(min(self.offset, 256))
                parts = (self.pending + chunk).split(b'\n')
                self.pending = parts.pop()
                for line in parts:
                    if self.skip_line:
                        self.skip_line = False
                        continue
                    if len(line) > self.max_line:
                        continue
                    try:
                        self._event(json.loads(line))
                    except (ValueError, UnicodeError):
                        # SDK logs, warnings and a malformed record are not metrics.
                        continue
                if len(self.pending) > self.max_line:
                    self.pending = b''
                    self.skip_line = True
                return {'available': True, 'updated_at': stat.st_mtime,
                        'age_seconds': max(0, time.time() - stat.st_mtime),
                        'catching_up': self.offset < stat.st_size, 'tail_limited': self.limited}
        except OSError:
            return {'available': False, 'updated_at': None, 'age_seconds': None,
                    'catching_up': False, 'tail_limited': self.limited}

    def snapshot(self):
        with self.lock:
            source = self._read()
            return {'generation': self.generation, 'revision': self.revision,
                    'source': source, 'phase': self.phase, 'reason': self.reason,
                    'run': dict(self.run), 'progress': dict(self.progress),
                    'progress_points': list(self.progress_points),
                    'validations': list(self.validations), 'events': list(self.events),
                    'max_points': self.max_points}


class BehaviorMonitor(LogMonitor):
    """Independent log lifetime/counters for new heads and frozen-world planning."""

    def _reset(self):
        super()._reset()
        self.fits = deque(maxlen=self.max_points)
        self.evaluations = deque(maxlen=self.max_points)

    def _event(self, row):
        if not isinstance(row, dict):
            return
        event = row.get('event')
        if event not in ('behavior_started', 'behavior_collection', 'behavior_fit',
                         'behavior_evaluation_progress', 'behavior_evaluation', 'behavior_world_verified', 'behavior_finished'):
            return
        if event == 'behavior_started':
            self._reset()
            self.run = {k: str(row[k])[:80] for k in ('game', 'device', 'mode', 'source') if k in row}
            self.run.update(numbers(row, ('level', 'source_actions', 'world_trainable')))
            if isinstance(row.get('levels'), list):
                self.run['levels'] = [v for v in row['levels'][:100] if type(v) is int and v > 0]
            self.run['compile'] = row.get('compile') is True
        point = numbers(row, ('round', 'epoch', 'epochs', 'updates', 'records', 'loss', 'event_ce',
                              'value_bce', 'q_bce', 'episodes', 'wins', 'actions', 'seconds', 'horizon',
                              'win_rate', 'actions_per_success', 'actions_per_second', 'decision_ms', 'level'))
        for key in ('policy', 'split', 'role'):
            if key in row:
                point[key] = str(row[key])[:40]
        if event == 'behavior_collection':
            # Counts are local to this collection round, never added to evaluation.
            if isinstance(row.get('per_level'), dict):
                point['per_level'] = {str(k)[:10]: numbers(v, ('episodes', 'wins', 'actions'))
                                      for k, v in list(row['per_level'].items())[:100] if isinstance(v, dict)}
            self.progress['collection'] = point
        elif event == 'behavior_fit':
            self.fits.append(point)
            self.progress['fit'] = point
        elif event in ('behavior_evaluation_progress', 'behavior_evaluation'):
            point.setdefault('split', 'unspecified')
            interval = row.get('wilson95')
            if isinstance(interval, list) and len(interval) == 2:
                point['wilson95'] = [number(v) for v in interval]
            self.progress['evaluation'] = point
            if event == 'behavior_evaluation':
                self.evaluations.append(point)
        elif event in ('behavior_world_verified', 'behavior_finished'):
            self.run['world_unchanged'] = row.get('world_unchanged') if type(row.get('world_unchanged')) is bool else None
            selected = row.get('selected')
            if isinstance(selected, dict):
                self.run['selected'] = {**numbers(selected, ('iteration', 'horizon')),
                                        'policy': str(selected.get('policy', ''))[:40]}
        self.phase = event
        self.revision += 1

    def snapshot(self):
        with self.lock:
            source = self._read()
            return {'generation': self.generation, 'revision': self.revision, 'source': source,
                    'phase': self.phase, 'run': dict(self.run), 'progress': dict(self.progress),
                    'fits': list(self.fits), 'evaluations': list(self.evaluations)}


class CampaignMonitor(LogMonitor):
    """Independent experiment lifetime; variant counters may restart at zero."""

    # A completed evaluation contains hundreds of episode records. Parse that
    # bounded line, then expose only the compact summaries above, never traces.
    max_line = 2 * 1024 * 1024

    def __init__(self, path, max_points=2000, report_path=None):
        self.report_path = Path(report_path) if report_path else None
        self.report_identity = None
        self.report_summary = None
        super().__init__(path, max_points)

    def _report(self):
        """Read the trainer's atomic report, exposing summaries rather than episodes."""
        if self.report_path is None:
            return {'available': False, 'reason': 'not_configured'}
        try:
            stat = self.report_path.stat()
            identity = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if identity != self.report_identity:
                with self.report_path.open('rb') as stream:
                    payload = stream.read(32 * 1024 * 1024 + 1)
                if len(payload) > 32 * 1024 * 1024:
                    raise ValueError('Report exceeds dashboard limit')
                report = json.loads(payload)
                if not isinstance(report, dict) or not isinstance(report.get('cells'), list):
                    raise ValueError('Invalid campaign report')
                behavior_keys = ('win_rate', 'wins', 'episodes', 'actions', 'actions_per_success', 'decision_ms')
                cells = []
                for cell in report['cells'][:100]:
                    cells.append({'variant': str(cell['variant'])[:80], 'seed': number(cell['seed']),
                        'world': numbers(cell.get('world'), VALIDATION_KEYS),
                        'behavior': numbers(cell.get('behavior'), behavior_keys),
                        'training': numbers(cell.get('training'), ('actions', 'updates', 'seconds', 'cumulative_actions'))})
                summary = {k: compact_values(report.get(k)) for k in (
                    'status', 'selected', 'processed_variants', 'controlled_comparisons',
                    'comparisons', 'deferred_variants')}
                config = report.get('config', {})
                summary['config'] = {k: compact_values(config.get(k)) for k in (
                    'variants', 'seeds', 'actions', 'dev_episodes', 'final_episodes', 'pilot')}
                summary['cells'] = cells
                summary['calibration'] = []
                for item in [*report['cells'][:100], report.get('pending_evaluation', {})]:
                    coverage = item.get('training', {}).get('calibration')
                    if not isinstance(coverage, dict):
                        continue
                    keys = ('episodes', 'actions', 'successes', 'failures', 'budget_ends', 'censored_episodes')
                    summary['calibration'].append({'variant': str(item['variant'])[:80], 'seed': number(item['seed']),
                        **numbers(coverage, (*keys, 'raw_bytes', 'records', 'positive_records', 'censored_records')),
                        'per_pair': [dict(numbers(p, ('level', *keys)), game=str(p['game'])[:80])
                                     for p in coverage.get('per_pair', [])[:100]]})
                summary['deferred_cells'] = [{k: compact_values(c.get(k)) for k in (
                    'variant', 'seed', 'reason', 'fallback')} for c in report.get('deferred_cells', [])[:100]]
                final = report.get('final', {})
                summary['final'] = {k: numbers(final.get(k), behavior_keys) for k in ('reference', 'selected')}
                summary['error'] = str(report.get('error', ''))[:500]
                summary.update(available=True, updated_at=stat.st_mtime)
                self.report_summary, self.report_identity = summary, identity
            return dict(self.report_summary, diagnostics=self._diagnostics(), learning_probe=self._learning_probe())
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            self.report_identity = None
            return {'available': False, 'reason': 'unreadable'}

    def _diagnostics(self):
        path = self.report_path.parent / 'diagnostics' / 'report.json'
        try:
            with path.open('rb') as stream:
                payload = stream.read(1024 * 1024 + 1)
            if len(payload) > 1024 * 1024:
                raise ValueError('Diagnostic report too large')
            report = json.loads(payload)
            count_keys = ('wrong', 'union', 'samples', 'delta_error', 'invented', 'missed', 'wrong_color', 'correct')
            results = []
            for row in report.get('results', [])[:10]:
                probes = []
                for p in row.get('probes', [])[:100]:
                    probes.append({'game': str(p['game'])[:80], 'level': number(p['level']),
                        'distinct_pairs': number(p.get('distinct_pairs')),
                        **{k: numbers(p.get(k), count_keys) for k in (
                            'predictor', 'no_change', 'wrong_action', 'distinct_correct', 'distinct_wrong')}})
                policies = {mode: {'complete': value.get('complete') is True,
                    'summary': numbers(value.get('summary'), ('episodes', 'wins', 'win_rate', 'actions',
                                                            'actions_per_success', 'decision_ms'))}
                    for mode, value in row.get('policies', {}).items() if mode in ('random', 'q', 'value_mpc')}
                results.append({'checkpoint': str(row['checkpoint'])[:80], 'probes': probes, 'policies': policies,
                                'weights_unchanged': row.get('weights_unchanged') is True})
            return {'available': True, 'status': str(report['status'])[:80],
                    'progress': compact_values(report.get('progress')), 'results': results,
                    'error': str(report.get('error', ''))[:240],
                    'age_seconds': max(0, time.time() - path.stat().st_mtime)}
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return {'available': False}

    def _learning_probe(self):
        path = self.report_path.parent / 'streaming-probe' / 'report.json'
        if not path.exists():
            path = self.report_path.parent / 'learning-probe' / 'report.json'
        try:
            with path.open('rb') as stream:
                payload = stream.read(4 * 1024 * 1024 + 1)
            if len(payload) > 4 * 1024 * 1024:
                raise ValueError('Learning probe report too large')
            report = json.loads(payload)
            results = []
            for row in report.get('results', [])[:4]:
                points = []
                for point in row.get('points', [])[-200:]:
                    values = numbers(point, ('updates', 'seconds', 'actions', 'selection_error'))
                    for split in ('fit', 'development'):
                        for horizon in ('1', '2'):
                            measure = point.get(split, {}).get('rollouts', {}).get(horizon, {})
                            values[f'{split}_{horizon}'] = number(measure.get('readouts', {}).get('contextual', {}).get('prediction_delta_error'))
                            values[f'{split}_{horizon}_samples'] = number(measure.get('samples'))
                    points.append(values)
                results.append({'variant': str(row['variant'])[:80], 'complete': row.get('complete') is True,
                                'frozen_representation_verified': row.get('frozen_representation_verified') is True,
                                'points': points, 'best': numbers(row.get('best'), ('actions', 'updates', 'selection_error'))})
            measurements = []
            def append_measurements(label, measured):
                for row in measured.get('per_pair', [])[:100]:
                    for horizon in ('1', '2'):
                        item = row.get('rollouts', {}).get(horizon, {})
                        changed = item.get('strata', {}).get('changed', {})
                        unchanged = item.get('strata', {}).get('unchanged', {})
                        samples = number(unchanged.get('samples'))
                        wrong = number(unchanged.get('readouts', {}).get('contextual', {}).get('prediction_wrong'))
                        measurements.append({'label': label, 'game': str(row['game'])[:80], 'level': number(row.get('level')),
                            'horizon': int(horizon), 'changed_samples': number(changed.get('samples')),
                            'changed_error': number(changed.get('readouts', {}).get('contextual', {}).get('prediction_delta_error')),
                            'unchanged_samples': samples, 'unchanged_pixels': wrong / samples if samples and wrong is not None else None})
            for row in report.get('results', [])[:4]:
                best = row.get('best')
                if best:
                    point = next((p for p in row.get('points', []) if p.get('actions') == best.get('actions')), {})
                    append_measurements(str(row['variant'])[:80], point.get('development', {}))
            confirmations = []
            for row in report.get('confirmation', [])[:4]:
                values = {'variant': str(row['variant'])[:80]}
                for horizon in ('1', '2'):
                    values[horizon] = number(row.get('rollouts', {}).get(horizon, {}).get('readouts', {}).get('contextual', {}).get('prediction_delta_error'))
                confirmations.append(values)
                append_measurements('confirmation_' + values['variant'], row)
            policies = [{'policy': str(mode)[:40], 'complete': row.get('complete') is True,
                         **numbers(row.get('summary'), ('wins', 'episodes', 'win_rate', 'actions_per_success', 'decision_ms'))}
                        for mode, row in list(report.get('policies', {}).items())[:4]]
            keys = ('episodes', 'actions', 'successes', 'failures', 'budget_ends', 'censored_episodes')
            coverage = numbers(report.get('coverage'), keys)
            coverage['per_pair'] = [dict(numbers(row, ('level', *keys)), game=str(row['game'])[:80])
                                   for row in report.get('coverage', {}).get('per_pair', [])[:100]]
            return {'available': True, 'status': str(report['status'])[:80],
                    'scope': str(report.get('scope', 'fixed_real_training_probe'))[:80],
                    'progress': compact_values(report.get('progress')), 'results': results, 'coverage': coverage,
                    'measurements': measurements, 'confirmation': confirmations, 'policies': policies,
                    'calibration': numbers(report.get('calibration'), ('records', 'positive_records', 'censored_records', 'successes')),
                    'selected': None if not report.get('selected') else dict(
                        numbers(report['selected'], ('actions', 'updates', 'selection_error')), variant=str(report['selected'].get('variant', ''))[:80]),
                    'error': str(report.get('error', ''))[:240],
                    'age_seconds': max(0, time.time() - path.stat().st_mtime)}
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return {'available': False}

    def _reset(self):
        super()._reset()
        self.evaluations = deque(maxlen=self.max_points)
        # Small summaries outlive the bounded curve window. They are rebuilt
        # from the existing log, never a second persistent training history.
        self.executions = {}

    def _event(self, row):
        if not isinstance(row, dict):
            return
        event = row.get('event')
        if event in ('behavior_evaluation_progress', 'behavior_evaluation'):
            # The campaign's evaluator emits inner events without variant/seed.
            # Bind only to an explicit active development-evaluation context.
            # Final confirmation belongs to report.json, not the last variant.
            if (self.progress.get('stage') != 'behavior_evaluation' or
                    self.progress.get('variant') is None or self.progress.get('seed') is None or
                    row.get('role') != 'campaign' or row.get('split') != 'dev'):
                return
            evaluation = numbers(row, ('level', 'episodes', 'wins', 'horizon', 'win_rate',
                                       'actions', 'decision_ms', 'actions_per_second',
                                       'total_episodes', 'current_episode', 'episode_actions',
                                       'episode_budget', 'seconds', 'episode_seconds', 'pair_index', 'pair_count'))
            evaluation.update(split='dev', policy=str(row.get('policy', ''))[:80],
                              level_complete=event == 'behavior_evaluation')
            if row.get('game'):
                evaluation['game'] = str(row['game'])[:80]
            evaluation['episode_complete'] = row.get('episode_complete') is True
            self.progress['evaluation'] = evaluation
            key = (self.progress['variant'], self.progress['seed'])
            self.executions.setdefault(key, {}).update(self.progress)
            self.revision += 1
            return
        if event not in ('campaign_started', 'campaign_phase', 'campaign_progress', 'campaign_evaluation',
                         'campaign_finished', 'campaign_error', 'campaign_imagination_gate',
                         'campaign_variant_deferred', 'campaign_paused'):
            return
        if event == 'campaign_started':
            self._reset()
            self.run['config'] = compact_values(row.get('config', {}))
        point = numbers(row, ('seed', 'actions', 'total_actions', 'level', 'updates',
                              'joint_updates', 'frozen_updates', 'actions_per_second'))
        for key in ('variant', 'stage', 'game', 'split'):
            if key in row:
                point[key] = str(row[key])[:80]
        if event == 'campaign_phase':
            if (point.get('variant'), point.get('seed')) != (self.progress.get('variant'), self.progress.get('seed')):
                self.progress = {}
            self.progress.update(point)
            self.progress.pop('evaluation', None)
            self.progress.pop('actions_per_second', None)
            self.progress['message'] = str(row.get('message', ''))[:240]
        elif event == 'campaign_progress':
            point['loss'] = numbers(row.get('loss'), tuple(row['loss'])[:80]) if isinstance(row.get('loss'), dict) else {}
            self.progress = point
            self.progress_points.append(dict(point))
        elif event == 'campaign_evaluation':
            if (point.get('variant'), point.get('seed')) == (self.progress.get('variant'), self.progress.get('seed')):
                self.progress.pop('evaluation', None)
            point['world'] = numbers(row.get('world'), VALIDATION_KEYS)
            world = row.get('world')
            if isinstance(world, dict) and isinstance(world.get('rollouts'), dict):
                point['world']['rollouts'] = compact_values(world['rollouts'])
            if isinstance(world, dict) and isinstance(world.get('per_pair'), list):
                point['world']['per_pair'] = [dict(numbers(r, VALIDATION_KEYS),
                    game=str(r.get('game', ''))[:80], level=number(r.get('level')))
                    for r in world['per_pair'][:100] if isinstance(r, dict)]
            point['behavior'] = numbers(row.get('behavior'),
                ('win_rate', 'wins', 'episodes', 'actions', 'actions_per_success', 'decision_ms',
                 'actions_per_second', 'seconds'))
            behavior = row.get('behavior')
            if isinstance(behavior, dict) and isinstance(behavior.get('wilson95'), list) and len(behavior['wilson95']) == 2:
                point['behavior']['wilson95'] = [number(v) for v in behavior['wilson95']]
            self.evaluations.append(point)
        elif event == 'campaign_imagination_gate':
            self.run['imagination_gate'] = {'variant': point.get('variant'), 'seed': point.get('seed'),
                                            'passed': row.get('passed') is True}
        elif event == 'campaign_variant_deferred':
            self.run.setdefault('deferred', {})[point.get('variant', 'unknown')] = {
                'reason': str(row.get('reason', ''))[:160],
                'fallback': str(row.get('fallback', ''))[:80]}
        elif event == 'campaign_finished':
            self.run['selected'] = compact_values(row.get('selected'))
            self.run['status'] = str(row.get('status', 'finished'))[:80]
        elif event == 'campaign_paused':
            self.progress.update(point)
            self.progress.pop('evaluation', None)
            self.run['status'] = 'paused'
        elif event == 'campaign_error':
            self.reason = str(row.get('error', row.get('message', row.get('reason', 'Error without details'))))[:500]
        key = (point.get('variant'), point.get('seed'))
        if key[0] is not None and key[1] is not None:
            summary = self.executions.setdefault(key, {})
            if event in ('campaign_phase', 'campaign_evaluation'):
                summary.pop('evaluation', None)
            if event in ('campaign_phase', 'campaign_progress'):
                summary.update(self.progress)
            elif event == 'campaign_evaluation':
                summary.update({k: v for k, v in point.items() if k in ('variant', 'seed', 'actions')})
            elif event == 'campaign_imagination_gate':
                summary.update(variant=key[0], seed=key[1], stage='imagination_gate',
                               admitted=row.get('passed') is True)
            if len(self.executions) > 100:
                self.executions.pop(next(iter(self.executions)))
        self.phase = event
        self.revision += 1

    def snapshot(self):
        with self.lock:
            source = self._read()
            return {'generation': self.generation, 'revision': self.revision, 'source': source,
                    'phase': self.phase, 'reason': self.reason, 'run': dict(self.run),
                    'progress': dict(self.progress), 'progress_points': list(self.progress_points),
                    'executions': [dict(p) for p in self.executions.values()],
                    'evaluations': list(self.evaluations), 'max_points': self.max_points,
                    'report': self._report()}


def subgoal_snapshot(path):
    try:
        if path.stat().st_size > MAX_READ:
            raise ValueError('Subgoal report exceeds dashboard limit')
        data = json.loads(path.read_text())
        if data.get('kind') != 'synthetic_subgoals':
            raise ValueError('Expected a synthetic-subgoal report')
        data['config'] = {k: v for k, v in data.get('config', {}).items() if k not in ('source', 'init_config', 'output')}
        data['age_seconds'] = max(0., time.time() - data.get('updated_at', path.stat().st_mtime))
        return data
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return {'kind': 'synthetic_subgoals', 'status': 'waiting', 'message': str(error), 'progress': {}}


def trajectory_snapshot(report_path):
    """Only the one sidecar belonging to the report's published sample."""
    try:
        metadata = subgoal_snapshot(report_path).get('planner_trajectory')
        if not metadata:
            return None
        path = report_path.with_name('planner_trajectory.json')
        if path.stat().st_size > MAX_READ:
            return None
        data = json.loads(path.read_text())
        return data if data.get('kind') == 'planner_trajectory' and data.get('id') == metadata['id'] else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def sequence_snapshot(path):
    try:
        if path.stat().st_size > MAX_READ:
            raise ValueError('Sequence report exceeds dashboard limit')
        data = json.loads(path.read_text())
        if data.get('kind') != 'game_sequences':
            raise ValueError('Expected a game-sequence report')
        data['age_seconds'] = max(0., time.time() - data.get('updated_at', path.stat().st_mtime))
        return data
    except (OSError, ValueError) as error:
        return dict(kind='game_sequences', status='waiting', message=str(error), history=[], progress={})


def make_server(address, monitor, behavior=None, campaign=None, subgoal_report=None, curriculum_service=None,
                sequence_report=None, history_subgoal_report=None):
    gpu = GpuMonitor()
    if sequence_report and any(x is not None for x in (monitor, behavior, campaign, subgoal_report, curriculum_service)):
        raise ValueError('Sequence report selects one independent run')
    if curriculum_service and not subgoal_report:
        raise ValueError('Curriculum controls require a subgoal report')
    if history_subgoal_report and not sequence_report:
        raise ValueError('Historical subgoal view requires a sequence report')
    control = CurriculumControl(curriculum_service, subgoal_report) if curriculum_service else None
    optimization = OptimizationControl(sequence_report)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if urlsplit(self.path).path == '/api/optimization':
                if not optimization.authorized(self.headers.get('X-Optimization-Token')):
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 1024 or self.headers.get('Content-Type') != 'application/json':
                        raise ValueError('Invalid JSON request.')
                    data = json.loads(self.rfile.read(length))
                    self.send_json(202, optimization.request(data))
                except (ValueError, TypeError):
                    self.send_json(400, {'error': 'Invalid diagnostic request.'})
                except (OSError, RuntimeError) as error:
                    self.send_json(409, {'error': str(error)})
                return
            if urlsplit(self.path).path != '/api/curriculum' or control is None:
                self.send_error(404)
                return
            if not control.authorized(self.headers.get('X-Curriculum-Token')):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 256 or self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('Send a valid JSON operation.')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or set(body) != {'operation'} or body['operation'] not in ('phase', 'level'):
                    raise ValueError('Invalid curriculum operation.')
            except (ValueError, TypeError):
                self.send_json(400, {'error': 'Invalid curriculum operation.'})
                return
            try:
                state = control.request(body['operation'], subgoal_snapshot(subgoal_report))
                self.send_json(202, state)
            except (OSError, ValueError, IndexError, StopIteration, subprocess.SubprocessError):
                self.send_json(409, {'error': 'Could not request advancement. Check trainer status.'})

        def send_json(self, status, payload, etag=None):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if etag is not None:
                self.send_header('ETag', etag)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == '/api/optimization':
                revision = optimization.revision()
                if revision is not None and self.headers.get('If-None-Match') == revision:
                    self.send_response(304)
                    self.send_header('ETag', revision)
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    return
                self.send_json(200, optimization.snapshot(), revision)
                return
            if route == '/api/state':
                state = ({'sequences': sequence_snapshot(sequence_report)} if sequence_report else
                         {'subgoals': subgoal_snapshot(subgoal_report)} if subgoal_report else monitor.snapshot())
                if control is not None:
                    state['curriculum_control'] = control.snapshot(state['subgoals'])
                if behavior is not None and subgoal_report is None:
                    state['behavior'] = behavior.snapshot()
                if campaign is not None and subgoal_report is None:
                    state['campaign'] = campaign.snapshot()
                state['gpu'] = gpu.snapshot()
                if sequence_report:
                    state['history_available'] = history_subgoal_report is not None
                data = json.dumps(state, allow_nan=False, separators=(',', ':')).encode()
                mime = 'application/json; charset=utf-8'
            elif route == '/api/history/state' and history_subgoal_report:
                history = subgoal_snapshot(history_subgoal_report)
                # The archived writer may have stopped while its last status was running.
                history['historical'] = True
                history['status'] = 'historical' if history.get('status') != 'waiting' else 'waiting'
                self.send_json(200, {'subgoals': history, 'curriculum_control': None})
                return
            elif route == '/api/history/trajectory' and history_subgoal_report:
                self.send_json(200, {'trajectory': trajectory_snapshot(history_subgoal_report)})
                return
            elif route == '/history.html' and history_subgoal_report:
                data = (ASSETS / 'subgoals.html').read_bytes()
                mime = 'text/html; charset=utf-8'
            elif route == '/api/trajectory' and subgoal_report:
                data = json.dumps({'trajectory': trajectory_snapshot(subgoal_report)}, allow_nan=False, separators=(',', ':')).encode()
                mime = 'application/json; charset=utf-8'
            elif route in ('/', '/index.html', '/dashboard.css', '/chart-tools.js', '/dashboard.js', '/campaign.html', '/campaign.js', '/subgoals.js', '/sequences.js',
                           '/optimization.html', '/optimization.js', '/optimization.css'):
                name = 'index.html' if route == '/' else route[1:]
                if subgoal_report and name in ('index.html', 'campaign.html'):
                    name = 'subgoals.html'
                if sequence_report and name in ('index.html', 'campaign.html'):
                    name = 'sequences.html'
                data = (ASSETS / name).read_bytes()
                mime = {'html': 'text/html', 'css': 'text/css', 'js': 'text/javascript'}[name.rsplit('.', 1)[1]] + '; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; frame-ancestors 'self'; base-uri 'none'")
            self.end_headers()
            if self.command != 'HEAD':
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        do_HEAD = do_GET

        def log_message(self, format, *args):
            pass  # Polling creates no second historical log.

    return ThreadingHTTPServer(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--behavior-log', type=Path, help='Independent goal/planner JSON-line log')
    parser.add_argument('--campaign-log', type=Path, help='Independent architecture comparison JSON-line log')
    parser.add_argument('--campaign-report', type=Path, help='Existing campaign report.json with completed comparisons')
    parser.add_argument('--subgoal-report', type=Path, help='Select the new synthetic-subgoal run for the entire dashboard')
    parser.add_argument('--sequence-report', type=Path, help='Select a game-sequence training report')
    parser.add_argument('--history-subgoal-report', type=Path,
                        help='Read-only previous subgoal run, kept separate from the active sequence run')
    parser.add_argument('--curriculum-service', help='Opt in to phase/level controls for this systemd user service')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--max-points', type=int, default=2000)
    args = parser.parse_args()
    if not args.log and not args.subgoal_report and not args.sequence_report:
        parser.error('Select --log, --subgoal-report or --sequence-report')
    if args.sequence_report and any((args.log, args.subgoal_report, args.behavior_log, args.campaign_log, args.campaign_report, args.curriculum_service)):
        parser.error('--sequence-report selects one run; do not combine report selectors')
    if args.curriculum_service and not args.subgoal_report:
        parser.error('--curriculum-service requires --subgoal-report')
    if args.history_subgoal_report and not args.sequence_report:
        parser.error('--history-subgoal-report requires --sequence-report')
    if args.subgoal_report and any((args.log, args.behavior_log, args.campaign_log, args.campaign_report)):
        parser.error('--subgoal-report selects one run; do not combine it with legacy log selectors')
    if not 2 <= args.max_points <= 10000:
        parser.error('--max-points must be between 2 and 10000')
    monitor = LogMonitor(args.log, args.max_points) if args.log else None
    behavior = BehaviorMonitor(args.behavior_log, args.max_points) if args.behavior_log else None
    campaign = CampaignMonitor(args.campaign_log, args.max_points, args.campaign_report) if args.campaign_log else None
    with make_server((args.host, args.port), monitor, behavior, campaign, args.subgoal_report, args.curriculum_service,
                     args.sequence_report, args.history_subgoal_report) as server:
        print(f'Training dashboard: http://{args.host}:{server.server_port}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
