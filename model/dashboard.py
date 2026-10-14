"""Read-only live dashboard for the trainer's existing JSON-line log.

Run with python -m model.dashboard --log model/runs/exploration/run.log.
No torch imports, training-process hooks, or persistent dashboard storage.
"""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit


MAX_READ = 8 * 1024 * 1024
MAX_LINE = 64 * 1024
ASSETS = Path(__file__).with_name('dashboard_web')
PROGRESS_KEYS = ('actions', 'updates', 'episodes', 'level_index', 'level_actions',
                 'passes', 'collapse_streak', 'actions_per_second', 'replay_episodes', 'gpu_peak_mb',
                 'episode_actions', 'episode_observations', 'device_cache_mb', 'trained_transitions', 'trained_transitions_per_second')
LOSS_KEYS = ('loss', 'latent_mse', 'sigreg', 'variance_floor', 'decoder_ce',
             'reconstruction_ce', 'local_reconstruction_ce', 'gradient_norm', 'samples', 'seconds',
             'rollout_mse', 'rollout_samples')
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
        if event not in ('campaign_started', 'campaign_phase', 'campaign_progress', 'campaign_evaluation',
                         'campaign_finished', 'campaign_error', 'campaign_imagination_gate',
                         'campaign_variant_deferred'):
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
            self.progress.pop('actions_per_second', None)
            self.progress['message'] = str(row.get('message', ''))[:240]
        elif event == 'campaign_progress':
            point['loss'] = numbers(row.get('loss'), tuple(row['loss'])[:80]) if isinstance(row.get('loss'), dict) else {}
            self.progress = point
            self.progress_points.append(dict(point))
        elif event == 'campaign_evaluation':
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
        elif event == 'campaign_error':
            self.reason = str(row.get('error', row.get('message', row.get('reason', 'Error sin detalle'))))[:500]
        key = (point.get('variant'), point.get('seed'))
        if key[0] is not None and key[1] is not None:
            summary = self.executions.setdefault(key, {})
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
                    'evaluations': list(self.evaluations), 'max_points': self.max_points}


def make_server(address, monitor, behavior=None, campaign=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path).path
            if route == '/api/state':
                state = monitor.snapshot()
                if behavior is not None:
                    state['behavior'] = behavior.snapshot()
                if campaign is not None:
                    state['campaign'] = campaign.snapshot()
                data = json.dumps(state, allow_nan=False, separators=(',', ':')).encode()
                mime = 'application/json; charset=utf-8'
            elif route in ('/', '/index.html', '/dashboard.css', '/dashboard.js'):
                name = 'index.html' if route == '/' else route[1:]
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
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--behavior-log', type=Path, help='Independent goal/planner JSON-line log')
    parser.add_argument('--campaign-log', type=Path, help='Independent architecture comparison JSON-line log')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--max-points', type=int, default=2000)
    args = parser.parse_args()
    if not 2 <= args.max_points <= 10000:
        parser.error('--max-points must be between 2 and 10000')
    monitor = LogMonitor(args.log, args.max_points)
    behavior = BehaviorMonitor(args.behavior_log, args.max_points) if args.behavior_log else None
    campaign = CampaignMonitor(args.campaign_log, args.max_points) if args.campaign_log else None
    with make_server((args.host, args.port), monitor, behavior, campaign) as server:
        print(f'Training dashboard: http://{args.host}:{server.server_port}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
