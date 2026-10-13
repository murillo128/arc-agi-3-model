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
                 'passes', 'collapse_streak', 'actions_per_second', 'replay_episodes', 'gpu_peak_mb')
LOSS_KEYS = ('loss', 'latent_mse', 'sigreg', 'variance_floor', 'decoder_ce',
             'reconstruction_ce', 'local_reconstruction_ce', 'gradient_norm')
VALIDATION_KEYS = ('delta_error', 'decoder_delta_error', 'latent_delta_mse', 'latent_std',
                   'transitions', 'changed_transitions', 'skipped', 'wins', 'game_overs')
CONFIG_KEYS = ('learning_rate', 'validate_every', 'consecutive_passes', 'delta_error',
               'latent_delta_mse', 'min_latent_std', 'collapse_std')


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def numbers(source, keys):
    if not isinstance(source, dict):
        return {}
    return {k: number(source[k]) for k in keys if k in source}


class LogMonitor:
    """Bounded incremental reader shared by browser clients; never writes the log."""

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
            self.run['compile'] = row.get('compile') is True
        elif event == 'ready':
            self.run['parameters'] = number(row.get('parameters'))
            self.phase = 'ready'
        elif event in ('progress', 'resumed', 'stopped'):
            actions = number(row.get('actions'))
            previous = number(self.progress.get('actions'))
            if actions is not None and previous is not None and actions < previous:
                self._reset()
            self.progress.update(numbers(row, PROGRESS_KEYS))
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
        elif event not in ('checkpoint', 'level_converged', 'encoder_collapsed'):
            return
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
                    if len(line) > MAX_LINE:
                        continue
                    try:
                        self._event(json.loads(line))
                    except (ValueError, UnicodeError):
                        # SDK logs, warnings and a malformed record are not metrics.
                        continue
                if len(self.pending) > MAX_LINE:
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


def make_server(address, monitor):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path).path
            if route == '/api/state':
                data = json.dumps(monitor.snapshot(), allow_nan=False, separators=(',', ':')).encode()
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
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--max-points', type=int, default=2000)
    args = parser.parse_args()
    if not 2 <= args.max_points <= 10000:
        parser.error('--max-points must be between 2 and 10000')
    monitor = LogMonitor(args.log, args.max_points)
    with make_server((args.host, args.port), monitor) as server:
        print(f'Training dashboard: http://{args.host}:{server.server_port}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
