"""Opt-in, bounded dashboard history; never writes the trainer's report."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading


MAX_REPORT = 8 * 1024 * 1024
MAX_ARCHIVE = 64 * 1024 * 1024
SCALARS = ('batch', 'actions', 'recent_tps', 'examples_per_second', 'peak_gpu_gib',
           'loss', 'reconstruction_loss', 'macro_delta_error', 'total_seconds',
           'collection_seconds', 'representation_seconds', 'encode_seconds',
           'predictor_seconds', 'evaluation_seconds', 'drift_seconds',
           'exploration_seconds', 'save_seconds', 'diagnostics_seconds')
SECTIONS = ('quality', 'diagnostics', 'predictor_terms', 'representation_terms', 'learning_rates')


def history_run(report_path, report):
    """Bind a snapshot to its source, training process and game cohort."""
    return hashlib.sha256(json.dumps([str(Path(report_path).resolve()), report.get('optimization_identity'),
        sorted(report.get('games', []))], sort_keys=True).encode()).hexdigest()[:24]


def numeric_tree(value, depth=0):
    if type(value) in (int, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict) and depth < 4:
        return {str(k)[:120]: numeric_tree(v, depth + 1)
                for k, v in list(value.items())[:132] if not isinstance(v, (list, str))}
    return None


def compact_row(row):
    result = {key: numeric_tree(row[key]) for key in SCALARS if key in row}
    result.update({key: numeric_tree(row[key]) for key in SECTIONS if key in row})
    drift_source = (row.get('diagnostics') or {}).get('drift_source')
    if isinstance(drift_source, str):
        result['diagnostics']['drift_source'] = drift_source[:80]
    for key in ('phase', 'measurement_source'):
        if isinstance(row.get(key), str):
            result[key] = row[key][:80]
    exploration = row.get('exploration') or {}
    result['exploration'] = numeric_tree({k: v for k, v in exploration.items() if k != 'games'})
    for key in ('policy', 'identity'):
        if isinstance(exploration.get(key), str):
            result['exploration'][key] = exploration[key][:120]
    result['sessions'] = [dict(game=s['game'], steps=s.get('steps', 0))
                          for s in row.get('sessions', [])[:128] if isinstance(s.get('game'), str)]
    return result


class SequenceHistory:
    def __init__(self, report, archive, max_points=2000):
        self.report, self.archive = Path(report).resolve(), Path(archive).resolve()
        if self.report == self.archive:
            raise ValueError('History must not overwrite the sequence report')
        if not 2 <= max_points <= 10000:
            raise ValueError('History limit must be between 2 and 10000')
        self.max_points = max_points
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.run, self.rows, self.stamp = None, [], None
        self.error, self.writable = None, True
        self.payload, self.revision = b'{}', None
        try:
            if self.archive.exists():
                if self.archive.stat().st_size > MAX_ARCHIVE:
                    raise ValueError('Oversized history')
                saved = json.loads(self.archive.read_text())
                if saved.get('kind') != 'sequence_dashboard_history' or saved.get('source') != str(self.report):
                    raise ValueError('History belongs to a different source')
                self.run = saved['run']
                self.rows = [compact_row(r) for r in saved['history'][-max_points:]]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            self.rows, self.run, self.writable = [], None, False
            self.error = 'Archive unavailable; collecting in memory only. Choose a new archive path to persist history.'
        self.refresh()

    def start(self):
        self.thread = threading.Thread(target=self._collect, name='sequence-history', daemon=True)
        self.thread.start()

    def _collect(self):
        while not self.stop.wait(5):
            self.refresh()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)

    def refresh(self):
        with self.lock:
            try:
                stat = self.report.stat()
                stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
                if stamp == self.stamp:
                    return
                if stat.st_size > MAX_REPORT:
                    raise ValueError('Oversized report')
                report = json.loads(self.report.read_text())
                if report.get('kind') != 'game_sequences':
                    raise ValueError('Not a sequence report')
                live = report.get('history', [])
                if not live:
                    return
                run = history_run(self.report, report)
                latest = live[-1]['batch']
                rollback = self.rows and latest < self.rows[-1]['batch']
                if self.run != run or rollback:
                    self.rows = []
                merged = {r['batch']: r for r in self.rows}
                for row in live:
                    if type(row.get('batch')) is int and 0 <= row['batch'] <= latest:
                        merged[row['batch']] = compact_row(row)
                self.rows = [merged[b] for b in sorted(merged)[-self.max_points:]]
                self.run, self.stamp = run, stamp
                saved = dict(kind='sequence_dashboard_history', source=str(self.report),
                             run=run, history=self.rows)
                disk = json.dumps(saved, allow_nan=False, separators=(',', ':')).encode()
                if len(disk) > MAX_ARCHIVE:
                    raise ValueError('History exceeds storage bound')
                if self.writable:
                    try:
                        self.archive.parent.mkdir(parents=True, exist_ok=True)
                        temporary = None
                        try:
                            with tempfile.NamedTemporaryFile(dir=self.archive.parent, prefix='.sequence-history-', delete=False) as stream:
                                temporary = stream.name
                                stream.write(disk)
                            os.replace(temporary, self.archive)
                        finally:
                            if temporary and os.path.exists(temporary):
                                os.unlink(temporary)
                        self.error = None
                    except OSError:
                        self.error = 'History could not be saved; current measurements remain available in memory.'
                self.payload = json.dumps(dict(run=run, history=self.rows), allow_nan=False,
                                          separators=(',', ':')).encode()
                self.revision = '"' + hashlib.sha256(self.payload).hexdigest()[:24] + '"'
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                self.error = 'Waiting for a complete sequence report; previously collected history is retained.'

    def metadata(self):
        with self.lock:
            return dict(enabled=True, run=self.run, revision=self.revision, points=len(self.rows),
                        limit=self.max_points, first_batch=self.rows[0]['batch'] if self.rows else None,
                        last_batch=self.rows[-1]['batch'] if self.rows else None, error=self.error)

    def snapshot(self):
        with self.lock:
            return self.payload, self.revision
