"""Torch-free transport for bounded diagnostic requests to one configured run."""
import hmac
import hashlib
import json
import math
from pathlib import Path
import secrets
import threading
import time


def _require(condition):
    if not condition:
        raise ValueError('Malformed optimization snapshot')


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**53 - 1


def _number(value, minimum=-math.inf, maximum=math.inf, *, nullable=False):
    return ((nullable and value is None) or
            (type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum))


def _validate_snapshot(state, session_count):
    """Validate the renderer's nested contract, including retained-history links.

    Missing captures/uncalibrated planes are valid; a partially malformed capture
    is not installed as live data. No Torch imports or tensor reconstruction.
    """
    _require(isinstance(state, dict) and state['kind'] == 'optimization' and type(state['version']) is int and state['version'] == 1)
    _require(isinstance(state['identity'], str) and bool(state['identity']))
    _require(_integer(state['batch']) and _integer(state['tick']) and _number(state['updated_at'], 0))
    _require(_integer(state['every'], 1) and _integer(state['sample_per_tensor'], 1) and state['sample_per_tensor'] <= 256)
    _require(_integer(session_count, 1))
    catalog = state['catalog']
    _require(isinstance(catalog, list) and bool(catalog))
    names = {}
    for meta in catalog:
        _require(isinstance(meta, dict) and isinstance(meta['name'], str) and bool(meta['name']))
        _require(meta['name'] not in names and meta['module'] in ('encoder', 'decoder', 'dynamics', 'goal_planner', 'path_planner'))
        _require(isinstance(meta['shape'], list) and len(meta['shape']) <= 64 and all(_integer(n, 1) for n in meta['shape']))
        _require(_integer(meta['size'], 1) and math.prod(meta['shape']) == meta['size'])
        names[meta['name']] = meta
    _require(isinstance(state['selected'], str) and state['selected'] in names)

    events, points, planes = state['events'], state['points'], state['planes']
    _require(isinstance(events, list) and len(events) <= 64)
    _require(isinstance(points, list) and len(points) <= 96 and isinstance(planes, dict))
    event_batches, last_batch, last_id = {}, 0, 0
    for event in events:
        _require(isinstance(event, dict) and _integer(event['id'], last_id + 1) and event['id'] <= state['tick'])
        _require(_integer(event['batch'], max(1, last_batch)) and event['batch'] <= state['batch'])
        _require(event['phase'] in ('predictor', 'representation', 'goal_planner', 'path_planner') and _number(event['clip_scale'], 0, 1))
        _require(isinstance(event['matrices'], dict) and event['matrices'].keys() <= names.keys())
        for row in event['matrices'].values():
            _require(isinstance(row, dict) and row['status'] in ('measured', 'no_gradient', 'frozen'))
            if row['status'] == 'measured':
                for key in ('gradient_rms', 'weight_rms', 'update_rms', 'relative_update', 'lr'):
                    _require(_number(row[key], 0, nullable=True))
                _require(_number(row['alignment'], nullable=True))
        detail = event.get('detail')
        if detail is not None:
            _require(isinstance(detail, dict) and isinstance(detail['name'], str) and detail['name'] in names)
            _require(event['matrices'].get(detail['name'], {}).get('status') == 'measured')
            meta = names[detail['name']]
            rows = meta['shape'][0] if meta['shape'] else 1
            cols = meta['size'] // rows
            for key in ('gradient', 'update'):
                grid = detail[key]
                _require(isinstance(grid, list) and len(grid) == min(16, rows))
                _require(all(isinstance(row, list) and len(row) == min(16, cols) and
                             all(_number(value, 0) for value in row) for row in grid))
            histogram = detail['histogram']
            _require(isinstance(histogram, dict))
            counts, edges = histogram['counts'], histogram['edges']
            _require(isinstance(counts, list) and 0 < len(counts) <= 16 and all(_integer(n) for n in counts))
            _require(sum(counts) == min(state['sample_per_tensor'], meta['size']))
            _require(isinstance(edges, list) and len(edges) == len(counts) + 1 and all(_number(n) for n in edges))
            _require(all(a <= b for a, b in zip(edges, edges[1:])))
        last_id, last_batch = event['id'], event['batch']
        event_batches[last_id] = last_batch

    last_batch = 0
    for point in points:
        _require(isinstance(point, dict) and _integer(point['batch'], last_batch + 1) and point['batch'] <= state['batch'])
        _require(_integer(point['event_id']) and point['event_id'] <= state['tick'])
        _require(_number(point['reference_loss'], 0, nullable=True))
        losses = point.get('module_losses', {})
        _require(isinstance(losses, dict) and losses.keys() <= {meta['module'] for meta in catalog})
        _require(all(_number(value, 0, nullable=True) for value in losses.values()))
        # Evicted events are deliberately absent, not reassigned to a newer batch.
        _require(point['event_id'] not in event_batches or event_batches[point['event_id']] <= point['batch'])
        last_batch = point['batch']
    _require(planes.keys() <= {meta['module'] for meta in catalog})
    for plane in planes.values():
        _require(isinstance(plane, dict) and _integer(plane['basis_id']))
        projections = plane['projections']
        _require(isinstance(projections, list) and len(projections) == (len(points) if plane['basis_id'] else 0))
        dimensions = set()
        for p in projections:
            _require(isinstance(p, dict) and _number(p['x']) and _number(p['y']))
            _require(_number(p['coverage'], 0, 1, nullable=True) and _number(p['angle'], 0, 180, nullable=True))
            _require(_number(p['step_norm'], 0))
            _require(('z' in p) == ('coverage_3d' in p))
            dimensions.add('z' in p)
            if 'z' in p:
                _require(_number(p['z']) and _number(p['coverage_3d'], 0, 1, nullable=True))
                if p['step_norm'] == 0:
                    _require(p['coverage'] is None and p['coverage_3d'] is None)
                else:
                    _require(p['coverage'] is not None and p['coverage_3d'] is not None
                             and p['coverage_3d'] + 1e-9 >= p['coverage'])
        _require(len(dimensions) <= 1)

    probe = state.get('probe')
    if probe is None:
        return
    _require(isinstance(probe, dict) and probe['status'] in ('ready', 'unavailable'))
    if probe['status'] == 'unavailable':
        _require(isinstance(probe['reason'], str))
        return
    _require(_integer(probe['session']) and probe['session'] < session_count and isinstance(probe['game'], str))
    _require(_integer(probe['batch'], 1) and probe['batch'] <= state['batch'])
    for key in ('session_steps', 'context_start', 'start', 'stop'):
        _require(_integer(probe[key]))
    _require(probe['context_start'] <= probe['start'] < probe['stop'] <= probe['session_steps'])
    _require(isinstance(probe['boundaries'], list) and all(
        _integer(t, probe['context_start']) and t < probe['stop'] for t in probe['boundaries']))
    _require(isinstance(probe['steps'], list) and 0 < len(probe['steps']) <= probe['stop'] - probe['context_start'])
    previous = probe['context_start'] - 1
    for step in probe['steps']:
        _require(isinstance(step, dict) and _integer(step['t'], previous + 1) and step['t'] < probe['stop'])
        _require(_number(step['rms'], 0, nullable=True))
        previous = step['t']
    for key in ('loss', 'cleared_loss', 'seconds'):
        _require(_number(probe[key], 0))


class OptimizationControl:
    def __init__(self, report_path):
        self.report_path = Path(report_path) if report_path else None
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()

    def revision(self):
        if self.report_path is None:
            return None
        try:
            capture = self.report_path.with_name('optimization.json').stat()
            age = time.time() - capture.st_mtime
            value = (capture.st_mtime_ns, self.report_path.stat().st_mtime_ns,
                     age >= 180, age >= 300, self.token)
            return '"' + hashlib.sha256(str(value).encode()).hexdigest()[:24] + '"'
        except OSError:
            return None

    def snapshot(self):
        if self.report_path is None:
            return dict(status='unavailable', message='This run does not record sequence optimization.')
        try:
            path = self.report_path.with_name('optimization.json')
            if path.stat().st_size > 8 * 1024**2 or self.report_path.stat().st_size > 8 * 1024**2:
                raise ValueError('The optimization record exceeds the read limit.')
            def reject_constant(value):
                raise ValueError('Non-finite diagnostic value')
            def finite_float(value):
                result = float(value)
                if not math.isfinite(result):
                    reject_constant(value)
                return result
            state = json.loads(path.read_text(), parse_constant=reject_constant, parse_float=finite_float)
            run = json.loads(self.report_path.read_text(), parse_constant=reject_constant, parse_float=finite_float)
            session_count = run['config']['batch_games']
            _validate_snapshot(state, session_count)
            if (run.get('kind') != 'game_sequences' or not _integer(run['progress']['batches']) or
                    state.get('identity') != run.get('optimization_identity') or
                    state['batch'] > run['progress']['batches']):
                raise ValueError('Waiting for a record compatible with the last saved batch.')
            age = max(0., time.time() - state['updated_at'])
            state.update(status='ready', age_seconds=age, running=run.get('status') == 'running',
                         session_count=session_count)
            state['control'] = dict(token=self.token, enabled=state['running'] and age < 300)
            return state
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
            return dict(status='waiting', message='Waiting for the first optimization capture. Earlier history contains no gradients.')

    def authorized(self, token):
        return isinstance(token, str) and hmac.compare_digest(self.token, token)

    def request(self, data):
        with self.lock:
            state = self.snapshot()
            if state.get('status') != 'ready' or not state['control']['enabled']:
                raise RuntimeError('The trainer is unavailable for captures.')
            if not isinstance(data, dict):
                raise ValueError('Invalid request.')
            operation = data.get('operation')
            if operation == 'matrix':
                if set(data) != {'operation', 'matrix'} or data['matrix'] not in {m['name'] for m in state['catalog']}:
                    raise ValueError('Select a registered matrix.')
            elif operation == 'probe':
                if (set(data) != {'operation', 'session'} or type(data['session']) is not int or
                        not 0 <= data['session'] < state['session_count']):
                    raise ValueError('Select a valid session.')
            elif operation != 'recalibrate' or set(data) != {'operation'}:
                raise ValueError('Invalid operation.')
            path = self.report_path.with_name('optimization_request.json')
            if path.exists():
                raise RuntimeError('A capture is already pending; it will be handled at the start of the next batch.')
            request_id = secrets.token_hex(8)
            payload = dict(data, identity=state['identity'], id=request_id)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, allow_nan=False))
            temporary.replace(path)
            return dict(id=request_id, status='pending', message='Request pending until the next batch.')
