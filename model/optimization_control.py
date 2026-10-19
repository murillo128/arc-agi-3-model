"""Torch-free transport for bounded diagnostic requests to one configured run."""
import hmac
import hashlib
import json
from pathlib import Path
import secrets
import threading
import time


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
            return dict(status='unavailable', message='Esta ejecución no registra optimización por secuencias.')
        try:
            path = self.report_path.with_name('optimization.json')
            if path.stat().st_size > 8 * 1024**2 or self.report_path.stat().st_size > 8 * 1024**2:
                raise ValueError('El registro de optimización supera el límite de lectura.')
            def reject_constant(value):
                raise ValueError('Non-finite diagnostic value')
            state = json.loads(path.read_text(), parse_constant=reject_constant)
            run = json.loads(self.report_path.read_text(), parse_constant=reject_constant)
            if (state.get('kind') != 'optimization' or state.get('version') != 1 or
                    not isinstance(state.get('catalog'), list) or not isinstance(state.get('events'), list) or
                    not isinstance(state.get('planes'), dict) or not isinstance(state.get('points'), list) or
                    not isinstance(state.get('identity'), str) or
                    state.get('identity') != run.get('optimization_identity') or
                    state.get('batch', 0) > run.get('progress', {}).get('batches', 0)):
                raise ValueError('Esperando un registro compatible con el último batch guardado.')
            age = max(0., time.time() - state['updated_at'])
            state.update(status='ready', age_seconds=age, running=run.get('status') == 'running',
                         session_count=run.get('config', {}).get('batch_games', 4))
            state['control'] = dict(token=self.token, enabled=state['running'] and age < 300)
            return state
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return dict(status='waiting', message='Esperando la primera captura de optimización. El historial anterior no contiene gradientes.')

    def authorized(self, token):
        return isinstance(token, str) and hmac.compare_digest(self.token, token)

    def request(self, data):
        with self.lock:
            state = self.snapshot()
            if state.get('status') != 'ready' or not state['control']['enabled']:
                raise RuntimeError('El entrenador no está disponible para capturas.')
            if not isinstance(data, dict):
                raise ValueError('Solicitud no válida.')
            operation = data.get('operation')
            if operation == 'matrix':
                if set(data) != {'operation', 'matrix'} or data['matrix'] not in {m['name'] for m in state['catalog']}:
                    raise ValueError('Selecciona una matriz registrada.')
            elif operation == 'probe':
                if (set(data) != {'operation', 'session'} or type(data['session']) is not int or
                        not 0 <= data['session'] < state['session_count']):
                    raise ValueError('Selecciona una sesión válida.')
            elif operation != 'recalibrate' or set(data) != {'operation'}:
                raise ValueError('Operación no válida.')
            path = self.report_path.with_name('optimization_request.json')
            if path.exists():
                raise RuntimeError('Ya hay una captura pendiente; se atenderá al comenzar el próximo batch.')
            request_id = secrets.token_hex(8)
            payload = dict(data, identity=state['identity'], id=request_id)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, allow_nan=False))
            temporary.replace(path)
            return dict(id=request_id, status='pending', message='Solicitud pendiente del próximo batch.')
