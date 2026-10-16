"""Optional dashboard bridge to one configured trainer's POSIX controls."""
import hmac
import os
from pathlib import Path
import secrets
import signal
import subprocess
import threading


class CurriculumControl:
    def __init__(self, service, report):
        self.service, self.report = service, Path(report).resolve()
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.pending = None
        self.result = None

    def authorized(self, token):
        return hmac.compare_digest((token or '').encode(), self.token.encode())

    def _target(self):
        """Pin process identity before inspecting its command and handlers."""
        result = subprocess.run(['systemctl', '--user', 'show', '--property=MainPID', '--value',
                                 '--', self.service], capture_output=True, text=True, timeout=2, check=True)
        pid = int(result.stdout.strip())
        if pid <= 1:
            raise ValueError('El entrenador está detenido.')
        fd = os.pidfd_open(pid)
        try:
            proc = Path('/proc') / str(pid)
            argv = (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if argv[1:3] != ['-m', 'model.subgoal_training']:
                # The live service uses Python's unbuffered flag.
                if argv[1:4] != ['-u', '-m', 'model.subgoal_training']:
                    raise ValueError('El servicio no ejecuta el entrenador esperado.')
            output = Path(argv[argv.index('--output') + 1])
            if not output.is_absolute():
                output = (proc / 'cwd').resolve() / output
            if (output / 'report.json').resolve() != self.report:
                raise ValueError('El servicio corresponde a otra campaña.')
            caught = next(line.split()[1] for line in (proc / 'status').read_text().splitlines()
                          if line.startswith('SigCgt:'))
            if not all(int(caught, 16) & (1 << (s - 1)) for s in (signal.SIGUSR2, signal.SIGHUP)):
                raise ValueError('El entrenador no tiene instalados los controles.')
            return pid, fd
        except BaseException:
            os.close(fd)
            raise

    def _state(self, report, pid):
        events = report.get('manual_controls', [])
        if self.pending:
            if pid != self.pending['pid'] or report.get('status') != 'running':
                self.result = dict(status='error', message='El entrenador se detuvo o reinició antes de confirmar la petición.')
                self.pending = None
            else:
                event = next((e for e in events if e not in self.pending['previous']
                              and e.get('operation') == self.pending['operation']), None)
                if event:
                    self.result = dict(status='applied', event=event)
                    self.pending = None
        p = report.get('progress', {})
        target = p.get('representation_level') or dict(game=p.get('game'), level=p.get('level'))
        levels = report.get('curriculum', {}).get('levels', [])
        row = next((r for r in levels if (r['game'], r['level']) == (target.get('game'), target.get('level'))), {})
        running = bool(pid and report.get('status') == 'running' and p.get('actions', 0) < p.get('total_actions', 0)
                       and report.get('control_signals') == dict(checkpoint='SIGUSR1', phase='SIGUSR2', level='SIGHUP'))
        return dict(token=self.token, target=target, pending=({'operation': self.pending['operation']} if self.pending else None),
                    result=self.result, phase=running and bool(row) and not row.get('planner') and not self.pending,
                    level=running and any(r['game'] == target.get('game') and not r.get('unlocked') for r in levels)
                    and not self.pending)

    def snapshot(self, report):
        with self.lock:
            try:
                pid, fd = self._target()
                os.close(fd)
                state = self._state(report, pid)
                state['message'] = '' if report.get('status') == 'running' else 'El entrenamiento no está activo.'
            except (OSError, ValueError, IndexError, StopIteration, subprocess.SubprocessError):
                state = self._state(report, None)
                state['message'] = 'Controles no disponibles: comprueba que el entrenador de esta campaña está activo.'
            return state

    def request(self, operation, report):
        with self.lock:
            pid, fd = self._target()
            try:
                state = self._state(report, pid)
                if not state[operation]:
                    raise ValueError('Ese avance no está disponible o hay una petición pendiente.')
                number = signal.SIGUSR2 if operation == 'phase' else signal.SIGHUP
                signal.pidfd_send_signal(fd, number)
                self.pending = dict(pid=pid, operation=operation, previous=list(report.get('manual_controls', [])))
                self.result = None
                return self._state(report, pid)
            finally:
                os.close(fd)
