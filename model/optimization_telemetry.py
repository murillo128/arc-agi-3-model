"""Opt-in, bounded measurements. Never changes a gradient, parameter or optimizer."""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .optimization_projection import ParameterPlane


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n')
    temporary.replace(path)


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def block_rms(tensor):
    """Actual RMS within spatial blocks, including all entries (not sampled pixels)."""
    x = tensor.detach().float().reshape(tensor.shape[0] if tensor.ndim else 1, -1)
    return F.adaptive_avg_pool2d(x.square()[None, None], (min(16, x.shape[0]), min(16, x.shape[1])))[0, 0].sqrt()


class OptimizationTelemetry:
    def __init__(self, model, directory, *, every=8, restored_batch=0, identity=''):
        if type(every) is not int or every < 1:
            raise ValueError('Optimization sampling interval must be positive')
        self.model, self.directory, self.every = model, Path(directory), every
        self.params = dict(model.named_parameters())
        self.catalog = [dict(name=n, module=n.split('.')[0], shape=list(p.shape), size=p.numel())
                        for n, p in self.params.items()]
        self.identity = hashlib.sha256((identity + json.dumps(self.catalog)).encode()).hexdigest()
        self.indices = {n: torch.linspace(0, p.numel()-1, min(256, p.numel()), device=p.device).long()
                        for n, p in self.params.items()}
        self.events, self.points, self.planes = [], [], {}
        self.tick, self.batch, self.predictor_ticks = 0, restored_batch, 0
        self.selected = next((n for n in self.params if n == 'dynamics.cell.gates.weight'), next(iter(self.params)))
        self.probe, self.probe_requested, self.probe_session = None, True, 0
        self.recalibrate, self.seconds, self.last_request = False, 0., None
        self._restore(restored_batch)

    def _restore(self, batch):
        try:
            path = self.directory / 'optimization.json'
            if path.stat().st_size > 8 * 1024**2:
                return
            state = json.loads(path.read_text())
            if state.get('identity') != self.identity or state.get('batch', 0) > batch:
                return  # A restored older checkpoint begins a new telemetry segment.
            with np.load(self.directory / 'optimization_samples.npz', allow_pickle=False) as saved:
                if int(saved['batch']) != state['batch']:
                    return
                planes = {}
                for name in ('encoder', 'decoder', 'dynamics'):
                    if name in saved:
                        planes[name] = ParameterPlane(saved[name], saved.get(name+'_basis'),
                            saved.get(name+'_origin'), state.get('planes', {}).get(name, {}).get('basis_id', 0))
            self.events, self.points = state.get('events', [])[-64:], state.get('points', [])[-96:]
            self.planes = planes
            self.tick = state.get('tick', 0)
            self.probe = state.get('probe')
            self.probe_requested = self.probe is None
            self.selected = state.get('selected', self.selected)
            if self.selected not in self.params:
                self.selected = next(iter(self.params))
        except (OSError, ValueError, KeyError, TypeError):
            pass  # Missing telemetry cannot prevent checkpoint resume.

    def start_batch(self, batch):
        self.batch, self.seconds = batch, 0.
        request = self.directory / 'optimization_request.json'
        try:
            if request.stat().st_size > 2048:
                return
            data = json.loads(request.read_text())
            request.unlink()
            if data.get('identity') != self.identity:
                return
            operation = data.get('operation')
            if operation == 'matrix' and data.get('matrix') in self.params:
                self.selected = data['matrix']
            elif operation == 'probe' and type(data.get('session')) is int and data['session'] >= 0:
                self.probe_session, self.probe_requested = data['session'], True
            elif operation == 'recalibrate':
                self.recalibrate = True
            self.last_request = data.get('id')
        except (OSError, ValueError, TypeError):
            pass

    @torch.no_grad()
    def before_step(self, phase, optimizer, **metadata):
        start = time.perf_counter()
        self.tick += 1
        if phase == 'predictor':
            self.predictor_ticks += 1
            if (self.predictor_ticks - 1) % self.every and not metadata.get('last'):
                return None
        rates = {id(p): float(g['lr']) for g in optimizer.param_groups for p in g['params']}
        active, scalars = {}, []
        for name, p in self.params.items():
            if id(p) not in rates:
                continue
            if p.grad is None:
                active[name] = None
                continue
            index = self.indices[name]
            w = p.detach().flatten()[index].float()
            g = p.grad.detach().flatten()[index].float()
            active[name] = (w, g)
            scalars.append(p.grad.detach().float().norm() / math.sqrt(p.numel()))
        detail = None
        if self.selected in active and active[self.selected] is not None:
            p = self.params[self.selected]
            detail = dict(name=self.selected, before=p.detach().clone(), gradient=block_rms(p.grad))
        token = dict(id=self.tick, batch=self.batch, phase=phase, active=active, rates=rates,
                     gradient_rms=torch.stack(scalars) if scalars else None, detail=detail, **metadata)
        self.seconds += time.perf_counter() - start
        return token

    @torch.no_grad()
    def after_step(self, token, norm, loss):
        if token is None:
            return
        start = time.perf_counter()
        rows, stats, names = {}, [], []
        for name, sample in token['active'].items():
            if sample is None:
                rows[name] = dict(status='no_gradient')
                continue
            p = self.params[name]
            before, gradient = sample
            delta = p.detach().flatten()[self.indices[name]].float() - before
            weight = before.square().mean().sqrt()
            update = delta.square().mean().sqrt()
            denominator = gradient.norm() * delta.norm()
            alignment = torch.where(denominator > 0, (-gradient @ delta) / denominator, float('nan'))
            stats.append(torch.stack((weight, update, update / weight.clamp_min(1e-12), alignment)))
            names.append(name)
        values = torch.stack(stats).cpu().tolist() if stats else []
        rms = token['gradient_rms'].cpu().tolist() if token['gradient_rms'] is not None else []
        for name, values_row, gradient in zip(names, values, rms):
            weight, update, relative, alignment = map(finite, values_row)
            rows[name] = dict(status='measured', gradient_rms=finite(gradient), weight_rms=weight,
                update_rms=update, relative_update=relative, alignment=alignment,
                lr=token['rates'][id(self.params[name])])
        detail = token['detail']
        if detail:
            delta = self.params[detail['name']].detach() - detail['before']
            sample = delta.flatten()[self.indices[detail['name']]].float().cpu().numpy()
            counts, edges = np.histogram(sample, bins=16)
            detail = dict(name=detail['name'], gradient=detail['gradient'].cpu().tolist(),
                          update=block_rms(delta).cpu().tolist(),
                          histogram=dict(counts=counts.tolist(), edges=edges.tolist()))
        self.events.append(dict(id=token['id'], batch=token['batch'], phase=token['phase'],
            start=token.get('start'), stop=token.get('stop'), context_start=token.get('context_start'),
            loss=finite(loss), gradient_norm=finite(norm), clip_scale=min(1., 1./(float(norm)+1e-6)),
            matrices=rows, detail=detail))
        self.events = self.events[-64:]
        self.seconds += time.perf_counter() - start

    @torch.no_grad()
    def finish_batch(self, reference_loss, actions):
        start = time.perf_counter()
        for module in ('encoder', 'decoder', 'dynamics'):
            values = [p.detach().flatten()[self.indices[n]].float()
                      for n, p in self.params.items() if n.split('.')[0] == module]
            if values:
                vector = torch.cat(values).cpu().numpy()
                self.planes.setdefault(module, ParameterPlane()).append(vector)
                if self.recalibrate:
                    self.planes[module].calibrate()
        self.recalibrate = False
        self.points.append(dict(batch=self.batch, actions=actions, event_id=self.tick,
                                reference_loss=finite(reference_loss)))
        self.points = self.points[-96:]
        self.seconds += time.perf_counter() - start
        self.publish()

    def publish(self):
        arrays = {'batch': np.array(self.batch)}
        planes = {}
        for name, plane in self.planes.items():
            arrays[name] = np.stack(plane.samples)
            if plane.basis is not None:
                arrays[name+'_basis'], arrays[name+'_origin'] = plane.basis, plane.origin
            planes[name] = dict(basis_id=plane.basis_id, projections=plane.project())
        temporary = self.directory / 'optimization_samples.tmp'
        with temporary.open('wb') as stream:
            np.savez(stream, **arrays)
        temporary.replace(self.directory / 'optimization_samples.npz')
        atomic_json(self.directory / 'optimization.json', dict(kind='optimization', version=1,
            identity=self.identity, updated_at=time.time(), batch=self.batch, tick=self.tick,
            every=self.every, sample_per_tensor=256, selected=self.selected, catalog=self.catalog,
            events=self.events, points=self.points, planes=planes, probe=self.probe,
            last_request=self.last_request, collection_seconds=self.seconds))
