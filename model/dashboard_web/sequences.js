'use strict';
const $ = id => document.getElementById(id);
const number = (x, digits = 2) => Number.isFinite(x) ? x.toLocaleString('es-ES', {maximumFractionDigits: digits}) : '—';
const svgNS = 'http://www.w3.org/2000/svg';
function element(name, attributes = {}, text = '') {
  const e = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attributes)) e.setAttribute(key, value);
  e.textContent = text; return e;
}
function chart(id, rows, series, ceiling) {
  const svg = $(id), legend = $(`${id}-legend`); svg.replaceChildren(); legend.replaceChildren();
  const values = rows.flatMap(r => series.map(s => s.value(r))).filter(Number.isFinite);
  if (!values.length) { svg.append(element('text', {x: 35, y: 65, fill: '#a9bcb7'}, 'Sin mediciones todavía')); return; }
  const max = ceiling || Math.max(1e-6, ...values) * 1.1;
  for (let i = 0; i <= 4; i++) {
    const y = 220 - i * 47;
    svg.append(element('line', {x1: 50, x2: 680, y1: y, y2: y, stroke: '#30443e'}));
    svg.append(element('text', {x: 42, y: y + 4, fill: '#a9bcb7', 'text-anchor': 'end', 'font-size': 11}, number(max*i/4, 1)));
  }
  series.forEach(s => {
    const label = document.createElement('span'); label.textContent = `${s.label}: ${number(s.value(rows.at(-1)))}`; label.style.color = s.color; legend.append(label);
    let points = [];
    const flush = () => { if (points.length) svg.append(element('polyline', {points: points.join(' '), fill: 'none', stroke: s.color, 'stroke-width': 2})); points = []; };
    rows.forEach((r, i) => {
      const value = s.value(r); if (!Number.isFinite(value)) { flush(); return; }
      const x = 50 + i * 630 / Math.max(1, rows.length - 1), y = 220 - 188 * value/max;
      points.push(`${x},${y}`); svg.append(element('circle', {cx: x, cy: y, r: 2, fill: s.color}));
    }); flush();
  });
  svg.append(element('text', {x: 50, y: 247, fill: '#a9bcb7', 'font-size': 11}, `Batch ${rows[0].batch}`));
  svg.append(element('text', {x: 680, y: 247, fill: '#a9bcb7', 'text-anchor': 'end', 'font-size': 11}, `Batch ${rows.at(-1).batch}`));
}
function render(data) {
  const c = data.config || {}, rows = data.history || [], last = rows.at(-1) || {};
  $('status').textContent = ({running: 'Entrenando', paused: 'Pausado', budget_complete: 'Presupuesto terminado', error: 'Error', waiting: 'Esperando'})[data.status] || data.status;
  $('run-info').textContent = data.message || `${(data.games || []).join(' · ')} · Actualizado hace ${number(data.age_seconds, 0)} s`;
  $('configuration').textContent = `${c.batch_games || '—'} sesiones independientes · ${c.session_steps || '—'} acciones por sesión · Bloque: ${c.block_steps || '—'} pasos + ${c.context_steps ?? 0} de contexto con gradiente · Peso inicial: ${number(c.loss_start_weight)} → 1 en ${c.loss_warmup_steps || '—'} acciones · ${c.game_schedule === 'blocked' ? 'Un juego por batch' : 'Juegos alternados'}`;
  $('actions').textContent = number(data.progress?.actions, 0); $('batches').textContent = `${number(data.progress?.batches, 0)} batches completados`;
  $('tps').textContent = `${number(last.recent_tps)} acciones/s`; $('examples').textContent = `${number(last.examples_per_second)} transiciones/s`; $('memory').textContent = `${number(last.peak_gpu_gib)} GiB`;
  const pct = value => Number.isFinite(value) ? value * 100 : null;
  chart('quality', rows, [{label: 'Inicio %', color: '#e9b86c', value: r => pct(r.quality?.early?.delta_error)}, {label: 'Después %', color: '#8fd8bf', value: r => pct(r.quality?.late?.delta_error)}], 100);
  chart('speed', rows, [{label: 'Acciones/s', color: '#8fd8bf', value: r => r.recent_tps}]);
  chart('loss', rows, [{label: 'Loss', color: '#bcaafa', value: r => r.loss}]);
  $('timings').replaceChildren();
  for (const [key, label] of Object.entries({collection_seconds: 'Juego', representation_seconds: 'Representación', encode_seconds: 'Codificación', predictor_seconds: 'Predictor y memoria', evaluation_seconds: 'Evaluación', save_seconds: 'Guardado'})) {
    const span = document.createElement('span'); span.textContent = `${label}: ${number(last[key])} s`; $('timings').append(span);
  }
  $('sessions').replaceChildren();
  (last.sessions || []).forEach((s, i) => { const row = document.createElement('tr');
    [i+1, s.game, s.steps, s.boundaries, s.end].forEach(value => { const cell = document.createElement('td'); cell.textContent = String(value); row.append(cell); }); $('sessions').append(row); });
  $('macro-error').textContent = `Media por juego: ${number(pct(last.macro_delta_error))} %`;
  $('game-quality').replaceChildren();
  (data.games || []).forEach(game => { const q = last.quality?.[game], row = document.createElement('tr');
    [game, `${number(pct(q?.delta_error))} %`, number(q?.transitions, 0)].forEach(value => { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); }); $('game-quality').append(row); });
}
async function refresh() {
  try { const response = await fetch('api/state', {cache: 'no-store'}); if (!response.ok) throw Error(response.status);
    const state = await response.json(); render(state.sequences || {status: 'waiting'});
  } catch { $('status').textContent = 'Sin conexión'; }
  setTimeout(refresh, 2000);
}
refresh();
