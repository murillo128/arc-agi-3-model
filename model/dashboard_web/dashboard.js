'use strict';

const $ = id => document.getElementById(id);
const finite = value => typeof value === 'number' && Number.isFinite(value);
const fmt = (value, digits = 0) => finite(value) ? value.toLocaleString('es-ES', {maximumFractionDigits: digits}) : '—';
const pct = value => finite(value) ? `${fmt(value * 100, 2)} %` : '—';
const percentValue = value => finite(value) ? value * 100 : null;
const scalar = value => !finite(value) ? '—' : value !== 0 && Math.abs(value) < .001 ? value.toExponential(2) : fmt(value, 4);
const levelName = value => value === null ? 'Nativo' : fmt(value);
const colors = ['#77dabb', '#f4b979', '#a9a6ef', '#94a7aa'];
const phases = {waiting: 'Esperando métricas', initializing: 'Inicializando', ready: 'Modelo preparado',
  progress: 'Entrenando · según el log', resumed: 'Entrenamiento reanudado', validation: 'Validación recibida',
  validation_finished: 'Validación terminada', stopped: 'Entrenamiento detenido'};
const events = {initializing: 'Inicio', ready: 'Modelo preparado', resumed: 'Reanudación', validation: 'Validación',
  validation_finished: 'Validación terminada', checkpoint: 'Checkpoint guardado', level_converged: 'Nivel superado',
  encoder_collapsed: 'Colapso detectado', stopped: 'Detenido'};
const reasons = {encoder_collapsed: 'colapso sostenido del encoder', level_converged: 'nivel convergido',
  curriculum_converged: 'currículo completado', interrupted: 'interrupción', action_limit: 'límite de acciones',
  level_limit_without_convergence: 'límite de acciones del nivel'};
let state = null;
let chartRevision = '';

class Chart {
  constructor(id) {
    this.canvas = $(id);
    this.legend = $(`${id}-legend`);
    this.tip = this.canvas.nextElementSibling;
    this.series = [];
    new ResizeObserver(() => this.draw()).observe(this.canvas);
    this.canvas.addEventListener('pointermove', event => this.hover(event));
    this.canvas.addEventListener('pointerleave', () => { this.tip.hidden = true; });
  }
  set(series, threshold, suffix = '') {
    this.series = series;
    this.threshold = threshold;
    this.suffix = suffix;
    this.legend.replaceChildren(...series.map((s, i) => {
      const span = document.createElement('span'); span.className = `series-${i}`;
      span.textContent = s.name; return span;
    }));
    if (finite(threshold)) {
      const span = document.createElement('span'); span.className = 'series-3';
      span.textContent = `Umbral: ${scalar(threshold)}${suffix}`; this.legend.append(span);
    }
    this.tip.hidden = true;
    this.draw();
  }
  draw() {
    const width = this.canvas.clientWidth, height = this.canvas.clientHeight;
    if (!width || !height) return;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(width * dpr); this.canvas.height = Math.round(height * dpr);
    const ctx = this.canvas.getContext('2d'); ctx.scale(dpr, dpr);
    const points = this.series.flatMap(s => s.points).filter(p => finite(p.x) && finite(p.y));
    if (!points.length) {
      ctx.fillStyle = '#8da39a'; ctx.font = '12px system-ui'; ctx.textAlign = 'center';
      ctx.fillText('Esperando suficientes datos de esta serie…', width / 2, height / 2); return;
    }
    const left = 62, top = 16, right = width - 16, bottom = height - 32;
    const minX = Math.min(...points.map(p => p.x));
    const maxX = Math.max(minX + 1, ...points.map(p => p.x));
    const maxY = Math.max(.000001, finite(this.threshold) ? this.threshold : 0, ...points.map(p => p.y)) * 1.12;
    const x = value => left + (value - minX) / (maxX - minX) * (right - left);
    const y = value => bottom - value / maxY * (bottom - top);
    this.bounds = {left, right, minX, maxX};
    ctx.font = '10px system-ui';
    for (let i = 0; i <= 4; i++) {
      const value = maxY * i / 4, py = y(value);
      ctx.strokeStyle = '#2a3835'; ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py); ctx.stroke();
      ctx.fillStyle = '#8da39a'; ctx.textAlign = 'right'; ctx.fillText(scalar(value) + this.suffix, left - 9, py + 3);
      const vx = minX + (maxX - minX) * i / 4;
      ctx.textAlign = i === 0 ? 'left' : i === 4 ? 'right' : 'center';
      ctx.fillText(fmt(vx), x(vx), height - 10);
    }
    if (finite(this.threshold)) {
      ctx.strokeStyle = '#788e86'; ctx.setLineDash([4, 4]); ctx.beginPath();
      ctx.moveTo(left, y(this.threshold)); ctx.lineTo(right, y(this.threshold)); ctx.stroke(); ctx.setLineDash([]);
    }
    this.series.forEach((s, i) => {
      ctx.strokeStyle = colors[i]; ctx.fillStyle = colors[i]; ctx.lineWidth = 1.8; ctx.beginPath();
      let connected = false;
      for (const p of s.points) {
        if (!finite(p.x) || !finite(p.y)) { connected = false; continue; }
        if (!connected) ctx.moveTo(x(p.x), y(p.y)); else ctx.lineTo(x(p.x), y(p.y));
        connected = true;
      }
      ctx.stroke();
      if (s.points.length < 80) for (const p of s.points) {
        if (!finite(p.x) || !finite(p.y)) continue;
        ctx.beginPath(); ctx.arc(x(p.x), y(p.y), 2.5, 0, Math.PI * 2); ctx.fill();
      }
    });
  }
  hover(event) {
    if (!this.bounds || !this.series.some(s => s.points.length)) return;
    const {left, right, minX, maxX} = this.bounds;
    const px = event.clientX - this.canvas.getBoundingClientRect().left;
    const desired = minX + Math.max(0, Math.min(1, (px - left) / (right - left))) * (maxX - minX);
    const lines = this.series.map(s => {
      const valid = s.points.filter(p => finite(p.x) && finite(p.y));
      const p = valid.reduce((best, point) => !best || Math.abs(point.x - desired) < Math.abs(best.x - desired) ? point : best, null);
      return p ? `${s.name}: ${scalar(p.y)}${this.suffix} · acción ${fmt(p.x)}` : `${s.name}: sin datos`;
    });
    this.tip.textContent = lines.join('\n'); this.tip.hidden = false;
  }
}
const charts = Object.fromEntries(['delta', 'changes', 'loss', 'speed', 'spread', 'latent', 'total'].map(id => [id, new Chart(id)]));

function render(data) {
  const p = data.progress, run = data.run, config = run.training || {};
  const clean = run.statistics_origin === 'after_first_action';
  $('statistics-origin').textContent = clean
    ? finite(p.elapsed_seconds)
      ? `Estadísticas desde el final de la primera jugada · ${fmt(p.elapsed_seconds, 1)} s medidos`
      : 'Preparando el punto limpio: esperando a que termine la primera jugada…'
    : 'Log anterior: contadores acumulados y velocidad media desde el arranque';
  $('tps-period').textContent = clean ? 'Media desde el punto limpio' : 'Media desde el arranque del proceso';
  $('axis-origin').textContent = clean ? 'Eje horizontal: acciones desde el punto limpio' : 'Eje horizontal: acciones acumuladas';
  $('speed-period').textContent = clean
    ? 'Media desde el punto limpio; incluye las pausas posteriores de validación y guardado.'
    : 'Media desde el arranque; incluye validación y guardado de checkpoints.';
  const currentLevel = Object.hasOwn(p, 'level') ? p.level : run.levels?.[p.level_index || 0];
  const latest = [...data.validations].reverse().find(v => v.level === currentLevel);
  $('run-info').textContent = [run.game || 'Juego pendiente', run.device?.toUpperCase(),
    finite(run.parameters) ? `${fmt(run.parameters / 1e6, 1)} M parámetros` : null,
    run.compile === true ? 'torch.compile' : run.compile === false ? 'Eager' : null,
    config.freeze_representation === true ? 'Solo predictor · encoder y decoder congelados' :
      config.freeze_representation === false ? 'Aprendizaje conjunto' : null].filter(Boolean).join(' · ');
  $('loss-note').textContent = config.freeze_representation === true
    ? 'Solo se entrena el predictor. La reconstrucción se sigue midiendo en validación; no tiene pérdida de entrenamiento activa.'
    : 'Supervisión del predictor y las dos reconstrucciones. Son pérdidas de aprendizaje, no porcentajes de error.';
  $('level').textContent = currentLevel === undefined ? '—' : levelName(currentLevel);
  $('level-actions').textContent = `${fmt(p.level_actions)} acciones en este nivel`;
  $('actions').textContent = fmt(p.actions); $('episodes').textContent = `${fmt(p.episodes)} ${clean ? 'partidas nuevas desde el punto limpio' : 'partidas iniciadas'}`;
  $('tps').textContent = fmt(p.actions_per_second, 2); $('updates').textContent = fmt(p.updates);
  if (finite(p.trained_transitions_per_second)) $('tps-period').textContent += ` · ${fmt(p.trained_transitions_per_second, 1)} transiciones entrenadas/s`;
  $('replay').textContent = `${fmt(p.replay_episodes)} partidas en memoria · ${fmt(p.device_cache_mb, 1)} MiB de entradas`;
  $('prediction').textContent = pct(latest?.delta_error); $('reconstruction').textContent = pct(latest?.decoder_delta_error);
  $('prediction-note').textContent = latest ? `Validado en acción ${fmt(latest.actions)}` : 'Todavía sin validación de este nivel';
  $('gpu').textContent = finite(p.gpu_peak_mb) ? `${fmt(p.gpu_peak_mb / 1024, 2)} GiB` : '—';
  $('learning-rate').textContent = scalar(config.learning_rate);
  $('false-change-weight').textContent = finite(config.false_change_weight) ? `×${fmt(config.false_change_weight, 2)}` : '—';
  $('passes').textContent = `${fmt(p.passes)} / ${fmt(config.consecutive_passes)}`;
  const checkpoint = [...data.events].reverse().find(e => e.event === 'checkpoint' || e.event === 'stopped');
  $('checkpoint').textContent = checkpoint ? `Acción ${fmt(checkpoint.actions)}` : 'Sin evento reciente';
  const event = data.events.at(-1);
  $('event').textContent = event ? (events[event.event] || event.event) : '—';
  const stale = data.source.age_seconds > 60;
  $('connection').textContent = !data.source.available ? 'Log no disponible' : data.source.catching_up ? 'Leyendo métricas…' :
    data.phase === 'stopped' ? phases.stopped : stale ? 'Sin datos recientes' : (phases[data.phase] || 'Recibiendo datos');
  $('connection').className = !data.source.available || stale || data.phase === 'stopped' ? 'warn' : '';
  $('freshness').textContent = data.source.available ? `Última escritura del log hace ${fmt(data.source.age_seconds)} s · consulta cada 2 s` : 'Reintentando automáticamente cada 2 s';
  const messages = [];
  if (!data.source.available) messages.push('No se puede leer el log. Los valores visibles son los últimos recibidos.');
  if (data.phase === 'stopped') messages.push(`Entrenamiento detenido: ${reasons[data.reason] || data.reason || 'motivo no indicado'}.`);
  else if (stale) messages.push('El log no recibe datos recientes. Puede estar validando, guardando pesos o detenido; el panel no confirma que el proceso siga activo.');
  if (data.source.tail_limited) messages.push('Se muestra el tramo reciente del log; los datos anteriores quedan fuera de la ventana de lectura.');
  $('notice').hidden = messages.length === 0; $('notice').textContent = messages.join(' ');
  $('gate-info').textContent = `Validación cada ${fmt(config.validate_every)} acciones. Umbrales: errores visuales ≤ ${pct(config.delta_error)}, MSE latente ≤ ${scalar(config.latent_delta_mse)}, diversidad ≥ ${scalar(config.min_latent_std)}; también se exige cobertura suficiente.`;
  $('retention').textContent = `Lee el log existente · hasta ${fmt(data.max_points)} puntos por tipo en memoria · no guarda un historial adicional.`;

  const levels = [...new Set([...(run.levels || []), ...data.validations.map(v => v.level)])];
  const levelSelect = $('validation-level'), selected = levelSelect.value;
  const values = ['current', ...levels.map(String)];
  if (JSON.stringify([...levelSelect.options].map(o => o.value)) !== JSON.stringify(values)) {
    levelSelect.replaceChildren(new Option('Actual', 'current'), ...levels.map(level => new Option(levelName(level), String(level))));
    if (values.includes(selected)) levelSelect.value = selected;
  }
  const lastByLevel = new Map(); for (const v of data.validations) lastByLevel.set(v.level, v);
  const rows = [...lastByLevel.values()].map(v => {
    const tr = document.createElement('tr');
    for (const value of [levelName(v.level), fmt(v.actions), pct(v.delta_error), pct(v.decoder_delta_error), fmt(v.transitions), v.passes ? 'Sí' : 'No']) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    return tr;
  });
  if (rows.length) $('validation-table').replaceChildren(...rows);
  else { const tr = document.createElement('tr'), td = document.createElement('td'); td.colSpan = 6; td.textContent = 'Todavía no hay validaciones.'; tr.append(td); $('validation-table').replaceChildren(tr); }
  const revision = `${data.generation}:${data.revision}:${levelSelect.value}:${$('window').value}`;
  if (revision !== chartRevision) { chartRevision = revision; drawCharts(currentLevel); }
}

function drawCharts(currentLevel) {
  if (!state) return;
  const p = state.progress, config = state.run.training || {};
  if (currentLevel === undefined) currentLevel = Object.hasOwn(p, 'level') ? p.level : state.run.levels?.[p.level_index || 0];
  const selected = $('validation-level').value;
  const level = selected === 'current' ? currentLevel : selected === 'null' ? null : Number(selected);
  const windowSize = $('window').value === 'all' ? Infinity : Number($('window').value);
  const cutoff = (p.actions || 0) - windowSize;
  const progress = state.progress_points.filter(row => row.actions >= cutoff);
  const validation = state.validations.filter(row => row.level === level && finite(row.actions) && row.actions >= cutoff);
  const series = (name, rows, value) => ({name, points: rows.map(row => ({x: row.actions, y: value(row)}))});
  charts.delta.set([series('Predictor', validation, r => percentValue(r.delta_error)), series('Decoder · latente real', validation, r => percentValue(r.decoder_delta_error))], percentValue(config.delta_error), ' %');
  const changeSeries = [['Inventados', 'extra_changes'], ['Omitidos', 'missed_changes'],
    ['Color incorrecto', 'wrong_replacement_colors'], ['Acertados', 'correct_changes']];
  charts.changes.set(changeSeries.map(([name, key]) => series(name, validation, r => r[key])), undefined, ' px');
  const detailed = validation.at(-1);
  $('changes-summary').textContent = detailed && changeSeries.every(([, key]) => finite(detailed[key]))
    ? `Última validación · ${fmt(detailed.validation_actions)} acciones: ${changeSeries.map(([name, key]) => `${name.toLowerCase()} ${fmt(detailed[key])}`).join(' · ')}.`
    : 'Esperando una validación con desglose de errores; los registros anteriores no contienen estos datos.';
  charts.loss.set([series('Predictor', progress, r => r.loss?.decoder_ce), series('Reconstrucción', progress, r => r.loss?.reconstruction_ce), series('Reconstrucción local', progress, r => r.loss?.local_reconstruction_ce)]);
  charts.speed.set([series(state.run.statistics_origin === 'after_first_action' ? 'Media desde el punto limpio' : 'Media del proceso', progress, r => r.actions_per_second)]);
  charts.spread.set([series('Diversidad observada', validation, r => r.latent_std)], config.min_latent_std);
  charts.latent.set([series('MSE del delta', validation, r => r.latent_delta_mse)], config.latent_delta_mse);
  charts.total.set([series('Pérdida total', progress, r => r.loss?.loss)]);
}
for (const id of ['window', 'validation-level']) $(id).addEventListener('change', () => { chartRevision = ''; if (state) render(state); });

async function poll() {
  try {
    const response = await fetch('api/state', {cache: 'no-store', signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const next = await response.json();
    if (!next.source || !next.progress || !Array.isArray(next.validations)) throw new Error('Respuesta no válida');
    state = next; render(state);
  } catch (error) {
    $('connection').textContent = 'Sin conexión · reintentando'; $('connection').className = 'warn';
    $('freshness').textContent = 'No se han podido actualizar los datos';
    $('notice').hidden = false; $('notice').textContent = 'No se puede conectar con el panel. Las métricas visibles no se están actualizando; se reintentará automáticamente.';
  } finally { setTimeout(poll, 2000); }
}
poll();
