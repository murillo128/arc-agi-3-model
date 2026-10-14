'use strict';

const $ = id => document.getElementById(id);
const finite = value => typeof value === 'number' && Number.isFinite(value);
const fmt = (value, digits = 0) => finite(value) ? value.toLocaleString('es-ES', {maximumFractionDigits: digits}) : '—';
const pct = value => finite(value) ? `${fmt(value * 100, 2)} %` : '—';
const percentValue = value => finite(value) ? value * 100 : null;
const scalar = value => !finite(value) ? '—' : value !== 0 && Math.abs(value) < .001 ? value.toExponential(2) : fmt(value, 4);
const levelName = value => value === null ? 'Nativo' : fmt(value);
const colors = ['#77dabb', '#f4b979', '#a9a6ef', '#94a7aa', '#ed94bb', '#72c8ec'];
const phases = {waiting: 'Esperando métricas', initializing: 'Inicializando', ready: 'Modelo preparado',
  progress: 'Entrenando · según el log', resumed: 'Entrenamiento reanudado', validation: 'Validación recibida',
  validation_finished: 'Validación terminada', stopped: 'Entrenamiento detenido'};
const events = {initializing: 'Inicio', ready: 'Modelo preparado', resumed: 'Reanudación', validation: 'Validación',
  validation_finished: 'Validación terminada', checkpoint: 'Checkpoint guardado', level_converged: 'Nivel superado',
  encoder_collapsed: 'Colapso detectado', stopped: 'Detenido', representation_gate: 'Evaluación del gate'};
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
  set(series, threshold, suffix = '', axis = 'acción') {
    this.axis = axis;
    this.series = series;
    this.threshold = threshold;
    this.suffix = suffix;
    this.legend.replaceChildren(...series.map((s, i) => {
      const span = document.createElement('span'); span.className = `series-${i}`;
      span.style.color = colors[i % colors.length];
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
      this.bounds = null;
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
      ctx.strokeStyle = colors[i % colors.length]; ctx.fillStyle = colors[i % colors.length]; ctx.lineWidth = 1.8; ctx.beginPath();
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
      return p ? `${s.name}: ${scalar(p.y)}${this.suffix} · ${this.axis} ${fmt(p.x)}` : `${s.name}: sin datos`;
    });
    this.tip.textContent = lines.join('\n'); this.tip.hidden = false;
  }
}
const charts = Object.fromEntries(['delta', 'changes', 'loss', 'speed', 'spread', 'latent', 'total'].map(id => [id, new Chart(id)]));

function render(data) {
  renderBehavior(data.behavior);
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
    config.auto_representation === true ? 'Gate automático' : null,
    config.freeze_representation === true ? 'Solo predictor · encoder y decoder congelados' :
      config.freeze_representation === false ? 'Aprendizaje conjunto' : null].filter(Boolean).join(' · ');
  const gate = run.representation_gate;
  const gateReasons = {insufficient_evidence: 'Evidencia insuficiente', hold: 'Mantiene el modo',
    reconstruction_high: 'Reactivados encoder y decoder', reconstruction_low: 'Congelados encoder y decoder'};
  $('representation-gate').textContent = config.auto_representation === true
    ? gate ? `${gateReasons[gate.reason] || 'Esperando validación'} · reconstrucción ${pct(gate.reconstruction_error)} · activar > ${pct(gate.unfreeze_error)}, congelar ≤ ${pct(gate.freeze_error)} · racha ${fmt(Math.max(gate.open_streak || 0, gate.close_streak || 0))}/${fmt(gate.patience)}`
      : `Esperando validación · activar > ${pct(config.representation_unfreeze_error)}, congelar ≤ ${pct(config.representation_freeze_error)}`
    : 'Manual';
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
  renderStatus(data);
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

function renderStatus(data) {
  $('world-status').textContent = !data.source.available ? 'Log del mundo no disponible.' : data.phase === 'stopped'
    ? `Entrenamiento del mundo detenido: ${reasons[data.reason] || data.reason || 'motivo no indicado'}. Las métricas de esta sección corresponden a esa ejecución.`
    : `Mundo: ${phases[data.phase] || 'Recibiendo métricas'}.`;
  const behavior = data.behavior?.source.available ? data.behavior : null;
  const current = behavior || data, source = current.source;
  const finished = behavior ? current.phase === 'behavior_finished' : current.phase === 'stopped';
  const stale = source.age_seconds > 60 && !finished;
  $('connection').textContent = !source.available ? 'Log no disponible' : source.catching_up ? 'Leyendo métricas…' :
    stale ? 'Sin datos recientes' : ((behavior ? behaviorPhases : phases)[current.phase] || 'Recibiendo datos');
  $('connection').className = !source.available || stale || (!behavior && finished) ? 'warn' : '';
  $('freshness').textContent = source.available ? `Última escritura del log hace ${fmt(source.age_seconds)} s · consulta cada 2 s` : 'Reintentando automáticamente cada 2 s';
  const messages = [];
  if (!source.available) messages.push('No se puede leer el log. Los valores visibles son los últimos recibidos.');
  if (!behavior && finished) messages.push(`Entrenamiento detenido: ${reasons[data.reason] || data.reason || 'motivo no indicado'}.`);
  else if (stale) messages.push('El log no recibe datos recientes. Puede estar validando, guardando pesos o detenido; el panel no confirma que el proceso siga activo.');
  if (source.tail_limited) messages.push('Se muestra el tramo reciente del log; los datos anteriores quedan fuera de la ventana de lectura.');
  $('notice').hidden = messages.length === 0; $('notice').textContent = messages.join(' ');
  if (behavior) {
    const levels = behavior.run.levels || (finite(behavior.run.level) ? [behavior.run.level] : []);
    $('run-info').textContent = ['Goal y planner', behavior.run.game, levels.length ? `Niveles ${levels.join(', ')}` : null,
      behavior.run.device?.toUpperCase(), behavior.run.compile ? 'torch.compile' : null].filter(Boolean).join(' · ');
    $('statistics-origin').textContent = 'Recogida y ajuste por rondas · las evaluaciones se cuentan por separado';
  }
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
const behaviorCharts = {loss: new Chart('behavior-loss-chart'), wins: new Chart('behavior-win-chart')};
const splitNames = {final: 'Final', development: 'Desarrollo', evaluation_only: 'Aislada', unspecified: 'No registrada'};
const behaviorPhases = {behavior_started: 'Preparando', behavior_collection: 'Recogiendo experiencia real',
  behavior_fit: 'Entrenando goal / valor', behavior_evaluation_progress: 'Evaluando',
  behavior_evaluation: 'Evaluación recibida', behavior_world_verified: 'Pesos del mundo verificados',
  behavior_finished: 'Experimento terminado', waiting: 'Esperando datos'};
function policyName(row) {
  const name = row.policy === 'random' ? 'Azar' : row.policy === 'q' ? 'Q' : `MPC · ${fmt(row.horizon)} pasos`;
  return row.role === 'initial' ? `Inicial · ${name}` : name;
}
let behaviorFilterManual = false;
function renderBehavior(b) {
  const levels = b?.run.levels || (finite(b?.run.level) ? [b.run.level] : []);
  const levelSelect = $('behavior-level');
  const options = ['all', ...levels.map(String)];
  if ([...levelSelect.options].map(o => o.value).join(',') !== options.join(',')) {
    const previous = levelSelect.value;
    levelSelect.replaceChildren(...options.map(v => new Option(v === 'all' ? 'Conjunto' : `Nivel ${v}`, v)));
    levelSelect.value = options.includes(previous) ? previous : 'all';
  }
  const levelRows = (b?.evaluations || []).filter(r =>
    levelSelect.value === 'all' ? levels.length < 2 || !finite(r.level) : r.level === Number(levelSelect.value));
  const evaluation = b?.progress.evaluation;
  if (!behaviorFilterManual) {
    const latest = levelRows.findLast(r => r.policy !== 'random' && r.role !== 'initial');
    const availableSplit = latest?.split
      || evaluation?.split || levelRows.at(-1)?.split
      || ({training: 'development', evaluation_only: 'evaluation_only'})[b?.run.mode];
    if (availableSplit) $('behavior-split').value = availableSplit;
  }
  const split = $('behavior-split').value;
  const rows = levelRows.filter(r => r.split === split);
  const learned = rows.findLast(r => r.policy !== 'random' && r.role !== 'initial'), baseline = rows.findLast(r => r.policy === 'random');
  const collection = b?.progress.collection, fit = b?.progress.fit;
  const complete = b?.phase === 'behavior_finished';
  const stale = b?.source.age_seconds > 60;
  $('behavior-status').textContent = !b ? 'No se ha configurado un log de comportamiento.' : !b.source.available
    ? 'Log de comportamiento no disponible; se reintentará.' : complete ? 'Experimento terminado'
    : stale ? 'Sin datos recientes; el log no confirma que siga activo.' : behaviorPhases[b.phase] || 'Recibiendo métricas';
  $('behavior-info').textContent = b ? [b.run.game, levels.length ? `Niveles ${levels.join(', ')}` : null,
    b.run.device?.toUpperCase(), b.run.world_trainable === 0 ? 'Encoder, predictor y decoder congelados' : null,
    b.run.world_unchanged === true ? 'Pesos del mundo verificados sin cambios' : null,
    b.run.world_unchanged === false ? 'FALLO: cambiaron los pesos del mundo' : null,
    b.source.available ? `Log actualizado hace ${fmt(b.source.age_seconds)} s` : null,
    b.phase === 'behavior_evaluation_progress' ? `${splitNames[evaluation?.split] || 'Fase no registrada'} · ${policyName(evaluation || {})} · ${fmt(evaluation?.wins)}/${fmt(evaluation?.episodes)} victorias` : null,
    b.source.tail_limited ? 'Solo tramo reciente del log' : null].filter(Boolean).join(' · ') : '';
  $('behavior-actions').textContent = fmt(collection?.actions);
  const perLevel = Object.entries(collection?.per_level || {}).map(([lv, c]) => `N${lv}: ${fmt(c.episodes)} partidas`).join(' · ');
  $('behavior-episodes').textContent = `Ronda ${fmt(collection?.round)} · ${fmt(collection?.episodes)} partidas · ${fmt(collection?.wins)} victorias · ${fmt(collection?.records)} transiciones en memoria${perLevel ? ' · ' + perLevel : ''}`;
  $('behavior-loss').textContent = scalar(fit?.loss);
  $('behavior-epoch').textContent = `Ronda ${fmt(fit?.round)} · época ${fmt(fit?.epoch)}/${fmt(fit?.epochs)} · ${fmt(fit?.updates)} actualizaciones de la ronda`;
  const evaluating = !learned && b?.source.available && b.phase === 'behavior_evaluation_progress' && evaluation?.split === split;
  const awaiting = !learned && b?.source.available && ['behavior_started', 'behavior_collection', 'behavior_fit',
    'behavior_world_verified', 'behavior_evaluation'].includes(b.phase);
  $('behavior-win').textContent = evaluating ? 'En curso' : awaiting ? 'Pendiente' : pct(learned?.win_rate);
  $('behavior-wins').textContent = learned ? `${policyName(learned)} · ${fmt(learned.wins)}/${fmt(learned.episodes)} · ${splitNames[split]}`
    : evaluating ? `Esperando el resultado completo de ${splitNames[split]}; progreso de la evaluación indicado arriba`
    : awaiting ? `Evaluación aprendida pendiente de ${splitNames[split]}; aún no se ha registrado un resultado completo para este nivel y fase`
    : 'Sin evaluación aprendida en esta fase';
  $('behavior-random').textContent = pct(baseline?.win_rate);
  $('behavior-random-wins').textContent = baseline ? `${fmt(baseline.wins)}/${fmt(baseline.episodes)} · ${splitNames[split]}` : 'Sin evaluación al azar en esta fase';
  $('behavior-ms').textContent = fmt(learned?.decision_ms, 2);
  $('behavior-tps').textContent = fmt(learned?.actions_per_second, 2);
  const fits = b?.fits || [];
  behaviorCharts.loss.set([['Total', 'loss'], ['Eventos', 'event_ce'], ['Valor V', 'value_bce'], ['Acción Q', 'q_bce']].map(([name, key]) =>
    ({name, points: fits.map((r, i) => ({x: i + 1, y: r[key]}))})), undefined, '', 'época registrada');
  behaviorCharts.wins.set([...new Set(rows.map(policyName))].map(name =>
    ({name, points: rows.flatMap((r, i) => policyName(r) === name ? [{x: i + 1, y: percentValue(r.win_rate)}] : [])})), undefined, ' %', 'evaluación');
  const body = $('behavior-table'); body.replaceChildren();
  for (const r of rows.slice(-100)) {
    const tr = document.createElement('tr');
    for (const value of [`${splitNames[r.split] || 'No registrada'} / ${fmt(r.round)}`, finite(r.level) ? fmt(r.level) : 'Conjunto', policyName(r),
      `${fmt(r.wins)}/${fmt(r.episodes)}`, `${pct(r.win_rate)} · ${r.wilson95?.map(pct).join('–') || '—'}`,
      fmt(r.actions_per_success, 1), fmt(r.decision_ms, 2), fmt(r.actions_per_second, 2)]) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    body.append(tr);
  }
  if (!rows.length) { const tr = document.createElement('tr'), td = document.createElement('td');
    td.colSpan = 8; td.textContent = 'Sin evaluaciones para este nivel y fase. Los logs antiguos se muestran en «fase no registrada».'; tr.append(td); body.append(tr); }
}
$('behavior-split').addEventListener('change', () => { behaviorFilterManual = true; renderBehavior(state?.behavior); });
$('behavior-level').addEventListener('change', () => renderBehavior(state?.behavior));

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
