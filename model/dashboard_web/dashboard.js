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
let selectedExecution = 'current';
let selectedTab = 'world';

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
      ctx.fillText('Sin datos registrados en esta ejecución', width / 2, height / 2); return;
    }
    const left = 62, top = 16, right = width - 16, bottom = height - 32;
    const minX = Math.min(...points.map(p => p.x));
    const maxX = Math.max(minX + 1, ...points.map(p => p.x));
    const maxY = Math.max(.000001, finite(this.threshold) ? this.threshold : 0, ...points.map(p => p.y)) * 1.12;
    const minY = Math.min(0, ...points.map(p => p.y)) * 1.12;
    const x = value => left + (value - minX) / (maxX - minX) * (right - left);
    const y = value => bottom - (value - minY) / (maxY - minY) * (bottom - top);
    this.bounds = {left, right, minX, maxX};
    ctx.font = '10px system-ui';
    for (let i = 0; i <= 4; i++) {
      const value = minY + (maxY - minY) * i / 4, py = y(value);
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
const charts = Object.fromEntries(['delta', 'changes', 'loss', 'speed', 'spread', 'latent', 'total',
  'rollouts', 'wins', 'behaviorLoss', 'actor'].map(id => [id, new Chart(id)]));
const splitNames = {dev: 'Desarrollo', development: 'Desarrollo', intermediate: 'Intermedia', final: 'Final',
  validation: 'Validación', evaluation_only: 'Aislada', unspecified: 'Fase no registrada'};
const stages = {initializing: 'Inicializando / compilando', world: 'Aprendiendo dinámica',
  head_calibration: 'Calibrando progreso y valor', world_evaluation: 'Evaluando el mundo',
  behavior_evaluation: 'Evaluando comportamiento', behavior: 'Aprendiendo valor y planificación',
  actor: 'Aprendiendo actor y crítico', imagination_gate: 'Comprobación de imaginación'};
const behaviorPhases = {behavior_started: 'Preparando', behavior_collection: 'Recogiendo experiencia real',
  behavior_fit: 'Aprendiendo progreso y valor', behavior_evaluation_progress: 'Evaluando comportamiento',
  behavior_evaluation: 'Evaluación recibida', behavior_finished: 'Experimento terminado',
  behavior_world_verified: 'Pesos del mundo verificados'};
const campaignKey = row => JSON.stringify([row.variant || 'preparando', row.seed ?? null]);
const scope = row => [row.game, finite(row.level) ? `Nivel ${fmt(row.level)}` : 'Conjunto'].filter(Boolean).join(' / ');
const splitName = row => splitNames[row.split] || row.split || 'Fase no registrada';
function policyName(row) {
  const policy = row.policy === 'random' ? 'Azar' : row.policy === 'q' ? 'Q'
    : finite(row.horizon) ? `MPC · ${fmt(row.horizon)} pasos` : row.policy || 'Agente';
  return row.role === 'initial' ? `Inicial · ${policy}` : policy;
}

// One catalogue and one selected view: no renderer reads another run's state.
// Only records already available in the configured logs are selectable.
function executionCatalogue(data) {
  const views = [];
  const base = (kind, raw, label) => ({kind, raw, label, source: raw.source || {}, phase: raw.phase,
    reason: raw.reason, config: raw.run?.training || {}, progress: raw.progress || {},
    worldPoints: [], validations: [], behaviorPoints: [], evaluations: [], fitAxis: 'acción',
    maxPoints: raw.max_points || 2000});
  const world = base('world', data, `Mundo · ${data.run?.game || 'registro disponible'}`);
  world.id = `world:${data.generation || 0}`;
  world.worldPoints = data.progress_points || [];
  world.validations = (data.validations || []).map(r => ({split: 'validation', game: data.run?.game, ...r}));
  views.push(world);
  if (data.behavior) {
    const raw = data.behavior;
    const behavior = base('behavior', raw, `Comportamiento · ${raw.run?.game || 'registro disponible'}`);
    behavior.id = `behavior:${raw.generation || 0}`;
    behavior.progress = {...raw.progress?.collection, updates: raw.progress?.fit?.updates};
    behavior.behaviorPoints = (raw.fits || []).map((loss, index) => ({actions: index + 1, loss}));
    behavior.fitAxis = 'época registrada';
    behavior.evaluations = (raw.evaluations || []).map((r, index) => ({...r, game: raw.run?.game, evaluationIndex: index + 1}));
    views.push(behavior);
  }
  if (data.campaign) {
    const raw = data.campaign, summaries = new Map();
    for (const row of raw.progress_points || []) summaries.set(campaignKey(row), row);
    for (const row of raw.evaluations || []) {
      const key = campaignKey(row);
      summaries.set(key, {...summaries.get(key), variant: row.variant, seed: row.seed, actions: row.actions});
    }
    for (const row of raw.executions || []) summaries.set(campaignKey(row), row);
    summaries.set(campaignKey(raw.progress || {}), raw.progress || {});
    for (const [key, progress] of summaries) {
      const view = base('campaign', raw, `${progress.variant || 'Preparando'} · semilla ${fmt(progress.seed)}`);
      view.id = `campaign:${raw.generation || 0}:${key}`;
      view.progress = progress;
      view.currentCell = key === campaignKey(raw.progress || {});
      if (!view.currentCell) { view.phase = 'archived'; view.reason = null; }
      if (progress.admitted === false) view.reason = 'Imaginación aplazada: el mundo no superó la comprobación de calidad a dos pasos.';
      const points = (raw.progress_points || []).filter(r => campaignKey(r) === key);
      view.worldPoints = points.filter(r => r.stage === 'world' || Object.hasOwn(r.loss || {}, 'latent_mse'));
      view.behaviorPoints = points.filter(r => ['behavior', 'actor'].includes(r.stage)
        || ['event_ce', 'value_bce', 'q_bce', 'imitation_loss', 'actor_loss', 'critic_loss'].some(k => Object.hasOwn(r.loss || {}, k)));
      for (const r of (raw.evaluations || []).filter(r => campaignKey(r) === key)) {
        if (Object.keys(r.world || {}).length) {
          view.validations.push({...r.world, actions: r.actions, split: r.split, aggregate: true});
          for (const pair of r.world.per_pair || []) view.validations.push({...pair, actions: r.actions, split: r.split});
        }
        if (Object.keys(r.behavior || {}).length) view.evaluations.push({...r.behavior,
          split: r.split, trainingActions: r.actions, policy: 'MPC'});
      }
      views.push(view);
    }
  }
  // Follow the most recently written configured source, even after it stops.
  // Missing logs keep their last-known identity; never substitute another run's metrics.
  const currentCandidates = views.filter(v => v.kind !== 'campaign' || v.currentCell);
  const current = currentCandidates.reduce((best, v) =>
    !best || (v.source.updated_at || 0) >= (best.source.updated_at || 0) ? v : best, null);
  return {views, current};
}

function statusFor(view, archived) {
  if (!view) return 'Registro no disponible';
  if (!view.source.available) return 'Log no disponible';
  if (archived) return 'Registro anterior · consulta';
  if (view.phase === 'campaign_error') return 'Ejecución detenida por error';
  if (view.phase === 'campaign_finished') return 'Campaña terminada';
  if (view.phase === 'campaign_paused') return 'Campaña pausada · diagnósticos en Comparación de variantes';
  if (view.phase === 'behavior_finished') return 'Experimento terminado';
  if (view.phase === 'stopped') return 'Entrenamiento detenido';
  const status = view.kind === 'campaign' ? stages[view.progress.stage] || 'Ejecución en curso según el log'
    : behaviorPhases[view.phase] || phases[view.phase] || 'Esperando datos';
  if (view.kind === 'campaign' && view.progress.stage?.endsWith('_evaluation')) {
    const granular = finite(view.progress.evaluation?.current_episode);
    return `${status} · ${granular && view.source.age_seconds > 60 ? 'sin datos recientes' : granular ? 'progreso por acciones' : 'progreso por bloques de partidas'}`;
  }
  return view.source.age_seconds > 60 ? `${status} · sin datos recientes` : status;
}
function setTable(id, rows, columns) {
  const table = $(id); table.replaceChildren();
  for (const values of rows) {
    const tr = document.createElement('tr');
    for (const value of values) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
    table.append(tr);
  }
  if (!rows.length) {
    const tr = document.createElement('tr'), td = document.createElement('td');
    td.colSpan = columns; td.textContent = 'Sin evaluaciones registradas para esta ejecución.';
    tr.append(td); table.append(tr);
  }
}
function line(name, points, value, x = row => row.actions) {
  return {name, points: points.map(row => ({x: x(row), y: value(row)}))};
}
function groupedLines(rows, fields, key, x = row => row.actions) {
  const groups = [...new Set(rows.map(key))];
  return groups.flatMap(group => fields.map(([name, value]) =>
    line(`${name} · ${group}`, rows.filter(r => key(r) === group), value, x)));
}
function aggregateDiagnostics(row) {
  const result = {...row}, pairs = row.per_pair || [];
  if (!pairs.length) return result;
  for (const key of ['extra_changes', 'missed_changes', 'wrong_replacement_colors', 'correct_changes']) {
    // Prefer an explicit total; never count it again alongside its per-level detail.
    if (!finite(result[key]) && pairs.every(p => finite(p[key])))
      result[key] = pairs.reduce((sum, p) => sum + p[key], 0);
  }
  if (!finite(result.latent_std) && pairs.every(p => finite(p.transitions) && p.transitions >= 0)) {
    const measured = pairs.filter(p => p.transitions > 0);
    const count = measured.reduce((sum, p) => sum + p.transitions, 0);
    if (count && measured.every(p => finite(p.latent_std)))
      result.latent_std = measured.reduce((sum, p) => sum + p.latent_std * p.transitions, 0) / count;
  }
  return result;
}
function evaluationDiagnostics(rows) {
  const groups = new Map();
  for (const row of rows) {
    const key = JSON.stringify([row.actions, row.split]);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return [...groups.values()].map(group => aggregateDiagnostics(group.length === 1 ? group[0]
    : {actions: group[0].actions, split: group[0].split, per_pair: group}));
}
function render(data) {
  const report = data.campaign?.report;
  const probe = report?.learning_probe, diagnostic = report?.diagnostics;
  const activity = [probe, diagnostic].find(a => a?.available && a.status === 'running');
  $('experiment-notice').hidden = !activity;
  $('experiment-summary').textContent = activity
    ? `${activity === probe ? 'Ensayo de aprendizaje' : 'Diagnóstico con pesos congelados'} en curso. Las métricas de abajo pertenecen al entrenamiento seleccionado.` : '';
  const {views, current} = executionCatalogue(data);
  const select = $('execution-select');
  const options = [new Option('En curso · seguimiento automático', 'current'),
    ...views.map(v => new Option(`${v.label}${v.id === current?.id ? ' · última ejecución' : ' · anterior'}`, v.id))];
  if (selectedExecution !== 'current' && !views.some(v => v.id === selectedExecution))
    options.push(new Option('Registro no disponible', selectedExecution));
  if (JSON.stringify([...select.options].map(o => [o.value, o.textContent])) !==
      JSON.stringify(options.map(o => [o.value, o.textContent]))) select.replaceChildren(...options);
  select.value = selectedExecution;
  const view = selectedExecution === 'current' ? current : views.find(v => v.id === selectedExecution);
  const archived = view && view.id !== current?.id;
  const p = view?.progress || {}, raw = view?.raw || {}, config = view?.config || {};
  const worldPoints = view?.worldPoints || [], validations = view?.validations || [];
  const fits = view?.behaviorPoints || [], evaluations = view?.evaluations || [];
  const campaignEvaluation = view?.kind === 'campaign' && p.stage?.endsWith('_evaluation');
  $('run-info').textContent = view?.label || 'Registro no disponible';
  $('connection').textContent = statusFor(view, archived);
  $('connection').className = !view || !view.source.available || view.phase === 'campaign_error'
    || (!archived && view.source.age_seconds > (campaignEvaluation && !finite(p.evaluation?.current_episode) ? 900 : 60) && !['stopped', 'behavior_finished', 'campaign_finished'].includes(view.phase)) ? 'warn' : '';
  $('freshness').textContent = archived ? 'Datos de la ejecución seleccionada'
    : `Última escritura hace ${fmt(view?.source.age_seconds)} s · consulta cada 2 s`;
  $('selection-note').textContent = selectedExecution === 'current'
    ? 'Toda la página sigue la ejecución más reciente. Las evaluaciones tienen contadores separados.'
    : 'Selección fija: todas las pestañas muestran este entrenamiento. No modifica el proceso en curso.';
  const reason = view?.reason;
  $('notice').textContent = reason ? `Ejecución seleccionada: ${reasons[reason] || reason}` : '';
  $('notice').hidden = !reason;
  $('actions').textContent = fmt(p.actions);
  $('updates').textContent = fmt(p.updates);
  $('tps').textContent = fmt(p.actions_per_second, 2);
  $('level').textContent = finite(p.level) ? `Nivel ${fmt(p.level)}` : '—';
  $('game').textContent = p.game || raw.run?.game || 'Sin juego actual registrado';
  const budget = p.total_actions ?? raw.run?.config?.actions;
  $('actions-note').textContent = finite(budget) ? `De ${fmt(budget)} · variante y semilla seleccionadas`
    : view?.kind === 'behavior' ? 'Ronda de recogida registrada · evaluación excluida'
    : raw.run?.statistics_origin === 'after_first_action' ? 'Desde el punto limpio · evaluación excluida' : 'Evaluación excluida';
  if (finite(p.episodes)) $('actions-note').textContent += ` · ${fmt(p.episodes)} partidas`;
  $('updates-note').textContent = finite(p.joint_updates)
    ? `${fmt(p.joint_updates)} conjuntas / ${fmt(p.frozen_updates)} con representación congelada`
    : view?.kind === 'behavior' ? 'Ronda de aprendizaje registrada' : 'Pasos del optimizador registrados';
  $('speed-note').textContent = `acciones/s · ${view?.kind === 'behavior' ? 'recogida' : 'aprendizaje'}`;
  if (campaignEvaluation) {
    $('speed-note').textContent = 'Evaluando con pesos congelados · no hay TPS de aprendizaje';
    $('level').textContent = finite(p.evaluation?.level) ? `Nivel ${fmt(p.evaluation.level)}` : 'Evaluando';
    $('game').textContent = p.evaluation?.game || 'Juego no identificado en el log de evaluación';
  }
  const aggregates = validations.filter(r => r.aggregate);
  const visualRows = aggregates.length ? aggregates : validations;
  const latestWorld = visualRows.at(-1);
  $('prediction').textContent = pct(latestWorld?.delta_error);
  $('reconstruction').textContent = pct(latestWorld?.decoder_delta_error);
  $('prediction-note').textContent = latestWorld
    ? `${splitName(latestWorld)} · ${scope(latestWorld)} · acción ${fmt(latestWorld.actions)}` : 'Sin evaluación del mundo en esta ejecución';
  const lastWorldPoint = worldPoints.findLast(r => finite(r.loss?.decoder_ce))
    || (p.stage === 'world' && finite(p.loss?.decoder_ce) ? p : null);
  const frozen = typeof config.freeze_representation === 'boolean' ? config.freeze_representation
    : raw.run?.world_trainable === 0 || ['behavior', 'actor'].includes(p.stage) ? true
    : lastWorldPoint ? !finite(lastWorldPoint.loss.reconstruction_ce) : null;
  $('representation').textContent = frozen === null ? 'Sin datos' : frozen ? 'Congelada' : 'Aprendiendo';
  $('representation-note').textContent = frozen === null ? 'El log no registra su estado'
    : 'Encoder y decoder · último estado registrado';
  $('world-status').textContent = worldPoints.length
    ? 'Métricas del predictor de esta ejecución. La reconstrucción se mide aparte, incluso con encoder y decoder congelados.'
    : validations.length ? 'Hay evaluaciones del mundo; no hay pasos de entrenamiento del mundo en el tramo disponible.'
    : 'Esta ejecución no tiene métricas del mundo registradas. No se usan datos de otros entrenamientos.';
  const curveGroup = r => `${splitName(r)} · ${scope(r)}`;
  charts.delta.set(groupedLines(visualRows, [['Predictor', r => percentValue(r.delta_error)],
    ['Decoder', r => percentValue(r.decoder_delta_error)]], curveGroup), finite(config.delta_error) ? config.delta_error * 100 : undefined, ' %');
  charts.loss.set([['Predicción', 'decoder_ce'], ['Reconstrucción', 'reconstruction_ce'], ['Reconstrucción local', 'local_reconstruction_ce'], ['Visual a 2 pasos · ponderada', 'rollout_visual']]
    .map(([name, key]) => line(name, worldPoints, r => r.loss?.[key])));
  charts.latent.set([['1 paso', 'latent_mse'], ['2 pasos', 'rollout_mse']]
    .map(([name, key]) => line(name, worldPoints, r => r.loss?.[key])));
  charts.rollouts.set(groupedLines(visualRows, [
    ...[1, 2, 4].map(h => [`${h} paso${h > 1 ? 's' : ''}`, r => r.rollouts?.[String(h)]?.latent_mse]),
    ['Identidad · 4 pasos', r => r.rollouts?.['4']?.identity_latent_mse]], curveGroup));
  const diagnostics = evaluationDiagnostics(visualRows);
  charts.changes.set([['Inventados', 'extra_changes'], ['Omitidos', 'missed_changes'],
    ['Color incorrecto', 'wrong_replacement_colors'], ['Acertados', 'correct_changes']]
    .map(([name, key]) => line(name, diagnostics, r => r[key])), undefined, ' px');
  charts.spread.set([line('Diversidad media', diagnostics, r => r.latent_std)], config.min_latent_std);
  charts.total.set([line('Pérdida total', worldPoints, r => r.loss?.loss)]);
  charts.speed.set([line('Media de la ejecución', view?.kind === 'campaign'
    ? (raw.progress_points || []).filter(r => campaignKey(r) === campaignKey(p)) : worldPoints, r => r.actions_per_second)]);
  $('resources').textContent = [finite(p.gpu_peak_mb) ? `Pico GPU del trainer: ${fmt(p.gpu_peak_mb)} MiB` : null,
    finite(p.replay_episodes) ? `Memoria: ${fmt(p.replay_episodes)} partidas` : null,
    finite(p.device_cache_mb) ? `Caché GPU: ${fmt(p.device_cache_mb)} MiB` : null,
    finite(p.trained_transitions_per_second) ? `Transiciones entrenadas/s: ${fmt(p.trained_transitions_per_second, 2)}` : null,
    finite(p.passes) ? `Validaciones consecutivas aprobadas: ${fmt(p.passes)}` : null,
    finite(config.learning_rate) ? `Learning rate: ${scalar(config.learning_rate)}` : null,
    finite(config.false_change_weight) ? `Penalización de cambios inventados: ${scalar(config.false_change_weight)}` : null,
    raw.run?.representation_gate ? `Gate: ${raw.run.representation_gate.reason}` : null].filter(Boolean).join(' · ');

  const learned = evaluations.filter(r => r.policy !== 'random' && r.role !== 'initial');
  const aggregateLearned = learned.filter(r => !finite(r.level));
  const latest = (aggregateLearned.length ? aggregateLearned : learned).at(-1);
  const pending = view?.kind === 'behavior' ? raw.progress?.evaluation : view?.kind === 'campaign' ? p.evaluation : null;
  const completed = ['behavior_finished', 'campaign_finished', 'campaign_error', 'campaign_paused', 'stopped'].includes(view?.phase) || archived;
  const ongoing = view?.source.available && !completed && ['behavior', 'campaign'].includes(view?.kind);
  const evaluating = ongoing && (view.phase === 'behavior_evaluation_progress'
    || (view.kind === 'campaign' && p.stage === 'behavior_evaluation'));
  const liveProgress = evaluating && finite(pending?.current_episode);
  $('evaluation-progress').hidden = !liveProgress;
  $('evaluation-progress').textContent = liveProgress
    ? `${pending.game || 'Evaluación'} · Nivel ${fmt(pending.level)}`
      + (finite(pending.pair_count) ? ` · bloque ${fmt(pending.pair_index)}/${fmt(pending.pair_count)}` : '')
      + ` · Partida ${fmt(pending.current_episode)}/${fmt(pending.total_episodes)}`
      + ` · ${fmt(pending.episode_actions)}/${fmt(pending.episode_budget)} acciones${pending.episode_complete ? ' · terminada' : ''}`
      + ` · ${fmt(pending.episodes)} partidas completadas · ${fmt(pending.wins)} victorias`
      + ` · ${fmt(pending.seconds)} s en este bloque (último registro)` : '';
  const partial = evaluating && view.kind === 'campaign' && pending?.episodes > 0 && finite(pending.wins);
  const baseline = evaluations.findLast(r => r.policy === 'random' && r.split === (latest?.split || pending?.split)
    && (r.level ?? null) === (latest?.level ?? null) && (r.game ?? null) === (latest?.game ?? raw.run?.game ?? null));
  $('behavior-win').textContent = partial ? pct(pending.wins / pending.episodes) : latest ? pct(latest.win_rate) : evaluating ? 'En curso' : ongoing && view.kind === 'behavior' ? 'Pendiente' : '—';
  $('behavior-wins').textContent = partial ? `${fmt(pending.wins)}/${fmt(pending.episodes)} partidas · Nivel ${fmt(pending.level)} · parcial de la campaña`
    : latest ? `${fmt(latest.wins)}/${fmt(latest.episodes)} · ${splitName(latest)} · ${scope(latest)} · ${policyName(latest)}`
    : evaluating ? `Evaluación ${splitName(pending || {})} en curso; esperando el resultado completo`
    : ongoing ? 'Evaluación aprendida pendiente' : 'Sin evaluación aprendida registrada';
  $('behavior-random').textContent = pct(baseline?.win_rate);
  $('behavior-random-note').textContent = baseline ? `${fmt(baseline.wins)}/${fmt(baseline.episodes)} · ${splitName(baseline)} · ${scope(baseline)}`
    : 'Sin referencia al azar para la misma fase y nivel';
  const measured = partial ? pending : latest;
  $('behavior-ms').textContent = fmt(measured?.decision_ms, 2);
  $('behavior-tps').textContent = fmt(measured?.actions_per_second, 2);
  $('behavior-status').textContent = evaluating && view.kind === 'campaign'
    ? liveProgress
      ? 'Evaluación con pesos congelados. Progreso aproximadamente cada 2 s al completar acciones y al terminar cada partida. Una decisión lenta puede tardar más; la antigüedad indica cuándo llegó el último registro.'
      : 'Evaluación con pesos congelados. Este registro sólo ofrece progreso por bloques de partidas. El resultado completo aparecerá al finalizar todos los juegos y niveles.'
    : fits.length ? `Aprendizaje de comportamiento de esta ejecución. Eje de pérdidas: ${view.fitAxis}.`
    : evaluations.length ? 'Resultados de evaluación; sin aprendizaje de comportamiento registrado en el tramo disponible.'
    : 'Sin métricas de comportamiento registradas en esta ejecución. El panel se completará si esta ejecución las produce.';
  $('behavior-baseline-card').hidden = !baseline;
  $('behavior-ms-card').hidden = !finite(measured?.decision_ms);
  $('behavior-tps-card').hidden = !finite(measured?.actions_per_second);
  $('behavior-wins-card').hidden = !evaluations.length;
  $('behavior-loss-card').hidden = !fits.some(r => ['loss', 'event_ce', 'value_bce', 'q_bce'].some(k => finite(r.loss?.[k])));
  $('behavior-actor-card').hidden = !fits.some(r => ['imitation_loss', 'actor_loss', 'critic_loss'].some(k => finite(r.loss?.[k])));
  charts.wins.set(groupedLines(evaluations, [['Éxito', r => percentValue(r.win_rate)]],
    r => `${policyName(r)} · ${splitName(r)} · ${scope(r)}`,
    r => view?.kind === 'behavior' ? r.evaluationIndex : r.trainingActions), undefined, ' %', view?.kind === 'behavior' ? 'evaluación' : 'acción');
  charts.behaviorLoss.set([['Total', 'loss'], ['Eventos', 'event_ce'], ['Estado V', 'value_bce'], ['Acción Q', 'q_bce']]
    .map(([name, key]) => line(name, fits, r => r.loss?.[key])), undefined, '', view?.fitAxis || 'acción');
  charts.actor.set([['Imitación', 'imitation_loss'], ['Política', 'actor_loss'], ['Crítico', 'critic_loss']]
    .map(([name, key]) => line(name, fits, r => r.loss?.[key])), undefined, '', view?.fitAxis || 'acción');
  setTable('world-table', validations.slice(-200).map(r => [splitName(r), scope(r), fmt(r.actions),
    pct(r.delta_error), pct(r.decoder_delta_error), scalar(r.latent_delta_mse), fmt(r.transitions)]), 7);
  setTable('behavior-table', evaluations.slice(-200).map(r => [splitName(r), scope(r), policyName(r),
    `${fmt(r.wins)}/${fmt(r.episodes)}`, `${pct(r.win_rate)}${r.wilson95 ? ` · ${r.wilson95.map(pct).join('–')}` : ''}`,
    fmt(r.actions), fmt(r.actions_per_success, 1), fmt(r.decision_ms, 2)]), 8);
  $('retention').textContent = `Logs disponibles · curvas limitadas a ${fmt(view?.maxPoints)} puntos recientes por fuente; los registros anteriores pueden conservar sólo resúmenes y evaluaciones. No se crea un historial adicional.${view?.source.tail_limited ? ' Lectura parcial: sólo el tramo reciente del log está disponible.' : ''}`;
  showTab(selectedTab);
}
function showTab(name, focus = false) {
  selectedTab = name;
  for (const tab of ['world', 'behavior', 'evaluations']) {
    const active = name === tab;
    $(`panel-${tab}`).hidden = !active;
    $(`tab-${tab}`).setAttribute('aria-selected', String(active));
    $(`tab-${tab}`).setAttribute('tabindex', active ? '0' : '-1');
  }
  if (focus) $(`tab-${name}`).focus();
  for (const chart of Object.values(charts)) chart.draw();
}
$('execution-select').addEventListener('change', () => {
  selectedExecution = $('execution-select').value;
  if (state) render(state);
});
const tabs = ['world', 'behavior', 'evaluations'];
for (const [index, name] of tabs.entries()) {
  $(`tab-${name}`).addEventListener('click', () => showTab(name));
  $(`tab-${name}`).addEventListener('keydown', event => {
    const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length
      : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length
      : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : null;
    if (next !== null) { event.preventDefault(); showTab(tabs[next], true); }
  });
}
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
