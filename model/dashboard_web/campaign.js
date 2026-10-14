'use strict';
const $ = id => document.getElementById(id);
const finite = v => typeof v === 'number' && Number.isFinite(v);
const fmt = (v, digits = 0) => finite(v) ? v.toLocaleString('es-ES', {maximumFractionDigits: digits}) : '—';
const pct = v => finite(v) ? `${fmt(v * 100, 2)} %` : '—';
const names = {world_reference: 'Referencia del mundo', rollout2: 'Predicción a dos pasos',
  local_decoder: 'Decoder local', staged_freeze: 'Congelación por etapas', sigreg_only: 'Sólo SIGReg',
  mc: 'Monte Carlo', double_q: 'Double-Q', click_mixed: 'Clics mixtos',
  actor_imitation: 'Actor por imitación', actor_imagined: 'Actor con imaginación'};
const stages = {initializing: 'Preparando / compilando', world: 'Entrenando predictor',
  head_calibration: 'Calibrando valor', world_evaluation: 'Evaluando mundo', behavior_evaluation: 'Evaluando planner',
  behavior: 'Entrenando comportamiento', actor: 'Entrenando actor', imagination_gate: 'Control de imaginación'};
const reasons = {success: 'Mejora de éxito', simplification: 'Simplificación con rendimiento conservado',
  measured_cost: 'Menor coste con rendimiento conservado', inconclusive_or_inferior: 'Sin mejora concluyente'};
const title = v => names[v] || v || '—';
function mean(cells, section, key) {
  const values = cells.map(c => c[section]?.[key]);
  return values.length && values.every(finite) ? values.reduce((a, b) => a + b, 0) / values.length : null;
}
function cell(text) { const td = document.createElement('td'); td.textContent = text; return td; }
function actionsPerWin(cells) {
  if (!cells.length || !cells.every(r => finite(r.behavior?.wins) && finite(r.behavior?.actions))) return null;
  const wins = cells.reduce((sum, r) => sum + r.behavior.wins, 0);
  return wins > 0 ? cells.reduce((sum, r) => sum + r.behavior.actions, 0) / wins : null;
}
function comparisonBars(id, variants, fields, totalSeeds, unit, fixedMax) {
  const host = $(id); host.replaceChildren();
  const maximum = fixedMax ?? Math.max(1, ...variants.flatMap(v => fields.map(f => f.value(v.cells))).filter(finite));
  const axis = document.createElement('p'); axis.className = 'comparison-axis';
  axis.textContent = `Escala: 0–${fmt(maximum, 2)} ${unit}`; host.append(axis);
  for (const variant of variants) {
    const group = document.createElement('div'); group.className = 'comparison-group';
    const heading = document.createElement('p');
    heading.textContent = `${title(variant.name)} · ${variant.cells.length}/${totalSeeds} semillas`
      + (variant.cells.length < totalSeeds ? ' · parcial' : '');
    group.append(heading);
    for (const [index, field] of fields.entries()) {
      const value = field.value(variant.cells);
      const row = document.createElement('div'); row.className = `comparison-series comparison-series-${index}`;
      const label = document.createElement('span'); label.textContent = field.name;
      const display = document.createElement('span'); display.textContent = finite(value) ? `${fmt(value, 2)} ${unit}` : 'Sin datos';
      row.append(label);
      if (finite(value)) {
        const meter = document.createElement('meter'); meter.min = 0; meter.max = maximum; meter.value = value;
        meter.setAttribute('aria-label', `${title(variant.name)} · ${field.name}`);
        meter.setAttribute('aria-valuetext', display.textContent);
        meter.title = `${heading.textContent} · ${field.name}: ${display.textContent}`;
        row.append(meter);
      } else { const gap = document.createElement('span'); row.append(gap); }
      row.append(display); group.append(row);
    }
    host.append(group);
  }
}
function renderComparisonCharts(report) {
  const variants = (report.config?.variants || []).map(name => ({name,
    cells: (report.cells || []).filter(r => r.variant === name)})).filter(v => v.cells.length);
  const seeds = report.config?.seeds?.length || 0;
  $('campaign-charts').hidden = !variants.length;
  $('campaign-charts-empty').hidden = !!variants.length;
  const percent = (cells, section, key) => { const value = mean(cells, section, key); return finite(value) ? value * 100 : null; };
  comparisonBars('campaign-success-chart', variants, [
    {name: 'Éxito', value: c => percent(c, 'behavior', 'win_rate')}], seeds, '%', 100);
  comparisonBars('campaign-error-chart', variants, [
    {name: 'Predictor', value: c => percent(c, 'world', 'delta_error')},
    {name: 'Decoder', value: c => percent(c, 'world', 'decoder_delta_error')}], seeds, '%', 100);
  comparisonBars('campaign-actions-chart', variants, [{name: 'Acciones', value: actionsPerWin}], seeds, 'acciones');
  comparisonBars('campaign-latency-chart', variants, [
    {name: 'Decisión', value: c => mean(c, 'behavior', 'decision_ms')}], seeds, 'ms');
}
function renderCampaign(data) {
  const campaign = data.campaign || {}, report = campaign.report || {}, source = campaign.source || {};
  const available = report.available === true;
  const p = campaign.progress || {}, config = report.config || {}, rows = report.cells || [];
  renderComparisonCharts(available ? report : {});
  const terminal = ['complete', 'pilot_complete', 'failed', 'interrupted'].includes(report.status);
  const failed = ['failed', 'interrupted'].includes(report.status) || campaign.phase === 'campaign_error';
  const finished = ['complete', 'pilot_complete'].includes(report.status);
  $('campaign-connection').textContent = !available ? 'Informe no disponible' : failed ? 'Campaña detenida'
    : finished ? 'Campaña terminada' : source.available ? 'Campaña en curso · según el log' : 'Sin log de progreso';
  $('campaign-connection').className = !available || failed || !source.available || source.age_seconds > 120 ? 'warn' : '';
  $('campaign-freshness').textContent = `Última escritura del log hace ${fmt(source.age_seconds)} s · consulta cada 2 s`;
  $('campaign-notice').hidden = available && !failed;
  $('campaign-notice').textContent = !available
    ? 'El informe de esta campaña todavía no está disponible. No se sustituyen sus resultados por los de otro entrenamiento.'
    : report.error || campaign.reason || 'La campaña está interrumpida; se conservan las evaluaciones completadas.';
  $('campaign-current').textContent = terminal ? (finished ? 'Terminada' : 'Detenida') : title(p.variant);
  $('campaign-stage').textContent = p.variant ? `Semilla ${fmt(p.seed)} · ${stages[p.stage] || p.stage || 'Esperando'} · ${fmt(p.actions)} / ${fmt(p.total_actions ?? config.actions)} acciones` : 'Esperando progreso';
  const variants = config.variants || [], seeds = config.seeds || [];
  $('campaign-completed').textContent = available ? `${rows.length} / ${variants.length * seeds.length}` : '—';
  $('campaign-budget').textContent = available ? `${fmt(config.actions)} acciones por variante/semilla · ${fmt(config.dev_episodes)} episodios por juego/nivel` : '';
  $('campaign-candidate').textContent = report.selected ? title(report.selected) : 'Pendiente';
  const final = report.final || {};
  const hasFinal = finite(final.selected?.win_rate) && finite(final.reference?.win_rate);
  $('campaign-confirmation').textContent = hasFinal ? 'Disponible' : config.pilot ? 'No aplica · piloto' : 'Pendiente';
  const table = $('campaign-table'); table.replaceChildren();
  for (const variant of variants) {
    const completed = rows.filter(r => r.variant === variant);
    const deferred = (report.deferred_cells || []).filter(r => r.variant === variant);
    const active = !terminal && p.variant === variant;
    const done = completed.length === seeds.length && seeds.length > 0;
    const comparison = report.controlled_comparisons?.[variant];
    const state = deferred.length ? 'Aplazada' : done ? 'Evaluada' : active ? 'En curso'
      : completed.length ? 'Parcial' : 'Pendiente';
    const tr = document.createElement('tr');
    tr.className = active ? 'campaign-active' : '';
    const label = cell(title(variant));
    const code = document.createElement('small'); code.textContent = variant; label.append(code);
    const progress = cell(`${completed.length}/${seeds.length} terminadas`);
    const bar = document.createElement('progress'); bar.max = Math.max(1, seeds.length); bar.value = completed.length;
    bar.setAttribute('aria-label', `Semillas terminadas de ${title(variant)}`); progress.append(bar);
    const details = document.createElement('small');
    details.textContent = seeds.map(seed => {
      const result = completed.find(r => r.seed === seed);
      if (result) return `S${seed}: éxito ${pct(result.behavior?.win_rate)}`;
      if (deferred.some(r => r.seed === seed)) return `S${seed}: aplazada`;
      return active && p.seed === seed ? `S${seed}: ${stages[p.stage] || p.stage} · ${fmt(p.actions)} acciones` : `S${seed}: pendiente`;
    }).join('\n'); progress.append(details);
    let verdict = done && !comparison ? 'Referencia / sin comparación controlada' : 'Esperando semillas';
    if (deferred.length) verdict = 'Control de imaginación no superado';
    if (comparison) verdict = `vs ${title(comparison.control)}: ${fmt(comparison.success_delta * 100, 2)} pp\n`
      + `IC 95 %: ${(comparison.paired95 || []).map(v => fmt(v * 100, 2)).join(' a ')} pp\n`
      + (reasons[comparison.reason] || comparison.reason || 'Resultado disponible');
    tr.append(label, progress, cell(state), cell(pct(mean(completed, 'world', 'delta_error'))),
      cell(pct(mean(completed, 'world', 'decoder_delta_error'))), cell(pct(mean(completed, 'behavior', 'win_rate'))),
      cell(fmt(actionsPerWin(completed), 1)),
      cell(fmt(mean(completed, 'behavior', 'decision_ms'), 2)), cell(verdict));
    table.append(tr);
  }
  if (!variants.length) { const tr = document.createElement('tr'), td = cell('Esperando la configuración de la campaña.'); td.colSpan = 9; tr.append(td); table.append(tr); }
  $('campaign-final').hidden = !hasFinal;
  const finalTable = $('campaign-final-table'); finalTable.replaceChildren();
  if (hasFinal) for (const [label, r] of [['Referencia', final.reference], [title(report.selected), final.selected]]) {
    const tr = document.createElement('tr'); tr.append(cell(label), cell(`${fmt(r.wins)} / ${fmt(r.episodes)}`),
      cell(pct(r.win_rate)), cell(fmt(r.actions_per_success, 1))); finalTable.append(tr);
  }
}
async function pollCampaign() {
  try {
    const response = await fetch('api/state', {cache: 'no-store', signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!data || !data.source) throw new Error('Invalid dashboard response');
    renderCampaign(data);
  } catch (error) {
    $('campaign-connection').textContent = 'Sin conexión · reintentando';
    $('campaign-connection').className = 'warn';
    $('campaign-notice').hidden = false;
    $('campaign-notice').textContent = 'Los resultados visibles no se están actualizando. Se reintentará automáticamente.';
  } finally { setTimeout(pollCampaign, 2000); }
}
pollCampaign();
