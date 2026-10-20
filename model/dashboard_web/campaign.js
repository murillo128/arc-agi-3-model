'use strict';
const $ = id => document.getElementById(id);
const finite = v => typeof v === 'number' && Number.isFinite(v);
const fmt = (v, digits = 0) => finite(v) ? v.toLocaleString('en-US', {maximumFractionDigits: digits}) : '—';
const pct = v => finite(v) ? `${fmt(v * 100, 2)} %` : '—';
const names = {world_reference: 'World-model reference', rollout2: 'Two-step prediction',
  rollout2_visual: 'Two steps with visual supervision', latent_2step: 'Two latent steps', visual_2step: 'Two latent + visual steps',
  local_decoder: 'Local decoder', staged_freeze: 'Staged freezing', sigreg_only: 'SIGReg only',
  mc: 'Monte Carlo', double_q: 'Double-Q', click_mixed: 'Mixed clicks',
  actor_imitation: 'Imitation actor', actor_imagined: 'Imagination actor'};
const stages = {initializing: 'Preparing / compiling', world: 'Training predictor',
  head_calibration: 'Calibrating value', world_evaluation: 'Evaluating world model', behavior_evaluation: 'Evaluating planner',
  behavior: 'Behavior training', actor: 'Actor training', imagination_gate: 'Imagination gate'};
const reasons = {success: 'Improved success', simplification: 'Simplification with preserved performance',
  measured_cost: 'Lower cost with preserved performance', inconclusive_or_inferior: 'No conclusive improvement'};
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
  axis.textContent = `Scale: 0–${fmt(maximum, 2)} ${unit}`; host.append(axis);
  for (const variant of variants) {
    const group = document.createElement('div'); group.className = 'comparison-group';
    const heading = document.createElement('p');
    heading.textContent = `${title(variant.name)} · ${variant.cells.length}/${totalSeeds} seeds`
      + (variant.cells.length < totalSeeds ? ' · partial' : '');
    group.append(heading);
    for (const [index, field] of fields.entries()) {
      const value = field.value(variant.cells);
      const row = document.createElement('div'); row.className = `comparison-series comparison-series-${index}`;
      const label = document.createElement('span'); label.textContent = field.name;
      const display = document.createElement('span'); display.textContent = finite(value) ? `${fmt(value, 2)} ${unit}` : 'No data';
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
    {name: 'Success', value: c => percent(c, 'behavior', 'win_rate')}], seeds, '%', 100);
  comparisonBars('campaign-error-chart', variants, [
    {name: 'Predictor', value: c => percent(c, 'world', 'delta_error')},
    {name: 'Decoder', value: c => percent(c, 'world', 'decoder_delta_error')}], seeds, '%', 100);
  comparisonBars('campaign-actions-chart', variants, [{name: 'Actions', value: actionsPerWin}], seeds, 'actions');
  comparisonBars('campaign-latency-chart', variants, [
    {name: 'Decision', value: c => mean(c, 'behavior', 'decision_ms')}], seeds, 'ms');
}
function renderCampaign(data) {
  const campaign = data.campaign || {}, report = campaign.report || {}, source = campaign.source || {};
  const available = report.available === true;
  const p = campaign.progress || {}, config = report.config || {}, rows = report.cells || [];
  renderComparisonCharts(available ? report : {});
  renderDiagnostics(report.diagnostics);
  renderLearningProbe(report.learning_probe);
  const calibration = report.calibration || [];
  $('calibration-coverage').hidden = !calibration.length;
  $('calibration-table').replaceChildren();
  for (const run of calibration) for (const pair of run.per_pair || []) {
    const tr = document.createElement('tr');
    tr.append(cell(`${title(run.variant)} · S${fmt(run.seed)}`), cell(`${pair.game} / ${fmt(pair.level)}`),
      ...['episodes', 'successes', 'failures', 'budget_ends', 'censored_episodes'].map(key => cell(fmt(pair[key]))));
    $('calibration-table').append(tr);
  }
  const terminal = ['complete', 'pilot_complete', 'failed', 'interrupted', 'paused'].includes(report.status);
  const failed = ['failed', 'interrupted'].includes(report.status) || campaign.phase === 'campaign_error';
  const finished = ['complete', 'pilot_complete'].includes(report.status);
  $('campaign-connection').textContent = !available ? 'Report unavailable' : failed ? 'Campaign stopped'
    : finished ? 'Campaign finished' : report.status === 'paused' ? 'Campaign paused · separate diagnostics' : source.available ? 'Campaign in progress · from log' : 'No progress log';
  $('campaign-connection').className = !available || failed || !source.available || source.age_seconds > 120 ? 'warn' : '';
  $('campaign-freshness').textContent = `Last log write ${fmt(source.age_seconds)} s ago · polling every 2 s`;
  $('campaign-notice').hidden = available && !failed;
  $('campaign-notice').textContent = !available
    ? 'This campaign report is not available yet. Results from another training run are not substituted.'
    : `Comparison campaign stopped: ${report.error || campaign.reason || 'completed evaluations are retained.'}`;
  const activities = [report.learning_probe, report.diagnostics].filter(a => a?.available);
  const activity = activities.find(a => a.status === 'running') || activities[0];
  if (activity && terminal) {
    const label = activity === report.learning_probe ? 'Learning trial' : 'Frozen-weight diagnostic';
    const stale = activity.status === 'running' && activity.age_seconds > 120;
    $('campaign-connection').textContent = `${label} · ${activity.status === 'complete' ? 'finished' : activity.status === 'failed' ? 'error' : stale ? 'no recent data' : 'in progress'}`;
    $('campaign-connection').className = stale || activity.status === 'failed' ? 'warn' : '';
    $('campaign-freshness').textContent = `Latest activity record ${fmt(activity.age_seconds)} s ago · polling every 2 s`;
  }
  $('campaign-current').textContent = terminal ? (finished ? 'Finished' : report.status === 'paused' ? 'Paused' : 'Stopped') : title(p.variant);
  $('campaign-stage').textContent = p.variant ? `Seed ${fmt(p.seed)} · ${stages[p.stage] || p.stage || 'Waiting'} · ${fmt(p.actions)} / ${fmt(p.total_actions ?? config.actions)} actions` : 'Waiting for progress';
  const variants = config.variants || [], seeds = config.seeds || [];
  $('campaign-completed').textContent = available ? `${rows.length} / ${variants.length * seeds.length}` : '—';
  $('campaign-budget').textContent = available ? `${fmt(config.actions)} actions per variant/seed · ${fmt(config.dev_episodes)} episodes per game/level` : '';
  $('campaign-candidate').textContent = report.selected ? title(report.selected) : 'Pending';
  const final = report.final || {};
  const hasFinal = finite(final.selected?.win_rate) && finite(final.reference?.win_rate);
  $('campaign-confirmation').textContent = hasFinal ? 'Available' : config.pilot ? 'Not applicable · pilot' : 'Pending';
  const table = $('campaign-table'); table.replaceChildren();
  for (const variant of variants) {
    const completed = rows.filter(r => r.variant === variant);
    const deferred = (report.deferred_cells || []).filter(r => r.variant === variant);
    const active = !terminal && p.variant === variant;
    const done = completed.length === seeds.length && seeds.length > 0;
    const comparison = report.controlled_comparisons?.[variant];
    const state = deferred.length ? 'Deferred' : done ? 'Evaluated' : active ? 'In progress'
      : completed.length ? 'Partial' : 'Pending';
    const tr = document.createElement('tr');
    tr.className = active ? 'campaign-active' : '';
    const label = cell(title(variant));
    const code = document.createElement('small'); code.textContent = variant; label.append(code);
    const progress = cell(`${completed.length}/${seeds.length} finished`);
    const bar = document.createElement('progress'); bar.max = Math.max(1, seeds.length); bar.value = completed.length;
    bar.setAttribute('aria-label', `Completed seeds for ${title(variant)}`); progress.append(bar);
    const details = document.createElement('small');
    details.textContent = seeds.map(seed => {
      const result = completed.find(r => r.seed === seed);
      if (result) return `S${seed}: success ${pct(result.behavior?.win_rate)}`;
      if (deferred.some(r => r.seed === seed)) return `S${seed}: deferred`;
      return active && p.seed === seed ? `S${seed}: ${stages[p.stage] || p.stage} · ${fmt(p.actions)} actions` : `S${seed}: pending`;
    }).join('\n'); progress.append(details);
    let verdict = done && !comparison ? 'Reference / no controlled comparison' : 'Waiting for seeds';
    if (deferred.length) verdict = 'Imagination gate not passed';
    if (comparison) verdict = `vs ${title(comparison.control)}: ${fmt(comparison.success_delta * 100, 2)} pp\n`
      + `95% CI: ${(comparison.paired95 || []).map(v => fmt(v * 100, 2)).join(' to ')} pp\n`
      + (reasons[comparison.reason] || comparison.reason || 'Results available');
    tr.append(label, progress, cell(state), cell(pct(mean(completed, 'world', 'delta_error'))),
      cell(pct(mean(completed, 'world', 'decoder_delta_error'))), cell(pct(mean(completed, 'behavior', 'win_rate'))),
      cell(fmt(actionsPerWin(completed), 1)),
      cell(fmt(mean(completed, 'behavior', 'decision_ms'), 2)), cell(verdict));
    table.append(tr);
  }
  if (!variants.length) { const tr = document.createElement('tr'), td = cell('Waiting for campaign configuration.'); td.colSpan = 9; tr.append(td); table.append(tr); }
  $('campaign-final').hidden = !hasFinal;
  const finalTable = $('campaign-final-table'); finalTable.replaceChildren();
  if (hasFinal) for (const [label, r] of [['Reference', final.reference], [title(report.selected), final.selected]]) {
    const tr = document.createElement('tr'); tr.append(cell(label), cell(`${fmt(r.wins)} / ${fmt(r.episodes)}`),
      cell(pct(r.win_rate)), cell(fmt(r.actions_per_success, 1))); finalTable.append(tr);
  }
}
function renderDiagnostics(diagnostic) {
  $('diagnostics').hidden = !diagnostic?.available;
  const policies = $('diagnostic-policies'), actions = $('diagnostic-actions');
  policies.replaceChildren(); actions.replaceChildren();
  if (!diagnostic?.available) return;
  const p = diagnostic.progress || {};
  const state = diagnostic.status === 'complete' ? 'Finished' : diagnostic.status === 'failed' ? `Error: ${diagnostic.error}` : 'In progress';
  $('diagnostic-status').textContent = `${state} · ${p.checkpoint || ''} · ${p.game || ''} ${finite(p.level) ? `level ${fmt(p.level)}` : ''}`
    + (p.phase === 'behavior' ? ` · ${p.policy} · episode ${fmt(p.current_episode)}/${fmt(p.total_episodes)} · ${fmt(p.episode_actions)}/${fmt(p.episode_budget)} actions` : '')
    + (p.phase === 'actions' ? ` · state ${fmt((p.anchor ?? 0) + 1)} · alternative ${fmt((p.action_index ?? 0) + 1)}/${fmt(p.candidates)}` : '')
    + ` · last record ${fmt(diagnostic.age_seconds)} s`;
  for (const result of diagnostic.results || []) {
    for (const [mode, policy] of Object.entries(result.policies || {})) {
      const r = policy.summary || {}, tr = document.createElement('tr');
      tr.append(cell(result.checkpoint), cell({random: 'Random', q: 'Direct Q', value_mpc: 'Planner'}[mode] || mode),
        cell(`${fmt(r.wins)} / ${fmt(r.episodes)}`), cell(pct(r.win_rate)), cell(fmt(r.actions_per_success, 1)),
        cell(fmt(r.decision_ms, 1)), cell(policy.complete ? 'Finished' : 'Partial'));
      policies.append(tr);
    }
    for (const r of result.probes || []) {
      const tr = document.createElement('tr');
      tr.append(cell(result.checkpoint), cell(`${r.game} / ${fmt(r.level)}`), cell(pct(r.predictor?.delta_error)),
        cell(pct(r.no_change?.delta_error)), cell(pct(r.distinct_correct?.delta_error)),
        cell(pct(r.distinct_wrong?.delta_error)), cell(fmt(r.distinct_pairs)));
      actions.append(tr);
    }
  }
}
const probeChartNodes = new Map();
function renderLearningProbe(probe) {
  $('learning-probe').hidden = !probe?.available;
  const table = $('probe-coverage'); table.replaceChildren();
  const streaming = probe?.scope === 'streaming_real_training_probe';
  $('probe-description').textContent = 'Same initial weights, data and updates. Encoder and decoder frozen. Both branches retain one-step visual loss; the second adds two-step visual supervision. '
    + (streaming ? 'New experiences are added progressively; the second branch receives exactly the same prefixes in the same order. Horizontal axis: actions seen. Only episodes separate from learning are evaluated.'
      : '“Trained data” measures fit; “New episodes” measures transfer to observations excluded from training. Horizontal axis: updates.');
  $('probe-selection').hidden = !streaming;
  $('probe-behavior').hidden = !probe?.calibration?.records;
  for (const id of ['probe-best', 'probe-confirmation', 'probe-measurements', 'probe-policies']) $(id).replaceChildren();
  for (const horizon of [1, 2]) {
    const id=`probe-chart-${horizon}`, host=$(id);
    if(!probeChartNodes.has(horizon)){
      const legend=document.createElement('div');legend.className='legend';
      const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
      svg.setAttribute('viewBox','0 0 540 245');svg.setAttribute('role','img');
      host.replaceChildren(legend,svg);probeChartNodes.set(horizon,{legend,svg});
    }
    const {legend,svg}=probeChartNodes.get(horizon);svg.replaceChildren();
    if (!probe?.available) {ChartTools.legend(legend,id,[],()=>{});continue;}
    const series = (probe.results || []).flatMap((row, index) => ['fit', 'development'].map((split, branch) => ({
      label: `${title(row.variant)} · ${split === 'fit' ? 'Trained data' : 'New episodes'}`,
      color: ['#77d8c4', '#f7bd76', '#b4a5ff', '#7cceff'][index * 2 + branch],
      points: row.points.map(p => [streaming ? p.actions : p.updates, finite(p[`${split}_${horizon}`]) ? p[`${split}_${horizon}`] * 100 : null])
    }))).filter(s => s.points.some(ChartTools.valid));
    const max = Math.max(1, ...series.flatMap(s => s.points.map(p => p[0])));
    const unit = streaming ? 'actions' : 'updates';
    svg.setAttribute('aria-label', `Visual error at ${horizon} steps. Horizontal axis: ${unit}; vertical: 0 to 100 percent.`);
    const add = (tag, attrs, text) => {
      const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      if (text !== undefined) node.textContent = text;
      svg.append(node);return node;
    };
    if (!series.length) {
      add('text',{x:42,y:120,fill:'#9bb0ac'},'Waiting for the first evaluation…');
      ChartTools.legend(legend,id,[],()=>{});continue;
    }
    for (const percent of [0, 25, 50, 75, 100]) {
      const y = 215 - percent * 2;
      add('line', {x1: 42, y1: y, x2: 525, y2: y, stroke: '#34413f'});
      add('text', {x: 0, y: y + 4, fill: '#9bb0ac', 'font-size': 12}, `${percent}%`);
    }
    add('text', {x: 42, y: 240, fill: '#9bb0ac', 'font-size': 12}, '0');
    add('text', {x: 525, y: 240, fill: '#9bb0ac', 'font-size': 12, 'text-anchor': 'end'}, `${fmt(max)} ${unit}`);
    const x=v=>42+v/max*483, y=v=>215-v*2;
    for (const s of series) {
      let segment=[];
      const flush=()=>{if(segment.length)add('polyline',{points:segment.join(' '),fill:'none',stroke:s.color,'stroke-width':2});segment=[];};
      for(const p of s.points){
        if(!ChartTools.valid(p)){flush();continue;}
        segment.push(`${x(p[0])},${y(p[1])}`);
        add('circle',{cx:x(p[0]),cy:y(p[1]),r:3,fill:s.color});
      }
      flush();
      svg.append(...ChartTools.svgEMA(s.points,x,y,s.color));
    }
    const guide=add('g',{class:'series-guide','pointer-events':'none'});
    const updateGuide=()=>{
      guide.replaceChildren();
      const selected=series.find(s=>s.label===ChartTools.selected(id)),last=selected?.points.findLast(ChartTools.valid);
      if(last)ChartTools.svgGuide(guide,{left:42,right:525,y:y(last[1]),lastX:x(last[0]),color:selected.color,text:`${fmt(last[1],2)} %`});
    };
    ChartTools.legend(legend,id,series.map(s=>({key:s.label,text:`${s.label}: ${fmt(s.points.findLast(ChartTools.valid)[1],2)} %`,color:s.color})),updateGuide);
    updateGuide();
  }
  if (!probe?.available) return;
  const p = probe.progress || {};
  const phase = {collection: 'Collecting data', loading: 'Preparing / compiling', training: 'Training predictor',
    evaluation: p.split === 'fit' ? 'Measuring trained data' : 'Measuring new episodes', confirmation: 'Independent confirmation',
    calibration_data: 'Preparing experiences for V/Q', head_calibration: 'Calibrating V/Q', behavior: 'Evaluating policies', complete: 'Finished'}[p.phase] || p.phase;
  const progress = p.phase === 'behavior' ? `${p.policy} · ${p.game} / ${p.level} · episode ${fmt(p.current_episode)}/${fmt(p.total_episodes)} · ${fmt(p.episode_actions)}/${fmt(p.episode_budget)} actions`
    : p.phase === 'head_calibration' ? `epoch ${fmt(p.epoch)}/${fmt(p.epochs)}`
    : streaming ? `${fmt(p.actions)} / ${fmt(p.total_actions)} actions · ${fmt(p.updates)} updates`
    : `${fmt(p.updates)} / ${fmt(p.total_updates)} updates`;
  $('probe-status').textContent = probe.status === 'failed' ? `Error: ${probe.error}` : `${title(p.variant)} · ${phase} · ${progress}`;
  if (streaming && probe.status === 'complete') {
    $('probe-status').textContent = `Trial finished · ${fmt(probe.coverage?.actions)} new actions · predictor and V/Q evaluated`;
  }
  for (const row of probe.results || []) {
    if (!finite(row.best?.selection_error)) continue;
    const tr = document.createElement('tr');
    tr.append(cell(title(row.variant)), cell(fmt(row.best.actions)), cell(pct(row.best.selection_error)));
    $('probe-best').append(tr);
  }
  $('probe-selected').textContent = probe.selected ? `Selected for this check: ${title(probe.selected.variant)}, action ${fmt(probe.selected.actions)}. Not promoted to the active agent.` : 'Selection pending.';
  for (const row of probe.confirmation || []) {
    const tr = document.createElement('tr');
    tr.append(cell(row.variant === 'initial' ? 'Initial' : 'Selected'), cell(pct(row['1'])), cell(pct(row['2'])));
    $('probe-confirmation').append(tr);
  }
  for (const row of probe.measurements || []) {
    const tr = document.createElement('tr');
    const label = {confirmation_initial: 'Initial · confirmation', confirmation_selected: 'Selected · confirmation'}[row.label] || `${title(row.label)} · validation`;
    tr.append(cell(label), cell(`${row.game} / ${fmt(row.level)}`), cell(fmt(row.horizon)), cell(fmt(row.changed_samples)), cell(pct(row.changed_error)), cell(fmt(row.unchanged_samples)), cell(fmt(row.unchanged_pixels, 2)));
    $('probe-measurements').append(tr);
  }
  $('probe-calibration').textContent = `${fmt(probe.calibration?.records)} real transitions; ${fmt(probe.calibration?.successes)} successes retained; ${fmt(probe.calibration?.positive_records)} positive targets; ${fmt(probe.calibration?.censored_records)} censored targets. Policies compared with the same seeds and budget.`;
  for (const row of probe.policies || []) {
    const tr = document.createElement('tr');
    tr.append(cell({random: 'Random', q: 'Direct Q', value_mpc: 'MPC'}[row.policy] || row.policy), cell(`${fmt(row.wins)} / ${fmt(row.episodes)}`), cell(pct(row.win_rate)), cell(fmt(row.actions_per_success, 1)), cell(fmt(row.decision_ms, 1)), cell(row.complete ? 'Complete' : 'Partial'));
    $('probe-policies').append(tr);
  }
  for (const r of probe.coverage?.per_pair || []) {
    const tr = document.createElement('tr');
    tr.append(cell(`${r.game} / ${fmt(r.level)}`), ...['episodes', 'actions', 'successes', 'failures', 'budget_ends'].map(k => cell(fmt(r[k]))));
    table.append(tr);
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
    $('campaign-connection').textContent = 'Disconnected · reintentando';
    $('campaign-connection').className = 'warn';
    $('campaign-notice').hidden = false;
    $('campaign-notice').textContent = 'Visible results are not updating. Retrying automatically.';
  } finally { setTimeout(pollCampaign, 2000); }
}
pollCampaign();
