// Functional rendering contract. No browser, layout assertions or screenshots.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const assets = path.join(__dirname, '../dashboard_web');
const html = fs.readFileSync(path.join(assets, 'index.html'), 'utf8');
const script = fs.readFileSync(path.join(assets, 'dashboard.js'), 'utf8');
function element(tag = 'div') {
  return {textContent: '', hidden: false, style: {}, className: '', value: '', options: [], children: [],
    clientWidth: 640, clientHeight: 240, nextElementSibling: {hidden: true}, listeners: {},
    addEventListener(name, fn) { this.listeners[name] = fn; },
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; if (tag === 'select') { this.options = items; this.value = items[0]?.value || ''; } },
    getContext() { return new Proxy({}, {get: (target, key) => target[key] || (() => {})}); }
  };
}
async function dashboard(data) {
  const nodes = new Map();
  for (const m of html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"[^>]*>/g)) nodes.set(m[2], element(m[1]));
  for (const m of html.matchAll(/<select[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
    const node = nodes.get(m[1]);
    node.options = [...m[2].matchAll(/<option value="([^"]+)"/g)].map(o => ({value: o[1]}));
    node.value = node.options[0].value;
  }
  const text = id => nodes.get(id).textContent;
  const change = (id, value) => { nodes.get(id).value = value; nodes.get(id).listeners.change(); };
  const context = vm.createContext({console, window: {devicePixelRatio: 1},
    document: {getElementById: id => nodes.get(id) || null, createElement: element},
    ResizeObserver: class { observe() {} }, Option: function(text, value) { this.value = value; this.textContent = text; },
    fetch: async () => ({ok: true, json: async () => data}), AbortSignal, setTimeout: () => {}});
  vm.runInContext(script, context);
  await new Promise(resolve => setImmediate(resolve)); // actual initial poll + render
  return {nodes, text, change, context};
}
const row = (split, policy, level, wins, episodes, role = 'candidate') =>
  ({split, policy, level, wins, episodes, role, horizon: 2, win_rate: wins / episodes, decision_ms: 2, actions_per_second: 100});
const behavior = {phase: 'behavior_evaluation_progress', source: {available: true, age_seconds: 2},
  run: {game: 'sc01-v1', levels: [3, 4, 5], world_trainable: 0, device: 'cuda', compile: true},
  fits: [{epoch: 1, loss: .2, event_ce: .1, value_bce: .05, q_bce: .05}],
  progress: {collection: {round: 2, episodes: 384, actions: 18614}, fit: {epoch: 12, loss: .08},
    evaluation: {split: 'final', policy: 'mpc', episodes: 16, wins: 4}},
  evaluations: [row('development', 'random', 3, 3, 32), row('development', 'q', 3, 3, 32),
    row('development', 'random', null, 6, 96), row('development', 'q', null, 8, 96),
    row('final', 'random', null, 30, 384), row('final', 'q', null, 0, 384, 'initial'),
    row('final', 'random', 3, 10, 128)]};
const data = {phase: 'stopped', reason: 'interrupted', source: {available: true, age_seconds: 1000},
  run: {game: 'old-world-run'}, progress: {}, progress_points: [], validations: [], events: [], behavior};
(async () => {
  // A completed random baseline precedes the first collection/fitting cycle.
  const firstCycle = {phase: 'stopped', reason: 'interrupted', source: {available: true, age_seconds: 1000},
    run: {}, progress: {}, progress_points: [], validations: [], events: [],
    behavior: {phase: 'behavior_evaluation', source: {available: true, age_seconds: 2},
      run: {game: 'sc01-v1', levels: [3, 4, 5], mode: 'training', world_trainable: 0},
      progress: {collection: {round: 1, episodes: 16, actions: 500},
        evaluation: {split: 'development', policy: 'random', level: null, episodes: 96, wins: 6, win_rate: .0625}},
      evaluations: [{split: 'development', policy: 'random', level: null, episodes: 96, wins: 6, win_rate: .0625}], fits: []}};
  const first = await dashboard(firstCycle);
  for (const [phase, status] of [
    ['behavior_evaluation', 'Evaluación recibida'],
    ['behavior_collection', 'Recogiendo experiencia real'], ['behavior_fit', 'Entrenando goal / valor'],
    ['behavior_world_verified', 'Pesos del mundo verificados']
  ]) {
    firstCycle.behavior.phase = phase;
    await vm.runInContext('poll()', first.context);
    assert.equal(first.nodes.get('behavior-split').value, 'development');
    assert.equal(first.text('connection'), status);
    assert.equal(first.text('behavior-win'), 'Pendiente', 'first candidate remains pending through collection and fitting');
    assert.match(first.text('behavior-wins'), /Evaluación aprendida pendiente.*Desarrollo/);
    assert.doesNotMatch(first.text('behavior-wins'), /progreso de la evaluación indicado arriba/);
    assert.equal(first.text('behavior-actions'), '500');
    assert.match(first.text('behavior-random-wins'), /6\/96.*Desarrollo/);
    assert.equal(first.text('behavior-ms'), '—');
    assert.equal(vm.runInContext('behaviorCharts.wins.series.length', first.context), 1);
    assert.equal(vm.runInContext('behaviorCharts.wins.series[0].points[0].y', first.context), 6.25);
  }
  const manual = await dashboard(firstCycle);
  manual.change('behavior-split', 'final');
  firstCycle.behavior.phase = 'behavior_evaluation_progress';
  firstCycle.behavior.progress.evaluation = {split: 'development', policy: 'q', episodes: 16, wins: 4};
  await vm.runInContext('poll()', first.context);
  await vm.runInContext('poll()', manual.context);
  assert.equal(first.text('behavior-win'), 'En curso', 'candidate evaluation has actually started');
  assert.match(first.text('behavior-wins'), /Esperando el resultado completo de Desarrollo/);
  assert.equal(manual.nodes.get('behavior-split').value, 'final');
  assert.equal(manual.text('behavior-win'), '—', 'development progress is not a final result');
  assert.equal(manual.text('behavior-random'), '—');
  firstCycle.behavior.phase = 'behavior_evaluation';
  const candidate = {split: 'development', policy: 'q', role: 'candidate', level: null,
    episodes: 96, wins: 24, win_rate: .25, decision_ms: 2};
  firstCycle.behavior.progress.evaluation = candidate;
  firstCycle.behavior.evaluations.push(candidate);
  await vm.runInContext('poll()', first.context);
  assert.equal(first.text('behavior-win'), '25 %');
  assert.match(first.text('behavior-wins'), /Q.*24\/96.*Desarrollo/);
  assert.equal(vm.runInContext('behaviorCharts.wins.series.find(s => s.name === "Q").points[0].y', first.context), 25);
  await vm.runInContext('poll()', manual.context);
  assert.equal(manual.nodes.get('behavior-split').value, 'final');
  assert.equal(manual.text('behavior-win'), 'Pendiente');
  assert.match(manual.text('behavior-wins'), /Evaluación aprendida pendiente.*Final/);
  assert.equal(vm.runInContext('behaviorCharts.wins.series.length', manual.context), 0);
  first.change('behavior-level', '3');
  assert.equal(first.text('behavior-win'), 'Pendiente', 'aggregate candidate is not a result for level 3');
  assert.equal(first.text('behavior-random'), '—');
  assert.equal(vm.runInContext('behaviorCharts.wins.series.length', first.context), 0);
  firstCycle.behavior.phase = 'behavior_finished';
  await vm.runInContext('poll()', first.context);
  assert.equal(first.text('behavior-win'), '—', 'finished run cannot claim a missing result is still pending');

  for (const [mode, split] of [['training', 'development'], ['evaluation_only', 'evaluation_only']]) {
    const preparation = {...firstCycle, behavior: {phase: 'behavior_started', source: {available: true, age_seconds: 2},
      run: {game: 'sc01-v1', levels: [3, 4, 5], mode}, progress: {}, evaluations: [], fits: []}};
    const preparing = await dashboard(preparation);
    assert.equal(preparing.nodes.get('behavior-split').value, split);
    assert.equal(preparing.text('connection'), 'Preparando');
    assert.equal(preparing.text('behavior-win'), 'Pendiente');
    assert.equal(preparing.text('behavior-random'), '—');
    preparation.behavior.source.available = false;
    await vm.runInContext('poll()', preparing.context);
    assert.equal(preparing.text('behavior-win'), '—', 'missing log cannot imply a pending evaluation');
  }
  for (const [evaluation, phaseName] of [
    [{split: 'development', policy: 'random', episodes: 16, wins: 2}, 'Desarrollo'],
    [{split: 'evaluation_only', policy: 'random', episodes: 16, wins: 1}, 'Aislada']
  ]) {
    const startup = {phase: 'stopped', reason: 'interrupted', source: {available: true, age_seconds: 1000},
      run: {}, progress: {}, progress_points: [], validations: [], events: [],
      behavior: {phase: 'behavior_evaluation_progress', source: {available: true, age_seconds: 2},
        run: {game: 'sc01-v1', levels: [3, 4, 5], world_trainable: 0},
        progress: {evaluation}, evaluations: [], fits: []}};
    const panel = await dashboard(startup);
    assert.equal(panel.nodes.get('behavior-split').value, evaluation.split, 'startup selects the phase being evaluated');
    assert.match(panel.text('connection'), /Evaluando/);
    assert.equal(panel.text('behavior-win'), 'En curso', 'no completed rows means pending evaluation');
    assert.ok(panel.text('behavior-wins').includes(`Esperando el resultado completo de ${phaseName}`));
    assert.equal(panel.text('behavior-random'), '—', 'progress is not a completed baseline');
    assert.equal(panel.text('behavior-ms'), '—');
    assert.equal(vm.runInContext('behaviorCharts.wins.series.length', panel.context), 0);
    startup.behavior.evaluations = [
      {split: 'final', policy: 'random', level: null, wins: 12, episodes: 16, win_rate: .75},
      {split: 'final', policy: 'q', role: 'initial', level: null, wins: 4, episodes: 16, win_rate: .25}
    ];
    await vm.runInContext('poll()', panel.context);
    assert.equal(panel.nodes.get('behavior-split').value, evaluation.split, 'running phase takes priority over other-phase baselines');
    assert.equal(panel.text('behavior-win'), 'En curso', 'an initial policy is not a completed learned candidate');
    assert.equal(panel.text('behavior-random'), '—');
    assert.equal(vm.runInContext('behaviorCharts.wins.series.length', panel.context), 0);
    panel.change('behavior-level', '3');
    await vm.runInContext('poll()', panel.context);
    assert.equal(panel.nodes.get('behavior-split').value, evaluation.split);
    assert.equal(panel.text('behavior-win'), 'En curso');
    panel.change('behavior-split', 'final');
    await vm.runInContext('poll()', panel.context);
    assert.equal(panel.nodes.get('behavior-split').value, 'final', 'startup fallback respects manual phase');
    assert.equal(panel.text('behavior-win'), '—');
  }
  const {nodes, text, change, context} = await dashboard(data);
  assert.match(text('connection'), /Evaluando/, 'header must describe the active behavior run');
  assert.doesNotMatch(text('notice'), /Entrenamiento detenido/);
  assert.match(text('world-status'), /detenido/i);
  assert.match(text('run-info'), /3, 4, 5/);
  assert.equal(nodes.get('behavior-split').value, 'development', 'pending final must preserve completed development');
  assert.match(text('behavior-wins'), /8\/96.*Desarrollo/);
  assert.match(text('behavior-random-wins'), /6\/96.*Desarrollo/);
  assert.match(text('behavior-info'), /Final.*16.*4\/16|Final.*4\/16/);
  assert.equal(text('behavior-actions'), '18.614');
  assert.ok(Math.abs(vm.runInContext('behaviorCharts.wins.series.find(s => s.name === "Q").points[0].y', context) - 100 * 8 / 96) < 1e-10);

  change('behavior-split', 'final');
  assert.equal(text('behavior-win'), 'En curso');
  assert.match(text('behavior-random-wins'), /30\/384.*Final/);
  assert.doesNotMatch(text('behavior-wins'), /8\/96|0\/384/, 'pending result cannot substitute another phase or initial policy');

  const automatic = await dashboard(data);
  assert.equal(automatic.nodes.get('behavior-split').value, 'development');
  behavior.evaluations.push(row('final', 'mpc', 3, 18, 128), row('final', 'mpc', null, 54, 384));
  behavior.phase = 'behavior_finished';
  await vm.runInContext('poll()', automatic.context);
  assert.equal(automatic.nodes.get('behavior-split').value, 'final', 'completed final candidate triggers automatic handover');
  assert.match(automatic.text('behavior-wins'), /54\/384.*Final/);
  assert.match(automatic.text('behavior-random-wins'), /30\/384.*Final/);
  await vm.runInContext('poll()', context);
  assert.match(text('behavior-wins'), /54\/384.*Final/);
  change('behavior-level', '3');
  assert.match(text('behavior-wins'), /18\/128.*Final/);
  assert.match(text('behavior-random-wins'), /10\/128.*Final/);
  assert.equal(nodes.get('behavior-table').children.length, 2);
  change('behavior-split', 'development');
  await vm.runInContext('poll()', context);
  assert.match(text('behavior-wins'), /3\/32.*Desarrollo/, 'manual phase selection is retained');
  assert.match(text('connection'), /terminado/i);

  behavior.phase = 'behavior_fit'; behavior.source.age_seconds = 200;
  await vm.runInContext('poll()', context);
  assert.match(text('connection'), /Sin datos recientes/);
  assert.equal(nodes.get('connection').className, 'warn');
  delete data.behavior;
  await vm.runInContext('poll()', context);
  assert.match(text('connection'), /Entrenamiento detenido/);
  assert.match(text('notice'), /interrupción/);
  const campaign = {phase: 'campaign_progress', source: {available: true, age_seconds: 2},
    run: {config: {variants: ['baseline', 'rollout'], seeds: [0], actions_per_variant: 100, device: 'cuda'}},
    progress: {variant: 'rollout', seed: 0, actions: 10, total_actions: 100, updates: 5,
      actions_per_second: 2, stage: 'world', game: 'synthetic', level: 2},
    progress_points: [{variant: 'baseline', seed: 0, actions: 100, loss: {latent_mse: .9}},
      {variant: 'rollout', seed: 0, actions: 10, loss: {latent_mse: .2, rollout_mse: .3}}],
    evaluations: [{variant: 'baseline', seed: 0, actions: 100, split: 'final',
      world: {delta_error: .5}, behavior: {win_rate: .25, actions: 88}},
      {variant: 'rollout', seed: 0, actions: 10, split: 'development',
        world: {delta_error: .1, rollouts: {'2': {latent_mse: .3}}},
        behavior: {win_rate: .75, actions: 18, decision_ms: 2}}]};
  const campaignData = {...data, campaign, behavior};
  const comparison = await dashboard(campaignData);
  assert.match(comparison.text('connection'), /Campaña.*en curso/);
  assert.equal(comparison.nodes.get('campaign-panel').hidden, false);
  assert.equal(comparison.text('campaign-actions'), '10');
  assert.equal(comparison.text('campaign-delta'), '10 %');
  assert.equal(comparison.text('campaign-win'), '75 %');
  assert.match(comparison.text('run-info'), /Campaña.*rollout/);
  assert.equal(vm.runInContext('campaignCharts.loss.series[0].points.length', comparison.context), 1);
  assert.equal(vm.runInContext('campaignCharts.loss.series[0].points[0].y', comparison.context), .2);
  comparison.change('campaign-run', JSON.stringify(['baseline', 0]));
  assert.equal(vm.runInContext('campaignCharts.loss.series[0].points[0].y', comparison.context), .9);
  assert.equal(comparison.text('campaign-actions'), '10', 'chart filter must not replace live counters');
  campaign.progress = {variant: 'rollout', seed: 1, actions: 0, total_actions: 100};
  await vm.runInContext('poll()', comparison.context);
  assert.equal(comparison.text('campaign-win'), '—', 'previous seed cannot masquerade as current evaluation');
  campaign.source.available = false;
  await vm.runInContext('poll()', comparison.context);
  assert.equal(comparison.text('connection'), 'Log no disponible', 'missing campaign cannot present an old trainer as active');
  campaign.source.available = true;
  campaign.phase = 'campaign_error'; campaign.reason = 'expected test error';
  await vm.runInContext('poll()', comparison.context);
  assert.match(comparison.text('connection'), /Campaña detenida por error/);
  assert.match(comparison.text('notice'), /expected test error/);
  campaign.phase = 'campaign_phase';
  campaign.source.age_seconds = 80;
  campaign.run.config = {actions: 40};
  campaign.progress = {variant: 'rollout', seed: 1, actions: 0, stage: 'initializing', message: 'Compiling kernels'};
  await vm.runInContext('poll()', comparison.context);
  assert.match(comparison.text('connection'), /Inicializando.*compilando.*sin datos recientes/);
  assert.equal(comparison.text('campaign-budget'), 'De 40 acciones de aprendizaje');
  assert.match(comparison.text('campaign-info'), /Compiling kernels/);
  console.log('Dashboard full poll/render: startup pending, active run, phase handover, level isolation and world-only fallback passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
