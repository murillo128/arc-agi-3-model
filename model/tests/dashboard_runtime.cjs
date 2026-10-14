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
  console.log('Dashboard full poll/render: startup pending, active run, phase handover, level isolation and world-only fallback passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
