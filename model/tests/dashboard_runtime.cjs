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
    clientWidth: 640, clientHeight: 240, nextElementSibling: {hidden: true}, listeners: {}, draws: [],
    set width(value){this._width=value;this.draws=[];},get width(){return this._width;},
    setAttribute(name, value) { this[name] = value; },
    addEventListener(name, fn) { this.listeners[name] = fn; },
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; if (tag === 'select') { this.options = items; this.value = items[0]?.value || ''; } },
    getContext() { return new Proxy({}, {get: (target, key) => target[key] || ((...args)=>this.draws.push([key,...args])),
      set:(target,key,value)=>{target[key]=value;this.draws.push(['set',key,value]);return true;}}); }
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
  vm.runInContext(fs.readFileSync(path.join(assets, 'chart-tools.js'), 'utf8'), context);
  vm.runInContext(script, context);
  await new Promise(resolve => setImmediate(resolve)); // actual initial poll + render
  return {nodes, text, change, context};
}

const source = {available: true, age_seconds: 2, updated_at: 300};
const evaluation = (split, policy, level, wins, episodes, role = 'candidate') =>
  ({split, policy, level, wins, episodes, role, horizon: 2, win_rate: wins / episodes, decision_ms: 2});
const data = {phase: 'stopped', reason: 'interrupted', source: {...source, updated_at: 100},
  run: {game: 'old-world', training: {freeze_representation: true, min_latent_std: .125}},
  progress: {actions: 900, updates: 80, level: 1, replay_episodes: 0,
    trained_transitions_per_second: 1234.5, passes: 0},
  progress_points: [{actions: 900, loss: {latent_mse: .9}}],
  validations: [{actions: 900, level: 1, delta_error: .8, decoder_delta_error: .2}], events: [],
  behavior: {phase: 'behavior_finished', source: {...source, updated_at: 200},
    run: {game: 'old-behavior', levels: [3, 4], world_trainable: 0},
    progress: {collection: {actions: 500, episodes: 20}, fit: {epoch: 12, updates: 7, loss: .08}},
    fits: [{epoch: 1, loss: .2, event_ce: .1, value_bce: .05, q_bce: .05}],
    evaluations: [evaluation('development', 'q', null, 8, 32),
      evaluation('final', 'random', null, 4, 32), evaluation('final', 'q', null, 0, 32, 'initial'),
      evaluation('final', 'mpc', 3, 6, 16), evaluation('final', 'mpc', null, 12, 32)]},
  campaign: {phase: 'campaign_progress', source, run: {config: {actions: 100, device: 'cuda'}},
    progress: {variant: 'rollout2', seed: 0, actions: 10, updates: 5, stage: 'world', game: 'synthetic', level: 2},
    progress_points: [{variant: 'world_reference', seed: 0, actions: 100, updates: 50, loss: {latent_mse: .7}},
      {variant: 'rollout2', seed: 0, actions: 10, updates: 5, loss: {latent_mse: .2, rollout_mse: .3}}],
    evaluations: [{variant: 'world_reference', seed: 0, actions: 100, split: 'dev',
      world: {delta_error: .5, decoder_delta_error: .1}, behavior: {win_rate: .25, actions: 88}},
      {variant: 'rollout2', seed: 0, actions: 10, split: 'intermediate',
        world: {delta_error: .1, decoder_delta_error: 0, rollouts: {'2': {latent_mse: .3}}, per_pair: [
          {game: 'first', level: 1, transitions: 1, latent_std: .2,
            extra_changes: 2, missed_changes: 3, wrong_replacement_colors: 4, correct_changes: 5},
          {game: 'second', level: 2, transitions: 3, latent_std: .6,
            extra_changes: 10, missed_changes: 20, wrong_replacement_colors: 30, correct_changes: 40}
        ]}}]}}
;
(async () => {
  assert.equal([...html.matchAll(/<select\b/g)].length, 1, 'one global execution selector, no independent selectors');
  assert.match(html, /id="execution-select"/);
  const ui = await dashboard(data);
  const {text, nodes, context} = ui;
  assert.equal(text('actions'), '10');
  assert.equal(text('updates'), '5');
  assert.equal(text('prediction'), '10 %');
  assert.equal(text('reconstruction'), '0 %', 'zero error is not missing');
  assert.equal(text('behavior-win'), '—', 'old behavior results cannot fill current campaign');
  assert.match(text('run-info'), /rollout2.*0/);
  assert.equal(vm.runInContext('charts.latent.series[0].points[0].y', context), .2);
  assert.equal(vm.runInContext('charts.changes.series.length', context), 4);
  assert.equal(vm.runInContext('JSON.stringify(charts.changes.series.map(s => s.points[0].y))', context), '[12,23,34,45]');
  assert.equal(vm.runInContext('charts.spread.series.length', context), 1);
  assert.ok(Math.abs(vm.runInContext('charts.spread.series[0].points[0].y', context) - .5) < 1e-12);
  assert.equal(vm.runInContext(`aggregateDiagnostics({extra_changes: 7, per_pair: [
    {extra_changes: 7}]}).extra_changes`, context), 7, 'existing totals are not counted twice');
  assert.equal(vm.runInContext(`aggregateDiagnostics({per_pair: [
    {extra_changes: 2, transitions: 1, latent_std: .2}, {transitions: 3}]}).extra_changes`, context),
    undefined, 'missing counts must not become a partial total');
  assert.equal(vm.runInContext(`aggregateDiagnostics({per_pair: [
    {transitions: 1, latent_std: .2}, {transitions: 3}]}).latent_std`, context),
    undefined, 'missing diversity must not silently exclude an evaluated level');
  assert.equal(nodes.get('world-table').children.length, 3, 'per-level detail remains available in the evaluation table');
  assert.equal(nodes.get('behavior-table').children.length, 1); // explicit empty row
  assert.match(nodes.get('behavior-table').children[0].children[0].textContent, /No evaluations/);
  // Tabs preserve the same selected execution and remain keyboard-operable native buttons.
  nodes.get('tab-behavior').listeners.click();
  assert.equal(nodes.get('panel-world').hidden, true);
  assert.equal(nodes.get('panel-behavior').hidden, false);
  assert.equal(text('actions'), '10');
  assert.equal(nodes.get('tab-behavior')['aria-selected'], 'true');
  // Inner evaluation events give provisional counters, never fabricated training losses.
  data.campaign.progress.stage = 'behavior_evaluation';
  data.campaign.progress.evaluation = {level: 1, episodes: 16, wins: 7, split: 'dev', policy: 'value_mpc'};
  data.campaign.source.age_seconds = 177;
  await vm.runInContext('poll()', context);
  assert.equal(text('actions'), '10', 'evaluation does not increment training actions');
  assert.equal(text('behavior-win'), '43.75 %');
  assert.match(text('behavior-wins'), /7\/16.*partial/);
  assert.match(text('behavior-status'), /progress by episode blocks/);
  assert.doesNotMatch(text('connection'), /no recent data/);
  assert.equal(nodes.get('connection').className, '');
  assert.equal(nodes.get('behavior-loss-card').hidden, true);
  assert.equal(nodes.get('behavior-actor-card').hidden, true);
  assert.equal(nodes.get('behavior-wins-card').hidden, true);
  assert.match(text('speed-note'), /no training TPS/);
  Object.assign(data.campaign.progress.evaluation, {current_episode: 17, total_episodes: 32,
    episode_actions: 24, episode_budget: 128, seconds: 194, game: 'si01-v1', pair_index: 6, pair_count: 10});
  data.campaign.source.age_seconds = 1;
  await vm.runInContext('poll()', context);
  assert.equal(nodes.get('evaluation-progress').hidden, false);
  assert.equal(text('evaluation-progress'),
    'si01-v1 · Level 1 · block 6/10 · Episode 17/32 · 24/128 actions · 16 completed episodes · 7 wins · 194 s in this block (last record)');
  assert.match(text('connection'), /progress per action/);
  assert.match(text('behavior-status'), /every 2 s/);
  assert.equal(text('game'), 'si01-v1');
  data.campaign.source.age_seconds = 70;
  await vm.runInContext('poll()', context);
  assert.match(text('connection'), /no recent data/);
  assert.equal(nodes.get('connection').className, 'warn');
  const old = nodes.get('execution-select').options.find(o => o.textContent.includes('world_reference'));
  ui.change('execution-select', old.value);
  assert.equal(text('actions'), '100', 'global selection changes counters as well as curves');
  assert.equal(text('updates'), '50');
  assert.equal(text('prediction'), '50 %');
  assert.equal(text('behavior-win'), '25 %');
  assert.match(text('connection'), /Previous record/);
  assert.equal(nodes.get('evaluation-progress').hidden, true);
  assert.equal(vm.runInContext('charts.latent.series[0].points[0].y', context), .7);
  assert.equal(nodes.get('world-table').children.length, 1);
  assert.equal(nodes.get('behavior-table').children.length, 1);
  // Explicit selection stays fixed when a new seed starts; auto follows it with no previous metrics.
  data.campaign.progress = {variant: 'rollout2', seed: 1, actions: 0, stage: 'initializing'};
  await vm.runInContext('poll()', context);
  assert.equal(text('actions'), '100');
  ui.change('execution-select', 'current');
  assert.equal(text('actions'), '0');
  assert.equal(text('prediction'), '—');
  assert.equal(text('behavior-win'), '—');
  assert.equal(vm.runInContext('charts.latent.series[0].points.length', context), 0);
  assert.match(text('connection'), /Initializing/);
  data.campaign.phase = 'campaign_error'; data.campaign.reason = 'expected failure';
  await vm.runInContext('poll()', context);
  assert.match(text('notice'), /expected failure/);
  assert.equal(nodes.get('notice').hidden, false);
  data.campaign.report = {learning_probe: {available: true, status: 'running'}};
  await vm.runInContext('poll()', context);
  assert.equal(nodes.get('experiment-notice').hidden, false);
  assert.match(text('experiment-summary'), /Learning trial in progress/);
  assert.match(text('notice'), /expected failure/, 'an active diagnostic does not hide the old failure');
  data.campaign.report.learning_probe.status = 'complete';
  await vm.runInContext('poll()', context);
  assert.equal(nodes.get('experiment-notice').hidden, true);
  ui.change('execution-select', old.value);
  assert.equal(nodes.get('notice').hidden, true, 'a different seed failure is not this historical run failure');
  assert.equal(text('actions'), '100');
  const world = nodes.get('execution-select').options.find(o => o.textContent.includes('old-world'));
  ui.change('execution-select', world.value);
  assert.equal(text('actions'), '900');
  assert.equal(text('resources'),
    'Replay memory: 0 episodes · Trained transitions/s: 1,234.5 · Consecutive validation passes: 0');
  assert.equal(nodes.get('spread-legend').children.at(-1).textContent, 'Threshold: 0.125');
  assert.equal(text('prediction'), '80 %');
  assert.equal(text('behavior-win'), '—');
  assert.doesNotMatch(text('notice'), /expected failure/, 'another run failure must not leak');
  const behavior = nodes.get('execution-select').options.find(o => o.textContent.includes('old-behavior'));
  ui.change('execution-select', behavior.value);
  assert.equal(text('actions'), '500');
  assert.equal(text('prediction'), '—');
  assert.equal(vm.runInContext('charts.delta.series.length', context), 0);
  assert.equal(text('behavior-win'), '37.5 %', 'latest aggregate learned policy, not per-level or initial');
  assert.match(text('behavior-wins'), /12\/32.*Final/);
  assert.equal(text('behavior-random'), '12.5 %', 'baseline uses same phase and level');
  assert.equal(nodes.get('behavior-table').children.length, 5, 'phases and levels remain explicitly labeled in table');
  assert.equal(nodes.get('behavior-table').children[2].children[2].textContent, 'Initial · Q');
  assert.equal(vm.runInContext('charts.behaviorLoss.series[0].points[0].y', context), .2);
  // Pending evaluation cannot borrow a final/random/initial/per-level result.
  data.behavior.source.updated_at = 400; // this run is now emitting new records
  data.behavior.phase = 'behavior_evaluation_progress';
  data.behavior.progress.evaluation = {split: 'development', policy: 'q', episodes: 16, wins: 4};
  data.behavior.evaluations = [evaluation('final', 'random', null, 2, 16), evaluation('final', 'q', null, 0, 16, 'initial')];
  await vm.runInContext('poll()', context);
  assert.equal(text('behavior-win'), 'In progress');
  assert.equal(text('behavior-random'), '—');
  assert.equal(text('behavior-ms'), '—');
  data.behavior.phase = 'behavior_fit';
  await vm.runInContext('poll()', context);
  assert.equal(text('behavior-win'), 'Pending');
  data.behavior.phase = 'behavior_finished';
  await vm.runInContext('poll()', context);
  assert.equal(text('behavior-win'), '—');
  data.behavior.source.available = false;
  await vm.runInContext('poll()', context);
  assert.match(text('connection'), /Log unavailable/);
  // Bounded history must not replace an explicit unavailable selection by the live run.
  delete data.behavior;
  await vm.runInContext('poll()', context);
  assert.match(text('connection'), /Record unavailable/);
  assert.equal(text('actions'), '—');
  // Current-source fallback without a campaign is still a complete world-only dashboard.
  const onlyWorld = {...data}; delete onlyWorld.campaign;
  onlyWorld.validations = [
    {actions: 100, level: 1, transitions: 1, latent_std: .2,
      extra_changes: 2, missed_changes: 3, wrong_replacement_colors: 4, correct_changes: 5},
    {actions: 100, level: 2, transitions: 3, latent_std: .6,
      extra_changes: 10, missed_changes: 20, wrong_replacement_colors: 30, correct_changes: 40},
    {actions: 200, level: 1, transitions: 1, latent_std: .8,
      extra_changes: 1, missed_changes: 2, wrong_replacement_colors: 3, correct_changes: 4}
  ];
  const single = await dashboard(onlyWorld);
  assert.equal(single.text('actions'), '900');
  assert.match(single.text('connection'), /stopped/i);
  assert.match(single.text('notice'), /interruption/);
  assert.equal(vm.runInContext('JSON.stringify(charts.changes.series.map(s => s.points.map(p => p.y)))', single.context),
    '[[12,1],[23,2],[34,3],[45,4]]', 'legacy per-level records aggregate only within each evaluation');
  assert.equal(vm.runInContext('charts.spread.series[0].points.length', single.context), 2);
  assert.ok(Math.abs(vm.runInContext('charts.spread.series[0].points[0].y', single.context) - .5) < 1e-12);
  assert.equal(vm.runInContext('charts.spread.series[0].points[1].y', single.context), .8);
  vm.runInContext("charts.speed.set([{name:'Prueba',points:[{x:0,y:0},{x:1,y:10},{x:2,y:20}]}])",single.context);
  const speed=single.nodes.get('speed'),button=single.nodes.get('speed-legend').children[0];
  assert(speed.draws.some(d=>d[0]==='set'&&d[1]==='lineWidth'&&d[2]===1.8));
  button.listeners.click();
  assert(speed.draws.some(d=>d[0]==='fillText'&&d[1]==='2.9'&&d[2]===8),'latest EMA value');
  vm.runInContext("charts.speed.set([{name:'Prueba',points:[{x:0,y:20},{x:1,y:0}]}])",single.context);
  assert.equal(single.nodes.get('speed-legend').children[0],button);
  assert.equal(button['aria-pressed'],'true');
  assert(speed.draws.some(d=>d[0]==='fillText'&&d[1]==='18'&&d[2]===8));
  button.listeners.click();
  assert(!speed.draws.some(d=>d[0]==='strokeRect'));
  single.context.fetch = async () => { throw Error('offline'); };
  await vm.runInContext('poll()', single.context);
  assert.equal(single.text('connection'), 'Disconnected · retrying');
  assert.equal(single.text('freshness'), 'Could not refresh data');
  single.context.fetch = async () => ({ok: true, json: async () => onlyWorld});
  await vm.runInContext('poll()', single.context);
  assert.equal(single.text('connection'), 'Training stopped');
  assert.equal(single.text('actions'), '900', 'reconnection preserves the selected run');
  console.log('Single execution selection: counters, charts, tables, tabs, source/seed isolation and pending evaluations passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
