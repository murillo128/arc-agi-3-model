// Campaign page: actual poll/render with literal summaries, no browser or visual tests.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assets = path.join(__dirname, '../dashboard_web');
const html = fs.readFileSync(path.join(assets, 'campaign.html'), 'utf8');
const script = fs.readFileSync(path.join(assets, 'campaign.js'), 'utf8');
function element() {
  return {textContent: '', hidden: false, children: [], className: '', style:{}, listeners:{},
    addEventListener(key,handler){this.listeners[key]=handler;},
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
    setAttribute(key, value) { this[key] = value; }};
}
const nodes = new Map([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1], element()]));
const report = {available: true, status: 'running', selected: null,
  config: {variants: ['world_reference', 'rollout2', 'actor_imagined'], seeds: [0, 1], actions: 100, dev_episodes: 32},
  cells: [{variant: 'world_reference', seed: 0, world: {delta_error: .2, decoder_delta_error: 0},
    behavior: {wins: 2, episodes: 4, win_rate: .5, actions: 20, decision_ms: 3}}]};
const data = {source: {}, campaign: {source: {available: true, age_seconds: 2}, phase: 'campaign_progress',
  progress: {variant: 'world_reference', seed: 1, actions: 40, total_actions: 100, stage: 'world'}, report}};
let offline = false;
const context = vm.createContext({console, AbortSignal, setTimeout: () => {},
  document: {getElementById: id => nodes.get(id), createElement: element, createElementNS: element},
  fetch: async () => { if (offline) throw new Error('offline'); return {ok: true, json: async () => data}; }});
const text = id => nodes.get(id).textContent;
const row = n => nodes.get('campaign-table').children[n].children;
const bars = id => nodes.get(id).children.slice(1); // axis followed by measured variants
(async () => {
  vm.runInContext(fs.readFileSync(path.join(assets, 'chart-tools.js'), 'utf8'), context);
  vm.runInContext(script, context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(text('campaign-completed'), '1 / 6');
  assert.equal(text('campaign-candidate'), 'Pending', 'no early winner from a partial seed');
  assert.equal(nodes.get('campaign-table').children.length, 3, 'queued variants remain visible');
  assert.equal(row(0)[3].textContent, '20 %');
  assert.equal(row(0)[4].textContent, '0 %');
  assert.equal(row(0)[6].textContent, '10');
  assert.match(row(0)[1].children[1].textContent, /S1: Training predictor.*40/);
  assert.equal(row(1)[3].textContent, '—', 'no borrowing reference metrics for pending variants');
  assert.equal(nodes.get('campaign-charts').hidden, false);
  assert.equal(bars('campaign-success-chart').length, 1, 'pending variants have no zero bars');
  assert.match(bars('campaign-success-chart')[0].children[0].textContent, /1\/2 seeds.*partial/);
  assert.equal(bars('campaign-success-chart')[0].children[1].children[1].value, 50);
  assert.equal(bars('campaign-success-chart')[0].children[1].children[1].max, 100);
  assert.equal(bars('campaign-error-chart')[0].children[2].children[1].value, 0, 'true zero decoder error remains measurable');
  assert.equal(bars('campaign-actions-chart')[0].children[1].children[1].value, 10);
  assert.equal(vm.runInContext('actionsPerWin([{behavior: {wins: 0, actions: 20}}])', context), null);
  report.cells.push({variant: 'world_reference', seed: 1, world: {delta_error: .4, decoder_delta_error: 0},
    behavior: {wins: 1, episodes: 4, win_rate: .25, actions: 40, decision_ms: 5}});
  report.selected = 'world_reference';
  data.campaign.progress = {variant: 'rollout2', seed: 0, actions: 0, stage: 'initializing'};
  await vm.runInContext('pollCampaign()', context);
  assert.equal(text('campaign-completed'), '2 / 6');
  assert.equal(row(0)[2].textContent, 'Evaluated');
  assert.equal(row(0)[3].textContent, '30 %');
  assert.equal(row(0)[5].textContent, '37.5 %');
  assert.equal(row(0)[6].textContent, '20', 'ratio of summed evaluation actions and wins');
  assert.equal(row(0)[7].textContent, '4');
  assert.equal(bars('campaign-success-chart')[0].children[1].children[1].value, 37.5);
  assert.equal(bars('campaign-actions-chart')[0].children[1].children[1].value, 20);
  assert.equal(bars('campaign-latency-chart')[0].children[1].children[1].value, 4);
  assert.doesNotMatch(bars('campaign-success-chart')[0].children[0].textContent, /partial/);
  report.controlled_comparisons = {rollout2: {control: 'world_reference', success_delta: .05,
    paired95: [-.01, .11], reason: 'inconclusive_or_inferior', promote: false}};
  report.deferred_cells = [{variant: 'actor_imagined', seed: 0}];
  await vm.runInContext('pollCampaign()', context);
  assert.match(row(1)[8].textContent, /5 pp.*\n95% CI: -1 to 11 pp/);
  assert.match(row(1)[8].textContent, /No conclusive improvement/);
  assert.equal(row(2)[2].textContent, 'Deferred');
  assert.equal(nodes.get('campaign-final').hidden, true);
  report.status = 'paused';
  report.diagnostics = {available: true, status: 'running', age_seconds: 1,
    progress: {checkpoint: 'reference', phase: 'behavior', policy: 'q', game: 'sc01-v1', level: 1,
      current_episode: 2, total_episodes: 8, episode_actions: 10, episode_budget: 128},
    results: [{checkpoint: 'reference', policies: {random: {complete: false, summary: {wins: 0, episodes: 8, win_rate: 0}}},
      probes: [{game: 'sc01-v1', level: 1, predictor: {delta_error: 0}, no_change: {delta_error: 1},
        distinct_pairs: 0, distinct_correct: {delta_error: null}, distinct_wrong: {delta_error: null}}]}]};
  await vm.runInContext('pollCampaign()', context);
  assert.match(text('campaign-connection'), /Frozen-weight diagnostic.*in progress/);
  assert.equal(text('campaign-current'), 'Paused');
  assert.equal(nodes.get('diagnostics').hidden, false);
  assert.match(text('diagnostic-status'), /episode 2\/8.*10\/128/);
  assert.equal(nodes.get('diagnostic-policies').children[0].children[3].textContent, '0 %');
  assert.equal(nodes.get('diagnostic-policies').children[0].children[6].textContent, 'Partial');
  assert.equal(nodes.get('diagnostic-actions').children[0].children[2].textContent, '0 %');
  assert.equal(nodes.get('diagnostic-actions').children[0].children[4].textContent, '—');
  report.status = 'failed'; report.error = 'compile failure';
  report.learning_probe = {available: true, status: 'running', age_seconds: 2,
    progress: {variant: 'visual_2step', phase: 'training', updates: 20, total_updates: 120},
    coverage: {per_pair: [{game: 'example', level: 1, episodes: 3, actions: 8, successes: 1, failures: 2, budget_ends: 0}]},
    results: [{variant: 'visual_2step', points: [{updates: 0, fit_1: .5, fit_2: .8, development_1: .6, development_2: .9},
      {updates: 20, fit_1: 0, fit_2: .1, development_1: .3, development_2: .5}]}]};
  await vm.runInContext('pollCampaign()', context);
  assert.match(text('campaign-connection'), /Learning trial.*in progress/);
  assert.match(text('campaign-notice'), /Comparison campaign.*stopped.*compile failure/);
  assert.match(text('probe-status'), /20 \/ 120/);
  assert.equal(nodes.get('learning-probe').hidden, false);
  assert.equal(nodes.get('probe-coverage').children[0].children[3].textContent, '1');
  assert.equal(nodes.get('probe-chart-1').children[0].children.length, 2, 'two named series in the interactive legend');
  assert.equal(nodes.get('probe-chart-1').children[1].role, 'img');
  const probeLegend=nodes.get('probe-chart-1').children[0], probeSvg=nodes.get('probe-chart-1').children[1];
  const probeButton=probeLegend.children[0];
  assert.equal(probeSvg.children.filter(n=>n.class==='series-ema').length,2);
  probeButton.listeners.click();
  const probeGuide=()=>probeSvg.children.find(n=>n.class==='series-guide');
  assert.equal(probeGuide().children[3].textContent,'0 %','guide uses actual last fit error, not EMA');
  await vm.runInContext('pollCampaign()',context);
  assert.equal(probeLegend.children[0],probeButton);
  assert.equal(probeButton['aria-pressed'],'true');
  probeButton.listeners.click();
  assert.equal(probeGuide().children.length,0);
  report.learning_probe.scope = 'streaming_real_training_probe';
  report.learning_probe.progress = {phase: 'behavior', policy: 'q', game: 'example', level: 1,
    current_episode: 2, total_episodes: 4, episode_actions: 17, episode_budget: 128};
  report.learning_probe.results[0].best = {actions: 1024, selection_error: .2};
  report.learning_probe.results[0].points.forEach((p, i) => p.actions = i * 1024);
  report.learning_probe.selected = {variant: 'visual_2step', actions: 1024};
  report.learning_probe.calibration = {records: 4096, successes: 8, positive_records: 30, censored_records: 3};
  report.learning_probe.policies = [{policy: 'q', complete: false, wins: 0, episodes: 2, win_rate: 0}];
  report.learning_probe.measurements = [{label: 'visual_2step', game: 'example', level: 1, horizon: 2,
    changed_samples: 0, changed_error: null, unchanged_samples: 4, unchanged_pixels: 1.5}];
  await vm.runInContext('pollCampaign()', context);
  assert.match(text('probe-status'), /episode 2\/4.*17\/128/);
  assert.equal(nodes.get('probe-selection').hidden, false);
  assert.match(text('probe-selected'), /1,024/);
  assert.equal(nodes.get('probe-policies').children[0].children[2].textContent, '0 %');
  assert.equal(nodes.get('probe-measurements').children[0].children[4].textContent, '—');
  assert.match(text('probe-calibration'), /4,096 real transitions.*8 successes/);
  report.learning_probe.age_seconds = 200;
  await vm.runInContext('pollCampaign()', context);
  assert.match(text('campaign-connection'), /no recent data/);
  report.learning_probe.status = 'complete';
  report.learning_probe.coverage.actions = 4096;
  await vm.runInContext('pollCampaign()', context);
  assert.doesNotMatch(text('campaign-connection'), /no recent data/);
  assert.match(text('probe-status'), /Trial finished.*4,096 new actions/);
  report.status = 'complete';
  report.final = {reference: {win_rate: .5, wins: 2, episodes: 4}, selected: {win_rate: .75, wins: 3, episodes: 4}};
  await vm.runInContext('pollCampaign()', context);
  assert.equal(text('campaign-confirmation'), 'Available');
  assert.equal(nodes.get('campaign-final').hidden, false);
  assert.equal(nodes.get('campaign-final-table').children.length, 2);
  offline = true;
  await vm.runInContext('pollCampaign()', context);
  assert.equal(text('campaign-connection'), 'Disconnected · retrying');
  assert.equal(nodes.get('campaign-notice').hidden, false);
  offline = false; data.campaign.report = {available: false};
  await vm.runInContext('pollCampaign()', context);
  assert.equal(text('campaign-completed'), '—');
  assert.equal(nodes.get('campaign-final').hidden, true);
  assert.match(text('campaign-connection'), /Report unavailable/);
  assert.equal(nodes.get('campaign-charts').hidden, true);
  assert.equal(nodes.get('learning-probe').hidden, true);
  assert.equal(bars('campaign-success-chart').length, 0, 'unavailable report clears stale chart data');
  console.log('Campaign partial/completed/deferred results, comparisons, confirmation and refresh passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
