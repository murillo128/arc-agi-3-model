// Campaign page: actual poll/render with literal summaries, no browser or visual tests.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assets = path.join(__dirname, '../dashboard_web');
const html = fs.readFileSync(path.join(assets, 'campaign.html'), 'utf8');
const script = fs.readFileSync(path.join(assets, 'campaign.js'), 'utf8');
function element() {
  return {textContent: '', hidden: false, children: [], className: '',
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
  document: {getElementById: id => nodes.get(id), createElement: element},
  fetch: async () => { if (offline) throw new Error('offline'); return {ok: true, json: async () => data}; }});
const text = id => nodes.get(id).textContent;
const row = n => nodes.get('campaign-table').children[n].children;
const bars = id => nodes.get(id).children.slice(1); // axis followed by measured variants
(async () => {
  vm.runInContext(script, context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(text('campaign-completed'), '1 / 6');
  assert.equal(text('campaign-candidate'), 'Pendiente', 'no early winner from a partial seed');
  assert.equal(nodes.get('campaign-table').children.length, 3, 'queued variants remain visible');
  assert.equal(row(0)[3].textContent, '20 %');
  assert.equal(row(0)[4].textContent, '0 %');
  assert.equal(row(0)[6].textContent, '10');
  assert.match(row(0)[1].children[1].textContent, /S1: Entrenando predictor.*40/);
  assert.equal(row(1)[3].textContent, '—', 'no borrowing reference metrics for pending variants');
  assert.equal(nodes.get('campaign-charts').hidden, false);
  assert.equal(bars('campaign-success-chart').length, 1, 'pending variants have no zero bars');
  assert.match(bars('campaign-success-chart')[0].children[0].textContent, /1\/2 semillas.*parcial/);
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
  assert.equal(row(0)[2].textContent, 'Evaluada');
  assert.equal(row(0)[3].textContent, '30 %');
  assert.equal(row(0)[5].textContent, '37,5 %');
  assert.equal(row(0)[6].textContent, '20', 'ratio of summed evaluation actions and wins');
  assert.equal(row(0)[7].textContent, '4');
  assert.equal(bars('campaign-success-chart')[0].children[1].children[1].value, 37.5);
  assert.equal(bars('campaign-actions-chart')[0].children[1].children[1].value, 20);
  assert.equal(bars('campaign-latency-chart')[0].children[1].children[1].value, 4);
  assert.doesNotMatch(bars('campaign-success-chart')[0].children[0].textContent, /parcial/);
  report.controlled_comparisons = {rollout2: {control: 'world_reference', success_delta: .05,
    paired95: [-.01, .11], reason: 'inconclusive_or_inferior', promote: false}};
  report.deferred_cells = [{variant: 'actor_imagined', seed: 0}];
  await vm.runInContext('pollCampaign()', context);
  assert.match(row(1)[8].textContent, /5 pp.*\nIC 95 %: -1 a 11 pp/);
  assert.match(row(1)[8].textContent, /Sin mejora concluyente/);
  assert.equal(row(2)[2].textContent, 'Aplazada');
  assert.equal(nodes.get('campaign-final').hidden, true);
  report.status = 'paused';
  report.diagnostics = {available: true, status: 'running', age_seconds: 1,
    progress: {checkpoint: 'reference', phase: 'behavior', policy: 'q', game: 'sc01-v1', level: 1,
      current_episode: 2, total_episodes: 8, episode_actions: 10, episode_budget: 128},
    results: [{checkpoint: 'reference', policies: {random: {complete: false, summary: {wins: 0, episodes: 8, win_rate: 0}}},
      probes: [{game: 'sc01-v1', level: 1, predictor: {delta_error: 0}, no_change: {delta_error: 1},
        distinct_pairs: 0, distinct_correct: {delta_error: null}, distinct_wrong: {delta_error: null}}]}]};
  await vm.runInContext('pollCampaign()', context);
  assert.match(text('campaign-connection'), /pausada/);
  assert.equal(nodes.get('diagnostics').hidden, false);
  assert.match(text('diagnostic-status'), /partida 2\/8.*10\/128/);
  assert.equal(nodes.get('diagnostic-policies').children[0].children[3].textContent, '0 %');
  assert.equal(nodes.get('diagnostic-policies').children[0].children[6].textContent, 'Parcial');
  assert.equal(nodes.get('diagnostic-actions').children[0].children[2].textContent, '0 %');
  assert.equal(nodes.get('diagnostic-actions').children[0].children[4].textContent, '—');
  report.status = 'complete';
  report.final = {reference: {win_rate: .5, wins: 2, episodes: 4}, selected: {win_rate: .75, wins: 3, episodes: 4}};
  await vm.runInContext('pollCampaign()', context);
  assert.equal(text('campaign-confirmation'), 'Disponible');
  assert.equal(nodes.get('campaign-final').hidden, false);
  assert.equal(nodes.get('campaign-final-table').children.length, 2);
  offline = true;
  await vm.runInContext('pollCampaign()', context);
  assert.match(text('campaign-connection'), /Sin conexión/);
  assert.equal(nodes.get('campaign-notice').hidden, false);
  offline = false; data.campaign.report = {available: false};
  await vm.runInContext('pollCampaign()', context);
  assert.equal(text('campaign-completed'), '—');
  assert.equal(nodes.get('campaign-final').hidden, true);
  assert.match(text('campaign-connection'), /Informe no disponible/);
  assert.equal(nodes.get('campaign-charts').hidden, true);
  assert.equal(bars('campaign-success-chart').length, 0, 'unavailable report clears stale chart data');
  console.log('Campaign partial/completed/deferred results, comparisons, confirmation and refresh passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
