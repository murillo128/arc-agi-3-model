// Data/render contract only: no browser, screenshots or visual assertions.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assets = path.join(__dirname, '../dashboard_web');
function element() {
  return {textContent: '', style: {}, children: [], dataset: {}, listeners: {},
    addEventListener(key, handler) { this.listeners[key]=handler; },
    focus() {},
    getContext() { return this.context??={pixels:[],clearRect(){this.pixels=[];},fillRect(x,y){this.pixels.push({x,y,color:this.fillStyle});}}; },
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
    setAttribute(key, value) { this[key] = value; }};
}
const nodes = new Map([...fs.readFileSync(path.join(assets, 'sequences.html'), 'utf8')
  .matchAll(/id="([^"]+)"/g)].map(m => [m[1], element()]));
const report = {status: 'paused', age_seconds: 2, games: ['<game>'],
  config: {batch_games: 4, session_steps: 512, block_steps: 16, context_steps: 16,
    loss_start_weight: .25, loss_warmup_steps: 32}, progress: {actions: 2048, batches: 1},
  history: [{batch: 1, recent_tps: 40, examples_per_second: 80, peak_gpu_gib: 4.2,
    macro_delta_error: .25, loss: 0, quality: {early: {delta_error: 0}, late: {delta_error: .5},
      '<game>': {delta_error: .25, transitions: 512}},
    sessions: [{game: '<game>', steps: 512, boundaries: 15, end: 'budget'}]}]};
let offline = false;
const context = vm.createContext({console, setTimeout() {}, fixture: report,
  document: {getElementById: id => nodes.get(id), createElement: element, createElementNS: element},
  fetch: async () => { if (offline) throw Error('offline'); return {ok: true, json: async () => ({sequences: report, history_available: true})}; }});
const text = id => nodes.get(id).textContent;
(async () => {
  vm.runInContext(fs.readFileSync(path.join(assets, 'sequences.js'), 'utf8'), context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(text('status'), 'Pausado');
  assert.equal(text('tps'), '40 acciones/s');
  assert.equal(text('examples'), '80 transiciones/s');
  assert.match(text('configuration'), /16 pasos \+ 16 de contexto/);
  assert.equal(nodes.get('quality-legend').children[0].textContent, 'Inicio %: 0');
  assert.equal(nodes.get('loss-legend').children[0].textContent, 'Loss: 0');
  assert.equal(text('macro-error'), 'Media por juego: 25 %');
  assert.equal(nodes.get('sessions').children[0].children[1].textContent, '<game>');
  assert.equal(nodes.get('game-quality').children[0].children[1].textContent, '25 %');
  assert.equal(nodes.get('history-link').hidden,false);
  nodes.get('tab-path').listeners.click();
  assert.equal(nodes.get('path').hidden,false);
  assert.equal(nodes.get('world').hidden,true);
  assert.equal(nodes.get('tab-path')['aria-selected'],'true');
  assert.equal(nodes.get('planner-history-link').hidden,false);
  report.history[0].predictor_terms={latent:.25,visual:.75};
  report.history[0].reconstruction_loss=0;
  report.history[0].representation_terms={color_ce:0};
  report.history[0].quality.all={delta_error:.5,transitions:2,invented:2,missed:3,color:0,correct:5};
  report.history[0].diagnostics={batch:1,every:4,sampled_transitions:2,rollouts:1,drift_reference_batch:1,
    reconstruction_error:0,rollout_error:.5,latent_drift:null,latent:{std_mean:0,within_episode_std:0,
      effective_rank:0,max_rank:1,inactive_fraction:1,changed_rms:.5,unchanged_rms:null}};
  report.prediction_sample_batch=1;
  report.prediction_samples=[{game:'<game>',level:1,step:0,action:4,previous:[[5,5,5,5]],
    actual:[[9,9,9,5]],predicted:[[9,5,14,14]],reconstruction:[[5,5,5,5]]}];
  await vm.runInContext('refresh()',context);
  assert.equal(nodes.get('path').hidden,false,'polling keeps the selected tab');
  assert.equal(nodes.get('predictor-components-legend').children[1].textContent,'Pérdida visual: 0,75');
  assert.equal(nodes.get('representation-loss-legend').children[0].textContent,'Loss encoder / decoder: 0');
  assert.equal(nodes.get('changes-legend').children[0].textContent,'Inventados: 1');
  assert.equal(nodes.get('changes-legend').children[1].textContent,'Omitidos: 1,5');
  assert.equal(nodes.get('latent-diversity-legend').children[0].textContent,'Entre observaciones: 0');
  assert.equal(nodes.get('latent-drift').children[0].textContent,'Sin mediciones todavía');
  assert.equal(nodes.get('evaluations').children[0].children[3].textContent,'0 %');
  assert.equal(nodes.get('preview-images').hidden,false);
  assert.deepEqual(nodes.get('preview-legend').children.map(n=>n.textContent),
    ['Inventados: 1 px','Omitidos: 1 px','Color incorrecto: 1 px','Acertados: 1 px']);
  assert.deepEqual(nodes.get('preview-real').context.pixels.map(p=>p.color),
    ['#1e93ff','#1e93ff','#1e93ff','#aaa']);
  report.prediction_samples=[];
  report.history = [];
  await vm.runInContext('refresh()', context);
  assert.equal(text('tps'), '— acciones/s');
  assert.equal(nodes.get('game-quality').children[0].children[1].textContent, '— %');
  assert.equal(nodes.get('sessions').children.length, 0);
  assert.equal(nodes.get('quality').children[0].textContent, 'Sin mediciones todavía');
  assert.equal(nodes.get('preview-images').hidden,true);
  assert.equal(nodes.get('preview-before').context.pixels.length,0);
  offline = true;
  await vm.runInContext('refresh()', context);
  assert.equal(text('status'), 'Sin conexión');
})().catch(error => { console.error(error); process.exitCode = 1; });
