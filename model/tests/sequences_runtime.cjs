// Data/render contract only: no browser, screenshots or visual assertions.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assets = path.join(__dirname, '../dashboard_web');
function element() {
  return {textContent: '', style: {}, children: [],
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
  fetch: async () => { if (offline) throw Error('offline'); return {ok: true, json: async () => ({sequences: report})}; }});
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
  report.history = [];
  await vm.runInContext('refresh()', context);
  assert.equal(text('tps'), '— acciones/s');
  assert.equal(nodes.get('game-quality').children[0].children[1].textContent, '— %');
  assert.equal(nodes.get('sessions').children.length, 0);
  assert.equal(nodes.get('quality').children[0].textContent, 'Sin mediciones todavía');
  offline = true;
  await vm.runInContext('refresh()', context);
  assert.equal(text('status'), 'Sin conexión');
})().catch(error => { console.error(error); process.exitCode = 1; });
