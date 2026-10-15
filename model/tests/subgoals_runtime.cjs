// Actual JS polling/rendering against literal measurements; no browser tests.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assets = path.join(__dirname, '../dashboard_web');
const html = fs.readFileSync(path.join(assets, 'subgoals.html'), 'utf8');
function element() {
  return {textContent:'', hidden:false, children:[], listeners:{},
    append(...items) {this.children.push(...items);},
    replaceChildren(...items) {this.children=items;},
    setAttribute(k,v) {this[k]=v;},
    addEventListener(k,v) {this.listeners[k]=v;},
    getContext() {return {clearRect(){}, fillRect(){}};}};
}
const nodes = new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],element()]));
const report = {status:'running', age_seconds:1, config:{seed:0,predictor_error:.1},
  progress:{phase:'training',actions:32,total_actions:65536,predictor_updates:1,path_updates:0,tps:2},
  curves:[{actions:1,predictor_loss:2,path_loss:null,tps:2,counts:{wrong:5,union:10,invented:2,missed:3}},
    {actions:32,predictor_loss:null,path_loss:0,tps:2,counts:{wrong:15,union:90,invented:10,missed:5}}],
  episodes:[{end_action:32,success:true,segments:[{success:true,actions:3,minimum_actions:2,reward:-1}]}],
  evaluations:[]};
let offline=false;
const context=vm.createContext({console,AbortSignal,setTimeout(){},
  fixture:report,
  document:{getElementById:id=>nodes.get(id),createElement:element,createElementNS:element},
  fetch:async()=>{if(offline)throw Error('offline');return {ok:true,json:async()=>({subgoals:report})};}});
const text=id=>nodes.get(id).textContent;
(async()=>{
  vm.runInContext(fs.readFileSync(path.join(assets,'subgoals.js'),'utf8'),context);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(text('actions'),'32');
  assert.equal(text('world-updates'),'1');
  assert.equal(text('path-updates'),'0');
  assert.equal(text('tps'),'2');
  // Aggregate pixel counts, not mean of the two error percentages (50%, 16.7%).
  assert.equal(vm.runInContext('blocks(fixture.curves)[0].error',context),20);
  assert.equal(vm.runInContext('blocks(fixture.curves)[0].count.invented',context),12);
  assert.equal(nodes.get('changes-legend').children.length,4,'one aggregate series per error kind');
  const circles=id=>nodes.get(id).children.filter(n=>n.r===2);
  assert.equal(circles('loss').length,1,'missing loss is not a fabricated zero');
  assert.match(circles('path-loss')[0].children[0].textContent,/Coste restante: 0/,'a genuine zero is retained');
  assert.equal(text('subgoals'),'1 / 1');
  nodes.get('tab-path').listeners.click();
  assert.equal(nodes.get('world').hidden,true);
  assert.equal(nodes.get('path').hidden,false);
  report.progress.phase='evaluation';
  Object.assign(report.progress,{policy:'path_mpc',evaluation_game:'si01-v1',evaluation_level:2,
    evaluation_episode:1,evaluation_episodes:2,evaluation_action:7});
  report.evaluations.push({actions:32,split:'dev',policy:'path_mpc',game:'si01-v1',level:2,
    successes:1,episodes:2,subgoals:3,attempted_subgoals:4,delta_error:.2,real_actions:50,decision_ms:6});
  await vm.runInContext('poll()',context);
  assert.equal(text('actions'),'32','evaluation does not advance the training counter');
  assert.match(text('progress'),/partida 1\/2 · acción 7/);
  assert.equal(nodes.get('evaluation-rows').children[0].children[6].textContent,'20 %');
  report.curves=[];report.episodes=[];report.evaluations=[];report.progress={phase:'loading',actions:0};
  await vm.runInContext('poll()',context);
  assert.equal(circles('delta').length,0,'no points borrowed from a previous run');
  assert.equal(nodes.get('evaluation-rows').children.length,0);
  offline=true;await vm.runInContext('poll()',context);
  assert.match(text('connection'),/Sin conexión.*offline/);
  console.log('Subgoal dashboard contracts passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
