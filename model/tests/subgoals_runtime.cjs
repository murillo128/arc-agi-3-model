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
    getContext() {return this.context??=( {pixels:[],clearRect(){this.pixels=[];},
      fillRect(x,y,w,h){this.pixels.push({x,y,color:this.fillStyle});}} );}};
}
const nodes = new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],element()]));
const report = {status:'running', age_seconds:1, config:{seed:0,predictor_error:.1,
    evaluation_pairs:[{game:'a',level:1},{game:'b',level:1}]},
  progress:{phase:'training',actions:32,total_actions:65536,predictor_updates:1,path_updates:0,tps:2},
  curves:[{actions:1,predictor_loss:2,path_loss:null,tps:2,counts:{wrong:5,union:10,invented:2,missed:3,color:0,correct:5}},
    {actions:32,predictor_loss:null,path_loss:0,tps:2,counts:{wrong:15,union:90,invented:10,missed:5,color:0,correct:75}}],
  episodes:[{game:'a',level:1,end_action:32,success:true,segments:[{success:true,actions:3,minimum_actions:2,reward:-1}]}],
  evaluations:[]};
let offline=false;
let gpu={devices:[{index:0,name:'Test GPU',utilization_percent:73,temperature_c:71}]};
const context=vm.createContext({console,AbortSignal,setTimeout(){},
  fixture:report,
  document:{getElementById:id=>nodes.get(id),createElement:element,createElementNS:element},
  fetch:async()=>{if(offline)throw Error('offline');return {ok:true,json:async()=>({subgoals:report,gpu})};}});
const text=id=>nodes.get(id).textContent;
(async()=>{
  vm.runInContext(fs.readFileSync(path.join(assets,'subgoals.js'),'utf8'),context);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(text('actions'),'32');
  assert.equal(text('world-updates'),'1');
  assert.equal(text('path-updates'),'0');
  assert.equal(text('tps'),'2');
  assert.equal(nodes.get('speed-legend').children[0].textContent,'Acciones/s: 2 acciones/s');
  assert.equal(nodes.get('path-loss-legend').children[0].textContent,'Media de actualizaciones · hasta 512 acciones: 0');
  assert.equal(nodes.get('loss-legend').children[1].textContent,'Actualización individual: 2');
  assert.match(nodes.get('loss-legend').children[1].title,/acción 1$/,'missing latest sample does not fabricate a zero');
  assert.equal(text('gpu-status'),'GPU · 73 % de uso · 71 °C');
  assert.match(nodes.get('gpu-status').title,/Test GPU/);
  gpu={devices:[{index:0,name:'Test GPU',utilization_percent:0,temperature_c:null},
    {index:1,name:'Second GPU',utilization_percent:null,temperature_c:65}]};
  await vm.runInContext('poll()',context);
  assert.equal(text('gpu-status'),'GPU 0 · 0 % de uso · — °C\nGPU 1 · — % de uso · 65 °C');
  gpu=undefined;
  await vm.runInContext('poll()',context);
  assert.equal(text('gpu-status'),'GPU · datos no disponibles','old backend data does not preserve stale readings');
  gpu={devices:[{index:0,name:'Test GPU',utilization_percent:88,temperature_c:72}]};
  await vm.runInContext('poll()',context);
  assert.equal(text('gpu-status'),'GPU · 88 % de uso · 72 °C');
  assert.equal(nodes.get('prediction-sample-images').hidden,true);
  assert.equal(nodes.get('latent-diversity-legend').children[0].textContent,'Entre observaciones: —');
  // One of each union category, plus unchanged pixels. Black (5) and white (0)
  // are replacement colors; only 16 means no change.
  report.prediction_sample={previous:[[9,5,5],[5,15,5]],real_delta:[[5,9,16],[0,16,16]],
    predicted_delta:[[5,16,14],[11,16,16]],wrong:3,union:4,captured_at:Date.now()/1000,
    game:'sc01-v1',level:1,episode_seed:123,episode_action:8,training_action:32,
    phase:'training',action:{id:4,data:{}}};
  await vm.runInContext('poll()',context);
  assert.equal(nodes.get('prediction-sample-images').hidden,false);
  assert.equal(nodes.get('prediction-reconstruction').hidden,true);
  assert.equal(nodes.get('prediction-reconstruction-empty').hidden,false);
  report.prediction_sample.reconstruction=[[9,5,5],[5,15,14]];
  await vm.runInContext('poll()',context);
  assert.equal(nodes.get('prediction-reconstruction').hidden,false);
  assert.equal(nodes.get('prediction-reconstruction-empty').hidden,true);
  assert.deepEqual(nodes.get('prediction-reconstruction').context.pixels.map(p=>p.color),
    ['#1e93ff','#000000','#000000','#000000','#a356d6','#4fcc30']);
  assert.equal(nodes.get('prediction-real').width,3);
  assert.equal(nodes.get('prediction-real').height,2);
  assert.deepEqual(nodes.get('prediction-real').context.pixels.map(p=>p.color),
    ['#000000','#1e93ff','#eee','#ffffff','#eee','#aaa']);
  assert.deepEqual(nodes.get('prediction-guessed').context.pixels.map(p=>p.color),
    ['#000000','#aaa','#4fcc30','#ffdc00','#eee','#aaa']);
  assert.equal(nodes.get('prediction-difference').width,3);
  assert.equal(nodes.get('prediction-difference').height,2);
  assert.deepEqual(nodes.get('prediction-difference').context.pixels,[
    {x:0,y:0,color:'#94a7aa'},{x:1,y:0,color:'#f4b979'},{x:2,y:0,color:'#77dabb'},
    {x:0,y:1,color:'#a9a6ef'},{x:1,y:1,color:'#eee'},{x:2,y:1,color:'#aaa'}]);
  assert.deepEqual(nodes.get('prediction-difference-legend').children.map(n=>[n.className,n.textContent]),[
    ['series-0','Inventados: 1 px'],['series-1','Omitidos: 1 px'],
    ['series-2','Color incorrecto: 1 px'],['series-3','Acertados: 1 px']]);
  assert.match(text('prediction-sample-info'),/sc01-v1 · nivel 1 · partida 123 · paso 8 · ACTION4/);
  assert.match(text('prediction-sample-error'),/75 % · 3 \/ 4/);
  report.prediction_sample={...report.prediction_sample,wrong:0,union:0,
    real_delta:[[16,16,16],[16,16,16]],predicted_delta:[[16,16,16],[16,16,16]],
    action:{id:6,data:{x:4,y:19}},phase:'evaluation'};
  await vm.runInContext('poll()',context);
  assert.match(text('prediction-sample-info'),/ACTION6 \(4, 19\) · Evaluación/);
  assert.match(text('prediction-sample-error'),/Sin cambios reales ni predichos/);
  assert.deepEqual(nodes.get('prediction-difference').context.pixels.map(p=>p.color),
    ['#eee','#aaa','#eee','#aaa','#eee','#aaa'],'no changes leaves the entire mask transparent');
  assert.ok(nodes.get('prediction-difference-legend').children.every(n=>n.textContent.endsWith(': 0 px')),
    'matching unchanged pixels do not count as correct changes');
  delete report.prediction_sample;
  await vm.runInContext('poll()',context);
  assert.equal(nodes.get('prediction-sample-images').hidden,true);
  assert.equal(nodes.get('prediction-real').context.pixels.length,0,'missing sample clears the old image');
  assert.equal(nodes.get('prediction-reconstruction').context.pixels.length,0,'missing sample clears the old reconstruction');
  assert.equal(nodes.get('prediction-difference').context.pixels.length,0,'missing sample clears the category mask');
  assert.equal(nodes.get('prediction-difference-legend').children.length,0,'missing sample clears category counts');
  // Aggregate pixel counts, not mean of the two error percentages (50%, 16.7%).
  assert.equal(vm.runInContext('blocks(fixture.curves)[0].error',context),20);
  assert.equal(vm.runInContext('blocks(fixture.curves)[0].count.invented',context),12);
  // No moving pixels is absence of delta evidence, not a measured 0% error.
  assert.equal(vm.runInContext('blocks([{actions:32,counts:{wrong:0,union:0,changed:0}}])[0].error',context),null);
  assert.equal(nodes.get('changes-legend').children.length,4,'one aggregate series per error kind');
  const circles=id=>nodes.get(id).children.filter(n=>n.r===2);
  // Partial blocks divide by actual measured actions, not 32. Both layers use
  // pixels/action, preserve each category's color and share one legend entry.
  const kinds=[['Inventados','#77dabb',6],['Omitidos','#f4b979',4],
    ['Color incorrecto','#a9a6ef',0],['Acertados','#94a7aa',40]];
  for(const [name,color,value] of kinds) {
    const points=circles('changes').filter(n=>n.fill===color);
    assert.equal(points.length,2,`${name} has raw and trend measurements`);
    assert.ok(points[0].opacity<points[1].opacity,'raw blocks are behind the stronger trend');
    for(const point of points)assert.ok(point.children[0].textContent.startsWith(`${name}: ${value} · acción 32`));
    assert.match(points[0].children[0].textContent,/px\/acción · bloque de 32/);
    assert.match(points[1].children[0].textContent,/px\/acción · media móvil de hasta 512/);
  }
  assert.equal(circles('loss').filter(n=>n.children[0].textContent.startsWith('Actualización individual:')).length,1,'missing loss is not a fabricated zero');
  assert.match(circles('loss')[0].children[0].textContent,/: 2 ·/,'mean includes only recorded updates');
  assert.match(circles('path-loss')[0].children[0].textContent,/: 0 ·/,'a genuine zero is retained');
  assert.equal(text('subgoals'),'1 / 1');
  nodes.get('tab-path').listeners.click();
  assert.equal(nodes.get('world').hidden,true);
  assert.equal(nodes.get('path').hidden,false);
  nodes.get('tab-curriculum').listeners.click();
  for(const tab of ['world','curriculum','path','evaluation']) {
    assert.equal(nodes.get(tab).hidden,tab!=='curriculum');
    assert.equal(nodes.get('tab-'+tab)['aria-selected'],String(tab==='curriculum'));
  }
  report.progress.phase='evaluation';
  Object.assign(report.progress,{policy:'path_mpc',evaluation_game:'si01-v1',evaluation_level:2,
    evaluation_episode:1,evaluation_episodes:2,evaluation_action:7});
  report.evaluations.push({actions:32,split:'dev',policy:'path_mpc',game:'si01-v1',level:2,
    successes:1,episodes:2,subgoals:3,attempted_subgoals:4,delta_error:.2,real_actions:50,decision_ms:6});
  await vm.runInContext('poll()',context);
  assert.equal(nodes.get('curriculum').hidden,false,'live polling preserves the selected tab');
  assert.equal(nodes.get('world').hidden,true);
  assert.equal(text('actions'),'32','evaluation does not advance the training counter');
  assert.match(text('progress'),/partida 1\/2 · acción 7/);
  assert.equal(nodes.get('evaluation-rows').children[0].children[6].textContent,'20 %');

  // Only the same complete, fixed-random development coverage is comparable.
  // Different pixel/action counts must not change each level's equal weight.
  const evaluation=(actions,game,error,extra={})=>({actions,split:'dev',policy:'random',game,level:1,
    episodes:2,delta_error:error,real_actions:10,...extra});
  report.evaluations=[evaluation(500,'a',.1),evaluation(500,'b',.5,{real_actions:1000}),
    evaluation(500,'a',1,{policy:'path_mpc'}),evaluation(1000,'a',.2),
    evaluation(2000,'a',.9,{split:'final'}),evaluation(2000,'b',.9,{split:'final'})];
  await vm.runInContext('poll()',context);
  assert.equal(circles('fixed-delta').length,1,'partial sweeps and new-seed final evaluations are not comparable points');
  assert.match(circles('fixed-delta')[0].children[0].textContent,/: 30 · acción 500 · 2 pares · 4 partidas/);
  assert.match(text('fixed-summary'),/En curso: 1\/2 pares/);
  report.evaluations.push(evaluation(1000,'b',.2));
  await vm.runInContext('poll()',context);
  assert.equal(circles('fixed-delta').length,2);
  assert.match(circles('fixed-delta')[1].children[0].textContent,/: 20 · acción 1000/);
  assert.doesNotMatch(text('fixed-summary'),/En curso/);
  // Returning to the quality chart fits its full panel immediately, preserving
  // the measured 20% point. A narrower panel changes geometry, not the data.
  Object.assign(nodes.get('fixed-delta'),{clientWidth:1200,clientHeight:300});
  nodes.get('tab-world').listeners.click();
  assert.equal(nodes.get('fixed-delta').viewBox,'0 0 1200 300');
  assert.equal(circles('fixed-delta').at(-1).cx,1178);
  assert.equal(circles('fixed-delta').at(-1).cy,212.8);
  assert.match(circles('fixed-delta').at(-1).children[0].textContent,/: 20 · acción 1000/);
  Object.assign(nodes.get('fixed-delta'),{clientWidth:360,clientHeight:240});
  await vm.runInContext('poll()',context);
  assert.equal(nodes.get('fixed-delta').viewBox,'0 0 360 240');
  assert.equal(circles('fixed-delta').at(-1).cx,338);
  assert.equal(circles('fixed-delta').at(-1).cy,164.8);
  assert.match(circles('fixed-delta').at(-1).children[0].textContent,/: 20 · acción 1000/);
  delete nodes.get('fixed-delta').clientWidth;delete nodes.get('fixed-delta').clientHeight;

  // Reports already being written by a running trainer have no pair metadata.
  delete report.config.evaluation_pairs;
  await vm.runInContext('poll()',context);
  assert.equal(circles('fixed-delta').length,2,'live legacy reports retain the fixed evaluation history');

  // Rolling ratios use summed pixels and exact action windows, not percentages
  // or a number of optimizer updates. A real zero loss remains a measurement.
  report.curves=[{actions:1,counts:{wrong:50,union:50},predictor_loss:16},
    {actions:512,counts:{wrong:0,union:50},predictor_loss:0},
    {actions:513,counts:{wrong:1,union:50},predictor_loss:null}];
  assert.equal(vm.runInContext('trend(fixture.curves,blocks(fixture.curves)).at(-1).error',context),1);
  assert.equal(vm.runInContext('trend(fixture.curves,blocks(fixture.curves)).at(-1).loss',context),0);

  // Representation and planner averages count measured updates, excluding
  // action 1 from the final 512-action window and retaining a real zero.
  report.curves=[{actions:1,representation_loss:100,path_loss:100},
    {actions:32,representation_loss:6,path_loss:8},
    {actions:33,representation_loss:null,path_loss:null},
    {actions:64,representation_loss:0,path_loss:null},
    {actions:512,representation_loss:2,path_loss:0},
    {actions:513,representation_loss:4,path_loss:4},
    {actions:514,representation_loss:null,path_loss:null}];
  await vm.runInContext('poll()',context);
  for(const [id,value,rawCount] of [['representation-loss',3,5],['path-loss',4,4]]) {
    const average=circles(id).filter(n=>n.opacity===1);
    assert.match(average.at(-1).children[0].textContent,new RegExp(`: ${value} · acción 514$`));
    assert.equal(circles(id).filter(n=>n.opacity<1).length,rawCount);
    assert.ok(circles(id).some(n=>n.opacity<1&&n.children[0].textContent.includes(': 0 ·')));
    assert.match(nodes.get(id+'-legend').children[0].textContent,new RegExp(`: ${value}$`));
  }
  report.curves=[{actions:1026,representation_loss:null,path_loss:null}];
  await vm.runInContext('poll()',context);
  for(const id of ['representation-loss','path-loss']) {
    assert.equal(circles(id).length,0,'no updates means no invented average');
    assert.match(nodes.get(id+'-legend').children[0].textContent,/: —$/);
  }

  // Loss contributions are already weighted. Display zeros, omit uncomputed
  // terms and clear old values when selecting a fresh run with no updates.
  report.curves=[{actions:3,representation_terms:{color_ce:1,delta_nll:2,delta_overlap:0,local_reconstruction:1.5},
    predictor_terms:{latent:.25,delta_nll:2,delta_overlap:.5}},
    {actions:4,representation_terms:null,predictor_terms:null}];
  await vm.runInContext('poll()',context);
  assert.deepEqual(nodes.get('representation-loss-terms').children.map(n=>n.textContent),
    ['Color: 1','Delta visual: 2','Unión del delta: 0','Reconstrucción local: 1,5']);
  assert.deepEqual(nodes.get('loss-terms').children.map(n=>n.textContent),
    ['Delta latente: 0,25','Delta visual: 2','Unión del delta: 0,5']);
  assert.match(nodes.get('loss-terms').children[0].title,/acción 3$/);
  report.progress.sigreg_enabled=true;
  await vm.runInContext('poll()',context);
  assert.equal(text('representation-regularization'),'SIGReg activo · fase inicial de representación');
  report.progress.sigreg_enabled=false;
  await vm.runInContext('poll()',context);
  assert.equal(text('representation-regularization'),'SIGReg desactivado · reconstrucción continúa');
  delete report.progress.sigreg_enabled;
  report.curves=[];
  await vm.runInContext('poll()',context);
  for(const id of ['representation-loss-terms','loss-terms']) {
    assert.equal(nodes.get(id).children.length,0);
    assert.equal(text(id),'Sin una actualización registrada');
  }
  assert.equal(text('representation-regularization'),'Estado de SIGReg no registrado');

  // Historical terms use recorded updates only, not missing terms as zeros.
  // At action 514 the 512-action window contains actions 3, 32 and 513.
  report.config.prediction_delta_loss='union';
  report.curves=[{actions:1,predictor_terms:{latent:100,delta_nll:100,delta_overlap:100,rollout:100}},
    {actions:3,predictor_terms:{latent:.25,delta_nll:2,delta_overlap:0}},
    {actions:32,predictor_terms:{latent:.5,delta_nll:4,delta_overlap:.5,rollout:6}},
    {actions:513,predictor_terms:{latent:0,delta_nll:0,delta_overlap:1,rollout:2}},
    {actions:514,predictor_terms:null}];
  await vm.runInContext('poll()',context);
  assert.match(text('predictor-objective'),/unión de cambios reales y predichos/);
  assert.deepEqual(nodes.get('predictor-components-legend').children.map(n=>n.textContent),
    ['MSE latente: 0,25','NLL visual: 2','Error de solapamiento: 0,5','Rollout · total: 4']);
  const termPoints=circles('predictor-components');
  assert.equal(termPoints.filter(n=>n.opacity<1).length,15,'3 terms at action 3; 4 at each other measured action');
  assert.ok(termPoints.some(n=>n.opacity<1&&n.children[0].textContent==='NLL visual: 0 · acción 513'));
  assert.ok(termPoints.some(n=>n.opacity===1&&n.children[0].textContent==='Rollout · total: 4 · acción 514'));
  assert.ok(termPoints.some(n=>n.opacity<1&&n.children[0].textContent==='Rollout · total: 6 · acción 32'));
  report.config.prediction_delta_loss='balanced';
  report.curves=[{actions:1026,predictor_terms:null}];
  await vm.runInContext('poll()',context);
  assert.match(text('predictor-objective'),/grupos equilibrados/);
  assert.equal(circles('predictor-components').length,0);
  assert.ok(nodes.get('predictor-components-legend').children.every(n=>n.textContent.endsWith(': —')));
  delete report.config.prediction_delta_loss;
  await vm.runInContext('poll()',context);
  assert.match(text('predictor-objective'),/no identifica el modo/);

  // Goal statistics average events in an action window, not the last N goals.
  // Simultaneous completions count equally, failures affect reward only, and a
  // censored segment without reward never dilutes the denominator.
  const oldEpisodes=report.episodes;
  const goal=(actions,minimum_actions,reward,success=true)=>({actions,minimum_actions,reward,success});
  report.episodes=[[1,[goal(100,50,-50)]],[32,[goal(6,3,-3)]],
    [33,[goal(160,0,-160,false)]],[64,[goal(0,0,0)]],[512,[goal(2,1,-1)]],
    [513,[goal(4,2,-2),goal(6,4,-2)]],[514,[goal(2,2,null,false)]]]
    .map(([end_action,segments])=>({game:'a',level:1,end_action,segments,success:segments.every(s=>s.success)}));
  await vm.runInContext('poll()',context);
  const efficiencyMean=circles('efficiency').filter(n=>n.opacity===1);
  assert.ok(efficiencyMean.some(n=>n.children[0].textContent.endsWith(': 3,6 · acción 513')));
  assert.ok(efficiencyMean.some(n=>n.children[0].textContent.endsWith(': 2 · acción 513')));
  assert.equal(efficiencyMean.filter(n=>n.children[0].textContent.includes('acción 513')).length,2,
    'one mean per action and metric includes all simultaneous goals');
  assert.ok(circles('efficiency').some(n=>n.opacity<1&&n.children[0].textContent==='Real: 6 · acción 513'));
  assert.equal(nodes.get('efficiency-legend').children.length,2,'individual lines share their metric colors');
  const rewardMean=circles('reward').filter(n=>n.opacity===1);
  assert.match(rewardMean.at(-1).children[0].textContent,/: -28 · acción 514$/);
  assert.ok(rewardMean.some(n=>n.children[0].textContent.endsWith(': -26,5 · acción 32')),'future goals cannot change an earlier mean');
  assert.ok(circles('reward').some(n=>n.opacity<1&&n.children[0].textContent==='Reward individual: -160 · acción 33'));
  assert.match(nodes.get('reward-legend').children[0].textContent,/: -28$/);
  report.episodes=oldEpisodes;

  // The trailing 512-action window excludes action 1. Missing measurements are
  // excluded, but measured zero changes count. Unequal blocks are not averaged.
  report.curves=[{actions:1,counts:{invented:100,missed:100,color:100,correct:100}},
    {actions:32,counts:{invented:6,missed:12,color:2,correct:30}},
    {actions:33,counts:null},
    {actions:64,counts:{invented:0,missed:0,color:0,correct:0}},
    {actions:512,counts:{invented:2,missed:4,color:0,correct:10}},
    {actions:513,counts:{invented:4,missed:8,color:1,correct:20}},
    {actions:514,counts:{invented:null,missed:null,color:null}}];
  await vm.runInContext('poll()',context);
  for(const [name,value] of [['Inventados','3'],['Omitidos','6'],['Color incorrecto','0,75'],['Acertados','15']]) {
    const point=circles('changes').find(n=>n.opacity===1&&n.children[0].textContent.startsWith(`${name}: ${value} · acción 514`));
    assert.ok(point,`${name} trend uses four measured actions in the exact window`);
  }
  const rawInvented=circles('changes').filter(n=>n.opacity<1&&n.fill==='#77dabb').map(n=>n.children[0].textContent);
  assert.ok(rawInvented.some(s=>s.startsWith('Inventados: 4 · acción 514')),'missing count does not dilute the last partial block');
  assert.ok(rawInvented.some(s=>s.startsWith('Inventados: 0 · acción 64')),'measured zero is kept');
  assert.equal(nodes.get('changes-legend').children.length,4);

  report.curves=[{actions:32,counts:{wrong:5,union:10,changed:10}},
    {actions:64,counts:{wrong:0,union:0,changed:0}},
    {actions:96,counts:{wrong:0,union:10,changed:10}},
    {actions:128,counts:null}];
  await vm.runInContext('poll()',context);
  const rawLines=nodes.get('delta').children.filter(n=>n.stroke==='#94a7aa'&&typeof n.points==='string');
  assert.equal(rawLines.length,2,'no line bridges the no-change gap');
  assert.ok(rawLines.every(n=>n.points.split(' ').length===1));
  assert.equal(nodes.get('delta').children.filter(n=>n.height===5).length,1,'no-change is marked separately from missing data');
  assert.match(text('interaction-summary'),/1 de 3 acciones medidas sin cambios/);
  const rawTitles=circles('delta').map(n=>n.children[0].textContent).filter(s=>s.startsWith('Bloque'));
  assert.equal(rawTitles.length,2);
  assert.ok(rawTitles.some(s=>s.includes(': 0 · acción 96')),'a perfect prediction of actual changes remains 0%');

  report.curves=[{actions:4096,counts:{wrong:0,union:0,changed:0}}];
  await vm.runInContext('poll()',context);
  assert.equal(circles('delta').length,0,'no-change-only window has no fabricated percentage');
  assert.equal(nodes.get('delta').children.filter(n=>n.height===5).length,1);
  assert.equal(circles('fixed-delta').length,2,'fixed evaluation history survives beyond the interaction window');
  report.curves=[{actions:32,counts:{wrong:4,union:4,changed:0}}];
  await vm.runInContext('poll()',context);
  assert.equal(nodes.get('delta').children.filter(n=>n.height===5).length,0,'invented changes are errors even when the real image did not move');
  assert.match(circles('delta')[0].children[0].textContent,/: 100 ·/);
  // Same run, growing coverage: aggregate curves must not bridge difficulty changes.
  report.curriculum={stage:2,unlocked:{a:2},thresholds:{patience:3,representation_open:.01,
    prediction_open:.1,latent_drift:.05,success:.8},levels:[
    {game:'a',level:1,unlocked:true,predictor:true,planner:true,promotion_passes:0,
      measurements:{reconstruction_error:0,latent_drift:.01,prediction_error:.04,rollout_error:.09,success:.5}},
    {game:'a',level:2,unlocked:true,predictor:false,planner:false,promotion_passes:0,measurements:{}},
    {game:'a',level:3,unlocked:false,promotion_passes:0,measurements:{}}]};
  report.progress.representation_updates=123;report.progress.predictor_enabled=true;
  const first=[{game:'a',level:1}], second=[...first,{game:'a',level:2}];
  report.evaluations=[evaluation(0,'a',.2,{coverage:first,stage:1}),
    evaluation(100,'a',.1,{coverage:first,stage:1}),
    evaluation(200,'a',.1,{coverage:second,stage:2}),
    evaluation(200,'a',.7,{coverage:second,stage:2,level:2})];
  await vm.runInContext('poll()',context);
  assert.equal(nodes.get('curriculum-panel').hidden,false);
  assert.equal(text('representation-updates'),'123');
  assert.match(text('gate'),/Activo · cada acción/);
  const cells=nodes.get('curriculum-rows').children.map(r=>r.children.map(c=>c.textContent));
  assert.equal(cells[0][1],'Representación + predictor + planner');
  assert.equal(cells[1][1],'Representación');
  assert.equal(cells[2][1],'Pendiente');
  assert.equal(cells[0][2],'0 %');assert.equal(cells[1][2],'—');
  const qualityLines=nodes.get('fixed-delta').children.filter(n=>n.stroke==='#77dabb'&&typeof n.points==='string');
  assert.deepEqual(qualityLines.map(n=>n.points.split(' ').length),[2,1]);
  assert.match(circles('fixed-delta').at(-1).children[0].textContent,/: 40 · acción 200/);
  // Live configured limits are thin dotted references; coincident limits merge.
  report.config.evaluate_every=256;
  Object.assign(report.curriculum.thresholds,{representation_open:.1,prediction_open:.1,latent_drift:.15});
  report.evaluations=[evaluation(256,'a',.2,{coverage:first,stage:1,
      quality:{reconstruction_error:.01,rollout_error:.3,latent_drift:.12}}),
    evaluation(512,'a',.1,{coverage:first,stage:1,
      quality:{reconstruction_error:0,rollout_error:.2,latent_drift:null}}),
    evaluation(768,'a',.05,{coverage:first,stage:1,
      quality:{reconstruction_error:0,rollout_error:.07,latent_drift:.11}})];
  await vm.runInContext('poll()',context);
  const labels=()=>nodes.get('fixed-delta-legend').children.map(n=>n.textContent);
  assert.deepEqual(labels(),['Predicción · 1 paso: 5 %','Reconstrucción: 0 %',
    'Predicción · 3 pasos: 7 %','Deriva latente: 11 %',
    'Límite · Reconstrucción / Predicción: 10 %','Límite · Deriva: 15 %']);
  const limits=()=>nodes.get('fixed-delta').children.filter(n=>n['stroke-dasharray']);
  assert.deepEqual(limits().map(n=>n.y1),[218.4,207.6]);
  assert.ok(limits().every(n=>n.y1===n.y2&&n['stroke-width']===1&&n['stroke-dasharray']==='2 4'));
  assert.match(text('quality-cadence'),/Calidad, deriva y activaciones cada 256 acciones/);
  const driftLines=()=>nodes.get('fixed-delta').children.filter(n=>n.stroke==='#94a7aa'&&typeof n.points==='string');
  assert.deepEqual(driftLines().map(n=>n.points.split(' ').length),[1,1],'missing drift measurements leave a gap');
  report.evaluations.pop();
  await vm.runInContext('poll()',context);
  assert.equal(labels()[0],'Predicción · 1 paso: 10 %');
  assert.equal(labels()[3],'Deriva latente: 12 %');
  assert.match(nodes.get('fixed-delta-legend').children[3].title,/acción 256/,'drift keeps the action of its actual measurement');
  report.curriculum.thresholds.representation_open=.02;
  await vm.runInContext('poll()',context);
  assert.equal(limits().length,3,'distinct limits are not hidden behind a hard-coded 10% line');
  // Diagnostics aggregate levels equally, even with unequal sample counts.
  // Older evaluations without this instrumentation are absent, not zero.
  const oldEvaluations=report.evaluations;
  const diagnosticCoverage=[{game:'a',level:1},{game:'b',level:1}];
  report.evaluations=[evaluation(0,'a',.1,{coverage:diagnosticCoverage}),
    evaluation(0,'b',.1,{coverage:diagnosticCoverage}),
    evaluation(100,'a',.1,{coverage:diagnosticCoverage,quality:{latent:{
      std_mean:2,within_episode_std:.2,inactive_fraction:0,effective_rank:2,max_rank:3,
      changed_rms:.004,unchanged_rms:0,sample_count:4,episode_count:1}}}),
    evaluation(100,'b',.1,{coverage:diagnosticCoverage,quality:{latent:{
      std_mean:4,within_episode_std:.4,inactive_fraction:.5,effective_rank:6,max_rank:7,
      changed_rms:.006,unchanged_rms:.002,sample_count:12,episode_count:2}}})];
  await vm.runInContext('poll()',context);
  assert.deepEqual(nodes.get('latent-diversity-legend').children.map(n=>n.textContent),
    ['Entre observaciones: 3','Dentro de cada partida: 0,3']);
  assert.deepEqual(nodes.get('latent-rank-legend').children.map(n=>n.textContent),
    ['Rango efectivo: 4','Máximo observable: 5']);
  assert.equal(nodes.get('latent-inactive-legend').children[0].textContent,'Casi constantes: 25 %');
  assert.deepEqual(nodes.get('latent-sensitivity-legend').children.map(n=>n.textContent),
    ['Bloques con cambios: 0,005','Bloques sin cambios: 0,001']);
  assert.equal(circles('latent-diversity').length,2,'only the instrumented evaluation contributes points');
  assert.ok(circles('latent-diversity').every(n=>n.cx===58),'unmeasured history is not a fake zero baseline');
  assert.match(text('latent-summary'),/16 observaciones de 3 partidas/);
  report.evaluations[3].quality.latent.changed_rms=null;
  report.evaluations[2].quality.latent.unchanged_rms=0;
  report.evaluations[3].quality.latent.unchanged_rms=0;
  await vm.runInContext('poll()',context);
  assert.deepEqual(nodes.get('latent-sensitivity-legend').children.map(n=>n.textContent),
    ['Bloques con cambios: —','Bloques sin cambios: 0']);
  assert.equal(circles('latent-sensitivity').length,1,'a missing class is not zero response');
  report.evaluations=oldEvaluations;
  report.config.game='sc01-v1';report.config.initialization='random';
  await vm.runInContext('poll()',context);
  assert.match(text('run-info'),/^sc01-v1 · semilla 0/);
  assert.match(text('run-info'),/Inicializado desde cero/);
  report.config.initialization='checkpoint';
  await vm.runInContext('poll()',context);
  assert.match(text('run-info'),/Pesos preentrenados/);
  report.curves=[];report.episodes=[];report.evaluations=[];report.progress={phase:'loading',actions:0};
  await vm.runInContext('poll()',context);
  assert.equal(circles('delta').length,0,'no points borrowed from a previous run');
  assert.equal(circles('fixed-delta').length,0,'fixed evaluations reset with the selected run');
  assert.equal(labels()[0],'Predicción · 1 paso: —','empty series has no stale last value');
  assert.equal(circles('latent-diversity').length,0);
  assert.equal(nodes.get('latent-rank-legend').children[0].textContent,'Rango efectivo: —');
  assert.equal(nodes.get('evaluation-rows').children.length,0);
  offline=true;await vm.runInContext('poll()',context);
  assert.match(text('connection'),/Sin conexión.*offline/);
  assert.equal(text('gpu-status'),'GPU · datos no disponibles','connection failure clears stale GPU readings');
  console.log('Subgoal dashboard contracts passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
