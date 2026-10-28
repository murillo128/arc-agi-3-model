// Actual JS polling/rendering against literal measurements; no browser tests.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assets = path.join(__dirname, '../dashboard_web');
const html = fs.readFileSync(path.join(assets, 'subgoals.html'), 'utf8');
function element() {
  return {textContent:'', hidden:false, children:[], listeners:{}, focus(){},
    append(...items) {for(const item of items){
      if(item.parentNode)item.parentNode.children=item.parentNode.children.filter(child=>child!==item);
      item.parentNode=this;this.children.push(item);
    }},
    replaceChildren(...items) {for(const child of this.children)child.parentNode=null;this.children=[];this.append(...items);},
    setAttribute(k,v) {this[k]=v;},
    addEventListener(k,v) {this.listeners[k]=v;},
    getContext() {return this.context??=( {pixels:[],clearRect(){this.pixels=[];},
      fillRect(x,y,w,h){this.pixels.push({x,y,color:this.fillStyle});}} );}};
}
const nodes = new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],element()]));
const report = {status:'running', age_seconds:1, config:{seed:0,predictor_error:.1,
    evaluation_pairs:[{game:'a',level:1},{game:'b',level:1}]},
  progress:{phase:'training',actions:32,total_actions:65536,predictor_updates:1,path_updates:0,tps:99,recent_tps:2},
  curves:[{actions:1,predictor_loss:2,path_loss:null,tps:99,recent_tps:2,counts:{wrong:5,union:10,invented:2,missed:3,color:0,correct:5}},
    {actions:32,predictor_loss:null,path_loss:0,tps:99,recent_tps:2,counts:{wrong:15,union:90,invented:10,missed:5,color:0,correct:75}}],
  episodes:[{game:'a',level:1,end_action:32,success:true,segments:[{success:true,actions:3,minimum_actions:2,reward:-1}]}],
  evaluations:[]};
let offline=false;
let controls;
let controlReply;
let trajectoryReply={ok:true,json:async()=>({trajectory:null})};
let trajectoryRequests=0;
const controlRequests=[];
let gpu={devices:[{index:0,name:'Test GPU',utilization_percent:73,temperature_c:71}]};
const context=vm.createContext({console,AbortSignal,setTimeout(){},clearTimeout(){},atob:value=>Buffer.from(value,'base64').toString('binary'),
  fixture:report,
  document:{getElementById:id=>nodes.get(id),createElement:element,createElementNS:element},
  fetch:async(url,options)=>{if(offline)throw Error('offline');if(url==='api/trajectory'){trajectoryRequests++;return trajectoryReply;}if(options?.method==='POST'){
    controlRequests.push({url,options});return controlReply;
  }return {ok:true,json:async()=>({subgoals:report,gpu,curriculum_control:controls})};}});
const text=id=>nodes.get(id).textContent;
(async()=>{
  vm.runInContext(fs.readFileSync(path.join(assets,'chart-tools.js'),'utf8'),context);
  vm.runInContext(fs.readFileSync(path.join(assets,'subgoals.js'),'utf8'),context);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(text('actions'),'32');
  assert.equal(text('world-updates'),'1');
  assert.equal(text('predictor-replay'),'No replay · one transition per update');
  report.config.predictor_replay_episodes=16;report.config.predictor_batch_size=4;
  await vm.runInContext('poll()',context);
  assert.equal(text('predictor-replay'),'Replay: up to 16 episodes · target batch 4');
  report.progress.predictor_replay={capacity:16,episodes:0,transitions:0,batch_size:4,actual_batch_size:0};
  await vm.runInContext('poll()',context);
  assert.equal(text('predictor-replay'),'Replay: 0/16 episodes · 0 transitions · batch 0/4');
  report.progress.predictor_replay={capacity:16,episodes:3,transitions:27,batch_size:4,actual_batch_size:3};
  await vm.runInContext('poll()',context);
  assert.equal(text('predictor-replay'),'Replay: 3/16 episodes · 27 transitions · batch 3/4');
  delete report.progress.predictor_replay;
  delete report.config.predictor_replay_episodes;delete report.config.predictor_batch_size;
  await vm.runInContext('poll()',context);
  assert.equal(text('predictor-replay'),'No replay · one transition per update');
  assert.equal(text('path-updates'),'0');
  assert.equal(text('tps'),'2');
  assert.equal(nodes.get('speed-legend').children[0].textContent,'Actions/s · last 60 s: 2 actions/s');
  assert.equal(nodes.get('path-loss-legend').children[0].textContent,'Update average · up to 512 actions: 0');
  assert.equal(nodes.get('loss-legend').children[1].textContent,'Individual update: 2');
  assert.equal(nodes.get('loss-legend').children[1].title,
    'Latest measurement · action 1 · Main line: EMA (α = 0.1). Fine translucent line: individual measurements. Select a legend label to focus a series and pin its latest EMA value.',
    'missing latest sample does not fabricate a zero');
  assert.equal(text('successes'),'Levels solved: 1 · recent window');
  assert.equal(text('gpu-status'),'GPU · 73 % utilization · 71 °C');
  assert.match(nodes.get('gpu-status').title,/Test GPU/);
  gpu={devices:[{index:0,name:'Test GPU',utilization_percent:0,temperature_c:null},
    {index:1,name:'Second GPU',utilization_percent:null,temperature_c:65}]};
  await vm.runInContext('poll()',context);
  assert.equal(text('gpu-status'),'GPU 0 · 0 % utilization · — °C\nGPU 1 · — % utilization · 65 °C');
  gpu=undefined;
  await vm.runInContext('poll()',context);
  assert.equal(text('gpu-status'),'GPU · data unavailable','old backend data does not preserve stale readings');
  gpu={devices:[{index:0,name:'Test GPU',utilization_percent:88,temperature_c:72}]};
  await vm.runInContext('poll()',context);
  assert.equal(text('gpu-status'),'GPU · 88 % utilization · 72 °C');
  assert.equal(nodes.get('prediction-sample-images').hidden,true);
  assert.equal(nodes.get('latent-diversity-legend').children[0].textContent,'Across observations: —');
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
    ['series-0','Invented: 1 px'],['series-1','Missed: 1 px'],
    ['series-2','Wrong color: 1 px'],['series-3','Correct: 1 px']]);
  assert.match(text('prediction-sample-info'),/sc01-v1 · level 1 · episode 123 · step 8 · ACTION4/);
  assert.match(text('prediction-sample-error'),/75 % · 3 \/ 4/);
  report.prediction_sample={...report.prediction_sample,wrong:0,union:0,
    real_delta:[[16,16,16],[16,16,16]],predicted_delta:[[16,16,16],[16,16,16]],
    action:{id:6,data:{x:4,y:19}},phase:'evaluation'};
  await vm.runInContext('poll()',context);
  assert.match(text('prediction-sample-info'),/ACTION6 \(4, 19\) · Evaluation/);
  assert.match(text('prediction-sample-error'),/No real or predicted changes/);
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
  const kinds=[['Invented','#77dabb',6],['Missed','#f4b979',4],
    ['Wrong color','#a9a6ef',0],['Correct','#94a7aa',40]];
  for(const [name,color,value] of kinds) {
    const points=circles('changes').filter(n=>n.fill===color);
    assert.equal(points.length,2,`${name} has raw and trend measurements`);
    assert.ok(points[0].opacity<points[1].opacity,'raw blocks are behind the stronger trend');
    for(const point of points)assert.ok(point.children[0].textContent.startsWith(`${name}: ${value} · action 32`));
    assert.match(points[0].children[0].textContent,/px\/action · block of 32/);
    assert.match(points[1].children[0].textContent,/px\/action · moving average of up to 512/);
  }
  assert.equal(circles('loss').filter(n=>n.children[0].textContent.startsWith('Individual update:')).length,1,'missing loss is not a fabricated zero');
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
  assert.match(text('progress'),/episode 1\/2 · action 7/);
  assert.equal(nodes.get('evaluation-rows').children[0].children[6].textContent,'20 %');

  // Success uses completed development episodes, independently per level/policy.
  // Literal wins establish percentages, including genuine zero and full success.
  report.config.success=.8;
  const winRow=(actions,policy,successes,episodes=8,level=1,split='dev')=>
    ({actions,policy,successes,episodes,level,split,game:'a',delta_error:.2});
  report.evaluations=[winRow(20,'random',3),winRow(200,'path_mpc',3),
    winRow(100,'random',3),winRow(100,'path_mpc',0),winRow(200,'random',4),
    winRow(300,'random',3),winRow(300,'path_mpc',2,2),
    winRow(400,'path_mpc',8,8,1,'final'),winRow(500,'unknown',8),
    winRow(100,'path_mpc',1,2,2),winRow(100,'random',0,8,2)];
  await vm.runInContext('poll()',context);
  const successLabels=()=>nodes.get('planner-success-legend').children.map(n=>n.textContent);
  const winPoints=()=>circles('planner-success').map(n=>n.children[0].textContent);
  assert.deepEqual(successLabels(),['Planner: 100 %','Random: 37.5 %','Threshold · Curriculum success: 80 %']);
  assert.deepEqual(winPoints(),[
    'Planner: 0 · action 100 · 0/8 wins','Planner: 37.5 · action 200 · 3/8 wins',
    'Planner: 100 · action 300 · 2/2 wins','Random: 37.5 · action 100 · 3/8 wins',
    'Random: 50 · action 200 · 4/8 wins','Random: 37.5 · action 300 · 3/8 wins']);
  assert.match(text('planner-success-summary'),/Planner: 2\/2 wins · action 300/);
  assert.equal(nodes.get('planner-success-level').children.length,2);
  report.evaluations.push(winRow(200,'path_mpc',4),winRow(400,'path_mpc',0,0),
    winRow(500,'path_mpc',9),winRow(600,'path_mpc',null));
  await vm.runInContext('poll()',context);
  assert.equal(winPoints().length,6,'invalid/empty counts are absent, not measured zeros');
  assert.equal(winPoints()[1],'Planner: 50 · action 200 · 4/8 wins','latest repeated sweep wins');

  const successSelector=nodes.get('planner-success-level');
  successSelector.value=JSON.stringify(['a',2]);successSelector.listeners.change();
  assert.deepEqual(winPoints(),['Planner: 50 · action 100 · 1/2 wins','Random: 0 · action 100 · 0/8 wins']);
  report.evaluations.push(winRow(200,'path_mpc',2,2,2));
  await vm.runInContext('poll()',context);
  assert.equal(successSelector.value,JSON.stringify(['a',2]),'polling keeps the selected level');
  assert.equal(winPoints()[1],'Planner: 100 · action 200 · 2/2 wins');
  report.evaluations.push(winRow(100,'random',3,8,3));
  await vm.runInContext('poll()',context);
  successSelector.value=JSON.stringify(['a',3]);successSelector.listeners.change();
  assert.equal(successLabels()[0],'Planner: —','random-only history is not planner success');
  assert.deepEqual(winPoints(),['Random: 37.5 · action 100 · 3/8 wins']);
  report.evaluations=[];
  await vm.runInContext('poll()',context);
  assert.equal(winPoints().length,0,'new empty reports clear success history');
  assert.equal(successSelector.value,'');
  assert.equal(successSelector.disabled,true);
  assert.match(text('planner-success-summary'),/Planner: no evaluation available/);
  // A single saved random score extends across later planner measurements. The
  // current evaluation level is the highest unlocked, even during old-level practice.
  const savedCurriculum=report.curriculum, savedGame=report.config.game;
  report.config.game='a';report.config.evaluation_protocol='current_level_cached_v1';
  report.curriculum={unlocked:{a:2}};
  report.evaluations=[winRow(0,'path_mpc',8),
    {...winRow(20,'random',3,8,2),fixed_baseline:true,delta_error:null,quality:null},
    winRow(100,'path_mpc',4,8,2),winRow(200,'path_mpc',6,8,2),
    {...winRow(200,'quality_probe',null,8,2),coverage:[{game:'a',level:2}],
      stage:2,delta_error:.1,real_actions:0,analyzed_actions:211,decision_ms:null,
      subgoals:null,quality:{reconstruction_error:0,rollout_error:.2,latent_drift:.01}}];
  await vm.runInContext('poll()',context);
  assert.equal(successSelector.value,JSON.stringify(['a',2]));
  assert.deepEqual(successLabels(),['Planner: 75 %','Random · fixed baseline: 37.5 %','Threshold · Curriculum success: 80 %']);
  assert.deepEqual(winPoints().filter(s=>s.startsWith('Random')), [
    'Random · fixed baseline: 37.5 · action 100 · 3/8 wins',
    'Random · fixed baseline: 37.5 · action 200 · 3/8 wins']);
  assert.match(text('planner-success-summary'),/measured once at action 20/);
  assert.equal(circles('fixed-delta').length,4,'quality probes provide fresh measurements, baseline does not');
  const storedRow=nodes.get('evaluation-rows').children.at(-1).children.map(n=>n.textContent);
  assert.deepEqual(storedRow.slice(3),['Quality · saved sample','—','—','10 %','0 new · 211 stored','—']);
  assert.equal(nodes.get('evaluation-rows').children[1].children[6].textContent,'—','unknown baseline error is not zero');
  report.curriculum=savedCurriculum;report.config.game=savedGame;
  delete report.config.evaluation_protocol;
  report.evaluations=[];await vm.runInContext('poll()',context);
  delete report.config.success;

  // Only the same complete, fixed-random development coverage is comparable.
  // Different pixel/action counts must not change each level's equal weight.
  const evaluation=(actions,game,error,extra={})=>({actions,split:'dev',policy:'random',game,level:1,
    episodes:2,delta_error:error,real_actions:10,...extra});
  report.evaluations=[evaluation(500,'a',.1),evaluation(500,'b',.5,{real_actions:1000}),
    evaluation(500,'a',1,{policy:'path_mpc'}),evaluation(1000,'a',.2),
    evaluation(2000,'a',.9,{split:'final'}),evaluation(2000,'b',.9,{split:'final'})];
  await vm.runInContext('poll()',context);
  assert.equal(circles('fixed-delta').length,1,'partial sweeps and new-seed final evaluations are not comparable points');
  assert.match(circles('fixed-delta')[0].children[0].textContent,/: 30 · action 500 · 2 pairs · 4 episodes/);
  assert.match(text('fixed-summary'),/In progress: 1\/2 pairs/);
  report.evaluations.push(evaluation(1000,'b',.2));
  await vm.runInContext('poll()',context);
  assert.equal(circles('fixed-delta').length,2);
  assert.match(circles('fixed-delta')[1].children[0].textContent,/: 20 · action 1000/);
  assert.doesNotMatch(text('fixed-summary'),/In progress/);
  // Returning to the quality chart fits its full panel immediately, preserving
  // the measured 20% point. A narrower panel changes geometry, not the data.
  Object.assign(nodes.get('fixed-delta'),{clientWidth:1200,clientHeight:300});
  nodes.get('tab-world').listeners.click();
  assert.equal(nodes.get('fixed-delta').viewBox,'0 0 1200 300');
  assert.equal(circles('fixed-delta').at(-1).cx,1178);
  assert.equal(circles('fixed-delta').at(-1).cy,212.8);
  assert.match(circles('fixed-delta').at(-1).children[0].textContent,/: 20 · action 1000/);
  Object.assign(nodes.get('fixed-delta'),{clientWidth:360,clientHeight:240});
  await vm.runInContext('poll()',context);
  assert.equal(nodes.get('fixed-delta').viewBox,'0 0 360 240');
  assert.equal(circles('fixed-delta').at(-1).cx,338);
  assert.equal(circles('fixed-delta').at(-1).cy,164.8);
  assert.match(circles('fixed-delta').at(-1).children[0].textContent,/: 20 · action 1000/);
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
    const average=circles(id).filter(n=>n.opacity===.5);
    assert.match(average.at(-1).children[0].textContent,new RegExp(`: ${value} · action 514$`));
    assert.equal(circles(id).filter(n=>n.opacity<.5).length,rawCount);
    assert.ok(circles(id).some(n=>n.opacity<.5&&n.children[0].textContent.includes(': 0 ·')));
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
    ['Color: 1','Visual delta: 2','Delta union: 0','Local reconstruction: 1.5']);
  assert.deepEqual(nodes.get('loss-terms').children.map(n=>n.textContent),
    ['Latent delta: 0.25','Visual delta: 2','Delta union: 0.5']);
  assert.match(nodes.get('loss-terms').children[0].title,/action 3$/);
  report.progress.sigreg_enabled=true;
  await vm.runInContext('poll()',context);
  assert.equal(text('representation-regularization'),'SIGReg active · initial representation phase');
  report.progress.sigreg_enabled=false;
  await vm.runInContext('poll()',context);
  assert.equal(text('representation-regularization'),'SIGReg disabled · reconstruction continues');
  report.progress.encoder_frozen=true;
  await vm.runInContext('poll()',context);
  assert.equal(text('representation-update-scope'),'Decoder · encoder frozen');
  assert.match(text('world-training-scope'),/^Encoder frozen:/);
  assert.equal(text('representation-regularization'),
    'Encoder frozen · encoder regularization disabled · decoder keeps learning');
  // Current progress wins over configuration; missing progress falls back to it.
  report.progress.encoder_frozen=false;
  report.config.freeze_encoder=true;
  await vm.runInContext('poll()',context);
  assert.equal(text('representation-update-scope'),'Encoder and decoder · each real action');
  delete report.progress.encoder_frozen;
  await vm.runInContext('poll()',context);
  assert.equal(text('representation-update-scope'),'Decoder · encoder frozen');
  delete report.config.freeze_encoder;
  delete report.progress.sigreg_enabled;
  report.progress.representation_learning_rates={encoder:3e-6,decoder:3e-5};
  await vm.runInContext('poll()',context);
  assert.match(text('world-training-scope'),/LR encoder: 3e-6 · decoder: 3e-5/);
  report.progress.representation_learning_rates.encoder=null;
  await vm.runInContext('poll()',context);
  assert.match(text('world-training-scope'),/LR encoder: — · decoder: 3e-5/);
  delete report.progress.representation_learning_rates;
  await vm.runInContext('poll()',context);
  assert.doesNotMatch(text('world-training-scope'),/LR encoder:/);
  report.curves=[];
  await vm.runInContext('poll()',context);
  for(const id of ['representation-loss-terms','loss-terms']) {
    assert.equal(nodes.get(id).children.length,0);
    assert.equal(text(id),'No update recorded');
  }
  assert.equal(text('representation-regularization'),'SIGReg status not recorded');
  assert.equal(text('representation-update-scope'),'Encoder and decoder · each real action');
  assert.match(text('world-training-scope'),/^Encoder and decoder learn/);

  // Historical terms use recorded updates only, not missing terms as zeros.
  // At action 514 the 512-action window contains actions 3, 32 and 513.
  report.config.prediction_delta_loss='union';
  report.curves=[{actions:1,predictor_terms:{latent:100,delta_nll:100,delta_overlap:100,rollout:100}},
    {actions:3,predictor_terms:{latent:.25,delta_nll:2,delta_overlap:0}},
    {actions:32,predictor_terms:{latent:.5,delta_nll:4,delta_overlap:.5,rollout:6}},
    {actions:513,predictor_terms:{latent:0,delta_nll:0,delta_overlap:1,rollout:2}},
    {actions:514,predictor_terms:null}];
  await vm.runInContext('poll()',context);
  assert.match(text('predictor-objective'),/union of real and predicted changes/);
  assert.deepEqual(nodes.get('predictor-components-legend').children.map(n=>n.textContent),
    ['Latent MSE: 0.25','Visual NLL: 2','Overlap error: 0.5','Rollout · total: 4']);
  const termPoints=circles('predictor-components');
  assert.equal(termPoints.filter(n=>n.opacity<.5).length,15,'3 terms at action 3; 4 at each other measured action');
  assert.ok(termPoints.some(n=>n.opacity<.5&&n.children[0].textContent==='Visual NLL: 0 · action 513'));
  assert.ok(termPoints.some(n=>n.opacity===.5&&n.children[0].textContent==='Rollout · total: 4 · action 514'));
  assert.ok(termPoints.some(n=>n.opacity<.5&&n.children[0].textContent==='Rollout · total: 6 · action 32'));
  // New predictor updates omit overlap; do not fabricate a zero contribution.
  report.curves=[{actions:1026,predictor_terms:{latent:.5,delta_nll:2,rollout:3}}];
  await vm.runInContext('poll()',context);
  assert.deepEqual(nodes.get('predictor-components-legend').children.map(n=>n.textContent),
    ['Latent MSE: 0.5','Visual NLL: 2','Overlap error: —','Rollout · total: 3']);
  assert.equal(circles('predictor-components').length,6);
  report.config.prediction_delta_loss='focal';
  await vm.runInContext('poll()',context);
  assert.match(text('predictor-objective'),/soft focal penalty/);
  assert.match(text('predictor-objective'),/γ=2, T=1/);
  assert.match(text('predictor-objective'),/without an argmax mask/);
  report.config.prediction_delta_loss='recall_focal';
  report.config.path_objective_version=2;report.config.path_success_margin=.02;
  report.objective_changes=[{actions:1026}];
  await vm.runInContext('poll()',context);
  assert.match(text('predictor-objective'),/2× color error on real changes/);
  assert.match(text('predictor-objective'),/1,026/);
  assert.match(text('path-objective'),/2 percentage points/);
  assert.match(text('path-objective-loss'),/Success\/failure BCE/);
  report.config.path_objective_version=4;
  report.config.path_exploration_warmup=2048;
  report.progress.path_replay={segments:32,successes:9,batch_segments:8,hindsight:true,predicted_inputs:true};
  report.path_recovery={training_episodes:10,training_actions:25};
  await vm.runInContext('poll()',context);
  assert.match(text('path-objective'),/2 percentage points/);
  assert.match(text('path-objective-loss'),/32 saved real attempts \(9 successes\)/);
  assert.match(text('path-objective-loss'),/without relabeling the original failures as wins/);
  assert.ok(text('path-objective-loss').includes(
    'Prior recovery: 10 episodes and 25 additional real actions; counted separately from these curves.'));
  assert.match(text('path-objective'),/from 100% to 20% over 2,048 planner actions per level/);
  report.config.prediction_delta_loss='recall_focal_capped';
  report.config.path_search={horizon:6,noop_cooldown:8};
  await vm.runInContext('poll()',context);
  assert.match(text('predictor-objective'),/max\(32, real changes\)/);
  assert.match(text('predictor-objective'),/only the current objective/);
  assert.match(text('path-objective'),/up to 6 steps; predictor and quality evaluation: 3 steps/);
  assert.match(text('path-objective'),/for 8 actions/);
  assert.ok(vm.runInContext('displayedReport.curves.length',context)>0,'history remains available');
  delete report.config.path_search;
  delete report.config.path_exploration_warmup;
  delete report.progress.path_replay;
  delete report.path_recovery;
  delete report.objective_changes;delete report.config.path_objective_version;
  report.config.prediction_delta_loss='balanced';
  report.curves=[{actions:1026,predictor_terms:null}];
  await vm.runInContext('poll()',context);
  assert.match(text('predictor-objective'),/balanced groups/);
  assert.equal(circles('predictor-components').length,0);
  assert.ok(nodes.get('predictor-components-legend').children.every(n=>n.textContent.endsWith(': —')));
  delete report.config.prediction_delta_loss;
  await vm.runInContext('poll()',context);
  assert.match(text('predictor-objective'),/does not identify the NLL mode/);

  // Goal statistics average events in an action window, not the last N goals.
  // Simultaneous completions count equally, efficiency excludes failures, and a
  // censored segment without reward never dilutes the denominator.
  const oldEpisodes=report.episodes;
  const goal=(actions,minimum_actions,reward,success=true)=>({actions,minimum_actions,reward,success});
  report.episodes=[[1,[goal(100,50,-50)]],[32,[goal(6,3,-3)]],
    [33,[goal(160,0,-160,false)]],[64,[goal(0,0,0)]],[512,[goal(2,1,-1)]],
    [513,[goal(4,2,-2),goal(6,4,-2)]],[514,[goal(2,2,null,false)]]]
    .map(([end_action,segments])=>({game:'a',level:1,end_action,segments,success:segments.every(s=>s.success)}));
  await vm.runInContext('poll()',context);
  const efficiencyMean=circles('efficiency').filter(n=>n.opacity===.5);
  assert.ok(efficiencyMean.some(n=>n.children[0].textContent.endsWith(': 3.6 · action 513')));
  assert.ok(efficiencyMean.some(n=>n.children[0].textContent.endsWith(': 2 · action 513')));
  assert.equal(efficiencyMean.filter(n=>n.children[0].textContent.includes('action 513')).length,2,
    'one mean per action and metric includes all simultaneous goals');
  assert.ok(circles('efficiency').some(n=>n.opacity<.5&&n.children[0].textContent==='Real: 6 · action 513'));
  assert.equal(nodes.get('efficiency-legend').children.length,2,'individual lines share their metric colors');
  const rewardMean=circles('reward').filter(n=>n.opacity===.5);
  assert.match(rewardMean.at(-1).children[0].textContent,/: -28 · action 514$/);
  assert.ok(rewardMean.some(n=>n.children[0].textContent.endsWith(': -26.5 · action 32')),'future goals cannot change an earlier mean');
  assert.ok(circles('reward').some(n=>n.opacity<.5&&n.children[0].textContent==='Individual reward: -160 · action 33'));
  assert.match(nodes.get('reward-legend').children[0].textContent,/: -28$/);
  const outcomeMeans=circles('subgoal-outcomes').map(n=>n.children[0].textContent);
  assert.ok(outcomeMeans.some(s=>s.includes(': 66.6667 · action 33 · 2 successes · 1 failure · 3 completed subgoals')),
    'two observed successes out of three completed goals');
  assert.ok(outcomeMeans.at(-1).includes(': 83.3333 · action 513 · 5 successes · 1 failure · 6 completed subgoals'),
    'cutoff excludes action 1 and includes both simultaneous goals plus the zero-action success');
  assert.match(text('subgoal-outcomes-summary'),/action 513: 5 successes · 1 failure · 6 completed subgoals/);
  assert.ok(!outcomeMeans.some(s=>s.includes('action 514')),'censored segment is not a completed failure');
  const outcomeDots=nodes.get('subgoal-outcomes').children.filter(n=>n.r===3);
  assert.equal(outcomeDots.length,7);
  assert.equal(outcomeDots.filter(n=>n.fill==='#f4b979').length,1);
  assert.ok(outcomeDots.some(n=>n.children[0].textContent==='Failure: 0 · action 33 · a / level 1 · Subgoal · 160 actions · minimum 0'));
  const outcomeLines=nodes.get('subgoal-outcomes').children.filter(n=>typeof n.points==='string'&&n.class!=='series-ema');
  assert.equal(outcomeLines.length,1,'binary outcomes are points, not connected routes');
  assert.equal(outcomeLines[0].stroke,'#a9a6ef');
  report.episodes=[{game:'a',level:1,end_action:1025,segments:[goal(5,2,-133,false)]}];
  await vm.runInContext('poll()',context);
  assert.match(nodes.get('subgoal-outcomes-legend').children[0].textContent,/: 0 %$/,'measured zero is not missing');
  assert.match(text('subgoal-outcomes-summary'),/0 successes · 1 failure · 1 completed subgoal/);
  report.episodes.push({game:'a',level:1,end_action:1030,segments:[goal(0,0,0)]});
  await vm.runInContext('poll()',context);
  assert.match(nodes.get('subgoal-outcomes-legend').children[0].textContent,/: 50 %$/);
  report.episodes=[{game:'a',level:1,end_action:1040,segments:[goal(2,2,null,false),{success:null,reward:0}]}];
  await vm.runInContext('poll()',context);
  assert.match(text('subgoal-outcomes-summary'),/No results recorded/);
  assert.match(nodes.get('subgoal-outcomes-legend').children[0].textContent,/: —$/);
  assert.equal(nodes.get('subgoal-outcomes').children.filter(n=>n.r===2||n.r===3).length,0,'empty refresh clears old results');
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
  for(const [name,value] of [['Invented','3'],['Missed','6'],['Wrong color','0.75'],['Correct','15']]) {
    const point=circles('changes').find(n=>n.opacity===.5&&n.children[0].textContent.startsWith(`${name}: ${value} · action 514`));
    assert.ok(point,`${name} trend uses four measured actions in the exact window`);
  }
  const rawInvented=circles('changes').filter(n=>n.opacity<.5&&n.fill==='#77dabb').map(n=>n.children[0].textContent);
  assert.ok(rawInvented.some(s=>s.startsWith('Invented: 4 · action 514')),'missing count does not dilute the last partial block');
  assert.ok(rawInvented.some(s=>s.startsWith('Invented: 0 · action 64')),'measured zero is kept');
  assert.equal(nodes.get('changes-legend').children.length,4);

  report.curves=[{actions:32,counts:{wrong:5,union:10,changed:10}},
    {actions:64,counts:{wrong:0,union:0,changed:0}},
    {actions:96,counts:{wrong:0,union:10,changed:10}},
    {actions:128,counts:null}];
  await vm.runInContext('poll()',context);
  const rawLines=nodes.get('delta').children.filter(n=>n.stroke==='#94a7aa'&&typeof n.points==='string'&&n.class!=='series-ema');
  assert.equal(rawLines.length,2,'no line bridges the no-change gap');
  assert.ok(rawLines.every(n=>n.points.split(' ').length===1));
  assert.equal(nodes.get('delta').children.filter(n=>n.height===5).length,1,'no-change is marked separately from missing data');
  assert.match(text('interaction-summary'),/1 of 3 measured actions without real or predicted changes/);
  const rawTitles=circles('delta').map(n=>n.children[0].textContent).filter(s=>s.startsWith('Block'));
  assert.equal(rawTitles.length,2);
  assert.ok(rawTitles.some(s=>s.includes(': 0 · action 96')),'a perfect prediction of actual changes remains 0%');

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
  assert.match(text('gate'),/Active · each action/);
  const cells=nodes.get('curriculum-rows').children.map(r=>r.children.map(c=>c.textContent));
  assert.equal(cells[0][1],'Representation + predictor + planner');
  assert.equal(cells[1][1],'Representation');
  assert.equal(cells[2][1],'Pending');
  assert.equal(cells[0][2],'0 %');assert.equal(cells[1][2],'—');
  report.config.freeze_representation_after_planner=true;
  report.progress.representation_frozen=true;
  report.progress.representation_learning_rates={encoder:null,decoder:null};
  report.manual_controls=[{actions:200,operation:'level',result:'advanced',game:'a',level:2,phase:'representation'}];
  await vm.runInContext('poll()',context);
  assert.equal(text('representation-update-scope'),'Encoder and decoder frozen');
  assert.equal(text('representation-regularization'),'Encoder and decoder frozen · reconstruction paused');
  assert.equal(text('gate'),'Active · representation frozen');
  assert.match(text('world-training-scope'),/LR encoder: — · decoder: —/);
  assert.match(text('manual-controls'),/level · a \/ 2 · representation · applied · action 200/);
  assert.match(nodes.get('curriculum-rows').children[0].children[1].textContent,/representation frozen/);
  assert.equal(nodes.get('curriculum-rows').children[1].children[1].textContent,'Representation');
  report.progress.representation_frozen=false;
  report.progress.representation_learning_rates={encoder:3e-5,decoder:3e-5};
  delete report.manual_controls;
  await vm.runInContext('poll()',context);
  assert.equal(text('representation-update-scope'),'Encoder and decoder · each real action');
  assert.match(text('world-training-scope'),/LR encoder: 3e-5 · decoder: 3e-5/);
  assert.equal(text('manual-controls'),'No manual advances recorded.');
  delete report.config.freeze_representation_after_planner;
  delete report.progress.representation_frozen;
  delete report.progress.representation_learning_rates;
  const qualityLines=nodes.get('fixed-delta').children.filter(n=>n.stroke==='#77dabb'&&typeof n.points==='string'&&n.class!=='series-ema');
  assert.deepEqual(qualityLines.map(n=>n.points.split(' ').length),[2,1]);
  assert.match(circles('fixed-delta').at(-1).children[0].textContent,/: 40 · action 200/);
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
  assert.deepEqual(labels(),['Prediction · 1 step: 5 %','Reconstruction: 0 %',
    'Prediction · 3 steps: 7 %','Latent drift: 11 %',
    'Threshold · Reconstruction / Prediction: 10 %','Threshold · Drift: 15 %']);
  const limits=()=>nodes.get('fixed-delta').children.filter(n=>n['stroke-dasharray']);
  assert.deepEqual(limits().map(n=>n.y1),[218.4,207.6]);
  assert.ok(limits().every(n=>n.y1===n.y2&&n['stroke-width']===1&&n['stroke-dasharray']==='2 4'));
  assert.match(text('quality-cadence'),/Quality, drift and activations every 256 actions/);
  const driftLines=()=>nodes.get('fixed-delta').children.filter(n=>n.stroke==='#94a7aa'&&typeof n.points==='string'&&n.class!=='series-ema');
  assert.deepEqual(driftLines().map(n=>n.points.split(' ').length),[1,1],'missing drift measurements leave a gap');
  report.evaluations.pop();
  await vm.runInContext('poll()',context);
  assert.equal(labels()[0],'Prediction · 1 step: 10 %');
  assert.equal(labels()[3],'Latent drift: 12 %');
  assert.match(nodes.get('fixed-delta-legend').children[3].title,/action 256/,'drift keeps the action of its actual measurement');
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
    ['Across observations: 3','Within each episode: 0.3']);
  assert.deepEqual(nodes.get('latent-rank-legend').children.map(n=>n.textContent),
    ['Effective rank: 4','Observable maximum: 5']);
  assert.equal(nodes.get('latent-inactive-legend').children[0].textContent,'Nearly constant: 25 %');
  assert.deepEqual(nodes.get('latent-sensitivity-legend').children.map(n=>n.textContent),
    ['Changed blocks: 0.005','Unchanged blocks: 0.001']);
  assert.equal(circles('latent-diversity').length,2,'only the instrumented evaluation contributes points');
  assert.ok(circles('latent-diversity').every(n=>n.cx===58),'unmeasured history is not a fake zero baseline');
  assert.match(text('latent-summary'),/16 observations from 3 episodes/);
  report.evaluations[3].quality.latent.changed_rms=null;
  report.evaluations[2].quality.latent.unchanged_rms=0;
  report.evaluations[3].quality.latent.unchanged_rms=0;
  await vm.runInContext('poll()',context);
  assert.deepEqual(nodes.get('latent-sensitivity-legend').children.map(n=>n.textContent),
    ['Changed blocks: —','Unchanged blocks: 0']);
  assert.equal(circles('latent-sensitivity').length,1,'a missing class is not zero response');
  report.evaluations=oldEvaluations;
  report.config.game='sc01-v1';report.config.initialization='random';
  await vm.runInContext('poll()',context);
  assert.match(text('run-info'),/^sc01-v1 · seed 0/);
  assert.match(text('run-info'),/Initialized from scratch/);
  report.config.initialization='checkpoint';
  await vm.runInContext('poll()',context);
  assert.match(text('run-info'),/Pretrained weights/);
  report.curves=[];report.episodes=[];report.evaluations=[];report.progress={phase:'loading',actions:0};
  await vm.runInContext('poll()',context);
  assert.equal(circles('delta').length,0,'no points borrowed from a previous run');
  assert.equal(circles('fixed-delta').length,0,'fixed evaluations reset with the selected run');
  assert.equal(labels()[0],'Prediction · 1 step: —','empty series has no stale last value');
  assert.equal(circles('latent-diversity').length,0);
  assert.equal(nodes.get('latent-rank-legend').children[0].textContent,'Effective rank: —');
  assert.equal(nodes.get('evaluation-rows').children.length,0);
  assert.equal(text('successes'),'Levels solved: 0 · recent window');
  assert.equal(nodes.get('advance-phase').disabled,true,'read-only dashboards cannot send controls');
  assert.equal(nodes.get('advance-level').disabled,true);
  report.curriculum.levels.forEach(row=>{row.game='sc01-v1';});
  controls={token:'session-token',target:{game:'sc01-v1',level:1},phase:true,level:true,pending:null,result:null};
  await vm.runInContext('poll()',context);
  assert.equal(text('control-target'),'Training level: sc01-v1 / 1');
  assert.equal(nodes.get('advance-phase').disabled,false);
  assert.equal(nodes.get('advance-phase').parentNode,nodes.get('curriculum-rows').children[0].children[1],'phase belongs to the current training level');
  assert.equal(nodes.get('advance-level').parentNode,nodes.get('curriculum-rows').children[1].children[0],'level advance belongs to the highest unlocked level, not the current practice level');
  let finishRequest;
  controlReply=new Promise(resolve=>{finishRequest=resolve;});
  const click=nodes.get('advance-phase').listeners.click();
  assert.equal(nodes.get('advance-phase').disabled,true,'disable before awaiting HTTP');
  assert.equal(nodes.get('advance-level').disabled,true);
  await nodes.get('advance-phase').listeners.click();
  assert.equal(controlRequests.length,1,'double click cannot send twice');
  assert.equal(controlRequests[0].url,'api/curriculum');
  assert.equal(controlRequests[0].options.headers['X-Curriculum-Token'],'session-token');
  assert.deepEqual(JSON.parse(controlRequests[0].options.body),{operation:'phase'});
  controls={...controls,phase:false,level:false,pending:{operation:'phase'}};
  finishRequest({ok:true,json:async()=>controls});await click;
  assert.match(text('control-status'),/phase pending/);
  await vm.runInContext('poll()',context);
  assert.equal(nodes.get('advance-phase').disabled,true,'poll preserves pending requests');
  controls={...controls,phase:true,level:true,pending:null,result:{status:'applied',event:{result:'advanced',level:1,phase:'predictor'}}};
  await vm.runInContext('poll()',context);
  assert.match(text('control-status'),/Advance applied · level 1 · phase predictor/);
  controls={...controls,target:{game:'sc01-v1',level:2}};
  await vm.runInContext('poll()',context);
  assert.equal(nodes.get('advance-phase').parentNode,nodes.get('curriculum-rows').children[1].children[1],'polling moves the same control to the new target');
  assert.equal(nodes.get('advance-level').parentNode,nodes.get('curriculum-rows').children[1].children[0]);
  controlReply={ok:false,json:async()=>({error:'Entrenador detenido'})};
  await nodes.get('advance-level').listeners.click();
  assert.deepEqual(JSON.parse(controlRequests[1].options.body),{operation:'level'});
  assert.match(text('control-status'),/Entrenador detenido.*will not be resent automatically/);
  assert.equal(nodes.get('advance-level').disabled,true);
  controls={...controls,phase:false,level:false,result:{status:'error',message:'El entrenador se detuvo.'}};
  await vm.runInContext('poll()',context);
  assert.equal(text('control-status'),'El entrenador se detuvo.');
  // Real palette bytes: initial white/gray, then two animation frames.
  const packed=(values,shape)=>({shape,pixels:Buffer.from(values).toString('base64')});
  const trajectory={kind:'planner_trajectory',id:'one',game:'sc01-v1',level:1,episode_seed:42,
    phase:'evaluation',actions:1,frame_count:3,reason:'game_over',
    initial:packed([0,1],[1,1,2]),steps:[{action:{id:4,data:{}},source:'planner',goal:'Cruzar',frames:packed([2,3,4,5],[2,1,2])}]};
  report.planner_trajectory={id:'one'};
  trajectoryReply={ok:true,json:async()=>({trajectory})};
  nodes.get('path').hidden=true;
  await vm.runInContext('poll()',context);
  assert.equal(trajectoryRequests,0,'hidden tab does not download frames');
  nodes.get('path').hidden=false;
  await vm.runInContext('poll()',context);
  assert.equal(trajectoryRequests,1);
  assert.match(text('trajectory-info'),/Evaluation.*1 actions.*Game over/);
  assert.deepEqual(nodes.get('observation').context.pixels.map(p=>p.color),['#ffffff','#cccccc']);
  await vm.runInContext('poll()',context);
  assert.equal(trajectoryRequests,1,'same sample is not downloaded twice');
  nodes.get('trajectory-next').listeners.click();
  assert.equal(nodes.get('trajectory-live').checked,false,'manual inspection holds the sample');
  assert.match(text('trajectory-step'),/Action 1\/1 · ACTION4 · Planner · frame 2\/3/);
  assert.deepEqual(nodes.get('observation').context.pixels.map(p=>p.color),['#999999','#666666']);
  nodes.get('trajectory-play').listeners.click();
  vm.runInContext('trajectoryTick()',context);
  assert.deepEqual(nodes.get('observation').context.pixels.map(p=>p.color),['#333333','#000000']);
  report.planner_trajectory={id:'two'};
  await vm.runInContext('poll()',context);
  assert.equal(trajectoryRequests,1,'polling cannot interrupt playback');
  nodes.get('trajectory-position').value='0';nodes.get('trajectory-position').listeners.input();
  assert.match(text('trajectory-step'),/Initial state/);
  const nextTrajectory={...trajectory,id:'two',phase:'training',steps:[{...trajectory.steps[0],source:'exploration'}]};
  trajectoryReply={ok:true,json:async()=>({trajectory:nextTrajectory})};
  nodes.get('trajectory-live').checked=true;
  await vm.runInContext('poll()',context);
  assert.equal(trajectoryRequests,2);
  nodes.get('trajectory-next').listeners.click();
  assert.match(text('trajectory-step'),/Random exploration/);
  report.planner_trajectory={id:'three'};nodes.get('trajectory-live').checked=true;
  trajectoryReply={ok:false,status:503};
  await vm.runInContext('poll()',context);
  assert.match(text('trajectory-info'),/Could not load trajectory.*503/);
  assert.doesNotMatch(text('connection'),/Disconnected/);
  trajectoryReply={ok:true,json:async()=>({trajectory:null})};
  await vm.runInContext('poll()',context);
  assert.match(text('trajectory-info'),/Waiting for the complete new sample/);
  assert.match(text('goal-detail'),/older sample/);
  // Every predicted destination is aligned with the actual time column. A
  // route from t0 uses different actions after step 1; later goals can change.
  const action=id=>({id,data:{}});
  const predicted=(id,color)=>({action:action(id),frames:packed([color,color],[1,1,2])});
  const modern={...trajectory,id:'matrix',format_version:2,actions:3,frame_count:5,steps:[
    {...trajectory.steps[0],goal_frame:packed([9,16],[1,1,2]),imagination_status:'planned',imagination_source:'searched_continuation',
      selected_estimate:{success_probability:.91,successful_actions:8,remaining_budget:128},
      forecast_estimate:{success_probability:.90,successful_actions:6,remaining_budget:128},
      imagination:[predicted(4,6),predicted(3,7),predicted(2,8)]},
    {action:action(2),source:'planner',goal:'Segunda meta',goal_frame:packed([10,10],[1,1,2]),frames:packed([8,8],[1,1,2]),
      imagination_status:'planned',imagination_source:'searched_alternative',imagination:[predicted(2,9),predicted(1,10),predicted(7,11)]},
    {action:action(1),source:'exploration',goal:'Segunda meta',goal_frame:packed([10,10],[1,1,2]),frames:packed([9,9],[1,1,2]),
      imagination_status:'exploration',imagination:[]}]};
  report.planner_trajectory={id:'matrix'};trajectoryReply={ok:true,json:async()=>({trajectory:modern})};
  await vm.runInContext('poll()',context);
  const descendants=node=>[node,...node.children.flatMap(descendants)];
  const matrix=nodes.get('trajectory-matrix'), all=descendants(matrix);
  assert.ok(all.some(n=>n.textContent==='Explored continuation'));
  assert.ok(all.some(n=>n.textContent==='Choice · 91 % · 8 actions'));
  assert.ok(all.some(n=>n.textContent==='Preview · 90 % · 6 actions'));
  assert.ok(all.some(n=>n.title?.includes('128 actions available')));
  assert.ok(all.some(n=>n.textContent==='Explored alternative'));
  const cell=(origin,time)=>all.find(n=>n.className==='trajectory-cell trajectory-prediction'&&n['data-origin']===origin&&n['data-time']===time);
  assert.equal(cell(0,1)['data-diverged'],'false');
  assert.equal(cell(0,2)['data-diverged'],'true');
  assert.equal(cell(0,3)['data-diverged'],'true','divergence persists after an earlier action differs');
  assert.equal(cell(1,2)['data-diverged'],'false');
  assert.equal(cell(1,3)['data-diverged'],'false');
  assert.equal(cell(1,4).children.at(-1).textContent,'No real outcome');
  assert.equal(cell(1,4).children[0]['aria-label'],'Prediction of t = 4 from t = 1 · A7 ↶ Undo');
  assert.equal(all.find(n=>n.className==='trajectory-origin-button')['aria-label'],'View origin t = 0');
  const realRow=all.find(n=>n.className==='trajectory-real');
  assert.equal(realRow.children[2].children[0]['aria-label'],'View real state t = 1 · A4 →');
  assert.equal(realRow.children[2].children[0].children[0]['aria-label'],'Real state t = 1');
  assert.equal(realRow.children.at(-1).children[0].textContent,'Not observed');
  assert.ok(all.some(n=>n.textContent==='Random exploration · no imagined route'));
  assert.deepEqual(realRow.children[2].children[0].children[0].context.pixels.map(p=>p.color),['#333333','#000000'],
    'real t1 thumbnail is the final animation frame, not its intermediate frame');
  assert.deepEqual(cell(0,1).children[0].context.pixels.map(p=>p.color),['#e53aa3','#e53aa3']);
  const selectedButton=realRow.children[3].children[0];
  selectedButton.listeners.click();
  assert.equal(nodes.get('trajectory-live').checked,false);
  assert.equal(selectedButton['aria-pressed'],'true');
  assert.equal(cell(0,2)['data-selected'],'true');
  assert.equal(cell(1,2)['data-selected'],'true');
  assert.equal(text('goal-context'),'Segunda meta');
  assert.deepEqual(nodes.get('observation').context.pixels.map(p=>p.color),['#f93c31','#f93c31']);
  const firstMatrixChild=matrix.children[0], downloads=trajectoryRequests;
  report.progress.observation=[[15]];report.progress.goal_frame=[[15]];
  await vm.runInContext('poll()',context);
  assert.equal(matrix.children[0],firstMatrixChild,'polling preserves matrix DOM and selected sample');
  assert.equal(trajectoryRequests,downloads);
  assert.deepEqual(nodes.get('observation').context.pixels.map(p=>p.color),['#f93c31','#f93c31']);
  nodes.get('trajectory-live').checked=true;
  await vm.runInContext('poll()',context);
  assert.equal(matrix.children[0],firstMatrixChild,'same ID does not rebuild thumbnails even in live mode');
  report.planner_trajectory={id:'broken'};
  trajectoryReply={ok:true,json:async()=>({trajectory:{...modern,id:'broken',steps:[
    {...modern.steps[0],imagination:[{...predicted(4,6),frames:packed([6],[1,1,2])}]},...modern.steps.slice(1)]}})};
  await vm.runInContext('poll()',context);
  assert.match(text('trajectory-info'),/Frames incomplete or too large/);
  assert.equal(matrix.children[0],firstMatrixChild,'malformed new sample preserves last complete view');
  delete report.planner_trajectory;
  delete report.progress.recent_tps;report.progress.tps=99;
  await vm.runInContext('poll()',context);
  assert.equal(nodes.get('trajectory-viewer').hidden,true);
  assert.equal(text('tps'),'—','never relabel legacy cumulative TPS as recent');
  controls={...controls,phase:false,level:false,result:{status:'applied',event:{result:'last_level',level:5,phase:'planner'}}};
  await vm.runInContext('poll()',context);
  assert.match(text('control-status'),/The final level was already unlocked/);
  assert.equal(nodes.get('advance-phase').disabled,true);
  assert.equal(nodes.get('advance-level').disabled,true);
  report.config.training_plan={version:1};
  report.curriculum={stage:2,index:1,unlocked:{'sc01-v1':2},thresholds:{patience:2,representation_open:.1,
    prediction_open:.2,latent_drift:.15,success:.8},
    ordered_stages:[{game:'sc01-v1',level:5,objective:'world_model'},
      {game:'sc01-v1',level:2,objective:'world_model'},{game:'bf01-v1',level:1,objective:'world_model'}],
    levels:[{game:'sc01-v1',level:5,unlocked:true,predictor:true,objective:'world_model'},
      {game:'sc01-v1',level:2,unlocked:true,predictor:true,objective:'world_model'},
      {game:'bf01-v1',level:1,unlocked:false,objective:'world_model'}]};
  report.status='running';report.progress.phase='training';
  controls={target:{game:'sc01-v1',level:2},phase:false,level:true};
  await vm.runInContext('poll()',context);
  assert.match(text('curriculum-summary'),/Stage 2 \/ 3/);
  assert.match(text('thresholds'),/no subgoal teacher/);
  assert.match(text('goal-detail'),/without a subgoal teacher/);
  assert.equal(nodes.get('advance-level').parentNode,nodes.get('curriculum-rows').children[1].children[0],
    'configured order owns the next-stage control, not the highest numeric visited level');
  report.status='budget_exhausted';report.progress.phase='budget_exhausted';
  await vm.runInContext('poll()',context);
  assert.match(text('connection'),/Budget exhausted/);
  offline=true;await vm.runInContext('poll()',context);
  assert.equal(nodes.get('advance-phase').parentNode,nodes.get('curriculum-action-parking'),'unavailable controls leave the table');
  assert.equal(nodes.get('advance-level').disabled,true);
  assert.match(text('connection'),/Disconnected.*offline/);
  assert.equal(text('gpu-status'),'GPU · data unavailable','connection failure clears stale GPU readings');
  // The full historical view uses separate endpoints and cannot send curriculum commands.
  const historyRequests=[];
  const history=vm.createContext({console,AbortSignal,setTimeout(){},clearTimeout(){},
    location:{pathname:'/arc3-training/history.html',hash:'#path'},document:context.document,
    fetch:async(url,options)=>{historyRequests.push({url,method:options?.method});
      return {ok:true,json:async()=>({subgoals:{...report,historical:true,status:'historical'},
        curriculum_control:{target:{game:'sc01-v1',level:2},phase:true,level:true}})};}});
  vm.runInContext(fs.readFileSync(path.join(assets,'chart-tools.js'),'utf8'),history);
  vm.runInContext(fs.readFileSync(path.join(assets,'subgoals.js'),'utf8'),history);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(text('connection'),'History · read only');
  assert.equal(nodes.get('path').hidden,false,'archived trajectory deep link opens its panel');
  nodes.get('tab-path').listeners.keydown({key:'End',preventDefault(){}});
  assert.equal(nodes.get('curriculum').hidden,false);
  await vm.runInContext('poll()',history);
  assert.equal(nodes.get('curriculum').hidden,false,'refresh preserves keyboard-selected panel');
  assert.match(text('notice'),/do not belong to the current sequence training/);
  assert.equal(nodes.get('advance-phase').disabled,true);
  assert.equal(nodes.get('advance-level').disabled,true);
  await vm.runInContext("advanceCurriculum('level')",history);
  assert.ok(historyRequests.every(r=>r.method!=='POST'));
  assert.ok(historyRequests.some(r=>r.url==='api/history/state'));
  assert.ok(historyRequests.every(r=>r.url!=='api/state'));
  // Embedded archives own one section, use only archived data and fit their contents.
  let resized, measuredHeight=900;
  const messages=[],embeddedRequests=[];
  const embeddedReport={...report,historical:true,status:'historical'};
  const embeddedNodes=new Map([...nodes.keys()].map(id=>[id,element()]));
  embeddedNodes.get('history-content').getBoundingClientRect=()=>({height:measuredHeight});
  const embedded=vm.createContext({console,AbortSignal,setTimeout(){},clearTimeout(){},
    location:{pathname:'/arc3-training/history.html',hash:'#path',origin:'https://dashboard.test'},
    document:{body:{dataset:{historySection:'world'}},getElementById:id=>embeddedNodes.get(id),
      createElement:element,createElementNS:element},
    ResizeObserver:class {constructor(callback){resized=callback;}
      observe(target){assert.equal(target,embeddedNodes.get('history-content'));}},
    parent:{postMessage(message,origin){messages.push({message,origin});}},
    fetch:async(url,options)=>{embeddedRequests.push({url,method:options?.method});
      return {ok:true,json:async()=>({subgoals:embeddedReport})};}});
  vm.runInContext(fs.readFileSync(path.join(assets,'chart-tools.js'),'utf8'),embedded);
  vm.runInContext(fs.readFileSync(path.join(assets,'subgoals.js'),'utf8'),embedded);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(embeddedNodes.get('world').hidden,false,'embedded section overrides an unrelated fragment');
  assert.equal(embeddedNodes.get('path').hidden,true);
  assert.match(embeddedNodes.get('archive-context').textContent,/Previous run.*read only/);
  assert.equal(embeddedNodes.get('advance-phase').disabled,true);
  assert.equal(embeddedNodes.get('advance-level').disabled,true);
  await vm.runInContext("advanceCurriculum('phase')",embedded);
  assert.ok(embeddedRequests.every(r=>r.method!=='POST'&&r.url.startsWith('api/history/')));
  resized();resized();
  assert.equal(messages.length,1,'unchanged height does not trigger another resize');
  assert.equal(messages[0].origin,'https://dashboard.test');
  assert.equal(messages[0].message.section,'world');
  assert.equal(messages[0].message.height,916);
  measuredHeight=1200;resized();
  assert.equal(messages[1].message.height,1216);
  embeddedReport.status='waiting';
  await vm.runInContext('poll()',embedded);
  assert.match(embeddedNodes.get('archive-context').textContent,/Archived data unavailable/);
  vm.runInContext("plot('speed',[{name:'Prueba',points:[[0,0],[1,10],[2,20]]}])",context);
  const legendButton=nodes.get('speed-legend').children[0];
  const lastGuide=()=>nodes.get('speed').children.find(n=>n.class==='series-guide');
  assert.equal(nodes.get('speed').children.filter(n=>n.class==='series-ema').length,1);
  legendButton.listeners.click();
  legendButton.listeners.blur();legendButton.listeners.pointerleave();
  assert.equal(lastGuide().children[3].textContent,'2.9','pinned EMA guide survives blur and pointer leave');
  vm.runInContext("plot('speed',[{name:'Prueba',points:[[0,0],[1,10],[2,20],[3,0]]}])",context);
  assert.equal(nodes.get('speed-legend').children[0],legendButton);
  assert.equal(lastGuide().children[3].textContent,'2.61');
  legendButton.listeners.click();
  assert.equal(lastGuide().children.length,0,'second click removes the guide');
  legendButton.listeners.click();
  vm.runInContext("plot('speed',[{name:'Prueba',points:[[0,null]]}])",context);
  assert.equal(legendButton.disabled,true);
  assert.equal(legendButton['aria-pressed'],'false');
  console.log('Subgoal dashboard contracts passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
