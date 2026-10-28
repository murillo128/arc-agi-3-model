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
let archiveState=null,archivePayload=null,archiveRequests=0,archiveOffline=false;
const context = vm.createContext({console, setTimeout() {}, fixture: report, location:{hash:'#curriculum'},
  document: {getElementById: id => nodes.get(id), createElement: element, createElementNS: element},
  fetch: async url => {
    if(offline)throw Error('offline');
    if(url==='api/sequence-history'){
      archiveRequests++;
      if(archiveOffline)throw Error('archive offline');
      return {ok:true,status:200,headers:{get:()=>archiveState.revision},json:async()=>archivePayload};
    }
    return {ok:true,json:async()=>({sequences:report,history_available:true,sequence_history:archiveState})};
  }});
const text = id => nodes.get(id).textContent;
(async () => {
  vm.runInContext(fs.readFileSync(path.join(assets,'chart-tools.js'),'utf8'),context);
  vm.runInContext(fs.readFileSync(path.join(assets,'sequence-trends.js'),'utf8'),context);
  vm.runInContext(fs.readFileSync(path.join(assets, 'sequences.js'), 'utf8'), context);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(text('status'), 'Paused');
  assert.equal(text('exploration-policy'), 'Uniform exploration');
  assert.equal(text('exploration-head-info'), 'No learned-head measurements yet');
  assert.equal(nodes.get('interaction-change-legend').children[0].textContent,'Changed actions %: —');
  assert.equal(nodes.get('planning-details').hidden,true);
  report.planning={goal_updates:16,path_updates:12,learned_probability:.125,
    config:{exploration:.25,horizon:2,beam:4}};
  report.history[0].planning={identity:'0',teacher_actions:8,episodes:2,censored:1,
    goal_loss:2.,path_loss:.5,goal_color_error:.2,goal_mask_error:.1,
    teacher_attempts:2,teacher_reached:1,teacher_steps:4,teacher_task_progress:1,teacher_level_successes:0,
    learned_attempts:1,learned_reached:1,learned_steps:2,learned_task_progress:0,learned_level_successes:0};
  report.planning_sample={batch:1,game:'sc01-v1',before:[[0,1]],target:[[2,3]],mask:[[true,false]],
    predicted:[[4,5]],predicted_mask:[[false,true]]};
  await vm.runInContext('refresh()',context);
  assert.equal(nodes.get('planning-details').hidden,false);
  assert.match(text('planning-status'),/Learned-goal share: 12.5%/);
  assert.equal(nodes.get('planning-success-legend').children[0].textContent,'Teacher %: 50');
  assert.equal(nodes.get('planning-success-legend').children[1].textContent,'Learned image match %: 100');
  assert.equal(nodes.get('planning-outcomes').children[1].children[3].textContent,'0',
    'Matching a learned image must not be displayed as genuine task progress');
  assert.equal(nodes.get('planning-target').context.pixels[1].color,'#aaa');
  assert.equal(nodes.get('planning-predicted').context.pixels[0].color,'#eee');
  report.history[0].planning.teacher_attempts=0;
  report.history[0].planning.teacher_reached=0;
  await vm.runInContext('refresh()',context);
  assert.equal(nodes.get('planning-success-legend').children[0].textContent,'Teacher %: —');
  assert.equal(nodes.get('planning-steps-legend').children[0].textContent,'Teacher: —');
  delete report.planning;delete report.planning_sample;delete report.history[0].planning;
  await vm.runInContext('refresh()',context);
  assert.equal(nodes.get('planning-details').hidden,true);
  report.exploration_config={policy:'learned',uniform_probability:.25};
  report.exploration_adoptions=[{batch:0}];
  report.history[0].exploration={policy:'learned',identity:'new',eligible_transitions:100,changed_actions:40,
    eligible_clicks:20,changed_clicks:12,head:{loss:.2,total_updates:16,examples_seen:100},games:[
      {game:'click-game',eligible_transitions:100,changed_actions:40,eligible_clicks:20,changed_clicks:12,wins:1},
      {game:'no-click',eligible_transitions:10,changed_actions:0,eligible_clicks:0,changed_clicks:0,wins:0}]};
  await vm.runInContext('refresh()',context);
  assert.equal(text('exploration-policy'),'Learned change prior · 25% uniform clicks · active from batch 1');
  assert.equal(nodes.get('interaction-change-legend').children[1].textContent,'Changed clicks %: 60');
  assert.equal(nodes.get('exploration-games').children[0].children[3].textContent,'60 %');
  assert.equal(nodes.get('exploration-games').children[1].children[3].textContent,'—');
  assert.match(text('exploration-head-info'),/16 auxiliary updates/);
  delete report.exploration_config; delete report.exploration_adoptions; delete report.history[0].exploration;
  await vm.runInContext('refresh()',context);

  assert.equal(text('run-info'), '1 game', 'configuration summarizes game count without IDs');
  assert.equal(text('curriculum-games'), '<game>', 'game IDs remain literal text in Curriculum');
  assert.equal(text('freshness'), 'Updated 2 s ago');
  assert.equal(nodes.get('curriculum').hidden, false, 'initial section hash is respected');
  nodes.get('run-configuration').open = true;
  report.age_seconds = 4;
  await vm.runInContext('refresh()',context);
  assert.equal(nodes.get('run-configuration').open, true, 'polling keeps the native disclosure open');
  assert.equal(nodes.get('curriculum').hidden, false, 'polling keeps the deep-linked section');
  assert.equal(text('freshness'), 'Updated 4 s ago');
  nodes.get('tab-curriculum').listeners.keydown({key:'Home',preventDefault(){}});
  assert.equal(nodes.get('world').hidden, false);
  assert.equal(nodes.get('tab-world')['aria-selected'], 'true');
  nodes.get('tab-world').listeners.keydown({key:'End',preventDefault(){}});
  assert.equal(nodes.get('curriculum').hidden, false);
  assert.equal(text('batches'), '1 completed batches');
  report.budget=32;
  await vm.runInContext('refresh()',context);
  assert.equal(text('batches'), '1 completed batches · budget 32');
  report.budget=null;
  await vm.runInContext('refresh()',context);
  assert.equal(text('batches'), '1 completed batches · Unlimited actions');
  delete report.budget;
  await vm.runInContext('refresh()',context);
  assert.equal(text('batches'), '1 completed batches');
  assert.equal(text('tps'), '40 actions/s');
  assert.equal(text('examples'), '80 transitions/s');
  assert.match(text('configuration'), /16 steps \+ 16 context steps/);
  report.config.game_schedule='shuffled';
  report.config.batch_games=8;
  await vm.runInContext('refresh()',context);
  assert.match(text('configuration'), /^Up to 8 independent sessions/);
  assert.match(text('configuration'), /shuffled without repeats/);
  assert.match(text('schedule-info'), /Each game once per round/);
  assert.match(text('schedule-info'), /last batch may be smaller/);
  report.game_weights={'<game>':3}; report.game_weight_adoptions=[{batch:0}]; report.sampling_start_batch=0;
  await vm.runInContext('refresh()',context);
  assert.match(text('configuration'), /Weighted shuffled rounds · 3 sessions per round/);
  assert.match(text('sampling-info'), /Current frequencies apply from batch 1/);
  assert.deepEqual(nodes.get('sampling-games').children[0].children.map(n=>n.textContent),
    ['<game>','3×','100 %','1','512']);
  report.sampling_start_batch=1;
  await vm.runInContext('refresh()',context);
  assert.equal(nodes.get('sampling-games').children[0].children[3].textContent,'0','old sessions do not count toward a new weighting');
  delete report.game_weights; delete report.game_weight_adoptions; delete report.sampling_start_batch;
  await vm.runInContext('refresh()',context);
  assert.equal(nodes.get('quality-legend').children[0].textContent, 'Early %: 0');
  assert.equal(nodes.get('loss-legend').children[0].textContent, 'Loss: 0');
  assert.equal(text('macro-error'), 'Average across games: 25 %');
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
  assert.equal(nodes.get('predictor-components-legend').children[1].textContent,'Visual loss: 0.75');
  assert.equal(nodes.get('representation-loss-legend').children[0].textContent,'Loss encoder / decoder: 0');
  assert.equal(nodes.get('changes-legend').children[0].textContent,'Invented: 1');
  assert.equal(nodes.get('changes-legend').children[1].textContent,'Missed: 1.5');
  assert.equal(nodes.get('latent-diversity-legend').children[0].textContent,'Across observations: 0');
  assert.equal(nodes.get('latent-drift').children[0].textContent,'No measurements yet');
  assert.equal(nodes.get('evaluations').children[0].children[3].textContent,'0 %');
  assert.equal(nodes.get('preview-images').hidden,false);
  assert.deepEqual(nodes.get('preview-legend').children.map(n=>n.textContent),
    ['Invented: 1 px','Missed: 1 px','Wrong color: 1 px','Correct: 1 px']);
  assert.deepEqual(nodes.get('preview-real').context.pixels.map(p=>p.color),
    ['#1e93ff','#1e93ff','#1e93ff','#aaa']);
  report.measurement_source='training_pre_update';
  report.phase_control={phase:'representation',window_batches:4,rows:[{game:'<game>',level:2,samples:8,changed:1,
    reconstruction_error:.2,latent_drift:.01,enough:false,passing:false,passes:0,
    thresholds:{minimum_samples:8,minimum_changed:4,representation_open:.1,latent_drift:.15,patience:3}}]};
  report.learning_rates={encoder:3e-5,decoder:3e-5,predictor:3e-5};
  await vm.runInContext('refresh()',context);
  assert.match(text('phase-status'),/predictor waiting/);
  assert.equal(text('tab-evaluation'),'Training measurements');
  assert.equal(nodes.get('phase-card').hidden,false);
  assert.equal(nodes.get('phase-rows').children[0].children[0].textContent,'<game> / 2');
  assert.equal(nodes.get('phase-rows').children[0].children[2].textContent,'20 % / 10 %');
  assert.equal(nodes.get('phase-rows').children[0].children[5].textContent,'Ignored · insufficient transitions');
  assert.match(text('phase-description'),/ignored for admission, not marked as passed/);
  assert.match(text('phase-description'),/At least one eligible level/);
  report.phase_control.rows[0].required=false;
  await vm.runInContext('refresh()',context);
  assert.equal(nodes.get('phase-rows').children[0].children[5].textContent,'Not reached yet');
  report.phase_control.rows[0].required=true;
  report.phase_control.rows[0].observed=false;
  await vm.runInContext('refresh()',context);
  assert.equal(nodes.get('phase-rows').children[0].children[5].textContent,'Awaiting first observation');
  report.phase_control.rows[0].observed=true;
  report.phase_control.rows[0].enough=true;
  await vm.runInContext('refresh()',context);
  assert.equal(nodes.get('phase-rows').children[0].children[5].textContent,'Outside threshold');
  report.phase_control.rows[0].passing=true;
  report.phase_control.rows[0].passes=2;
  await vm.runInContext('refresh()',context);
  assert.equal(nodes.get('phase-rows').children[0].children[5].textContent,'Awaiting confirmations');
  report.phase_control.rows[0].passes=3;
  await vm.runInContext('refresh()',context);
  assert.equal(nodes.get('phase-rows').children[0].children[5].textContent,'Ready');
  assert.match(text('preview-info'),/before training batch/);
  assert.match(text('drift-description'),/before and after updating/);
  // The phase metric has a different denominator from global reconstruction.
  const guides=id=>nodes.get(id).children.filter(n=>n['stroke-dasharray']);
  const labels=id=>nodes.get(id).children.map(n=>n.textContent);
  const diagnostic=report.history[0].diagnostics;
  Object.assign(diagnostic,{reconstruction_error:.00675,reconstruction_delta_error:.73,latent_drift:.02});
  report.phase_control.rows[0].thresholds.prediction_open=.99;
  await vm.runInContext('refresh()',context);
  assert(labels('fixed-quality-legend').includes('Global reconstruction %: 0.675'));
  assert(labels('fixed-quality-legend').includes('Reconstruction on changes %: 73'));
  let [limit]=guides('fixed-quality');
  assert.equal(guides('fixed-quality').length,1);
  assert.equal(limit.y1,limit.y2);
  assert(Math.abs(limit.y1-201.2)<1e-9,'10% on the 0–100 quality axis');
  assert.equal(limit.children[0].textContent,'Reconstruction-on-changes threshold: 10 %');
  const driftGuide=guides('latent-drift')[0];
  assert.equal(driftGuide.children[0].textContent,'Drift threshold: 15 %');
  assert(driftGuide.y1>=32&&driftGuide.y1<220,'threshold remains within the plot when measured drift is only 2%');
  assert.equal(driftGuide.y1,driftGuide.y2);
  // Configuration changes, duplicate limits, and differing game/level limits.
  const control=report.phase_control, row=control.rows[0];
  row.thresholds.representation_open=.12;
  row.thresholds.latent_drift=.2;
  control.rows.push({...row,level:3});
  await vm.runInContext('refresh()',context);
  assert.equal(guides('fixed-quality').length,1,'identical limits share one line');
  assert.equal(guides('latent-drift').length,1);
  assert.equal(guides('latent-drift')[0].children[0].textContent,'Drift threshold: 20 %');
  assert(Math.abs(guides('fixed-quality')[0].y1-197.44)<1e-9,'12%, not a hard-coded 10%');
  control.rows.push({...row,game:'other',level:1,thresholds:{...row.thresholds,representation_open:.08}});
  await vm.runInContext('refresh()',context);
  assert.deepEqual(guides('fixed-quality').map(n=>n.children[0].textContent),[
    'Reconstruction-on-changes threshold · other / 1: 8 %',
    'Reconstruction-on-changes threshold · <game> / 2, <game> / 3: 12 %']);
  // Absence must not manufacture either zero error or a default phase gate.
  delete diagnostic.reconstruction_delta_error;
  delete report.phase_control;
  await vm.runInContext('refresh()',context);
  assert(labels('fixed-quality-legend').includes('Reconstruction on changes %: —'));
  assert.equal(guides('fixed-quality').length,0);
  assert.equal(guides('latent-drift').length,0);
  diagnostic.reconstruction_delta_error=0;
  report.phase_control={...control,rows:[{...row,thresholds:{...row.thresholds,representation_open:null,latent_drift:2}}]};
  await vm.runInContext('refresh()',context);
  assert(labels('fixed-quality-legend').includes('Reconstruction on changes %: 0'));
  assert.equal(guides('fixed-quality').length,0);
  assert.equal(guides('latent-drift').length,0);
  report.phase_control=control;
  report.phase_control.phase='predictor';
  report.learning_rates.encoder=report.learning_rates.decoder=3e-7;
  await vm.runInContext('refresh()',context);
  assert.match(text('phase-status'),/reduced representation LR/);
  assert.match(text('learning-rates'),/LR decoder: 3.0e-7/);
  report.phase_control.representation_frozen_at_batch=1;
  report.phase_control.rows[0].representation_frozen=true;
  report.learning_rates.encoder=report.learning_rates.decoder=null;
  await vm.runInContext('refresh()',context);
  assert.match(text('phase-status'),/representation frozen after batch 1/);
  assert.match(text('learning-rates'),/LR encoder: Frozen/);
  assert.match(text('learning-rates'),/LR decoder: Frozen/);
  assert.match(text('learning-rates'),/LR predictor: 3.0e-5/);
  assert.match(text('drift-description'),/Drift measurement is skipped/);
  assert.equal(nodes.get('phase-rows').children[0].children[3].textContent,'Frozen');
  delete report.phase_control.representation_frozen_at_batch;
  delete report.phase_control.rows[0].representation_frozen;
  report.learning_rates.encoder=report.learning_rates.decoder=3e-7;
  report.prediction_samples=[];
  report.history = [];
  await vm.runInContext('refresh()', context);
  assert.equal(text('tps'), '— actions/s');
  assert.equal(nodes.get('game-quality').children[0].children[1].textContent, '— %');
  assert.equal(nodes.get('sessions').children.length, 0);
  assert.equal(nodes.get('quality').children[0].textContent, 'No measurements yet');
  assert.equal(nodes.get('preview-images').hidden,true);
  assert.equal(nodes.get('preview-before').context.pixels.length,0);
  offline = true;
  await vm.runInContext('refresh()', context);
  assert.equal(text('status'), 'Disconnected');
  // Actual renderer: EMA 0 -> 10 -> 19; guide and badge use its endpoint.
  context.chartRows=[{batch:1,v:0},{batch:2,v:100},{batch:3,v:100}];
  const draw=()=>vm.runInContext("chart('quality',chartRows,[series('Prueba',r=>r.v)],100)",context);
  draw();
  const smooth=()=>nodes.get('quality').children.filter(n=>n.class==='series-ema');
  assert.equal(smooth().length,1);
  assert.equal(smooth()[0]['stroke-width'],1.8);
  assert.equal(smooth()[0].points,'50,220 365,201.2 680,184.28');
  assert.equal(smooth()[0].stroke,nodes.get('quality-legend').children[0].style.color);
  const selected=nodes.get('quality-legend').children[0];
  const guide=()=>nodes.get('quality').children.find(n=>n.class==='series-guide');
  selected.listeners.click();
  assert.equal(selected['aria-pressed'],'true');
  assert.equal(guide().children[0].y1,184.28);
  assert.equal(guide().children[0]['stroke-dasharray'],'2 4');
  assert.equal(guide().children[2].x,2);
  assert.equal(guide().children[3].textContent,'19');
  context.chartRows.push({batch:4,v:0});
  draw();
  assert.equal(nodes.get('quality-legend').children[0],selected,'polling retains the focused button');
  assert.equal(selected['aria-pressed'],'true');
  assert.equal(guide().children[0].y1,187.852);
  assert.equal(guide().children[3].textContent,'17.1','zero contributes to the EMA');
  selected.listeners.click();
  assert.equal(guide().children.length,0);
  selected.listeners.click();
  context.chartRows=[{batch:1,v:null}];
  draw();
  assert.equal(selected.disabled,true);
  assert.equal(selected['aria-pressed'],'false');
  // Separate measurement sources never blend; missing rows break, explicit skips do not.
  context.chartRows=[{batch:1,v:0},{batch:2,v:100},{batch:3,v:100,measurement_source:'training'},
    {batch:4,v:null,measurement_source:'training'},{batch:5,v:50,measurement_source:'training'}];
  draw();
  assert.equal(smooth().length,3);
  assert.equal(smooth()[1].points,'365,32','new source begins at its actual value');
  context.chartRows=[{batch:1,v:0},{batch:2,v:null,skip:true},{batch:3,v:100}];
  vm.runInContext("chart('quality',chartRows,[series('Prueba',r=>r.v,{skip:r=>r.skip})],100)",context);
  assert.equal(smooth().length,1);
  assert.equal(smooth()[0].points,'50,220 680,201.2');
  // Periodic diagnostics skip present, unsampled batches without resetting their EMA.
  // 0 -> 1 -> 1 with alpha=.1 ends at .19 (19 for percentage plots).
  offline=false;
  const periodic=Array.from({length:8},(_,i)=>({batch:i+1}));
  for(const batch of [1,4,8]) {
    const value=batch===1?0:1;
    periodic[batch-1].diagnostics={latent_drift:value,drift_reference_batch:1,
      latent:{std_mean:value,effective_rank:value,inactive_fraction:value,changed_rms:value}};
  }
  report.history=periodic;
  await vm.runInContext('refresh()',context);
  const diagnosticCharts=['latent-drift','latent-diversity','latent-rank','latent-inactive','latent-sensitivity'];
  const diagnosticLines=id=>nodes.get(id).children.filter(n=>n.class==='series-ema');
  for(const id of diagnosticCharts) {
    assert.equal(diagnosticLines(id).length,1,`${id}: unsampled batches keep one segment`);
    nodes.get(id+'-legend').children[0].listeners.click();
    const selectedGuide=nodes.get(id).children.find(n=>n.class==='series-guide');
    assert.equal(selectedGuide.children[3].textContent,id==='latent-drift'||id==='latent-inactive'?'19':'0.19');
  }
  // Actual missing batches and regime changes on unsampled rows still break the line.
  for(const interruption of ['gap','measurement','exploration','phase','lr','reference']) {
    report.history=structuredClone(periodic);
    if(interruption==='gap')report.history.splice(5,1);
    else if(interruption==='reference')report.history[7].diagnostics.drift_reference_batch=2;
    else Object.assign(report.history[5],{
      measurement:{measurement_source:'training_pre_update'},exploration:{exploration:{policy:'learned'}},
      phase:{phase:'predictor'},lr:{learning_rates:{predictor:1e-4}}
    }[interruption]);
    await vm.runInContext('refresh()',context);
    for(const id of interruption==='reference'?['latent-drift']:diagnosticCharts)
      assert.equal(diagnosticLines(id).length,2,`${id}: ${interruption} breaks the segment`);
  }
  // The archive expands charts, not live counters; window controls persist through polling.
  offline=false;
  const history=Array.from({length:1200},(_,i)=>({batch:i+1,actions:(i+1)*8,loss:.2,
    measurement_source:'training_pre_update',quality:{'<game>':{wrong:i<1100?2:1,union:10,transitions:8},all:{delta_error:.2}}}));
  archiveState={enabled:true,run:'test-run',revision:'"1"',limit:2000};
  archivePayload={run:'test-run',history};
  report.history_run='test-run';
  report.history=history.slice(-200);report.progress={actions:9600,batches:1200};
  nodes.get('history-range').value='all';nodes.get('trend-window').value='100';
  await vm.runInContext('refresh()',context);
  assert.equal(archiveRequests,1);
  assert.equal(text('actions'),'9,600');
  assert.equal(text('trend-balanced'),'10 %');
  assert.equal(text('trend-change'),'-10 pp');
  assert.match(text('trend-comparison'),/1,101–1,200 vs batches 1,001–1,100/);
  assert.match(text('history-scope'),/1,200 retained batches, starting at 1/);
  assert(nodes.get('loss').children.filter(n=>n.cx!==undefined).length<=351,'markers stay bounded on long histories');
  nodes.get('history-range').value='200';nodes.get('history-range').listeners.change();
  assert.match(text('history-scope'),/Showing 1,001–1,200/);
  nodes.get('trend-zoom').checked=true;nodes.get('trend-zoom').listeners.change();
  assert.match(text('trend-comparison'),/Y-axis zoomed/);
  await vm.runInContext('refresh()',context);
  assert.equal(archiveRequests,1,'unchanged archive is not re-downloaded on each live poll');
  assert.equal(nodes.get('history-range').value,'200');
  archiveState.revision='"2"';archiveOffline=true;
  await vm.runInContext('refresh()',context);
  assert.equal(text('status'),'Paused','archive failure does not disconnect live measurements');
  assert.match(text('history-scope'),/temporarily unavailable/);
  archiveState.run='other-run';
  await vm.runInContext('refresh()',context);
  assert.match(text('history-scope'),/200 retained batches/,'a new run cannot inherit the previous archive');
  report.updated_at=1234;
  await vm.runInContext('refresh()',context);
  const priorPlot=nodes.get('loss').children[0];
  report.age_seconds=10;
  await vm.runInContext('refresh()',context);
  assert.equal(nodes.get('loss').children[0],priorPlot,'unchanged metrics do not rebuild the charts');
  assert.equal(text('freshness'),'Updated 10 s ago');
  offline=true;await vm.runInContext('refresh()',context);
  assert.equal(text('status'),'Disconnected');
  offline=false;await vm.runInContext('refresh()',context);
  assert.equal(text('status'),'Paused','recovery restores status even if no new batch was written');
  // The live report can change run/cohort before the independent collector notices.
  const counts=(wrong)=>({wrong,union:10,transitions:1});
  const oldHistory=Array.from({length:100},(_,i)=>({batch:i+1,quality:{a:counts(0),b:counts(0)}}));
  for(const {games,updatedAt} of [{games:['a','c'],updatedAt:2001},{games:['a','b'],updatedAt:2000}]) {
    archiveOffline=false;
    archiveState={enabled:true,run:'old-run',revision:'"old"',limit:2000};
    archivePayload={run:'old-run',history:oldHistory};
    Object.assign(report,{games:['a','b'],history_run:'old-run',updated_at:2000,history:oldHistory.slice(-1)});
    await vm.runInContext('refresh()',context);
    assert.equal(text('trend-balanced'),'0 %');
    const requests=archiveRequests;
    // Even if updated_at is unchanged, a new identity invalidates the rendered data.
    Object.assign(report,{games,history_run:'new-run',updated_at:updatedAt,history:[{batch:101,quality:{a:counts(10),[games[1]]:counts(10)}}],
      progress:{actions:808,batches:101}});
    await vm.runInContext('refresh()',context);
    assert.equal(text('trend-balanced'),'—','one new-run batch cannot complete an old-run window');
    assert.match(text('trend-comparison'),/1 consecutive, compatible batches available/);
    assert.match(text('history-scope'),/1 retained batches, starting at 101/);
    assert.equal(text('actions'),'808','latest counters remain live while the collector lags');
    assert.equal(archiveRequests,requests,'incompatible collector metadata does not trigger an archive fetch');
    nodes.get('history-range').listeners.change();
    assert.equal(text('trend-balanced'),'—','control changes cannot reintroduce incompatible rows');
    // Metadata can also race the separate history response; reject its mismatched identity.
    archiveState={...archiveState,run:'new-run',revision:'"new"'};
    await vm.runInContext('refresh()',context);
    assert.equal(text('trend-balanced'),'—');
    assert.match(text('history-scope'),/temporarily unavailable/);
    // After collection catches up, only the new run supplies a complete window.
    archivePayload={run:'new-run',history:Array.from({length:100},(_,i)=>({batch:101+i,
      quality:{a:counts(10),[games[1]]:counts(10)}}))};
    report.history=archivePayload.history.slice(-1);report.updated_at++;
    await vm.runInContext('refresh()',context);
    assert.equal(text('trend-balanced'),'100 %');
    assert.equal(text('trend-change'),'—','no previous window is inherited from the old run');
    assert.match(text('history-scope'),/100 retained batches, starting at 101/);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
