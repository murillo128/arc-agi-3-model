'use strict';
const $ = id => document.getElementById(id);
const number = (v, digits=2) => Number.isFinite(v) ? v.toLocaleString('en-US', {maximumFractionDigits: digits}) : '—';
const pct = v => Number.isFinite(v) ? v*100 : null;
const colors = ['#8fd8bf', '#e9b86c', '#bcaafa', '#94a7aa', '#ec91b9', '#83bdea'];
const palette = ['#ffffff','#cccccc','#999999','#666666','#333333','#000000','#e53aa3','#ff7bcc','#f93c31','#1e93ff','#88d8f1','#ffdc00','#ff851b','#921231','#4fcc30','#a356d6'];
const kinds = [['invented','Invented'],['missed','Missed'],['color','Wrong color'],['correct','Correct']];
let displayedReport = null;
let retainedHistory = {history:[], revision:null, run:null}, historyMetadata = null, historyError = '';
let renderedRevision = null;
function element(tag, attrs, text='') {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  node.textContent = text; return node;
}
function chart(id, rows, series, ceiling=null, limits=[], zoom=false) {
  const svg = $(id), legend = $(`${id}-legend`); svg.replaceChildren();
  const data=series.map((s,index)=>{
    const points=[]; let previousGroup, previousRow, last=null;
    for(const r of rows) {
      const gap=previousRow && !SequenceTrends.connected(previousRow,r);
      previousRow=r;
      if(gap)points.push([r.batch,null]);
      if(s.skip?.(r))continue;
      const collection=r.exploration?.policy==='uniform'?'uniform':r.exploration?.identity??'uniform';
      const group=`${s.group?.(r)??r.measurement_source??'fixed_evaluation'}:${collection}`;
      if(previousGroup!==undefined&&group!==previousGroup)points.push([r.batch,null]);
      previousGroup=group;
      const value=s.value(r); points.push([r.batch,value,r]);
      if(Number.isFinite(value))last={value,row:r};
    }
    return {s,points,last,color:s.color||colors[index%colors.length]};
  });
  let updateGuide=()=>{};
  ChartTools.legend(legend,id,[...data.map(({s,last,color})=>({key:s.label,
    text:`${s.label}: ${number(last?.value, s.digits??4)}`,color,disabled:!last,
    hint:s.mean?'Window mean; select to focus this series and mark its latest value.':undefined,
    title:last?`Latest measurement: batch ${last.row.batch} · action ${number(last.row.actions,0)}`:'No measurements'})),
    ...limits.map(l=>({key:`limit:${l.label}`,text:`┄ ${l.label}: ${number(l.value,4)} %`,color:l.color,passive:true}))],()=>updateGuide());
  const values=data.flatMap(d=>d.points.map(p=>p[1])).filter(Number.isFinite);
  if (!values.length) { svg.append(element('text', {x: 35, y: 65, fill: '#a9bcb7'}, 'No measurements yet')); return; }
  let max = Math.max(ceiling || 0, 1e-6, ...values, ...limits.map(l=>l.value)) * (ceiling ? 1 : 1.1), min=0;
  if(zoom){
    const low=Math.min(...values), high=Math.max(...values), padding=Math.max(1,(high-low)*.15);
    min=Math.max(0,low-padding);max=Math.min(100,high+padding);
  }
  const start=rows[0].batch, end=Math.max(start+1, rows.at(-1).batch);
  const x=v=>50+(v-start)*630/(end-start), y=v=>220-188*(v-min)/(max-min);
  for (let i = 0; i <= 4; i++) {
    const py = 220 - i * 47;
    svg.append(element('line', {x1: 50, x2: 680, y1: py, y2: py, stroke: '#30443e'}));
    svg.append(element('text', {x: 42, y: py + 4, fill: '#a9bcb7', 'text-anchor': 'end', 'font-size': 11}, number(min+(max-min)*i/4, max<.1?4:1)));
  }
  const layers=data.map(()=>[]);
  data.forEach(({s,points,color},i)=>{
    const append=(node,ema=false)=>{layers[i].push({node,ema});svg.append(node);};
    let segment=[];
    const flush=()=>{if(segment.length)append(element('polyline',{points:segment.join(' '),fill:'none',stroke:color}),s.mean===true);segment=[];};
    const stride=Math.max(1,Math.ceil(points.length/350));
    for(const [index,p] of points.entries()) {
      if(!ChartTools.valid(p)){flush();continue;}
      segment.push(`${x(p[0])},${y(p[1])}`);
      if(index%stride!==0&&index!==points.length-1)continue;
      const dot=element('circle',{cx:x(p[0]),cy:y(p[1]),r:2.5,fill:color});
      dot.append(element('title',{},`${s.label}: ${number(p[1],4)} · batch ${p[0]} · action ${number(p[2]?.actions,0)}`));append(dot);
    }
    flush();if(!s.mean)ChartTools.svgEMA(points,x,y,color).forEach(node=>append(node,true));
  });
  limits.forEach(({value,label,color})=>{
    const line=element('line',{x1:50,x2:680,y1:y(value),y2:y(value),stroke:color,'stroke-width':1.5,'stroke-dasharray':'5 5'});
    line.append(element('title',{},`${label}: ${number(value,4)} %`));svg.append(line);
  });
  svg.append(element('text', {x: 50, y: 247, fill: '#a9bcb7', 'font-size': 11}, `Batch ${rows[0].batch}`));
  svg.append(element('text', {x: 680, y: 247, fill: '#a9bcb7', 'text-anchor': 'end', 'font-size': 11}, `Batch ${rows.at(-1).batch}`));
  const guide=element('g',{class:'series-guide','pointer-events':'none'});svg.append(guide);
  updateGuide=()=>{
    const selected=data.find(d=>d.s.label===ChartTools.selected(id));guide.replaceChildren();
    ChartTools.svgFocus(layers,data.indexOf(selected));
    const last=selected&&(selected.s.mean?selected.points.findLast(ChartTools.valid):ChartTools.latestEMA(selected.points));
    if(last)ChartTools.svgGuide(guide,{left:50,right:680,y:y(last[1]),lastX:x(last[0]),
      color:selected.color,text:number(last[1],selected.s.digits??4)});
  };
  updateGuide();
}

function table(id, rows) {
  $(id).replaceChildren(...rows.map(values=>{const row=document.createElement('tr');
    values.forEach(value=>{const cell=document.createElement('td');cell.textContent=String(value);row.append(cell);});return row;}));
}
function series(label, value, extra={}) { return {label,value,...extra}; }
function phaseThresholds(data, key, label, color) {
  const groups=new Map();
  for(const row of data.phase_control?.rows||[]){
    const value=row.thresholds?.[key];
    if(!Number.isFinite(value)||value<0||value>1)continue;
    if(!groups.has(value))groups.set(value,[]);
    groups.get(value).push(`${row.game} / ${row.level}`);
  }
  return [...groups].sort(([a],[b])=>a-b).map(([value,scope])=>({value:pct(value),color,
    label:groups.size>1?`${label} · ${scope.join(', ')}`:label}));
}
const timingLabels = {collection_seconds:'Game',representation_seconds:'Representation',encode_seconds:'Encoding',
  predictor_seconds:'Predictor and memory',evaluation_seconds:'Measurement',drift_seconds:'Encoder drift',exploration_seconds:'Exploration head',save_seconds:'Saving'};
function availableHistory(data) {
  const live=data.history||[], latest=live.at(-1)?.batch;
  if(!Number.isFinite(latest))return [];
  const archived=data.history_run&&retainedHistory.run===data.history_run?retainedHistory.history:[];
  const merged=new Map(archived.filter(r=>r.batch<=latest).map(r=>[r.batch,r]));
  live.forEach(r=>merged.set(r.batch,r));
  return [...merged.values()].sort((a,b)=>a.batch-b.batch);
}
function historyRange(rows) {
  const range=$('history-range').value||'1000';
  return range==='all'?rows:rows.slice(-Number(range));
}
const errorText=value=>Number.isFinite(value)?`${number(value,2)} %`:'—';
const changeText=value=>Number.isFinite(value)?`${value>0?'+':''}${number(value,2)} pp`:'—';
function renderTrend(data, all, visible) {
  const games=data.games||[], size=Number($('trend-window').value||100);
  const comparison=SequenceTrends.compare(all,games,size), {current,previous}=comparison;
  const change=Number.isFinite(current?.balanced)&&Number.isFinite(previous?.balanced)?current.balanced-previous.balanced:null;
  $('trend-balanced').textContent=errorText(current?.balanced);
  $('trend-change').textContent=changeText(change);
  $('trend-change').className=change<0?'trend-down':change>0?'trend-up':'';
  $('trend-coverage').textContent=`${current?.coverage??'—'} / ${games.length}`;
  const range=w=>`batches ${number(w.start,0)}–${number(w.end,0)}`;
  $('trend-comparison').textContent=current
    ?`${range(current)}${previous?` vs ${range(previous)}`:` · Need ${2*size} consecutive, compatible batches for comparison (${comparison.available} available)`}. ${current.coverage<games.length?'Incomplete game coverage; aggregate unavailable. ':''}Negative change means lower error; new scenes can still affect the comparison.${$('trend-zoom').checked?' Y-axis zoomed to the observed range.':''}`
    :`Collecting a full ${size}-batch window (${comparison.available} consecutive, compatible batches available). Gaps and training-regime changes restart the window.`;
  const points=SequenceTrends.rolling(all,games,size).filter(r=>r.batch>=(visible[0]?.batch??Infinity));
  chart('prediction-trend',points,[series(`Equal game weight · ${size} batches`,r=>r.balanced,{mean:true}),
    series(`Pooled pixels · ${size} batches`,r=>r.pooled,{mean:true,color:colors[2]})],100,[],$('trend-zoom').checked===true);
  const before=new Map((previous?.games||[]).map(g=>[g.game,g]));
  const perGame=(current?.games||games.map(game=>({game,error:null}))).map(g=>({current:g,previous:before.get(g.game)}))
    .sort((a,b)=>(b.current.error??-1)-(a.current.error??-1));
  table('trend-games',perGame.map(({current:c,previous:p})=>[c.game,errorText(p?.error),errorText(c.error),
    changeText(Number.isFinite(c.error)&&Number.isFinite(p?.error)?c.error-p.error:null),
    `${number(p?.transitions,0)} / ${number(c.transitions,0)}`]));
  perGame.forEach(({current:c,previous:p},i)=>{
    const cell=$('trend-games').children[i].children[3];
    if(Number.isFinite(c.error)&&Number.isFinite(p?.error))cell.className=c.error<p.error?'trend-down':c.error>p.error?'trend-up':'';
  });
}
function render(data) {
  displayedReport=data;
  const c = data.config || {}, all=availableHistory(data), rows=historyRange(all), last = (data.history||[]).at(-1) || {};
  const gaps=all.reduce((n,r,i)=>n+(i>0&&r.batch!==all[i-1].batch+1?1:0),0);
  $('history-scope').textContent=all.length
    ?`Showing ${number(rows[0].batch,0)}–${number(rows.at(-1).batch,0)} · ${number(all.length,0)} retained batches, starting at ${number(all[0].batch,0)}${gaps?` · ${gaps} gaps`:''}. ${historyMetadata?.enabled?`Archive limit: ${number(historyMetadata.limit,0)}; older discarded batches cannot be recovered.`:'Only the trainer’s retained history is available; persistence is not enabled.'}${historyError||historyMetadata?.error?` ${historyError||historyMetadata.error}`:''}`
    :'Waiting for recorded batches.';
  renderTrend(data,all,rows);
  const training=data.measurement_source==='training_pre_update', control=data.phase_control;
  $('phase-status').textContent=control?`Phase: ${control.phase==='predictor'?'Predictor · reduced representation LR':'Representation · predictor waiting'}${last.phase&&last.phase!==control.phase?' · Change takes effect next batch':''}`:'';
  $('quality-source').textContent=$('early-source').textContent=training?'% · batch before training':'% · fixed evaluation';
  $('tab-evaluation').textContent=training?'Training measurements':'Fixed evaluation';
  $('evaluation-description').textContent=training?'Normal training trajectories, measured before weight updates. Each game has equal weight in the average. These are learning diagnostics, not measures of generalization or wins.':'Separate seeds, frozen weights and causal memory rebuilt from the start of each session. Each game has equal weight in the average; this does not measure wins.';
  $('evaluations-title').textContent=training?'Measurements before learning':'Fixed evaluations';
  $('evaluations-description').textContent=training?'One record per training batch. Scenes change between batches; these are not fixed episodes. Reconstruction and rollout use a sample from each game and level.':'One record per batch using the same real reference sessions. Reconstruction and rollout are measured every four batches; gaps do not mean zero.';
  $('drift-description').textContent=training?'Relative encoder change on the same batch images, before and after updating representation. Each batch provides new images; object movement is kept separate from encoder drift.':'Relative latent change from the first measurement in this process. Resuming establishes a new reference; lines are separated between references.';
  $('phase-card').hidden=!control;
  if(control){
    $('phase-description').textContent=`Window of ${control.window_batches} batches. First observe every synthetic level and each SDK game's first level; higher SDK levels join when reached. Observed levels with insufficient transitions are ignored for admission, not marked as passed. At least one eligible level is required, and every eligible level must meet reconstruction, drift and confirmation requirements. All games continue training. Predictor admission is retained until the curriculum is extended.`;
    table('phase-rows',control.rows.map(r=>[`${r.game} / ${r.level}`,`${r.samples} / ${r.changed} · minimum ${r.thresholds.minimum_samples} / ${r.thresholds.minimum_changed}`,
      `${number(pct(r.reconstruction_error))} % / ${number(pct(r.thresholds.representation_open))} %`,
      `${number(pct(r.latent_drift))} % / ${number(pct(r.thresholds.latent_drift))} %`,`${r.passes} / ${r.thresholds.patience}`,
      r.required===false?'Not reached yet':r.observed===false?'Awaiting first observation':
      !r.enough?'Ignored · insufficient transitions':!r.passing?'Outside threshold':
      r.passes<r.thresholds.patience?'Awaiting confirmations':'Ready']));
  }
  $('status').textContent = ({running: 'Training', paused: 'Paused', budget_complete: 'Budget reached', error: 'Error', waiting: 'Waiting'})[data.status] || data.status;
  const gameCount=(data.games||[]).length;
  $('run-info').textContent = data.message || `${number(gameCount,0)} ${gameCount===1?'game':'games'}`;
  $('freshness').textContent = Number.isFinite(data.age_seconds) ? `Updated ${number(data.age_seconds,0)} s ago` : '';
  $('curriculum-games').textContent=(data.games||[]).join(' · ');
  const shuffled=c.game_schedule==='shuffled';
  const weights=data.game_weights||{}, weighted=Object.values(weights).some(w=>w>1);
  const sessionsPerRound=(data.games||[]).reduce((total,game)=>total+(weights[game]??1),0);
  $('configuration').textContent = `${shuffled?'Up to ':''}${c.batch_games || '—'} independent sessions · ${c.session_steps || '—'} actions per session · Block: ${c.block_steps || '—'} steps + ${c.context_steps ?? 0} context steps with gradients · Initial loss weight: ${number(c.loss_start_weight)} → 1 over ${c.loss_warmup_steps || '—'} actions · ${shuffled?(weighted?`Weighted shuffled rounds · ${sessionsPerRound} sessions per round`:'Games shuffled without repeats'):c.game_schedule === 'blocked' ? 'One game per batch' : 'Alternating games'}`;
  $('sampling-card').hidden=!shuffled;
  const exposureRows=rows.filter(r=>r.batch>(data.sampling_start_batch??0)), exposure=new Map();
  for(const row of exposureRows) for(const session of row.sessions||[]){
    const total=exposure.get(session.game)||[0,0]; total[0]++; total[1]+=session.steps||0; exposure.set(session.game,total);
  }
  const weightAdoption=(data.game_weight_adoptions||[]).at(-1);
  $('sampling-info').textContent=`${sessionsPerRound} independent sessions per round. Weights change collection frequency; per-session loss weighting is unchanged.${weightAdoption?` Current frequencies apply from batch ${number(weightAdoption.batch+1,0)}; earlier chart history uses the previous mix.`:''} Collected counts cover ${exposureRows.length} retained batches${exposureRows.length?` (${exposureRows[0].batch}–${exposureRows.at(-1).batch})`:''} and may include partial rounds. Repeated games start with separate seeds and memory.`;
  table('sampling-games',(data.games||[]).map(game=>[game,`${weights[game]??1}×`,`${number(100*(weights[game]??1)/sessionsPerRound,2)} %`,
    number(exposure.get(game)?.[0]??0,0),number(exposure.get(game)?.[1]??0,0)]));
  $('learning-rates').textContent=Object.entries(data.learning_rates||{}).map(([key,value])=>`LR ${key}: ${Number.isFinite(value)?value.toExponential(1):'—'}`).join(' · ');
  if(data.network?.rotation_adapter_width) $('learning-rates').textContent+=` · Rotation adapter: ${data.network.rotation_adapter_width} filters × 4 orientations`;
  if(data.click_map) $('learning-rates').textContent+=` · Gaussian click σ: ${number(data.click_map.sigma_pixels,4)} px (${data.click_map.learnable?'learned':'fixed'})`;
  const budgetLabel=data.budget===null?' · Unlimited actions':data.budget?` · budget ${number(data.budget,0)}`:'';
  $('actions').textContent = number(data.progress?.actions, 0); $('batches').textContent = `${number(data.progress?.batches, 0)} completed batches${budgetLabel}`;
  $('tps').textContent = `${number(last.recent_tps)} actions/s`; $('examples').textContent = `${number(last.examples_per_second)} transitions/s`; $('memory').textContent = `${number(last.peak_gpu_gib)} GiB`;
  chart('quality', rows, [series('Early %',r=>pct(r.quality?.early?.delta_error),{color:colors[1]}),series('Later %',r=>pct(r.quality?.late?.delta_error),{color:colors[0]})],100);
  chart('speed', rows, [series('Actions/s',r=>r.recent_tps)]);
  chart('loss', rows, [series('Loss',r=>r.loss,{color:colors[2]})]);
  chart('representation-loss',rows,[series('Loss encoder / decoder',r=>r.reconstruction_loss)]);
  chart('predictor-components',rows,[series('Latent MSE',r=>r.predictor_terms?.latent),series('Visual loss',r=>r.predictor_terms?.visual)]);
  const representationLabels={color_ce:'Color',delta_nll:'Visual delta',delta_overlap:'Overlap',local_reconstruction:'Local reconstruction',sigreg:'SIGReg',variance_floor:'Variance floor'};
  const terms=[...new Set(rows.flatMap(r=>Object.keys(r.representation_terms||{})))].sort();
  chart('representation-components',rows,terms.map(key=>series(representationLabels[key]||key,r=>r.representation_terms?.[key])));
  chart('changes',rows,kinds.map(([key,label])=>series(label,r=>{
    const q=r.quality?.all;return q?.transitions>0&&Number.isFinite(q[key])?q[key]/q.transitions:null;
  })));
  const diagnostic=rows.findLast(r=>r.diagnostics)?.diagnostics;
  // Keep every batch for gap/regime detection, but skip intentionally unsampled diagnostics.
  const diagnosticSeries=(label,value,extra={})=>series(label,value,{...extra,skip:r=>!r.diagnostics});
  chart('fixed-quality',rows,[series('1-step prediction %',r=>pct(r.quality?.all?.delta_error)),
    series('Global reconstruction %',r=>pct(r.diagnostics?.reconstruction_error),{skip:r=>!r.diagnostics}),
    series('3-step prediction %',r=>pct(r.diagnostics?.rollout_error),{skip:r=>!r.diagnostics}),
    series('Reconstruction on changes %',r=>pct(r.diagnostics?.reconstruction_delta_error),{color:colors[4]})],
    100,phaseThresholds(data,'representation_open','Reconstruction-on-changes threshold',colors[4]));
  $('diagnostic-scope').textContent=training
    ?`Normal batch data before weight updates · diagnostics stratified by game/level${diagnostic?` · batch ${diagnostic.batch}, ${diagnostic.sampled_transitions} transitions · ${diagnostic.rollouts} three-step routes`:''}. Drift compares the same images before and after the encoder update.`
    :diagnostic
    ?`Fixed evaluation every batch. Additional diagnostics every ${diagnostic.every} batches · latest measurement: batch ${diagnostic.batch}, ${diagnostic.sampled_transitions} transitions and ${diagnostic.latent?.sample_count??0} real states. Rollout: ${diagnostic.rollouts} three-step routes. Drift: process reference at batch ${diagnostic.drift_reference_batch}.`
    :'Fixed evaluation every batch. Additional diagnostics appear after a new measurement; historical gaps are not filled with zeros.';
  chart('latent-drift',rows,[diagnosticSeries('Drift %',r=>pct(r.diagnostics.latent_drift),{group:r=>r.diagnostics.drift_source??r.diagnostics.drift_reference_batch})],
    null,phaseThresholds(data,'latent_drift','Drift threshold',colors[0]));
  chart('latent-diversity',rows,[diagnosticSeries('Across observations',r=>r.diagnostics.latent?.std_mean),diagnosticSeries('Within each episode',r=>r.diagnostics.latent?.within_episode_std)]);
  chart('latent-rank',rows,[diagnosticSeries('Effective rank',r=>r.diagnostics.latent?.effective_rank),diagnosticSeries('Observable maximum',r=>r.diagnostics.latent?.max_rank)]);
  chart('latent-inactive',rows,[diagnosticSeries('Nearly constant %',r=>pct(r.diagnostics.latent?.inactive_fraction))],100);
  chart('latent-sensitivity',rows,[diagnosticSeries('Changed blocks',r=>r.diagnostics.latent?.changed_rms),diagnosticSeries('Unchanged blocks',r=>r.diagnostics.latent?.unchanged_rms)]);
  const exploration=data.exploration_config||{policy:'uniform'}, interaction=last.exploration;
  const policyLabel={uniform:'Uniform exploration',mixed:'Visible-element clicks',learned:'Learned change prior'}[exploration.policy]||exploration.policy;
  const adoption=data.exploration_adoptions?.at(-1);
  $('exploration-policy').textContent=`${policyLabel}${exploration.policy==='uniform'?'':` · ${number(100*exploration.uniform_probability,0)}% uniform clicks`}${adoption?` · active from batch ${adoption.batch+1}`:''}`;
  if(exploration.allow_undo===false) $('exploration-policy').textContent+=' · Undo excluded';
  if(Number.isFinite(interaction?.undo_actions)) $('exploration-policy').textContent+=` · Undo in last batch: ${number(interaction.undo_actions,0)}`;
  const adaptation=data.predictor_adaptations?.at(-1);
  if(adaptation) $('exploration-policy').textContent+=` · Current metrics from batch ${adaptation.batch+1}`;
  const changeRate=(row,numerator,denominator)=>{
    const e=row.exploration;
    return e&&Number.isFinite(e[numerator])&&e[denominator]>0?100*e[numerator]/e[denominator]:null;
  };
  chart('interaction-change',rows,[series('Changed actions %',r=>changeRate(r,'changed_actions','eligible_transitions')),
    series('Changed clicks %',r=>changeRate(r,'changed_clicks','eligible_clicks'))],100);
  chart('exploration-loss',rows,[series('Change-head BCE',r=>r.exploration?.head?.loss)]);
  const head=interaction?.head;
  $('exploration-head-info').textContent=head?.total_updates!==undefined?`${number(head.total_updates,0)} auxiliary updates · ${number(head.examples_seen,0)} real examples seen · ${number(last.exploration_seconds)} s last batch`:'No learned-head measurements yet';
  table('exploration-games',(interaction?.games||[]).map(g=>[g.game,
    `${number(g.changed_actions,0)} / ${number(g.eligible_transitions,0)}`,
    `${number(g.changed_clicks,0)} / ${number(g.eligible_clicks,0)}`,
    g.eligible_clicks>0?`${number(100*g.changed_clicks/g.eligible_clicks)} %`:'—',number(g.wins,0)]));
  chart('timing-chart',rows,Object.entries(timingLabels).map(([key,label])=>series(label,r=>r[key])));
  $('timings').replaceChildren();
  for (const [key, label] of Object.entries({...timingLabels,diagnostics_seconds:'Within evaluation: extra diagnostics'})) {
    const span=document.createElement('span');span.textContent=`${label}: ${number(last[key])} s`;$('timings').append(span);
  }
  table('sessions',(last.sessions||[]).map((s,i)=>[i+1,s.game,s.steps,s.boundaries,s.end]));
  $('macro-error').textContent = `Average across games: ${number(pct(last.macro_delta_error))} %`;
  table('game-quality',(data.games||[]).map(game=>{const q=last.quality?.[game];return [game,`${number(pct(q?.delta_error))} %`,number(q?.transitions,0)];}));
  table('evaluations',[...rows].reverse().map(r=>[number(r.actions,0),r.batch,`${number(pct(r.quality?.all?.delta_error))} %`,
    `${number(pct(r.diagnostics?.reconstruction_error))} %`,`${number(pct(r.diagnostics?.rollout_error))} %`,
    number(r.quality?.all?.transitions,0),number(r.diagnostics?.sampled_transitions,0)]));
  table('schedule',(data.schedule||[]).map((s,i)=>[i+1,s.game,s.level,s.source,control?.phase==='representation'?'Encoder · decoder':'Encoder · decoder · predictor']));
  $('schedule-info').textContent=`${shuffled?(weighted?'Weighted sessions per shuffled round; the last batch may be smaller':'Each game once per round in shuffled order; the last batch may be smaller'):c.game_schedule==='blocked'?'One game per batch':'Games interleaved in each batch'} · rotation every ${c.level_steps??'—'} moves in synthetic games; SDK games progress natively. Memory is preserved across levels within a session. This mode does not use thresholds to activate the planner.`;
  renderPreview(data);
}
function draw(id, grid) {
  const canvas=$(id), ctx=canvas.getContext('2d');
  if(!grid?.length){ctx.clearRect(0,0,canvas.width,canvas.height);return;}
  canvas.height=grid.length;canvas.width=grid[0].length;
  for(let y=0;y<grid.length;y++)for(let x=0;x<grid[y].length;x++){
    const value=grid[y][x];ctx.fillStyle=typeof value==='string'?value:palette[value]||((x+y)%2?'#aaa':'#eee');ctx.fillRect(x,y,1,1);
  }
}
function renderPreview(data) {
  const samples=data.prediction_samples||[], select=$('preview-game'), previous=select.value;
  const signature=JSON.stringify(samples.map(s=>s.game));
  if(select.dataset?.signature!==signature){
    select.replaceChildren(...samples.map(s=>{const o=document.createElement('option');o.value=s.game;o.textContent=s.game;return o;}));
    select.dataset.signature=signature;
    select.value=samples.some(s=>s.game===previous)?previous:(samples[0]?.game||'');
  }
  select.disabled=samples.length<2;
  const sample=samples.find(s=>s.game===select.value);
  $('preview-images').hidden=!sample;$('preview-legend').replaceChildren();
  if(!sample){
    $('preview-info').textContent=data.measurement_source==='training_pre_update'?'Waiting for a training-batch sample.':'Waiting for a fixed-evaluation sample.';
    for(const id of ['before','reconstruction','next-real','next-predicted','real','predicted','difference'])draw(`preview-${id}`,null);
    return;
  }
  const arrows={1:'↑',2:'↓',3:'←',4:'→',5:'Space',6:'Click',7:'Undo'};
  $('preview-info').textContent=`${sample.game} · level ${sample.level} · step ${sample.step} · A${sample.action} ${arrows[sample.action]||''} · ${data.measurement_source==='training_pre_update'?'before training batch':'fixed evaluation of batch'} ${data.prediction_sample_batch}`;
  draw('preview-before',sample.previous);draw('preview-reconstruction',sample.reconstruction);
  draw('preview-next-real',sample.actual);draw('preview-next-predicted',sample.predicted);
  const real=[], predicted=[], difference=[], counts=[0,0,0,0];
  sample.previous.forEach((row,y)=>{real[y]=[];predicted[y]=[];difference[y]=[];row.forEach((before,x)=>{
    const actual=sample.actual[y][x], guessed=sample.predicted[y][x], changed=actual!==before, invented=guessed!==before;
    real[y][x]=changed?actual:16;predicted[y][x]=invented?guessed:16;
    const category=!changed&&invented?0:changed&&!invented?1:changed&&guessed!==actual?2:changed?3:null;
    difference[y][x]=category===null?16:colors[category];if(category!==null)counts[category]++;
  });});
  draw('preview-real',real);draw('preview-predicted',predicted);draw('preview-difference',difference);
  kinds.forEach(([,label],i)=>{const node=document.createElement('span');node.textContent=`${label}: ${counts[i]} px`;node.style.color=colors[i];$('preview-legend').append(node);});
}
$('preview-game').addEventListener('change',()=>{if(displayedReport)renderPreview(displayedReport);});
const tabs=['world','path','evaluation','curriculum'];
function selectTab(name) {
  if(!tabs.includes(name))return;
  for(const tab of tabs){$(tab).hidden=tab!==name;$('tab-'+tab).setAttribute('aria-selected',String(tab===name));$('tab-'+tab).tabIndex=tab===name?0:-1;}
}
tabs.forEach((name,i)=>{
  $('tab-'+name).addEventListener('click',()=>selectTab(name));
  $('tab-'+name).addEventListener('keydown',event=>{
    const next=event.key==='ArrowRight'?(i+1)%tabs.length:event.key==='ArrowLeft'?(i+tabs.length-1)%tabs.length:event.key==='Home'?0:event.key==='End'?tabs.length-1:null;
    if(next===null)return;event.preventDefault();selectTab(tabs[next]);$('tab-'+tabs[next]).focus();
  });
});
selectTab(globalThis.location?.hash?.slice(1)||'world');
for(const id of ['history-range','trend-window','trend-zoom'])$(id).addEventListener('change',()=>{
  if(displayedReport)render(displayedReport);
});
async function refresh() {
  try { const response = await fetch('api/state', {cache: 'no-store'}); if (!response.ok) throw Error(response.status);
    const state=await response.json();historyMetadata=state.sequence_history;
    const data=state.sequences||{status:'waiting'};
    const compatible=historyMetadata?.enabled&&data.history_run&&historyMetadata.run===data.history_run;
    if(!compatible||retainedHistory.run!==data.history_run||
       (retainedHistory.history.at(-1)?.batch??0)>(data.history?.at(-1)?.batch??Infinity))
      retainedHistory={history:[],revision:null,run:null};
    historyError=historyMetadata?.enabled&&!compatible?'History is waiting for the current run; latest measurements remain live.':'';
    if(compatible&&historyMetadata.revision&&historyMetadata.revision!==retainedHistory.revision){
      try{
        const result=await fetch('api/sequence-history',{cache:'no-store',headers:retainedHistory.revision?{'If-None-Match':retainedHistory.revision}:{}});
        if(result.status!==304){
          if(!result.ok)throw Error(result.status);
          const saved=await result.json();
          if(saved.run!==data.history_run||!Array.isArray(saved.history))throw Error('History changed');
          retainedHistory={...saved,revision:result.headers.get('ETag')};
        }
      }catch{historyError='History temporarily unavailable; latest measurements remain live.';}
    }
    const revision=Number.isFinite(data.updated_at)?JSON.stringify([data.updated_at,data.status,
      data.history_run,historyMetadata?.revision,historyMetadata?.error,historyError]):null;
    if(revision===null||revision!==renderedRevision){render(data);renderedRevision=revision;}
    else{
      displayedReport=data;
      $('freshness').textContent=Number.isFinite(data.age_seconds)?`Updated ${number(data.age_seconds,0)} s ago`:'';
    }
    $('history-link').hidden=$('planner-history-link').hidden=!state.history_available;
    $('planner-history-status').textContent=state.history_available?'Earlier charts are preserved in a separate history view.':'Planner history is not configured on this server.';
    $('gpu-status').textContent=(state.gpu?.devices||[]).map(g=>`GPU · ${number(g.utilization_percent,0)} % · ${number(g.temperature_c,0)} °C`).join(' · ');
  } catch { renderedRevision=null; $('status').textContent = 'Disconnected'; }
  setTimeout(refresh, 2000);
}
refresh();
