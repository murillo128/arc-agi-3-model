'use strict';
const $ = id => document.getElementById(id);
const number = (v, digits=2) => Number.isFinite(v) ? v.toLocaleString('en-US', {maximumFractionDigits: digits}) : '—';
const pct = v => Number.isFinite(v) ? v*100 : null;
const colors = ['#8fd8bf', '#e9b86c', '#bcaafa', '#94a7aa', '#ec91b9', '#83bdea'];
const palette = ['#ffffff','#cccccc','#999999','#666666','#333333','#000000','#e53aa3','#ff7bcc','#f93c31','#1e93ff','#88d8f1','#ffdc00','#ff851b','#921231','#4fcc30','#a356d6'];
const kinds = [['invented','Invented'],['missed','Missed'],['color','Wrong color'],['correct','Correct']];
let displayedReport = null;
function element(tag, attrs, text='') {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  node.textContent = text; return node;
}
function chart(id, rows, series, ceiling=null, limits=[]) {
  const svg = $(id), legend = $(`${id}-legend`); svg.replaceChildren();
  const data=series.map((s,index)=>{
    const points=[]; let previousGroup, last=null;
    for(const r of rows) {
      if(s.skip?.(r))continue;
      const group=s.group?.(r)??r.measurement_source??'fixed_evaluation';
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
    title:last?`Latest measurement: batch ${last.row.batch} · action ${number(last.row.actions,0)}`:'No measurements'})),
    ...limits.map(l=>({key:`limit:${l.label}`,text:`┄ ${l.label}: ${number(l.value,4)} %`,color:l.color,passive:true}))],()=>updateGuide());
  const values=data.flatMap(d=>d.points.map(p=>p[1])).filter(Number.isFinite);
  if (!values.length) { svg.append(element('text', {x: 35, y: 65, fill: '#a9bcb7'}, 'No measurements yet')); return; }
  const max = Math.max(ceiling || 0, 1e-6, ...values, ...limits.map(l=>l.value)) * (ceiling ? 1 : 1.1);
  const start=rows[0].batch, end=Math.max(start+1, rows.at(-1).batch);
  const x=v=>50+(v-start)*630/(end-start), y=v=>220-188*v/max;
  for (let i = 0; i <= 4; i++) {
    const py = 220 - i * 47;
    svg.append(element('line', {x1: 50, x2: 680, y1: py, y2: py, stroke: '#30443e'}));
    svg.append(element('text', {x: 42, y: py + 4, fill: '#a9bcb7', 'text-anchor': 'end', 'font-size': 11}, number(max*i/4, max<.1?4:1)));
  }
  data.forEach(({s,points,color})=>{
    let segment=[];
    const flush=()=>{if(segment.length)svg.append(element('polyline',{points:segment.join(' '),fill:'none',stroke:color,'stroke-width':2}));segment=[];};
    for(const p of points) {
      if(!ChartTools.valid(p)){flush();continue;}
      segment.push(`${x(p[0])},${y(p[1])}`);
      const dot=element('circle',{cx:x(p[0]),cy:y(p[1]),r:2.5,fill:color});
      dot.append(element('title',{},`${s.label}: ${number(p[1],4)} · batch ${p[0]} · action ${number(p[2]?.actions,0)}`));svg.append(dot);
    }
    flush();svg.append(...ChartTools.svgEMA(points,x,y,color));
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
    if(selected?.last)ChartTools.svgGuide(guide,{left:50,right:680,y:y(selected.last.value),lastX:x(selected.last.row.batch),
      color:selected.color,text:number(selected.last.value,selected.s.digits??4)});
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
  predictor_seconds:'Predictor and memory',evaluation_seconds:'Measurement',drift_seconds:'Encoder drift',save_seconds:'Saving'};
function render(data) {
  displayedReport=data;
  const c = data.config || {}, rows = data.history || [], last = rows.at(-1) || {};
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
  $('configuration').textContent = `${shuffled?'Up to ':''}${c.batch_games || '—'} independent sessions · ${c.session_steps || '—'} actions per session · Block: ${c.block_steps || '—'} steps + ${c.context_steps ?? 0} context steps with gradients · Initial loss weight: ${number(c.loss_start_weight)} → 1 over ${c.loss_warmup_steps || '—'} actions · ${shuffled?'Games shuffled without repeats':c.game_schedule === 'blocked' ? 'One game per batch' : 'Alternating games'}`;
  $('learning-rates').textContent=Object.entries(data.learning_rates||{}).map(([key,value])=>`LR ${key}: ${Number.isFinite(value)?value.toExponential(1):'—'}`).join(' · ');
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
  const measured=rows.filter(r=>r.diagnostics), diagnostic=measured.at(-1)?.diagnostics;
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
  chart('latent-drift',measured,[series('Drift %',r=>pct(r.diagnostics.latent_drift),{group:r=>r.diagnostics.drift_source??r.diagnostics.drift_reference_batch})],
    null,phaseThresholds(data,'latent_drift','Drift threshold',colors[0]));
  chart('latent-diversity',measured,[series('Across observations',r=>r.diagnostics.latent?.std_mean),series('Within each episode',r=>r.diagnostics.latent?.within_episode_std)]);
  chart('latent-rank',measured,[series('Effective rank',r=>r.diagnostics.latent?.effective_rank),series('Observable maximum',r=>r.diagnostics.latent?.max_rank)]);
  chart('latent-inactive',measured,[series('Nearly constant %',r=>pct(r.diagnostics.latent?.inactive_fraction))],100);
  chart('latent-sensitivity',measured,[series('Changed blocks',r=>r.diagnostics.latent?.changed_rms),series('Unchanged blocks',r=>r.diagnostics.latent?.unchanged_rms)]);
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
  $('schedule-info').textContent=`${shuffled?'Each game once per round in shuffled order; the last batch may be smaller':c.game_schedule==='blocked'?'One game per batch':'Games interleaved in each batch'} · rotation every ${c.level_steps??'—'} moves in synthetic games; SDK games progress natively. Memory is preserved across levels within a session. This mode does not use thresholds to activate the planner.`;
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
async function refresh() {
  try { const response = await fetch('api/state', {cache: 'no-store'}); if (!response.ok) throw Error(response.status);
    const state=await response.json();render(state.sequences||{status:'waiting'});
    $('history-link').hidden=$('planner-history-link').hidden=!state.history_available;
    $('planner-history-status').textContent=state.history_available?'Earlier charts are preserved in a separate history view.':'Planner history is not configured on this server.';
    $('gpu-status').textContent=(state.gpu?.devices||[]).map(g=>`GPU · ${number(g.utilization_percent,0)} % · ${number(g.temperature_c,0)} °C`).join(' · ');
  } catch { $('status').textContent = 'Disconnected'; }
  setTimeout(refresh, 2000);
}
refresh();
