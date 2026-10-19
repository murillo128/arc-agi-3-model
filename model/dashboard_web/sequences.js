'use strict';
const $ = id => document.getElementById(id);
const number = (v, digits=2) => Number.isFinite(v) ? v.toLocaleString('es-ES', {maximumFractionDigits: digits}) : '—';
const pct = v => Number.isFinite(v) ? v*100 : null;
const colors = ['#8fd8bf', '#e9b86c', '#bcaafa', '#94a7aa', '#ec91b9', '#83bdea'];
const palette = ['#ffffff','#cccccc','#999999','#666666','#333333','#000000','#e53aa3','#ff7bcc','#f93c31','#1e93ff','#88d8f1','#ffdc00','#ff851b','#921231','#4fcc30','#a356d6'];
const kinds = [['invented','Inventados'],['missed','Omitidos'],['color','Color incorrecto'],['correct','Acertados']];
let displayedReport = null;
function element(tag, attrs, text='') {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  node.textContent = text; return node;
}
function chart(id, rows, series, ceiling=null) {
  const svg = $(id), legend = $(`${id}-legend`); svg.replaceChildren(); legend.replaceChildren();
  const values = rows.flatMap(r => series.map(s => s.value(r))).filter(Number.isFinite);
  if (!values.length) { svg.append(element('text', {x: 35, y: 65, fill: '#a9bcb7'}, 'Sin mediciones todavía')); return; }
  const max = Math.max(ceiling || 0, 1e-6, ...values) * (ceiling ? 1 : 1.1);
  const start=rows[0].batch, end=Math.max(start+1, rows.at(-1).batch);
  for (let i = 0; i <= 4; i++) {
    const y = 220 - i * 47;
    svg.append(element('line', {x1: 50, x2: 680, y1: y, y2: y, stroke: '#30443e'}));
    svg.append(element('text', {x: 42, y: y + 4, fill: '#a9bcb7', 'text-anchor': 'end', 'font-size': 11}, number(max*i/4, max<.1?4:1)));
  }
  series.forEach((s,index) => {
    const color=s.color||colors[index%colors.length], last=rows.findLast(r=>Number.isFinite(s.value(r)));
    const label = document.createElement('span'); label.textContent = `${s.label}: ${number(last?s.value(last):null, s.digits??4)}`; label.style.color = color;
    label.title=last?`Última medición: batch ${last.batch} · acción ${number(last.actions,0)}`:'Sin mediciones';legend.append(label);
    let points = [], previousGroup;
    const flush = () => { if (points.length) svg.append(element('polyline', {points: points.join(' '), fill: 'none', stroke: color, 'stroke-width': 2})); points = []; };
    rows.forEach(r => {
      if(s.skip?.(r))return;
      const group=s.group?.(r);if(previousGroup!==undefined&&group!==previousGroup)flush();previousGroup=group;
      const value = s.value(r); if (!Number.isFinite(value)) { flush(); return; }
      const x = 50 + (r.batch-start) * 630 / (end-start), y = 220 - 188 * value/max;
      points.push(`${x},${y}`);
      const dot=element('circle', {cx: x, cy: y, r: 2.5, fill: color});
      dot.append(element('title',{},`${s.label}: ${number(value,4)} · batch ${r.batch} · acción ${number(r.actions,0)}`));svg.append(dot);
    }); flush();
  });
  svg.append(element('text', {x: 50, y: 247, fill: '#a9bcb7', 'font-size': 11}, `Batch ${rows[0].batch}`));
  svg.append(element('text', {x: 680, y: 247, fill: '#a9bcb7', 'text-anchor': 'end', 'font-size': 11}, `Batch ${rows.at(-1).batch}`));
}
function table(id, rows) {
  $(id).replaceChildren(...rows.map(values=>{const row=document.createElement('tr');
    values.forEach(value=>{const cell=document.createElement('td');cell.textContent=String(value);row.append(cell);});return row;}));
}
function series(label, value, extra={}) { return {label,value,...extra}; }
const timingLabels = {collection_seconds:'Juego',representation_seconds:'Representación',encode_seconds:'Codificación',
  predictor_seconds:'Predictor y memoria',evaluation_seconds:'Evaluación',save_seconds:'Guardado'};
function render(data) {
  displayedReport=data;
  const c = data.config || {}, rows = data.history || [], last = rows.at(-1) || {};
  $('status').textContent = ({running: 'Entrenando', paused: 'Pausado', budget_complete: 'Presupuesto terminado', error: 'Error', waiting: 'Esperando'})[data.status] || data.status;
  $('run-info').textContent = data.message || `${(data.games || []).join(' · ')} · Actualizado hace ${number(data.age_seconds, 0)} s`;
  $('configuration').textContent = `${c.batch_games || '—'} sesiones independientes · ${c.session_steps || '—'} acciones por sesión · Bloque: ${c.block_steps || '—'} pasos + ${c.context_steps ?? 0} de contexto con gradiente · Peso inicial: ${number(c.loss_start_weight)} → 1 en ${c.loss_warmup_steps || '—'} acciones · ${c.game_schedule === 'blocked' ? 'Un juego por batch' : 'Juegos alternados'}`;
  $('learning-rates').textContent=Object.entries(data.learning_rates||{}).map(([key,value])=>`LR ${key}: ${Number.isFinite(value)?value.toExponential(1):'—'}`).join(' · ');
  $('actions').textContent = number(data.progress?.actions, 0); $('batches').textContent = `${number(data.progress?.batches, 0)} batches completados${data.budget?` · presupuesto ${number(data.budget,0)}`:''}`;
  $('tps').textContent = `${number(last.recent_tps)} acciones/s`; $('examples').textContent = `${number(last.examples_per_second)} transiciones/s`; $('memory').textContent = `${number(last.peak_gpu_gib)} GiB`;
  chart('quality', rows, [series('Inicio %',r=>pct(r.quality?.early?.delta_error),{color:colors[1]}),series('Después %',r=>pct(r.quality?.late?.delta_error),{color:colors[0]})],100);
  chart('speed', rows, [series('Acciones/s',r=>r.recent_tps)]);
  chart('loss', rows, [series('Loss',r=>r.loss,{color:colors[2]})]);
  chart('representation-loss',rows,[series('Loss encoder / decoder',r=>r.reconstruction_loss)]);
  chart('predictor-components',rows,[series('MSE latente',r=>r.predictor_terms?.latent),series('Pérdida visual',r=>r.predictor_terms?.visual)]);
  const representationLabels={color_ce:'Color',delta_nll:'Delta visual',delta_overlap:'Solapamiento',local_reconstruction:'Reconstrucción local',sigreg:'SIGReg',variance_floor:'Varianza mínima'};
  const terms=[...new Set(rows.flatMap(r=>Object.keys(r.representation_terms||{})))].sort();
  chart('representation-components',rows,terms.map(key=>series(representationLabels[key]||key,r=>r.representation_terms?.[key])));
  chart('changes',rows,kinds.map(([key,label])=>series(label,r=>{
    const q=r.quality?.all;return q?.transitions>0&&Number.isFinite(q[key])?q[key]/q.transitions:null;
  })));
  const measured=rows.filter(r=>r.diagnostics), diagnostic=measured.at(-1)?.diagnostics;
  chart('fixed-quality',rows,[series('Predicción 1 paso %',r=>pct(r.quality?.all?.delta_error)),
    series('Reconstrucción %',r=>pct(r.diagnostics?.reconstruction_error),{skip:r=>!r.diagnostics}),
    series('Predicción 3 pasos %',r=>pct(r.diagnostics?.rollout_error),{skip:r=>!r.diagnostics})],100);
  $('diagnostic-scope').textContent=diagnostic
    ?`Evaluación fija en cada batch. Diagnósticos adicionales cada ${diagnostic.every} batches · última medición: batch ${diagnostic.batch}, ${diagnostic.sampled_transitions} transiciones y ${diagnostic.latent?.sample_count??0} estados reales. Rollout: ${diagnostic.rollouts} rutas de tres pasos. Deriva: referencia del proceso en batch ${diagnostic.drift_reference_batch}.`
    :'Evaluación fija en cada batch. Los diagnósticos adicionales aparecerán tras una nueva medición; no se rellenan datos anteriores con ceros.';
  chart('latent-drift',measured,[series('Deriva %',r=>pct(r.diagnostics.latent_drift),{group:r=>r.diagnostics.drift_reference_batch})]);
  chart('latent-diversity',measured,[series('Entre observaciones',r=>r.diagnostics.latent?.std_mean),series('Dentro de cada episodio',r=>r.diagnostics.latent?.within_episode_std)]);
  chart('latent-rank',measured,[series('Rango efectivo',r=>r.diagnostics.latent?.effective_rank),series('Máximo observable',r=>r.diagnostics.latent?.max_rank)]);
  chart('latent-inactive',measured,[series('Casi constantes %',r=>pct(r.diagnostics.latent?.inactive_fraction))],100);
  chart('latent-sensitivity',measured,[series('Bloques con cambios',r=>r.diagnostics.latent?.changed_rms),series('Bloques sin cambios',r=>r.diagnostics.latent?.unchanged_rms)]);
  chart('timing-chart',rows,Object.entries(timingLabels).map(([key,label])=>series(label,r=>r[key])));
  $('timings').replaceChildren();
  for (const [key, label] of Object.entries({...timingLabels,diagnostics_seconds:'De la evaluación: diagnósticos adicionales'})) {
    const span=document.createElement('span');span.textContent=`${label}: ${number(last[key])} s`;$('timings').append(span);
  }
  table('sessions',(last.sessions||[]).map((s,i)=>[i+1,s.game,s.steps,s.boundaries,s.end]));
  $('macro-error').textContent = `Media por juego: ${number(pct(last.macro_delta_error))} %`;
  table('game-quality',(data.games||[]).map(game=>{const q=last.quality?.[game];return [game,`${number(pct(q?.delta_error))} %`,number(q?.transitions,0)];}));
  table('evaluations',[...rows].reverse().map(r=>[number(r.actions,0),r.batch,`${number(pct(r.quality?.all?.delta_error))} %`,
    `${number(pct(r.diagnostics?.reconstruction_error))} %`,`${number(pct(r.diagnostics?.rollout_error))} %`,
    number(r.quality?.all?.transitions,0),number(r.diagnostics?.sampled_transitions,0)]));
  table('schedule',(data.schedule||[]).map((s,i)=>[i+1,s.game,s.level,s.source,'Encoder · decoder · predictor']));
  $('schedule-info').textContent=`${c.game_schedule==='blocked'?'Un juego por batch':'Juegos intercalados en cada batch'} · rotación cada ${c.level_steps??'—'} movimientos en juegos sintéticos; el SDK progresa de forma nativa. Se conserva la memoria entre niveles de una sesión. Este modo no utiliza umbrales para activar el planner.`;
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
    $('preview-info').textContent='Esperando una muestra de la evaluación fija.';
    for(const id of ['before','reconstruction','real','predicted','difference'])draw(`preview-${id}`,null);
    return;
  }
  const arrows={1:'↑',2:'↓',3:'←',4:'→',5:'Space',6:'Click',7:'Undo'};
  $('preview-info').textContent=`${sample.game} · nivel ${sample.level} · paso ${sample.step} · A${sample.action} ${arrows[sample.action]||''} · evaluación fija del batch ${data.prediction_sample_batch}`;
  draw('preview-before',sample.previous);draw('preview-reconstruction',sample.reconstruction);
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
    $('planner-history-status').textContent=state.history_available?'Las gráficas anteriores se conservan en una vista histórica separada.':'El histórico del planner no está configurado en este servidor.';
    $('gpu-status').textContent=(state.gpu?.devices||[]).map(g=>`GPU · ${number(g.utilization_percent,0)} % · ${number(g.temperature_c,0)} °C`).join(' · ');
  } catch { $('status').textContent = 'Sin conexión'; }
  setTimeout(refresh, 2000);
}
refresh();
