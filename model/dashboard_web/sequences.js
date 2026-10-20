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
function chart(id, rows, series, ceiling=null, limits=[]) {
  const svg = $(id), legend = $(`${id}-legend`); svg.replaceChildren(); legend.replaceChildren();
  const values = rows.flatMap(r => series.map(s => s.value(r))).filter(Number.isFinite);
  if (!values.length) { svg.append(element('text', {x: 35, y: 65, fill: '#a9bcb7'}, 'Sin mediciones todavía')); return; }
  const max = Math.max(ceiling || 0, 1e-6, ...values, ...limits.map(l=>l.value)) * (ceiling ? 1 : 1.1);
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
      const group=s.group?.(r)??r.measurement_source??'fixed_evaluation';if(previousGroup!==undefined&&group!==previousGroup)flush();previousGroup=group;
      const value = s.value(r); if (!Number.isFinite(value)) { flush(); return; }
      const x = 50 + (r.batch-start) * 630 / (end-start), y = 220 - 188 * value/max;
      points.push(`${x},${y}`);
      const dot=element('circle', {cx: x, cy: y, r: 2.5, fill: color});
      dot.append(element('title',{},`${s.label}: ${number(value,4)} · batch ${r.batch} · acción ${number(r.actions,0)}`));svg.append(dot);
    }); flush();
  });
  limits.forEach(({value,label,color})=>{
    const y=220-188*value/max;
    const line=element('line',{x1:50,x2:680,y1:y,y2:y,stroke:color,'stroke-width':1.5,'stroke-dasharray':'5 5'});
    const caption=`${label}: ${number(value,4)} %`;
    line.append(element('title',{},caption));svg.append(line);
    const item=document.createElement('span');item.textContent=`┄ ${caption}`;item.style.color=color;legend.append(item);
  });
  svg.append(element('text', {x: 50, y: 247, fill: '#a9bcb7', 'font-size': 11}, `Batch ${rows[0].batch}`));
  svg.append(element('text', {x: 680, y: 247, fill: '#a9bcb7', 'text-anchor': 'end', 'font-size': 11}, `Batch ${rows.at(-1).batch}`));
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
const timingLabels = {collection_seconds:'Juego',representation_seconds:'Representación',encode_seconds:'Codificación',
  predictor_seconds:'Predictor y memoria',evaluation_seconds:'Medición',drift_seconds:'Deriva encoder',save_seconds:'Guardado'};
function render(data) {
  displayedReport=data;
  const c = data.config || {}, rows = data.history || [], last = rows.at(-1) || {};
  const training=data.measurement_source==='training_pre_update', control=data.phase_control;
  $('phase-status').textContent=control?`Fase: ${control.phase==='predictor'?'Predictor · representación a LR reducido':'Representación · predictor en espera'}${last.phase&&last.phase!==control.phase?' · Cambio efectivo en el siguiente batch':''}`:'';
  $('quality-source').textContent=$('early-source').textContent=training?'% · batch antes de entrenar':'% · evaluación fija';
  $('tab-evaluation').textContent=training?'Mediciones del entrenamiento':'Evaluación congelada';
  $('evaluation-description').textContent=training?'Trayectorias del batch normal, medidas antes de actualizar los pesos. La media da el mismo peso a cada juego presente. Son diagnósticos de aprendizaje; no miden generalización ni victorias.':'Semillas separadas, pesos congelados y memoria causal reconstruida desde el inicio de cada sesión. La media da el mismo peso a cada juego; no mide victorias.';
  $('evaluations-title').textContent=training?'Mediciones antes de aprender':'Evaluaciones congeladas';
  $('evaluations-description').textContent=training?'Un registro por batch de entrenamiento. Las escenas cambian entre batches; no son partidas fijas. Reconstrucción y rollout utilizan una muestra por juego y nivel.':'Un registro por batch con las mismas sesiones reales de referencia. Reconstrucción y rollout se miden cada cuatro batches; los huecos no significan cero.';
  $('drift-description').textContent=training?'Cambio relativo del encoder sobre las mismas imágenes del batch, antes y después de actualizar representación. Cada batch aporta nuevas imágenes; no se confunde movimiento de objetos con deriva del encoder.':'Cambio relativo del latente frente a la primera medición del proceso. Al reanudar se establece una nueva referencia; se separan las líneas entre referencias.';
  $('phase-card').hidden=!control;
  if(control){
    $('phase-description').textContent=`Ventana de ${control.window_batches} batches. Todos los juegos y niveles deben cumplir reconstrucción y deriva con suficiente evidencia. La reconstrucción usa la unión de cambios reales y reconstruidos; se incluyen cambios observados poco frecuentes. La admisión del predictor se conserva; las métricas posteriores siguen visibles.`;
    table('phase-rows',control.rows.map(r=>[`${r.game} / ${r.level}`,`${r.samples} / ${r.changed} · mínimo ${r.thresholds.minimum_samples} / ${r.thresholds.minimum_changed}`,
      `${number(pct(r.reconstruction_error))} % / ${number(pct(r.thresholds.representation_open))} %`,
      `${number(pct(r.latent_drift))} % / ${number(pct(r.thresholds.latent_drift))} %`,`${r.passes} / ${r.thresholds.patience}`,
      !r.enough?'Recogiendo evidencia':r.passing?'Cumple':'Fuera del umbral']));
  }
  $('status').textContent = ({running: 'Entrenando', paused: 'Pausado', budget_complete: 'Presupuesto terminado', error: 'Error', waiting: 'Esperando'})[data.status] || data.status;
  $('run-info').textContent = data.message || `${(data.games || []).join(' · ')} · Actualizado hace ${number(data.age_seconds, 0)} s`;
  const shuffled=c.game_schedule==='shuffled';
  $('configuration').textContent = `${shuffled?'Hasta ':''}${c.batch_games || '—'} sesiones independientes · ${c.session_steps || '—'} acciones por sesión · Bloque: ${c.block_steps || '—'} pasos + ${c.context_steps ?? 0} de contexto con gradiente · Peso inicial: ${number(c.loss_start_weight)} → 1 en ${c.loss_warmup_steps || '—'} acciones · ${shuffled?'Juegos barajados sin repetición':c.game_schedule === 'blocked' ? 'Un juego por batch' : 'Juegos alternados'}`;
  $('learning-rates').textContent=Object.entries(data.learning_rates||{}).map(([key,value])=>`LR ${key}: ${Number.isFinite(value)?value.toExponential(1):'—'}`).join(' · ');
  const budgetLabel=data.budget===null?' · Sin límite de acciones':data.budget?` · presupuesto ${number(data.budget,0)}`:'';
  $('actions').textContent = number(data.progress?.actions, 0); $('batches').textContent = `${number(data.progress?.batches, 0)} batches completados${budgetLabel}`;
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
    series('Reconstrucción global %',r=>pct(r.diagnostics?.reconstruction_error),{skip:r=>!r.diagnostics}),
    series('Predicción 3 pasos %',r=>pct(r.diagnostics?.rollout_error),{skip:r=>!r.diagnostics}),
    series('Reconstrucción sobre cambios %',r=>pct(r.diagnostics?.reconstruction_delta_error),{color:colors[4]})],
    100,phaseThresholds(data,'representation_open','Límite de reconstrucción sobre cambios',colors[4]));
  $('diagnostic-scope').textContent=training
    ?`Datos del batch normal antes de actualizar pesos · diagnóstico estratificado por juego/nivel${diagnostic?` · batch ${diagnostic.batch}, ${diagnostic.sampled_transitions} transiciones · ${diagnostic.rollouts} rutas de tres pasos`:''}. La deriva compara las mismas imágenes antes y después de actualizar el encoder.`
    :diagnostic
    ?`Evaluación fija en cada batch. Diagnósticos adicionales cada ${diagnostic.every} batches · última medición: batch ${diagnostic.batch}, ${diagnostic.sampled_transitions} transiciones y ${diagnostic.latent?.sample_count??0} estados reales. Rollout: ${diagnostic.rollouts} rutas de tres pasos. Deriva: referencia del proceso en batch ${diagnostic.drift_reference_batch}.`
    :'Evaluación fija en cada batch. Los diagnósticos adicionales aparecerán tras una nueva medición; no se rellenan datos anteriores con ceros.';
  chart('latent-drift',measured,[series('Deriva %',r=>pct(r.diagnostics.latent_drift),{group:r=>r.diagnostics.drift_source??r.diagnostics.drift_reference_batch})],
    null,phaseThresholds(data,'latent_drift','Límite de deriva',colors[0]));
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
  table('schedule',(data.schedule||[]).map((s,i)=>[i+1,s.game,s.level,s.source,control?.phase==='representation'?'Encoder · decoder':'Encoder · decoder · predictor']));
  $('schedule-info').textContent=`${shuffled?'Cada juego una vez por vuelta, en orden barajado; el último batch puede ser menor':c.game_schedule==='blocked'?'Un juego por batch':'Juegos intercalados en cada batch'} · rotación cada ${c.level_steps??'—'} movimientos en juegos sintéticos; el SDK progresa de forma nativa. Se conserva la memoria entre niveles de una sesión. Este modo no utiliza umbrales para activar el planner.`;
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
    $('preview-info').textContent=data.measurement_source==='training_pre_update'?'Esperando una muestra del batch de entrenamiento.':'Esperando una muestra de la evaluación fija.';
    for(const id of ['before','reconstruction','real','predicted','difference'])draw(`preview-${id}`,null);
    return;
  }
  const arrows={1:'↑',2:'↓',3:'←',4:'→',5:'Space',6:'Click',7:'Undo'};
  $('preview-info').textContent=`${sample.game} · nivel ${sample.level} · paso ${sample.step} · A${sample.action} ${arrows[sample.action]||''} · ${data.measurement_source==='training_pre_update'?'antes de entrenar el batch':'evaluación fija del batch'} ${data.prediction_sample_batch}`;
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
