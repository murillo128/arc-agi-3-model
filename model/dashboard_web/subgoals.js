'use strict';
const $ = id => document.getElementById(id);
const finite = x => typeof x === 'number' && Number.isFinite(x);
const fmt = (x, n=0) => finite(x) ? x.toLocaleString('es-ES', {maximumFractionDigits:n}) : '—';
// Same pinned SDK palette as viewer/src/canvas.ts.
const palette = ['#ffffff','#cccccc','#999999','#666666','#333333','#000000','#e53aa3','#ff7bcc','#f93c31','#1e93ff','#88d8f1','#ffdc00','#ff851b','#921231','#4fcc30','#a356d6'];
const colors = ['#77dabb','#f4b979','#a9a6ef','#94a7aa'];
const deltaKinds = [['Inventados','invented'],['Omitidos','missed'],['Color incorrecto','color'],['Acertados','correct']];
const predictorLossKinds = [['MSE latente','latent'],['NLL visual','delta_nll'],
  ['Error de solapamiento','delta_overlap'],['Rollout · total','rollout']];
const namespace = 'http://www.w3.org/2000/svg';
const chartStates = new Map();
function gpuStatus(sample) {
  const devices=sample?.devices||[], status=$('gpu-status');
  status.textContent=devices.length ? devices.map(gpu=>
    `GPU${devices.length>1?' '+gpu.index:''} · ${fmt(gpu.utilization_percent)} % de uso · ${fmt(gpu.temperature_c)} °C`).join('\n')
    : 'GPU · datos no disponibles';
  status.title=devices.length
    ? `${devices.map(gpu=>gpu.name).join(', ')}. Lectura actual del servidor; incluye otros procesos. Se consulta cada 2 s.`
    : 'No se ha podido consultar la GPU del servidor.';
}
function element(tag, attributes={}, text='') {
  const node = document.createElementNS(namespace, tag);
  for (const [k,v] of Object.entries(attributes)) node.setAttribute(k, v);
  node.textContent=text; return node;
}
function plot(id, series, {markers=[], thresholds=[], unit='', percent=false, responsive=false, digits=2, minimumRange=1, empty='Pendiente de datos de esta ejecución'}={}) {
  const svg=$(id), legend=$(`${id}-legend`); svg.replaceChildren();
  if(!chartStates.has(id))chartStates.set(id,{hovered:null,focused:null});
  const state=chartStates.get(id), lastPoints=series.map(s=>s.points.findLast(p=>finite(p[0])&&finite(p[1])));
  state.update=()=>{};
  const width=responsive?(svg.clientWidth||700):700, height=responsive?(svg.clientHeight||280):280;
  svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  const suffix=unit?` ${unit}`:percent?' %':'';
  const labels=[];
  series.forEach((s,i)=>{
    if(s.legend===false)return;
    const last=lastPoints[i];
    labels.push({key:`series:${s.name}`,index:i,text:`${s.name}: ${last?fmt(last[1],digits)+suffix:'—'}`,
      title:last?`Último dato medido · acción ${fmt(last[0])}${last[2]?' · '+last[2]:''}`:'Sin mediciones',
      className:s.muted?'series-muted':`series-${s.colorIndex??i%4}`});
  });
  // Coincident limits share one line and label, rather than hiding each other.
  const byValue=new Map();
  for(const threshold of thresholds.filter(t=>finite(t.value))) {
    const previous=byValue.get(threshold.value);
    if(previous){previous.name+=` / ${threshold.name}`;previous.colorIndex=3;}
    else byValue.set(threshold.value,{...threshold});
  }
  const references=[...byValue.values()];
  for(const ref of references) {
    labels.push({key:`threshold:${ref.name}`,text:`Límite · ${ref.name}: ${fmt(ref.value,2)}${suffix}`,
      className:`threshold-label series-${ref.colorIndex??3}`});
  }
  if(markers.length)labels.push({key:'no-change',text:'Banda inferior: sin cambios reales ni predichos',className:'series-muted'});
  // Keep the actual buttons across polling redraws, preserving hover and focus.
  const signature=JSON.stringify(labels.map(l=>l.key));
  if(state.signature!==signature) {
    state.signature=signature;state.hovered=state.focused=null;
    legend.replaceChildren(...labels.map(label=>{
      const button=label.index!==undefined, node=document.createElement(button?'button':'span');
      if(button) {
        node.type='button';node.setAttribute('aria-controls',id);
        node.addEventListener('pointerenter',event=>{if(event.pointerType!=='touch'){state.hovered=label.index;state.update();}});
        node.addEventListener('pointerleave',()=>{state.hovered=null;state.update();});
        node.addEventListener('focus',()=>{state.focused=label.index;state.update();});
        node.addEventListener('blur',()=>{state.focused=null;state.update();});
        node.addEventListener('click',()=>node.focus()); // Also expose the guide on touch screens.
      }
      return node;
    }));
  }
  labels.forEach((label,i)=>{
    const node=legend.children[i];node.textContent=label.text;node.title=label.title||'';node.className=label.className;
    if(label.index!==undefined)node.disabled=!lastPoints[label.index];
    node.setAttribute('data-highlighted','false');
  });
  if(!lastPoints[state.hovered])state.hovered=null;
  if(!lastPoints[state.focused])state.focused=null;
  const points=series.flatMap(s=>s.points).filter(p=>finite(p[0])&&finite(p[1]));
  if (!points.length&&!markers.length) {svg.append(element('text',{x:width/2,y:height/2,fill:'#94a7aa','text-anchor':'middle'},empty));return;}
  const times=series.flatMap(s=>s.points.map(p=>p[0])).concat(markers.flatMap(m=>[m.start,m.end])).filter(finite);
  const minX=Math.min(...times), maxX=Math.max(minX+1,...times);
  const minY=Math.min(0,...points.map(p=>p[1])), maxY=percent?100:Math.max(minY+minimumRange,...points.map(p=>p[1]),...references.map(r=>r.value));
  const x=v=>58+(v-minX)/(maxX-minX)*(width-80), y=v=>height-40-(v-minY)/(maxY-minY)*(height-64);
  for(let i=0;i<=4;i++) {const v=minY+(maxY-minY)*i/4;
    svg.append(element('line',{x1:58,y1:y(v),x2:width-22,y2:y(v),stroke:'#2a3835'}));
    svg.append(element('text',{x:50,y:y(v)+4,fill:'#94a7aa','font-size':11,'text-anchor':'end'},fmt(v,digits)));
    const a=minX+(maxX-minX)*i/4;svg.append(element('text',{x:x(a),y:height-14,fill:'#94a7aa','font-size':11,'text-anchor':'middle'},fmt(a)));
  }
  for(const ref of references) {
    const line=element('line',{x1:58,y1:y(ref.value),x2:width-22,y2:y(ref.value),
      stroke:colors[ref.colorIndex??3],'stroke-width':1,'stroke-dasharray':'2 4',opacity:.75});
    line.append(element('title',{},`Límite · ${ref.name}: ${fmt(ref.value,2)}${suffix}`));svg.append(line);
  }
  const layers=series.map(()=>[]);
  series.forEach((s,i)=>{
    const colorIndex=s.colorIndex??i%4;
    const color=s.muted?'#94a7aa':colors[colorIndex], opacity=s.faint ? .2 : s.muted ? .45 : 1;
    const append=node=>{layers[i].push({node,opacity});svg.append(node);};
    let segment=[];
    const flush=()=>{if(segment.length)append(element('polyline',{points:segment.map(p=>`${x(p[0])},${y(p[1])}`).join(' '),fill:'none',stroke:color,'stroke-width':s.muted||s.faint?1:2,opacity}));segment=[];};
    for(const p of s.points) {
      // Leave a gap when there was no measurement; never join across it.
      if(!finite(p[0])||!finite(p[1])) {flush();continue;}
      segment.push(p);
      const circle=element('circle',{cx:x(p[0]),cy:y(p[1]),r:2,fill:color,opacity});
      circle.append(element('title',{},`${s.name}: ${fmt(p[1],4)} · acción ${p[0]}${p[2]?' · '+p[2]:''}`));append(circle);
    }
    flush();
  });
  if(markers.length) {
    for(const m of markers) {const mark=element('rect',{x:x(m.start),y:height-35,width:Math.max(2,x(m.end)-x(m.start)),height:5,fill:'#94a7aa'});mark.append(element('title',{},`Sin cambios reales ni predichos · acciones ${m.start}–${m.end}`));svg.append(mark);}
    if(!points.length)svg.append(element('text',{x:width/2,y:height/2,fill:'#94a7aa','text-anchor':'middle'},'Sin cambios: no hay porcentaje de error que medir'));
  }
  const guide=element('g',{'class':'series-guide','pointer-events':'none','aria-hidden':'true'});svg.append(guide);
  state.update=()=>{
    const selected=state.hovered??state.focused, last=lastPoints[selected];
    layers.forEach((layer,i)=>layer.forEach(({node,opacity})=>node.setAttribute('opacity',
      !last?opacity:i===selected?1:opacity*.2)));
    labels.forEach((label,i)=>legend.children[i].setAttribute('data-highlighted',String(!!last&&label.index===selected)));
    guide.replaceChildren();
    if(!last)return;
    const s=series[selected], color=s.muted?'#94a7aa':colors[s.colorIndex??selected%4], position=y(last[1]);
    guide.append(element('line',{x1:58,y1:position,x2:width-22,y2:position,stroke:color,'stroke-width':1.2,'stroke-dasharray':'3 4'}),
      element('circle',{cx:x(last[0]),cy:position,r:4,fill:'#172122',stroke:color,'stroke-width':2}),
      element('text',{x:width-22,y:Math.max(16,position-7),fill:color,stroke:'#172122','stroke-width':3,'paint-order':'stroke',
        'font-size':11,'text-anchor':'end'},`${fmt(last[1],digits)}${suffix}`));
  };
  state.update();
}
function picture(id, frame, mask, pixelColors=palette) {
  const canvas=$(id), ctx=canvas.getContext('2d');
  canvas.height=Array.isArray(frame)&&frame.length?frame.length:64;
  canvas.width=Array.isArray(frame?.[0])&&frame[0].length?frame[0].length:64;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!Array.isArray(frame))return;
  frame.forEach((row,y)=>row.forEach((value,x)=>{ctx.fillStyle=value===16||(mask&&!mask[y]?.[x]) ? ((x+y)%2?'#aaa':'#eee') : pixelColors[value]||'#000';ctx.fillRect(x,y,1,1);}));
}
function predictionSample(sample) {
  $('prediction-sample-images').hidden=!sample;
  picture('prediction-before',sample?.previous);
  picture('prediction-reconstruction',sample?.reconstruction);
  const reconstructed=Array.isArray(sample?.reconstruction);
  $('prediction-reconstruction').hidden=!reconstructed;
  $('prediction-reconstruction-empty').hidden=reconstructed;
  picture('prediction-real',sample?.real_delta);
  picture('prediction-guessed',sample?.predicted_delta);
  const counts=deltaKinds.map(()=>0);
  const difference=sample?.real_delta.map((row,y)=>row.map((real,x)=>{
    const predicted=sample.predicted_delta[y][x];
    if(real===16&&predicted===16)return 16;
    // Categories partition the union; matching unchanged pixels are excluded.
    const kind=real===16?0:predicted===16?1:real!==predicted?2:3;
    counts[kind]++;return kind;
  }));
  picture('prediction-difference',difference,null,colors);
  const legend=$('prediction-difference-legend');legend.replaceChildren();
  if(!sample) {
    $('prediction-sample-info').textContent='Esperando una predicción y su resultado real…';
    $('prediction-sample-error').textContent='';return;
  }
  deltaKinds.forEach(([name],i)=>{
    const span=document.createElement('span');span.className=`series-${i}`;
    span.textContent=`${name}: ${fmt(counts[i])} px`;legend.append(span);
  });
  const command=sample.action, coordinates=command.id===6?` (${command.data.x}, ${command.data.y})`:'';
  const age=Math.max(0,(Date.now()/1000)-sample.captured_at);
  $('prediction-sample-info').textContent=`${sample.game} · nivel ${fmt(sample.level)} · partida ${sample.episode_seed} · paso ${fmt(sample.episode_action)} · ACTION${command.id}${coordinates} · ${sample.phase==='training'?'Entrenamiento':'Evaluación'} · acción de entrenamiento ${fmt(sample.training_action)} · hace ${fmt(age)} s`;
  $('prediction-sample-error').textContent=sample.union>0
    ? `Error de esta muestra: ${fmt(100*sample.wrong/sample.union,2)} % · ${fmt(sample.wrong)} / ${fmt(sample.union)} píxeles cambiados en cualquiera de los dos deltas.`
    : 'Sin cambios reales ni predichos en esta acción.';
}
function deltaRates(rows) {
  return Object.fromEntries(deltaKinds.map(([,key])=>{
    const values=rows.map(r=>r.counts?.[key]).filter(finite);
    return [key,values.length?values.reduce((sum,v)=>sum+v,0)/values.length:null];
  }));
}
function blocks(rows) {
  const grouped=new Map();
  for(const row of rows) {const key=Math.floor((row.actions-1)/32);if(!grouped.has(key))grouped.set(key,[]);grouped.get(key).push(row);}
  return [...grouped.values()].map(group=>{const count={};for(const row of group)for(const [k,v]of Object.entries(row.counts||{}))count[k]=(count[k]||0)+v;
    const measured=group.filter(r=>r.counts);
    return {start:group[0].actions,x:group.at(-1).actions,count,rates:deltaRates(group),
      noChange:measured.length===group.length&&count.union===0,
      error:count.union>0?100*count.wrong/count.union:null};});
}
function trend(rows, grouped, window=512) {
  return grouped.map(b=>{
    const samples=rows.filter(r=>r.actions>b.x-window&&r.actions<=b.x);
    const count=samples.reduce((a,r)=>({wrong:a.wrong+(r.counts?.wrong||0),union:a.union+(r.counts?.union||0)}),{wrong:0,union:0});
    const mean=key=>{const values=samples.map(r=>r[key]).filter(finite);return values.length?values.reduce((sum,v)=>sum+v,0)/values.length:null;};
    return {x:b.x,rates:deltaRates(samples),error:count.union>0?100*count.wrong/count.union:null,
      loss:mean('predictor_loss'),representationLoss:mean('representation_loss'),pathLoss:mean('path_loss')};
  });
}
function movingAverage(points, window=512) {
  const samples=points.filter(p=>finite(p[0])).sort((a,b)=>a[0]-b[0]);
  let start=0,end=0,sum=0,count=0;
  // Several subgoals can be published at the same episode-end action. Include
  // all of them once in that window; their array order must not change the mean.
  return [...new Set(samples.map(p=>p[0]))].map(x=>{
    while(end<samples.length&&samples[end][0]<=x){if(finite(samples[end][1])){sum+=samples[end][1];count++;}end++;}
    while(start<end&&samples[start][0]<=x-window){if(finite(samples[start][1])){sum-=samples[start][1];count--;}start++;}
    return [x,count?sum/count:null];
  });
}
function lossBreakdown(id, rows, key) {
  const target=$(id);target.replaceChildren();
  const row=rows.findLast(r=>r[key]&&Object.values(r[key]).some(finite));
  if(!row){target.textContent='Sin una actualización registrada';return;}
  const labels={color_ce:'Color',delta_nll:'Delta visual',delta_overlap:'Unión del delta',
    local_reconstruction:'Reconstrucción local',sigreg:'SIGReg',variance_floor:'Varianza',
    latent:'Delta latente',rollout:'Predicción encadenada'};
  Object.entries(row[key]).filter(([,v])=>finite(v)).forEach(([name,value],i)=>{
    const span=document.createElement('span');span.className=`series-${i%4}`;
    span.textContent=`${labels[name]||name}: ${fmt(value,4)}`;
    span.title=`Contribución al total, con su peso aplicado · acción ${fmt(row.actions)}`;
    target.append(span);
  });
}
function fixedEvaluations(report) {
  const pair=r=>`${r.game}/${r.level}`;
  // Older live reports have no declared pair list; their completed training
  // episodes and evaluation rows identify the campaign's observed coverage.
  const expected=new Set((report.config?.evaluation_pairs||[...(report.episodes||[]),...(report.evaluations||[])]).map(pair));
  const grouped=new Map();
  for(const row of report.evaluations||[]) {
    if(row.split!=='dev'||row.policy!=='random'||!finite(row.actions))continue;
    if(!grouped.has(row.actions))grouped.set(row.actions,new Map());
    grouped.get(row.actions).set(pair(row),row);
  }
  const points=[];let pending=null;
  for(const [actions,byPair] of [...grouped].sort((a,b)=>a[0]-b[0])) {
    const coverage=[...byPair.values()][0]?.coverage;
    const required=coverage?new Set(coverage.map(pair)):expected;
    const rows=[...required].map(k=>byPair.get(k));
    if(!rows.length||rows.some(r=>!r||!finite(r.delta_error))) {pending={actions,expected:required.size,pairs:rows.filter(r=>r&&finite(r.delta_error)).length};continue;}
    const mean=key=>rows.every(r=>finite(r.quality?.[key]))?100*rows.reduce((s,r)=>s+r.quality[key],0)/rows.length:null;
    const latentMean=(key,scale=1)=>rows.every(r=>finite(r.quality?.latent?.[key]))
      ?scale*rows.reduce((s,r)=>s+r.quality.latent[key],0)/rows.length:null;
    points.push({actions,stage:rows[0].stage,error:100*rows.reduce((s,r)=>s+r.delta_error,0)/rows.length,
      reconstruction:mean('reconstruction_error'),rollout:mean('rollout_error'),drift:mean('latent_drift'),
      latentStd:latentMean('std_mean'),latentWithin:latentMean('within_episode_std'),
      latentInactive:latentMean('inactive_fraction',100),latentRank:latentMean('effective_rank'),
      latentMaxRank:latentMean('max_rank'),latentChanged:latentMean('changed_rms'),latentUnchanged:latentMean('unchanged_rms'),
      latentSamples:latentMean('sample_count',rows.length),latentEpisodes:latentMean('episode_count',rows.length),
      pairs:rows.length,episodes:rows.reduce((s,r)=>s+r.episodes,0)});
  }
  return {points,expected:expected.size,pending};
}
let displayedReport;
function render(report) {
  displayedReport=report;
  const p=report.progress||{}, c=report.config||{};
  const phases={loading:'Cargando pesos',warmup:'Preparando torch.compile y la primera acción',training:'Entrenando',evaluation:'Evaluando con pesos congelados',checkpoint:'Guardando pesos',complete:'Terminado',failed:'Detenido por error'};
  $('connection').textContent=phases[p.phase]||'Esperando entrenamiento';
  $('connection').className=report.status==='failed'||report.age_seconds>60?'warn':'';
  $('freshness').textContent=`Última escritura hace ${fmt(report.age_seconds)} s · consulta cada 2 s`;
  const initialization=c.initialization==='random'?' · Inicializado desde cero':c.initialization==='checkpoint'?' · Pesos preentrenados':'';
  $('run-info').textContent=`${c.game||'Una configuración'} · semilla ${fmt(c.seed)} · currículo acumulativo${initialization}`;
  const curriculum=report.curriculum, limits=curriculum?.thresholds||{};
  const pct=value=>finite(value)?`${fmt(value*100,2)} %`:'—';
  $('curriculum-panel').hidden=!curriculum;
  $('curriculum-summary').textContent=curriculum?`Etapa ${fmt(curriculum.stage)} · `+Object.entries(curriculum.unlocked).map(([g,n])=>`${g}: niveles 1–${n}`).join(' · '):'';
  $('thresholds').textContent=curriculum?`Activación tras ${limits.patience} evaluaciones consecutivas: reconstrucción ≤ ${pct(limits.representation_open)}, deriva latente ≤ ${pct(limits.latent_drift)}, predicción a 1 y 3 pasos ≤ ${pct(limits.prediction_open)}. Avance: además éxito ≥ ${pct(limits.success)} en TODOS los niveles desbloqueados del juego. Se exige cobertura de cambios; si ningún píxel cambia, hace falta observar una terminación nativa.`:'';
  $('curriculum-rows').replaceChildren(...(curriculum?.levels||[]).map(row=>{
    const m=row.measurements||{}, tr=document.createElement('tr');
    const next=!row.unlocked?'Nivel bloqueado':row.planner?'Todos activos':row.predictor?`Planner: ${fmt(row.prediction_passes)} / ${limits.patience}`:`Predictor: ${fmt(row.representation_passes)} / ${limits.patience}`;
    const active=!row.unlocked?'Pendiente':row.planner?'Representación + predictor + planner':row.predictor?'Representación + predictor':'Representación';
    for(const value of [`${row.game} / ${row.level}`,active,pct(m.reconstruction_error),pct(m.latent_drift),pct(m.prediction_error),pct(m.rollout_error),pct(m.success),next,row.unlocked?`${Math.min(row.promotion_passes,limits.patience)} / ${limits.patience}`:'—']){const td=document.createElement('td');td.textContent=value;tr.append(td);}return tr;
  }));
  $('representation-updates').textContent=fmt(p.representation_updates);
  const encoderFrozen=(p.encoder_frozen??c.freeze_encoder)===true;
  $('representation-update-scope').textContent=encoderFrozen
    ? 'Decoder · encoder congelado' : 'Encoder y decoder · cada acción real';
  $('world-training-scope').textContent=(encoderFrozen
    ? 'Encoder congelado: los latentes se mantienen fijos para una misma observación. El decoder sigue aprendiendo.'
    : 'Encoder y decoder aprenden desde el principio.')
    +' El predictor se añade cuando la representación está preparada. Las evaluaciones no actualizan pesos.';
  $('notice').hidden=!report.error;$('notice').textContent=report.error||'';
  $('actions').textContent=fmt(p.actions);$('budget').textContent=`De ${fmt(p.total_actions)} · evaluación excluida`;
  $('world-updates').textContent=fmt(p.predictor_updates);$('path-updates').textContent=fmt(p.path_updates);
  $('gate').textContent=p.predictor_enabled?'Activo · cada acción junto a representación':'Pendiente de representación estable';
  const replay=p.predictor_replay;
  $('predictor-replay').textContent=replay
    ? `Replay: ${fmt(replay.episodes)}/${fmt(replay.capacity)} partidas · ${fmt(replay.transitions)} transiciones · lote ${fmt(replay.actual_batch_size)}/${fmt(replay.batch_size)}`
    : c.predictor_replay_episodes>0 ? `Replay: hasta ${fmt(c.predictor_replay_episodes)} partidas · lote objetivo ${fmt(c.predictor_batch_size)}`
    : 'Sin replay · una transición por actualización';
  predictionSample(report.prediction_sample);
  $('tps').textContent=fmt(p.tps,2);
  $('progress').textContent=p.phase==='evaluation'
    ? `${p.policy} · ${p.evaluation_game} / nivel ${fmt(p.evaluation_level)} · partida ${fmt(p.evaluation_episode)}/${fmt(p.evaluation_episodes)} · acción ${fmt(p.evaluation_action)}`
    : `${p.game||c.game||'Preparando juego'} / nivel ${fmt(p.level)} · subobjetivo ${fmt(p.goal_index)}/${fmt(p.goals)} · ${p.goal||''} · ${fmt(p.goal_actions)} acciones (mínimo inicial ${fmt(p.minimum_actions)}) · exploración ${fmt(p.epsilon*100)} %`;
  const fixed=fixedEvaluations(report), last=fixed.points.at(-1);
  const qualityPoints=key=>{let stage=null;const points=[];for(const r of fixed.points){if(stage!==null&&r.stage!==stage)points.push([r.actions,null]);points.push([r.actions,r[key],`${r.pairs} pares · ${r.episodes} partidas`]);stage=r.stage;}return points;};
  const qualityThresholds=[{name:'Reconstrucción',value:100*limits.representation_open,colorIndex:1},
    {name:'Predicción',value:100*limits.prediction_open,colorIndex:0},
    {name:'Deriva',value:100*limits.latent_drift,colorIndex:3}];
  // Relative latent drift can exceed 100%; allow this chart to expand its axis.
  plot('fixed-delta',[{name:'Predicción · 1 paso',points:qualityPoints('error')},
    {name:'Reconstrucción',points:qualityPoints('reconstruction')},
    {name:'Predicción · 3 pasos',points:qualityPoints('rollout')},
    {name:'Deriva latente',points:qualityPoints('drift')}],{unit:'%',thresholds:qualityThresholds,percent:!fixed.points.some(r=>r.drift>100),responsive:true,empty:'Pendiente de una evaluación fija completa'});
  $('quality-cadence').textContent=finite(c.evaluate_every)
    ? `Calidad, deriva y activaciones cada ${fmt(c.evaluate_every)} acciones · al terminar la partida.`:'';
  $('fixed-summary').textContent=(last?`Última medición: acción ${fmt(last.actions)} · ${fmt(last.pairs)} pares · ${fmt(last.episodes)} partidas.`:'Todavía no hay una evaluación fija completa.')+
    (fixed.pending&&(!last||fixed.pending.actions>last.actions)?` En curso: ${fmt(fixed.pending.pairs)}/${fmt(fixed.pending.expected??fixed.expected)} pares; el punto se añadirá al completarlos.`:'');
  const latentPoints=key=>{
    const points=qualityPoints(key), first=points.findIndex(p=>finite(p[1]));
    return first<0?[]:points.slice(first); // Older reports have no latent diagnostics.
  };
  const latentLast=fixed.points.findLast(r=>r.latentSamples>0);
  $('latent-summary').textContent=latentLast
    ?`Última medición: acción ${fmt(latentLast.actions)} · ${fmt(latentLast.latentSamples)} observaciones de ${fmt(latentLast.latentEpisodes)} partidas. Curvas agregadas con igual peso por nivel; una buena reconstrucción o diversidad alta no garantizan buena predicción.`
    :'Las medidas aparecerán tras una evaluación fija con los nuevos diagnósticos. Los datos anteriores no se rellenan con ceros.';
  plot('latent-diversity',[{name:'Entre observaciones',points:latentPoints('latentStd')},
    {name:'Dentro de cada partida',points:latentPoints('latentWithin')}],{digits:4,minimumRange:.0001});
  plot('latent-rank',[{name:'Rango efectivo',points:latentPoints('latentRank')},
    {name:'Máximo observable',muted:true,points:latentPoints('latentMaxRank')}]);
  plot('latent-inactive',[{name:'Casi constantes',points:latentPoints('latentInactive')}],{percent:true});
  plot('latent-sensitivity',[{name:'Bloques con cambios',points:latentPoints('latentChanged')},
    {name:'Bloques sin cambios',points:latentPoints('latentUnchanged')}],{digits:4,minimumRange:.0001});
  const rows=report.curves||[], b=blocks(rows), t=trend(rows,b), series=(name,key)=>({name,points:rows.map(r=>[r.actions,r[key]])});
  const measured=rows.filter(r=>r.counts), unchanged=measured.filter(r=>r.counts.union===0).length;
  $('interaction-summary').textContent=rows.length?`Ventana reciente: acciones ${fmt(rows[0].actions)}–${fmt(rows.at(-1).actions)}. ${fmt(unchanged)} de ${fmt(measured.length)} acciones medidas sin cambios reales ni predichos.`:'Esperando acciones reales de esta ejecución.';
  plot('delta',[{name:'Media móvil · hasta 512 acciones',points:t.map(r=>[r.x,r.error])},
    {name:'Bloque de 32 acciones',muted:true,points:b.map(r=>[r.x,r.error])}],
    {percent:true,markers:b.filter(r=>r.noChange).map(r=>({start:r.start,end:r.x}))});
  plot('changes',[
    ...deltaKinds.map(([name,key],colorIndex)=>({name,colorIndex,faint:true,legend:false,
      points:b.map(r=>[r.x,r.rates[key],'px/acción · bloque de 32'])})),
    ...deltaKinds.map(([name,key],colorIndex)=>({name,colorIndex,
      points:t.map(r=>[r.x,r.rates[key],'px/acción · media móvil de hasta 512 acciones'])}))
  ],{unit:'px/acción'});
  plot('loss',[{name:'Media de actualizaciones · hasta 512 acciones',points:t.map(r=>[r.x,r.loss])},
    {...series('Actualización individual','predictor_loss'),muted:true}]);
  plot('representation-loss',[{name:'Media de actualizaciones · hasta 512 acciones',points:t.map(r=>[r.x,r.representationLoss])},
    {...series('Actualización individual','representation_loss'),muted:true}]);
  lossBreakdown('representation-loss-terms',rows,'representation_terms');
  lossBreakdown('loss-terms',rows,'predictor_terms');
  const lossTerms=predictorLossKinds.map(([name,key],colorIndex)=>({name,colorIndex,
    points:rows.map(r=>[r.actions,r.predictor_terms?.[key]])}));
  plot('predictor-components',[
    ...lossTerms.map(s=>({...s,faint:true,legend:false})),
    ...lossTerms.map(s=>({...s,points:movingAverage(s.points)}))
  ],{digits:4,minimumRange:.001,empty:'Todavía no hay componentes de pérdida registrados'});
  const lossMode=report.config?.prediction_delta_loss;
  $('predictor-objective').textContent=lossMode==='focal'
    ? 'NLL activa: cambios reales y penalización focal suave de píxeles estáticos (γ=2, T=1). Normalizada por cambios reales, sin máscara de argmax.'
    : lossMode==='union'
    ? 'NLL activa: unión de cambios reales y predichos. El fondo correctamente conservado queda fuera de esa NLL.'
    : lossMode==='balanced' ? 'NLL activa: grupos equilibrados de píxeles cambiados y estáticos.'
    : 'Este reporte no identifica el modo de NLL.';
  $('representation-regularization').textContent=encoderFrozen
    ? 'Encoder congelado · regularización del encoder desactivada · decoder sigue aprendiendo'
    : p.sigreg_enabled===true
    ? 'SIGReg activo · fase inicial de representación'
    : p.sigreg_enabled===false ? 'SIGReg desactivado · reconstrucción continúa' : 'Estado de SIGReg no registrado';
  plot('speed',[series('Acciones/s','tps')],{unit:'acciones/s'});
  plot('path-loss',[{name:'Media de actualizaciones · hasta 512 acciones',points:t.map(r=>[r.x,r.pathLoss])},
    {...series('Actualización individual','path_loss'),muted:true}]);
  const episodes=report.episodes||[], goals=episodes.flatMap(e=>e.segments.map(g=>({...g,x:e.end_action})));
  $('episodes').textContent=fmt(episodes.length);$('successes').textContent=`${fmt(episodes.filter(e=>e.success).length)} niveles resueltos · ventana reciente`;
  $('subgoals').textContent=`${fmt(goals.filter(g=>g.success).length)} / ${fmt(goals.length)}`;$('goal-label').textContent=p.goal||'';
  picture('observation',p.observation);picture('goal',p.goal_frame,p.goal_mask);
  const efficiency=[['Real','actions'],['Mínimo','minimum_actions']].map(([name,key],colorIndex)=>({name,colorIndex,points:goals.filter(g=>g.success).map(g=>[g.x,g[key]])}));
  plot('efficiency',[
    ...efficiency.map(s=>({...s,faint:true,legend:false})),
    ...efficiency.map(s=>({...s,name:`${s.name} · media de hasta 512 acciones`,points:movingAverage(s.points)}))
  ],{unit:'acciones'});
  const rewards=goals.map(g=>[g.x,g.reward]);
  plot('reward',[{name:'Media por subobjetivo · hasta 512 acciones',points:movingAverage(rewards)},
    {name:'Reward individual',points:rewards,muted:true}]);
  const evaluations=report.evaluations||[];$('evaluation-empty').hidden=!!evaluations.length;
  $('evaluation-rows').replaceChildren(...evaluations.map(r=>{const tr=document.createElement('tr');for(const value of [fmt(r.actions),r.split,`${r.game} / ${r.level}`,r.policy,`${r.successes}/${r.episodes}`,`${r.subgoals}/${r.attempted_subgoals}`,`${fmt(r.delta_error*100,2)} %`,fmt(r.real_actions),fmt(r.decision_ms,2)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}return tr;}));
}
const tabs=['world','curriculum','path','evaluation'];
for(const name of tabs)$('tab-'+name).addEventListener('click',()=>{
  for(const tab of tabs){$(tab).hidden=tab!==name;$('tab-'+tab).setAttribute('aria-selected',String(tab===name));}
  // A hidden SVG has no measured width; fit it as soon as its tab is visible.
  if(name==='world'&&displayedReport)render(displayedReport);
});
async function poll(){try{const response=await fetch('api/state',{cache:'no-store',signal:AbortSignal.timeout(10000)});if(!response.ok)throw Error('HTTP '+response.status);const data=await response.json();gpuStatus(data.gpu);if(!data.subgoals)throw Error('Esperando contrato de subobjetivos');render(data.subgoals);}catch(error){gpuStatus(null);$('connection').textContent='Sin conexión · '+error.message;$('connection').className='warn';}finally{setTimeout(poll,2000);}}
poll();
