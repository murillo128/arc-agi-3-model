'use strict';
const $ = id => document.getElementById(id);
const finite = x => typeof x === 'number' && Number.isFinite(x);
const fmt = (x, n=0) => finite(x) ? x.toLocaleString('es-ES', {maximumFractionDigits:n}) : '—';
// Same pinned SDK palette as viewer/src/canvas.ts.
const palette = ['#ffffff','#cccccc','#999999','#666666','#333333','#000000','#e53aa3','#ff7bcc','#f93c31','#1e93ff','#88d8f1','#ffdc00','#ff851b','#921231','#4fcc30','#a356d6'];
const colors = ['#77dabb','#f4b979','#a9a6ef','#94a7aa'];
const deltaKinds = [['Inventados','invented'],['Omitidos','missed'],['Color incorrecto','color'],['Acertados','correct']];
const namespace = 'http://www.w3.org/2000/svg';
function element(tag, attributes={}, text='') {
  const node = document.createElementNS(namespace, tag);
  for (const [k,v] of Object.entries(attributes)) node.setAttribute(k, v);
  node.textContent=text; return node;
}
function plot(id, series, {markers=[], percent=false, empty='Pendiente de datos de esta ejecución'}={}) {
  const svg=$(id), legend=$(`${id}-legend`); svg.replaceChildren(); legend.replaceChildren();
  const points=series.flatMap(s=>s.points).filter(p=>finite(p[0])&&finite(p[1]));
  if (!points.length&&!markers.length) {svg.append(element('text',{x:350,y:140,fill:'#94a7aa','text-anchor':'middle'},empty));return;}
  const times=series.flatMap(s=>s.points.map(p=>p[0])).concat(markers.flatMap(m=>[m.start,m.end])).filter(finite);
  const minX=Math.min(...times), maxX=Math.max(minX+1,...times);
  const minY=Math.min(0,...points.map(p=>p[1])), maxY=percent?100:Math.max(minY+1,...points.map(p=>p[1]));
  const x=v=>58+(v-minX)/(maxX-minX)*620, y=v=>240-(v-minY)/(maxY-minY)*216;
  for(let i=0;i<=4;i++) {const v=minY+(maxY-minY)*i/4;
    svg.append(element('line',{x1:58,y1:y(v),x2:678,y2:y(v),stroke:'#2a3835'}));
    svg.append(element('text',{x:50,y:y(v)+4,fill:'#94a7aa','font-size':11,'text-anchor':'end'},fmt(v,2)));
    const a=minX+(maxX-minX)*i/4;svg.append(element('text',{x:x(a),y:266,fill:'#94a7aa','font-size':11,'text-anchor':'middle'},fmt(a)));
  }
  series.forEach((s,i)=>{
    const colorIndex=s.colorIndex??i%4;
    if(s.legend!==false) {const span=document.createElement('span');span.textContent=s.name;span.className=s.muted?'series-muted':`series-${colorIndex}`;legend.append(span);}
    const color=s.muted?'#94a7aa':colors[colorIndex], opacity=s.faint ? .2 : s.muted ? .45 : 1;
    let segment=[];
    const flush=()=>{if(segment.length)svg.append(element('polyline',{points:segment.map(p=>`${x(p[0])},${y(p[1])}`).join(' '),fill:'none',stroke:color,'stroke-width':s.muted||s.faint?1:2,opacity}));segment=[];};
    for(const p of s.points) {
      // Leave a gap when there was no measurement; never join across it.
      if(!finite(p[0])||!finite(p[1])) {flush();continue;}
      segment.push(p);
      const circle=element('circle',{cx:x(p[0]),cy:y(p[1]),r:2,fill:color,opacity});
      circle.append(element('title',{},`${s.name}: ${fmt(p[1],4)} · acción ${p[0]}${p[2]?' · '+p[2]:''}`));svg.append(circle);
    }
    flush();
  });
  if(markers.length) {
    const span=document.createElement('span');span.textContent='Banda inferior: sin cambios reales ni predichos';span.className='series-muted';legend.append(span);
    for(const m of markers) {const mark=element('rect',{x:x(m.start),y:245,width:Math.max(2,x(m.end)-x(m.start)),height:5,fill:'#94a7aa'});mark.append(element('title',{},`Sin cambios reales ni predichos · acciones ${m.start}–${m.end}`));svg.append(mark);}
    if(!points.length)svg.append(element('text',{x:350,y:140,fill:'#94a7aa','text-anchor':'middle'},'Sin cambios: no hay porcentaje de error que medir'));
  }
}
function picture(id, frame, mask) {
  const canvas=$(id), ctx=canvas.getContext('2d');ctx.clearRect(0,0,64,64);
  if(!Array.isArray(frame))return;
  frame.forEach((row,y)=>row.forEach((value,x)=>{ctx.fillStyle=mask&&!mask[y]?.[x] ? ((x+y)%2?'#aaa':'#eee') : palette[value]||'#000';ctx.fillRect(x,y,1,1);}));
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
    const losses=samples.map(r=>r.predictor_loss).filter(finite);
    return {x:b.x,rates:deltaRates(samples),error:count.union>0?100*count.wrong/count.union:null,
      loss:losses.length?losses.reduce((a,v)=>a+v,0)/losses.length:null};
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
    const rows=[...expected].map(k=>byPair.get(k));
    if(!rows.length||rows.some(r=>!r||!finite(r.delta_error))) {pending={actions,pairs:rows.filter(r=>r&&finite(r.delta_error)).length};continue;}
    points.push({actions,error:100*rows.reduce((s,r)=>s+r.delta_error,0)/rows.length,
      pairs:rows.length,episodes:rows.reduce((s,r)=>s+r.episodes,0)});
  }
  return {points,expected:expected.size,pending};
}
function render(report) {
  const p=report.progress||{}, c=report.config||{};
  const phases={loading:'Cargando pesos',warmup:'Preparando torch.compile y la primera acción',training:'Entrenando',evaluation:'Evaluando con pesos congelados',checkpoint:'Guardando pesos',complete:'Terminado',failed:'Detenido por error'};
  $('connection').textContent=phases[p.phase]||'Esperando entrenamiento';
  $('connection').className=report.status==='failed'||report.age_seconds>60?'warn':'';
  $('freshness').textContent=`Última escritura hace ${fmt(report.age_seconds)} s · consulta cada 2 s`;
  $('run-info').textContent=`Una configuración · semilla ${fmt(c.seed)} · predictor online · objetivos sintéticos`;
  $('notice').hidden=!report.error;$('notice').textContent=report.error||'';
  $('actions').textContent=fmt(p.actions);$('budget').textContent=`De ${fmt(p.total_actions)} · evaluación excluida`;
  $('world-updates').textContent=fmt(p.predictor_updates);$('path-updates').textContent=fmt(p.path_updates);
  $('gate').textContent=`Actualizar si error > ${fmt(c.predictor_error*100)} %`;
  $('tps').textContent=fmt(p.tps,2);
  $('progress').textContent=p.phase==='evaluation'
    ? `${p.policy} · ${p.evaluation_game} / nivel ${fmt(p.evaluation_level)} · partida ${fmt(p.evaluation_episode)}/${fmt(p.evaluation_episodes)} · acción ${fmt(p.evaluation_action)}`
    : `${p.game||'Preparando juegos'} / nivel ${fmt(p.level)} · subobjetivo ${fmt(p.goal_index)}/${fmt(p.goals)} · ${p.goal||''} · ${fmt(p.goal_actions)} acciones (mínimo inicial ${fmt(p.minimum_actions)}) · exploración ${fmt(p.epsilon*100)} %`;
  const fixed=fixedEvaluations(report), last=fixed.points.at(-1);
  plot('fixed-delta',[{name:'Evaluación fija · media por juego/nivel',points:fixed.points.map(r=>[r.actions,r.error,`${r.pairs} pares · ${r.episodes} partidas`])}],{percent:true,empty:'Pendiente de una evaluación fija completa'});
  $('fixed-summary').textContent=(last?`Última completa: ${fmt(last.error,2)} % · acción ${fmt(last.actions)} · ${fmt(last.pairs)} pares · ${fmt(last.episodes)} partidas.`:'Todavía no hay una evaluación fija completa.')+
    (fixed.pending&&(!last||fixed.pending.actions>last.actions)?` En curso: ${fmt(fixed.pending.pairs)}/${fmt(fixed.expected)} pares; el punto se añadirá al completarlos.`:'');
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
  ]);
  plot('loss',[{name:'Media de actualizaciones · hasta 512 acciones',points:t.map(r=>[r.x,r.loss])},
    {...series('Actualización individual','predictor_loss'),muted:true}]);
  plot('speed',[series('Acciones/s','tps')]);plot('path-loss',[series('Coste restante','path_loss')]);
  const episodes=report.episodes||[], goals=episodes.flatMap(e=>e.segments.map(g=>({...g,x:e.end_action})));
  $('episodes').textContent=fmt(episodes.length);$('successes').textContent=`${fmt(episodes.filter(e=>e.success).length)} niveles resueltos · ventana reciente`;
  $('subgoals').textContent=`${fmt(goals.filter(g=>g.success).length)} / ${fmt(goals.length)}`;$('goal-label').textContent=p.goal||'';
  picture('observation',p.observation);picture('goal',p.goal_frame,p.goal_mask);
  plot('efficiency',[['Real','actions'],['Mínimo','minimum_actions']].map(([name,key])=>({name,points:goals.filter(g=>g.success).map(g=>[g.x,g[key]])})));
  plot('reward',[{name:'Reward real',points:goals.map(g=>[g.x,g.reward])}]);
  const evaluations=report.evaluations||[];$('evaluation-empty').hidden=!!evaluations.length;
  $('evaluation-rows').replaceChildren(...evaluations.map(r=>{const tr=document.createElement('tr');for(const value of [fmt(r.actions),r.split,`${r.game} / ${r.level}`,r.policy,`${r.successes}/${r.episodes}`,`${r.subgoals}/${r.attempted_subgoals}`,`${fmt(r.delta_error*100,2)} %`,fmt(r.real_actions),fmt(r.decision_ms,2)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}return tr;}));
}
for(const name of ['world','path','evaluation'])$('tab-'+name).addEventListener('click',()=>{for(const tab of ['world','path','evaluation']){$(tab).hidden=tab!==name;$('tab-'+tab).setAttribute('aria-selected',String(tab===name));}});
async function poll(){try{const response=await fetch('api/state',{cache:'no-store',signal:AbortSignal.timeout(10000)});if(!response.ok)throw Error('HTTP '+response.status);const data=await response.json();if(!data.subgoals)throw Error('Esperando contrato de subobjetivos');render(data.subgoals);}catch(error){$('connection').textContent='Sin conexión · '+error.message;$('connection').className='warn';}finally{setTimeout(poll,2000);}}
poll();
