'use strict';
const $ = id => document.getElementById(id);
const finite = x => typeof x === 'number' && Number.isFinite(x);
const fmt = (x, n=0) => finite(x) ? x.toLocaleString('es-ES', {maximumFractionDigits:n}) : '—';
// Same pinned SDK palette as viewer/src/canvas.ts.
const palette = ['#ffffff','#cccccc','#999999','#666666','#333333','#000000','#e53aa3','#ff7bcc','#f93c31','#1e93ff','#88d8f1','#ffdc00','#ff851b','#921231','#4fcc30','#a356d6'];
const colors = ['#77dabb','#f4b979','#a9a6ef','#94a7aa'];
const namespace = 'http://www.w3.org/2000/svg';
function element(tag, attributes={}, text='') {
  const node = document.createElementNS(namespace, tag);
  for (const [k,v] of Object.entries(attributes)) node.setAttribute(k, v);
  node.textContent=text; return node;
}
function plot(id, series) {
  const svg=$(id), legend=$(`${id}-legend`); svg.replaceChildren(); legend.replaceChildren();
  const points=series.flatMap(s=>s.points).filter(p=>finite(p[0])&&finite(p[1]));
  if (!points.length) {svg.append(element('text',{x:350,y:140,fill:'#94a7aa','text-anchor':'middle'},'Pendiente de datos de esta ejecución'));return;}
  const minX=Math.min(...points.map(p=>p[0])), maxX=Math.max(minX+1,...points.map(p=>p[0]));
  const minY=Math.min(0,...points.map(p=>p[1])), maxY=Math.max(minY+1,...points.map(p=>p[1]));
  const x=v=>58+(v-minX)/(maxX-minX)*620, y=v=>240-(v-minY)/(maxY-minY)*216;
  for(let i=0;i<=4;i++) {const v=minY+(maxY-minY)*i/4;
    svg.append(element('line',{x1:58,y1:y(v),x2:678,y2:y(v),stroke:'#2a3835'}));
    svg.append(element('text',{x:50,y:y(v)+4,fill:'#94a7aa','font-size':11,'text-anchor':'end'},fmt(v,2)));
    const a=minX+(maxX-minX)*i/4;svg.append(element('text',{x:x(a),y:266,fill:'#94a7aa','font-size':11,'text-anchor':'middle'},fmt(a)));
  }
  series.forEach((s,i)=>{const span=document.createElement('span');span.textContent=s.name;span.className=`series-${i%4}`;legend.append(span);
    const ps=s.points.filter(p=>finite(p[0])&&finite(p[1]));
    svg.append(element('polyline',{points:ps.map(p=>`${x(p[0])},${y(p[1])}`).join(' '),fill:'none',stroke:colors[i%4],'stroke-width':2}));
    for(const p of ps) {const circle=element('circle',{cx:x(p[0]),cy:y(p[1]),r:2,fill:colors[i%4]});circle.append(element('title',{},`${s.name}: ${fmt(p[1],4)} · acción ${p[0]}`));svg.append(circle);}
  });
}
function picture(id, frame, mask) {
  const canvas=$(id), ctx=canvas.getContext('2d');ctx.clearRect(0,0,64,64);
  if(!Array.isArray(frame))return;
  frame.forEach((row,y)=>row.forEach((value,x)=>{ctx.fillStyle=mask&&!mask[y]?.[x] ? ((x+y)%2?'#aaa':'#eee') : palette[value]||'#000';ctx.fillRect(x,y,1,1);}));
}
function blocks(rows) {
  const grouped=new Map();
  for(const row of rows) {const key=Math.floor((row.actions-1)/32);if(!grouped.has(key))grouped.set(key,[]);grouped.get(key).push(row);}
  return [...grouped.values()].map(group=>{const count={};for(const row of group)for(const [k,v]of Object.entries(row.counts||{}))count[k]=(count[k]||0)+v;
    return {x:group.at(-1).actions,count,error:group.some(r=>r.counts)?100*(count.wrong||0)/Math.max(1,count.union||0):null};});
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
  const rows=report.curves||[], b=blocks(rows), series=(name,key)=>({name,points:rows.map(r=>[r.actions,r[key]])});
  plot('delta',[{name:'Error antes de actualizar · %',points:b.map(r=>[r.x,r.error])}]);
  plot('changes',[['Inventados','invented'],['Omitidos','missed'],['Color incorrecto','color'],['Acertados','correct']].map(([name,key])=>({name,points:b.map(r=>[r.x,r.count[key]])})));
  plot('loss',[series('Predictor','predictor_loss')]);plot('speed',[series('Acciones/s','tps')]);plot('path-loss',[series('Coste restante','path_loss')]);
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
