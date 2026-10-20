'use strict';
const $ = id => document.getElementById(id);
const historicalView = globalThis.location?.pathname?.endsWith('/history.html') === true;
const reportApi = historicalView ? 'api/history/' : 'api/';
const finite = x => typeof x === 'number' && Number.isFinite(x);
const fmt = (x, n=0) => finite(x) ? x.toLocaleString('en-US', {maximumFractionDigits:n}) : '—';
// Same pinned SDK palette as viewer/src/canvas.ts.
const palette = ['#ffffff','#cccccc','#999999','#666666','#333333','#000000','#e53aa3','#ff7bcc','#f93c31','#1e93ff','#88d8f1','#ffdc00','#ff851b','#921231','#4fcc30','#a356d6'];
const colors = ['#77dabb','#f4b979','#a9a6ef','#94a7aa'];
const deltaKinds = [['Invented','invented'],['Missed','missed'],['Wrong color','color'],['Correct','correct']];
const predictorLossKinds = [['Latent MSE','latent'],['Visual NLL','delta_nll'],
  ['Overlap error','delta_overlap'],['Rollout · total','rollout']];
const namespace = 'http://www.w3.org/2000/svg';
const chartStates = new Map();
function gpuStatus(sample) {
  const devices=sample?.devices||[], status=$('gpu-status');
  status.textContent=devices.length ? devices.map(gpu=>
    `GPU${devices.length>1?' '+gpu.index:''} · ${fmt(gpu.utilization_percent)} % utilization · ${fmt(gpu.temperature_c)} °C`).join('\n')
    : 'GPU · data unavailable';
  status.title=devices.length
    ? `${devices.map(gpu=>gpu.name).join(', ')}. Current server reading; includes other processes. Polled every 2 s.`
    : 'Could not read server GPU status.';
}
function element(tag, attributes={}, text='') {
  const node = document.createElementNS(namespace, tag);
  for (const [k,v] of Object.entries(attributes)) node.setAttribute(k, v);
  node.textContent=text; return node;
}
function plot(id, series, {markers=[], thresholds=[], unit='', percent=false, responsive=false, digits=2, minimumRange=1, empty='Waiting for data from this run'}={}) {
  const svg=$(id), legend=$(`${id}-legend`); svg.replaceChildren();
  if(!chartStates.has(id))chartStates.set(id,{hovered:null,focused:null,selected:null});
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
      title:last?`Último dato medido · action ${fmt(last[0])}${last[2]?' · '+last[2]:''}`:'No measurements',
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
    labels.push({key:`threshold:${ref.name}`,text:`Threshold · ${ref.name}: ${fmt(ref.value,2)}${suffix}`,
      className:`threshold-label series-${ref.colorIndex??3}`});
  }
  if(markers.length)labels.push({key:'no-change',text:'Bottom band: no real or predicted changes',className:'series-muted'});
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
        node.addEventListener('click',()=>{
          node.focus();
          state.selected=state.selected===label.key?null:label.key;
          state.hovered=state.focused=null;state.update();
        });
      }
      return node;
    }));
  }
  labels.forEach((label,i)=>{
    const node=legend.children[i];node.textContent=label.text;node.title=[label.title,label.index!==undefined&&ChartTools.hint].filter(Boolean).join(' · ');node.className=label.className;
    if(label.index!==undefined)node.disabled=!lastPoints[label.index];
    node.setAttribute('data-highlighted','false');
  });
  if(!lastPoints[state.hovered])state.hovered=null;
  if(!lastPoints[state.focused])state.focused=null;
  if(!labels.some(l=>l.key===state.selected&&lastPoints[l.index]))state.selected=null;
  labels.forEach((label,i)=>{
    if(label.index!==undefined)legend.children[i].setAttribute('aria-pressed',String(label.key===state.selected));
  });
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
    line.append(element('title',{},`Threshold · ${ref.name}: ${fmt(ref.value,2)}${suffix}`));svg.append(line);
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
      if(!s.pointsOnly)segment.push(p);
      const circle=element('circle',{cx:x(p[0]),cy:y(p[1]),r:s.pointsOnly?3:2,fill:color,opacity});
      circle.append(element('title',{},`${s.name}: ${fmt(p[1],4)} · action ${p[0]}${p[2]?' · '+p[2]:''}`));append(circle);
    }
    flush();
    if(!s.pointsOnly)for(const node of ChartTools.svgEMA(s.points,x,y,color,opacity))append(node);
  });
  if(markers.length) {
    for(const m of markers) {const mark=element('rect',{x:x(m.start),y:height-35,width:Math.max(2,x(m.end)-x(m.start)),height:5,fill:'#94a7aa'});mark.append(element('title',{},`No real or predicted changes · actions ${m.start}–${m.end}`));svg.append(mark);}
    if(!points.length)svg.append(element('text',{x:width/2,y:height/2,fill:'#94a7aa','text-anchor':'middle'},'No changes: no error percentage to measure'));
  }
  const guide=element('g',{'class':'series-guide','pointer-events':'none','aria-hidden':'true'});svg.append(guide);
  state.update=()=>{
    const selected=state.hovered??state.focused??labels.find(l=>l.key===state.selected)?.index, last=lastPoints[selected];
    layers.forEach((layer,i)=>layer.forEach(({node,opacity})=>node.setAttribute('opacity',
      !last?opacity:i===selected?1:opacity*.2)));
    labels.forEach((label,i)=>{
      legend.children[i].setAttribute('data-highlighted',String(!!last&&label.index===selected));
      if(label.index!==undefined)legend.children[i].setAttribute('aria-pressed',String(label.key===state.selected));
    });
    guide.replaceChildren();
    if(!last)return;
    const s=series[selected], color=s.muted?'#94a7aa':colors[s.colorIndex??selected%4], position=y(last[1]);
    ChartTools.svgGuide(guide,{left:58,right:width-22,y:position,color,lastX:x(last[0]),text:`${fmt(last[1],digits)}${suffix}`});
  };
  state.update();
}
function picture(id, frame, mask, pixelColors=palette) {
  const canvas=typeof id==='string'?$(id):id, ctx=canvas.getContext('2d');
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
    $('prediction-sample-info').textContent='Waiting for a prediction and its actual outcome…';
    $('prediction-sample-error').textContent='';return;
  }
  deltaKinds.forEach(([name],i)=>{
    const span=document.createElement('span');span.className=`series-${i}`;
    span.textContent=`${name}: ${fmt(counts[i])} px`;legend.append(span);
  });
  const command=sample.action, coordinates=command.id===6?` (${command.data.x}, ${command.data.y})`:'';
  const age=Math.max(0,(Date.now()/1000)-sample.captured_at);
  $('prediction-sample-info').textContent=`${sample.game} · level ${fmt(sample.level)} · episode ${sample.episode_seed} · step ${fmt(sample.episode_action)} · ACTION${command.id}${coordinates} · ${sample.phase==='training'?'Training':'Evaluation'} · training action ${fmt(sample.training_action)} · ${fmt(age)} s ago`;
  $('prediction-sample-error').textContent=sample.union>0
    ? `Error in this sample: ${fmt(100*sample.wrong/sample.union,2)} % · ${fmt(sample.wrong)} / ${fmt(sample.union)} pixels changed in either delta.`
    : 'No real or predicted changes in this action.';
}
const trajectoryPlayer={sample:null,frames:[],states:[],steps:[],extent:0,position:0,playing:false,timer:null,fetching:false,matrixId:null,cells:[]};
function stopTrajectory(){
  trajectoryPlayer.playing=false;
  clearTimeout(trajectoryPlayer.timer);
}
function frameImage(frame){
  return frame?Array.from({length:frame.height},(_,y)=>Array.from({length:frame.width},(_,x)=>frame.pixels.charCodeAt(frame.offset+y*frame.width+x))):null;
}
function actionLabel(action){
  if(!action)return 'Start';
  // Official input controls (https://docs.arcprize.org/actions), not inferred game effects.
  const names={0:'Reset',1:'↑',2:'↓',3:'←',4:'→',5:'Space',6:'Click',7:'↶ Deshacer'};
  return `A${action.id} ${names[action.id]||''}${action.id===6?` (${action.data.x}, ${action.data.y})`:''}`.trim();
}
function sameAction(a,b){
  return !!a&&!!b&&a.id===b.id&&Object.keys(a.data||{}).length===Object.keys(b.data||{}).length
    &&Object.entries(a.data||{}).every(([k,v])=>b.data?.[k]===v);
}
function renderObservationContext(){
  const player=trajectoryPlayer, sample=player.sample, frame=player.frames[player.position], p=displayedReport?.progress||{};
  if(!sample){
    picture('observation',p.observation);picture('goal',p.goal_frame,p.goal_mask);
    $('observation-context').textContent='Current interaction';
    $('observation-detail').textContent=p.game?`${p.game} / level ${fmt(p.level)}`:'Waiting for an observation…';
    $('goal-context').textContent=p.goal||'Current goal';
    const curriculum=displayedReport?.curriculum;
    $('goal-detail').textContent=curriculum?.ordered_stages?.[curriculum.index]?.objective==='world_model'
      ?'Stage without a subgoal teacher: dynamics are learned from random actions.'
      :'Teacher-provided image. Checkerboard: area excluded from the goal.';
    return;
  }
  picture('observation',frameImage(frame));picture('goal',frameImage(frame.goalImage));
  $('observation-context').textContent=`Selected sample · t = ${frame.step}`;
  $('observation-detail').textContent=`${sample.game} / level ${sample.level} · episode ${sample.episode_seed} · ${sample.phase==='training'?'Training':'Evaluation'} · frame ${player.position+1}/${player.frames.length}`;
  $('goal-context').textContent=frame.goal||'Sample subgoal';
  $('goal-detail').textContent=frame.goalImage?'Goal for this action. Checkerboard: area excluded from the goal.':'This older sample does not include a subgoal image.';
}
function buildTrajectoryMatrix(){
  const player=trajectoryPlayer, table=$('trajectory-matrix');
  table.replaceChildren();player.cells=[];
  if(!player.sample){player.matrixId=null;return;}
  const make=(tag,className='',text='')=>{const node=document.createElement(tag);node.className=className;node.textContent=text;return node;};
  const track=(node,t)=>{player.cells.push({node,t});return node;};
  const thumb=(frame,label)=>{const canvas=make('canvas','trajectory-thumb');canvas.setAttribute('role','img');canvas.setAttribute('aria-label',label);picture(canvas,frameImage(frame));return canvas;};
  const caption=make('caption','sr-only','Time advances from left to right. Each imagined row starts from a real state; columns align the destination time.');
  const cols=make('colgroup');cols.append(make('col','trajectory-label-column'));
  const head=make('thead'), headRow=make('tr');headRow.append(make('th','trajectory-row-label','Time'));
  for(let t=0;t<=player.extent;t++){
    cols.append(track(make('col'),t));
    const cell=track(make('th','trajectory-time',`t = ${t}`),t);cell.setAttribute('scope','col');headRow.append(cell);
  }
  head.append(headRow);table.append(caption,cols,head);
  const body=make('tbody'), real=make('tr','trajectory-real');
  const realLabel=make('th','trajectory-row-label','REAL');realLabel.setAttribute('scope','row');real.append(realLabel);
  for(let t=0;t<=player.extent;t++){
    const cell=track(make('td','trajectory-cell'),t), frame=player.states[t];
    cell.setAttribute('data-time',t);
    if(frame){
      const button=make('button','trajectory-state');button.type='button';button.setAttribute('aria-label',`Ver state real t = ${t} · ${actionLabel(frame.action)}`);
      button.append(thumb(frame,`Status real t = ${t}`),make('span','action-badge',actionLabel(frame.action)));
      if(frame.source==='exploration')button.append(make('small','trajectory-source','Exploration'));
      button.addEventListener('click',()=>seekTrajectory(frame.frameIndex));track(button,t);cell.append(button);
    }else cell.append(make('span','trajectory-absent','Not observed'));
    real.append(cell);
  }
  body.append(real);
  const separator=make('tr','trajectory-section'), section=make('th','','IMAGINATION · one row per origin');
  section.colSpan=player.extent+2;separator.append(section);body.append(separator);
  const spacer=count=>{const node=make('td','trajectory-spacer');node.colSpan=count;return node;};
  player.steps.forEach((step,origin)=>{
    const row=make('tr','trajectory-imagined');row.setAttribute('data-origin',origin);
    const label=make('th','trajectory-row-label',`From t = ${origin}`);label.setAttribute('scope','row');row.append(label);
    const routeLabel={selected_route:'Selected route',searched_continuation:'Explored continuation',searched_alternative:'Explored alternative'}[step.imagination_source];
    if(routeLabel)label.append(make('small','trajectory-route-kind',routeLabel));
    for(const [name,estimate] of [['Choice',step.selected_estimate],['Preview',step.forecast_estimate]]){
      if(estimate&&finite(estimate.success_probability)&&estimate.success_probability>=0&&estimate.success_probability<=1
          &&finite(estimate.successful_actions)&&estimate.successful_actions>=0&&finite(estimate.remaining_budget)){
        const summary=`${name} · ${fmt(100*estimate.success_probability)} % · ${fmt(estimate.successful_actions)} actions`;
        const detail=make('small','trajectory-route-kind',summary);
        detail.title=`Success probability and duration if successful. Estimated at decision time, with ${estimate.remaining_budget} actions available; includes the imagined prefix and remaining duration. This is not an observed outcome.`;
        label.append(detail);
      }
    }
    if(origin)row.append(spacer(origin));
    const originCell=track(make('td','trajectory-cell trajectory-origin'),origin), button=make('button','trajectory-origin-button','Source');
    button.type='button';button.setAttribute('aria-label',`Ver origen t = ${origin}`);
    button.addEventListener('click',()=>seekTrajectory(player.states[origin].frameIndex));originCell.append(button);row.append(originCell);
    let diverged=false;
    step.imagination.forEach((prediction,index)=>{
      const destination=origin+index+1, actual=player.sample.steps[origin+index]?.action;
      if(actual&&!sameAction(prediction.action,actual))diverged=true;
      const cell=track(make('td','trajectory-cell trajectory-prediction'),destination);
      cell.setAttribute('data-origin',origin);cell.setAttribute('data-time',destination);
      cell.setAttribute('data-diverged',String(diverged));
      const label=`Prediction of t = ${destination} from t = ${origin} · ${actionLabel(prediction.action)}`;
      cell.append(thumb(prediction.image,label),make('span','action-badge',`+${index+1} · ${actionLabel(prediction.action)}`));
      if(diverged)cell.append(make('small','trajectory-route','Different route'));
      else if(!actual)cell.append(make('small','trajectory-source','No real outcome'));
      cell.title=label+(diverged?' · The imagined sequence differs from the executed sequence':'');row.append(cell);
    });
    const remaining=player.extent-origin-step.imagination.length;
    if(remaining){
      const empty=spacer(remaining);
      if(!step.imagination.length){
        empty.className+=' trajectory-missing';
        empty.textContent=step.imagination_status==='exploration'?'Random exploration · no imagined route'
          :player.sample.format_version===2?'No selected route':'Older sample · no saved predictions';
      }
      row.append(empty);
    }
    body.append(row);
  });
  table.append(body);player.matrixId=player.sample.id;
}
function renderTrajectory(){
  const player=trajectoryPlayer, sample=player.sample, frame=player.frames[player.position];
  $('trajectory-viewer').hidden=!sample;
  $('trajectory-mode').textContent=$('trajectory-live').checked?'Auto-refresh':'Sample pinned';
  $('trajectory-mode').setAttribute('data-pinned',String(!$('trajectory-live').checked));
  renderObservationContext();
  if(!sample){if(player.matrixId)buildTrajectoryMatrix();return;}
  if(player.matrixId!==sample.id)buildTrajectoryMatrix();
  for(const {node,t} of player.cells){node.setAttribute('data-selected',String(t===frame.step));if(node.className==='trajectory-state')node.setAttribute('aria-pressed',String(t===frame.step));}
  const reasons={success:'Level solved',game_over:'Game over',budget:'Action limit',interrupted:'Interrupted',unreachable_subgoal:'Unreachable goal'};
  $('trajectory-info').textContent=`${sample.game} / level ${sample.level} · episode ${sample.episode_seed} · ${sample.phase==='training'?'Training':'Evaluation'} · ${sample.actions} actions · ${reasons[sample.reason]||sample.reason} · captured at action ${fmt(sample.training_action)}`;
  $('trajectory-play').textContent=player.playing?'Pause animation':'Play animation';
  $('trajectory-prev').disabled=player.position===0;
  $('trajectory-next').disabled=player.position===player.frames.length-1;
  $('trajectory-position').max=String(player.frames.length-1);
  $('trajectory-position').value=String(player.position);
  const command=frame.action;
  const action=command?`Action ${frame.step}/${sample.actions} · ACTION${command.id}${command.id===6?` (${command.data.x}, ${command.data.y})`:''} · ${frame.source==='planner'?'Planner':'Random exploration'}`:'Initial state';
  $('trajectory-step').textContent=`${action} · frame ${player.position+1}/${player.frames.length}${frame.goal?' · '+frame.goal:''}`;
  $('trajectory-position').setAttribute('aria-valuetext',$('trajectory-step').textContent);
}
function unpackTrajectory(sample){
  if(!Number.isInteger(sample.actions)||sample.actions<1||sample.actions>128||!Array.isArray(sample.steps)||sample.steps.length!==sample.actions)throw Error('Incomplete trajectory');
  const frames=[], states=[], steps=[];let bytes=0, extent=sample.actions;
  const unpack=(packed,maxColor=15)=>{
    if(!packed||!Array.isArray(packed.shape)||packed.shape.length!==3)throw Error('Incomplete frames');
    const [count,height,width]=packed.shape;
    if(![count,height,width].every(n=>Number.isInteger(n)&&n>0)||count*height*width>4*1024*1024)throw Error('Invalid dimensions');
    const pixels=atob(packed.pixels);bytes+=pixels.length;
    if(pixels.length!==count*height*width||bytes>4*1024*1024)throw Error('Frames incomplete or too large');
    for(let i=0;i<pixels.length;i++)if(pixels.charCodeAt(i)>maxColor)throw Error('Invalid color');
    return Array.from({length:count},(_,i)=>({width,height,pixels,offset:i*height*width}));
  };
  const single=(packed,maxColor=15)=>{const result=unpack(packed,maxColor);if(result.length!==1)throw Error('Expected one final state');return result[0];};
  const validAction=action=>action&&Number.isInteger(action.id)&&action.id>=0&&action.id<=7&&action.data&&typeof action.data==='object';
  for(const [origin,step] of sample.steps.entries()){
    if(!validAction(step.action))throw Error('Invalid action');
    const predictions=step.imagination??[];
    if(!Array.isArray(predictions)||predictions.length>3)throw Error('Invalid imagined route');
    const imagination=predictions.map(p=>{if(!validAction(p.action))throw Error('Invalid imagined action');return {...p,image:single(p.frames)};});
    if(step.source==='exploration'&&imagination.length)throw Error('Exploration has no selected route');
    steps.push({...step,goalImage:step.goal_frame?single(step.goal_frame,16):null,imagination});
    extent=Math.max(extent,origin+imagination.length);
  }
  for(const [t,entry] of [{frames:sample.initial,goal:steps[0].goal,goalImage:steps[0].goalImage},...steps].entries()){
    for(const frame of unpack(entry.frames))frames.push({...entry,...frame,step:t,frameIndex:frames.length});
    states.push(frames.at(-1));
  }
  if(frames.length!==sample.frame_count)throw Error('Incomplete trajectory');
  return {frames,states,steps,extent};
}
async function refreshTrajectory(metadata){
  const player=trajectoryPlayer;
  $('trajectory-warning').textContent=displayedReport?.trajectory_error||'';
  if(!metadata){
    stopTrajectory();player.sample=null;player.frames=[];player.states=[];player.steps=[];renderTrajectory();
    $('trajectory-info').textContent=displayedReport?.trajectory_error||'Waiting for a completed trajectory with the planner active…';
    return;
  }
  if($('path').hidden||!$('trajectory-live').checked||player.fetching||player.sample?.id===metadata.id)return;
  player.fetching=true;
  try{
    const response=await fetch(reportApi+'trajectory',{cache:'no-store',signal:AbortSignal.timeout(10000)});
    if(!response.ok)throw Error('HTTP '+response.status);
    const sample=(await response.json()).trajectory;
    if(displayedReport?.planner_trajectory?.id!==metadata.id||!$('trajectory-live').checked)return;
    if(!sample||sample.id!==metadata.id){$('trajectory-info').textContent='Waiting for the complete new sample…';return;}
    const decoded=unpackTrajectory(sample);
    stopTrajectory();Object.assign(player,decoded,{sample,position:0});renderTrajectory();
  }catch(error){$('trajectory-info').textContent='Could not load trajectory: '+error.message;}
  finally{player.fetching=false;}
}
function trajectoryTick(){
  const player=trajectoryPlayer;
  if(!player.playing)return;
  if($('path').hidden||player.position>=player.frames.length-1){stopTrajectory();renderTrajectory();return;}
  player.position++;renderTrajectory();
  player.timer=setTimeout(trajectoryTick,250);
}
$('trajectory-live').checked=true;
$('trajectory-live').addEventListener('change',()=>{
  stopTrajectory();renderTrajectory();refreshTrajectory(displayedReport?.planner_trajectory);
});
$('trajectory-play').addEventListener('click',()=>{
  if(!trajectoryPlayer.sample)return;
  $('trajectory-live').checked=false;
  if(trajectoryPlayer.playing)stopTrajectory();
  else{
    if(trajectoryPlayer.position===trajectoryPlayer.frames.length-1)trajectoryPlayer.position=0;
    trajectoryPlayer.playing=true;trajectoryPlayer.timer=setTimeout(trajectoryTick,250);
  }
  renderTrajectory();
});
function seekTrajectory(position){
  if(!trajectoryPlayer.sample)return;
  $('trajectory-live').checked=false;stopTrajectory();
  trajectoryPlayer.position=Math.max(0,Math.min(trajectoryPlayer.frames.length-1,position));renderTrajectory();
}
$('trajectory-prev').addEventListener('click',()=>seekTrajectory(trajectoryPlayer.position-1));
$('trajectory-next').addEventListener('click',()=>seekTrajectory(trajectoryPlayer.position+1));
$('trajectory-position').addEventListener('input',()=>seekTrajectory(Number($('trajectory-position').value)));
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
function renderGoalOutcomes(goals){
  // Finished training segments have a boolean outcome and a terminal reward.
  // Missing/censored records must not silently become failures.
  const completed=goals.filter(g=>finite(g.x)&&typeof g.success==='boolean'&&finite(g.reward)).sort((a,b)=>a.x-b.x);
  const outcomes=completed.map(g=>[g.x,g.success?100:0,
    `${g.game||'Unknown game'} / level ${fmt(g.level)} · ${g.goal||'Subgoal'} · ${fmt(g.actions)} actions · minimum ${fmt(g.minimum_actions)}`]);
  let start=0,end=0,wins=0;
  const rates=[...new Set(completed.map(g=>g.x))].map(x=>{
    while(end<completed.length&&completed[end].x<=x){wins+=Number(completed[end].success);end++;}
    while(start<end&&completed[start].x<=x-512){wins-=Number(completed[start].success);start++;}
    const total=end-start;
    return [x,100*wins/total,`${fmt(wins)} ${wins===1?'success':'successes'} · ${fmt(total-wins)} ${total-wins===1?'failure':'failures'} · ${fmt(total)} ${total===1?'completed subgoal':'completed subgoals'}`];
  });
  plot('subgoal-outcomes',[
    {name:'Success',colorIndex:0,legend:false,pointsOnly:true,points:outcomes.filter(p=>p[1]===100)},
    {name:'Failure',colorIndex:1,legend:false,pointsOnly:true,points:outcomes.filter(p=>p[1]===0)},
    {name:'Success · up to 512 actions',colorIndex:2,points:rates}
  ],{percent:true,empty:'No completed subgoal results'});
  const last=rates.at(-1);
  $('subgoal-outcomes-summary').textContent=last
    ?`Window at action ${fmt(last[0])}: ${last[2]}. Each subgoal counts once.`
    :'No results recorded. Incomplete attempts do not count as failures.';
}
function lossBreakdown(id, rows, key) {
  const target=$(id);target.replaceChildren();
  const row=rows.findLast(r=>r[key]&&Object.values(r[key]).some(finite));
  if(!row){target.textContent='No update recorded';return;}
  const labels={color_ce:'Color',delta_nll:'Visual delta',delta_overlap:'Delta union',
    local_reconstruction:'Local reconstruction',sigreg:'SIGReg',variance_floor:'Variance',
    latent:'Latent delta',rollout:'Chained prediction'};
  Object.entries(row[key]).filter(([,v])=>finite(v)).forEach(([name,value],i)=>{
    const span=document.createElement('span');span.className=`series-${i%4}`;
    span.textContent=`${labels[name]||name}: ${fmt(value,4)}`;
    span.title=`Contribution to total, with weight applied · action ${fmt(row.actions)}`;
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
    if(row.split!=='dev'||!['random','quality_probe'].includes(row.policy)||row.fixed_baseline||!finite(row.actions))continue;
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
    points.push({actions,stage:`${rows[0].stage}:${rows.map(pair).sort().join(',')}`,error:100*rows.reduce((s,r)=>s+r.delta_error,0)/rows.length,
      reconstruction:mean('reconstruction_error'),rollout:mean('rollout_error'),drift:mean('latent_drift'),
      latentStd:latentMean('std_mean'),latentWithin:latentMean('within_episode_std'),
      latentInactive:latentMean('inactive_fraction',100),latentRank:latentMean('effective_rank'),
      latentMaxRank:latentMean('max_rank'),latentChanged:latentMean('changed_rms'),latentUnchanged:latentMean('unchanged_rms'),
      latentSamples:latentMean('sample_count',rows.length),latentEpisodes:latentMean('episode_count',rows.length),
      pairs:rows.length,episodes:rows.reduce((s,r)=>s+r.episodes,0)});
  }
  return {points,expected:expected.size,pending};
}
let successPairSelection=null, successOptionsKey='';
function renderPlannerSuccess(report) {
  const pair=r=>JSON.stringify([r.game,r.level]);
  const rows=(report.evaluations||[]).filter(r=>r.split==='dev'&&['path_mpc','random'].includes(r.policy)
    &&typeof r.game==='string'&&r.game&&Number.isInteger(r.level)&&r.level>0&&finite(r.actions)&&r.actions>=0);
  const pairs=[...new Map(rows.map(r=>[pair(r),r])).values()]
    .sort((a,b)=>a.game.localeCompare(b.game)||a.level-b.level);
  const keys=pairs.map(pair), selector=$('planner-success-level');
  if(!keys.includes(successPairSelection))successPairSelection=null;
  const latest=report.curriculum?.unlocked?.[report.config?.game];
  const current=pair(!report.config?.training_plan&&report.config?.evaluation_protocol==='current_level_cached_v1'&&latest
    ?{game:report.config.game,level:latest}:report.progress||{});
  const selected=successPairSelection||(keys.includes(current)?current:keys[0]);
  const signature=JSON.stringify(keys);
  if(signature!==successOptionsKey) {
    successOptionsKey=signature;
    selector.replaceChildren(...pairs.map(r=>{const option=document.createElement('option');
      option.value=pair(r);option.textContent=`${r.game} / level ${r.level}`;return option;}));
  }
  selector.value=selected||'';selector.disabled=pairs.length<2;
  const byAction=new Map();
  for(const row of rows.filter(r=>pair(r)===selected)) {
    if(!byAction.has(row.actions))byAction.set(row.actions,new Map());
    byAction.get(row.actions).set(row.policy,row); // A repeated sweep replaces its earlier measurement.
  }
  const valid=r=>r&&Number.isInteger(r.episodes)&&r.episodes>0&&Number.isInteger(r.successes)
    &&r.successes>=0&&r.successes<=r.episodes;
  const baseline=rows.find(r=>pair(r)===selected&&r.policy==='random'&&r.fixed_baseline&&valid(r));
  let sweeps=[...byAction].sort((a,b)=>a[0]-b[0]);
  const firstPlanner=sweeps.findIndex(([,policies])=>valid(policies.get('path_mpc')));
  // Earlier random-only evaluations would compress the planner's history to a sliver.
  if(firstPlanner>=0)sweeps=sweeps.slice(firstPlanner);
  const series=[['Planner','path_mpc'],[baseline?'Random · fixed baseline':'Random','random']].map(([name,policy])=>({name,
    points:sweeps.map(([actions,policies])=>{const r=policy==='random'&&baseline?baseline:policies.get(policy);
      return [actions,valid(r)?100*r.successes/r.episodes:null,valid(r)?`${r.successes}/${r.episodes} wins`:''];})}));
  const success=report.curriculum?.thresholds?.success??report.config?.success;
  plot('planner-success',series,{percent:true,responsive:true,
    thresholds:finite(success)&&success>=0&&success<=1?[{name:'Curriculum success',value:100*success,colorIndex:2}]:[],
    empty:'Waiting for success evaluations for this level'});
  $('planner-success-summary').textContent=series.map((s,i)=>{const last=s.points.findLast(p=>finite(p[1]));
    return last?`${s.name}: ${last[2]} · ${i===1&&baseline?'measured once at action':'action'} ${fmt(i===1&&baseline?baseline.actions:last[0])}`:`${s.name}: no evaluation available`;}).join(' · ');
}
$('planner-success-level').addEventListener('change',()=>{
  successPairSelection=$('planner-success-level').value;
  if(displayedReport)renderPlannerSuccess(displayedReport);
});
let displayedReport;
let curriculumCells=[];
function render(report) {
  displayedReport=report;
  $('training-nav').hidden=!historicalView;
  $('training-nav').setAttribute('aria-current',historicalView?'false':'page');
  const p=report.progress||{}, c=report.config||{};
  const representationFrozen=p.representation_frozen===true;
  const encoderFrozen=representationFrozen||(p.encoder_frozen??c.freeze_encoder)===true;
  const phases={loading:'Loading weights',warmup:'Preparing torch.compile and the first action',training:'Training',evaluation:'Evaluating with frozen weights',checkpoint:'Saving weights',complete:'Finished',converged:'Curriculum passed',budget_exhausted:'Budget exhausted · curriculum pending',failed:'Stopped due to an error'};
  $('connection').textContent=phases[p.phase]||'Waiting for training';
  $('connection').className=report.status==='failed'||report.age_seconds>60?'warn':'';
  $('freshness').textContent=`Last write ${fmt(report.age_seconds)} s ago · polling every 2 s`;
  const initialization=c.initialization==='random'?' · Initialized from scratch':c.initialization==='checkpoint'?' · Pretrained weights':'';
  $('run-info').textContent=`${p.game||c.game||'One configuration'} · seed ${fmt(c.seed)} · ${c.training_plan?'configured curriculum':'cumulative curriculum'}${initialization}`;
  const curriculum=report.curriculum, limits=curriculum?.thresholds||{};
  const pct=value=>finite(value)?`${fmt(value*100,2)} %`:'—';
  $('curriculum-panel').hidden=!curriculum;
  $('curriculum-summary').textContent=curriculum?`Stage ${fmt(curriculum.stage)}${curriculum.ordered_stages?` / ${curriculum.ordered_stages.length}`:''} · `+Object.entries(curriculum.unlocked).map(([g,n])=>`${g}: ${curriculum.ordered_stages?'level':'levels 1–'}${n}`).join(' · '):'';
  const control=report.manual_controls?.at(-1);
  const results={advanced:'applied',already_planner:'already in planner phase',last_level:'final level already unlocked',budget_exhausted:'budget exhausted'};
  $('manual-controls').textContent=control
    ? `Last manual request: ${control.operation==='phase'?'phase':'level'} · ${control.game} / ${control.level} · ${control.phase} · ${results[control.result]||control.result} · action ${fmt(control.actions)}. This does not count as a passed evaluation.`
    : 'No manual advances recorded.';
  const worldOnly=curriculum?.ordered_stages?.[curriculum.index]?.objective==='world_model';
  $('thresholds').textContent=curriculum?`Activation after ${limits.patience} consecutive evaluations: reconstruction ≤ ${pct(limits.representation_open)}, latent drift ≤ ${pct(limits.latent_drift)}, 1- and 3-step prediction ≤ ${pct(limits.prediction_open)}. ${worldOnly?'World-model stage: prediction quality; no subgoal teacher or planner training.':`Advancement also requires success ≥ ${pct(limits.success)} on the active evaluation level.`} Change coverage is required; if no pixel changes, a native termination must be observed.`:'';
  const focusedControl=[phaseButton,levelButton].includes(document.activeElement)?document.activeElement:null;
  curriculumCells=[];
  $('curriculum-rows').replaceChildren(...(curriculum?.levels||[]).map(row=>{
    const m=row.measurements||{}, tr=document.createElement('tr'), rowLimits=row.thresholds||limits;
    const next=!row.unlocked?'Level pending':row.objective==='world_model'&&row.predictor?'Validating predictor':row.planner?'Planner active':row.predictor?`Planner: ${fmt(row.prediction_passes)} / ${rowLimits.patience}`:`Predictor: ${fmt(row.representation_passes)} / ${rowLimits.patience}`;
    const active=!row.unlocked?'Pending':c.freeze_representation_after_planner&&row.planner
      ? 'Predictor + planner · representation frozen'
      : row.planner?'Representation + predictor + planner':row.predictor?'Representation + predictor':'Representation';
    for(const value of [`${row.game} / ${row.level}`,active,pct(m.reconstruction_error),pct(m.latent_drift),pct(m.prediction_error),pct(m.rollout_error),pct(m.success),next,row.unlocked?`${Math.min(row.promotion_passes,rowLimits.patience)} / ${rowLimits.patience}`:'—']){const td=document.createElement('td');td.textContent=value;tr.append(td);}
    curriculumCells.push({game:row.game,level:row.level,unlocked:row.unlocked,levelCell:tr.children[0],phaseCell:tr.children[1]});
    return tr;
  }));
  renderControls(controlState);
  if(focusedControl&&!focusedControl.disabled)focusedControl.focus({preventScroll:true});
  $('representation-updates').textContent=fmt(p.representation_updates);
  $('representation-update-scope').textContent=representationFrozen ? 'Encoder and decoder frozen'
    : encoderFrozen
    ? 'Decoder · encoder frozen' : 'Encoder and decoder · each real action';
  $('world-training-scope').textContent=(representationFrozen
    ? 'Encoder and decoder are frozen on this level with the planner. They resume learning on new levels.'
    : encoderFrozen
    ? 'Encoder frozen: latents stay fixed for the same observation. The decoder keeps learning.'
    : 'Encoder and decoder learn from the start.')
    +' The predictor starts when representation is ready. Evaluations do not update weights.';
  const rates=p.representation_learning_rates;
  if(rates) $('world-training-scope').textContent+=` LR encoder: ${finite(rates.encoder)?rates.encoder.toExponential():'—'} · decoder: ${finite(rates.decoder)?rates.decoder.toExponential():'—'}.`;
  $('notice').hidden=!report.error;$('notice').textContent=report.error||'';
  $('actions').textContent=fmt(p.actions);$('budget').textContent=`Of ${fmt(p.total_actions)} · evaluation excluded`;
  $('world-updates').textContent=fmt(p.predictor_updates);$('path-updates').textContent=fmt(p.path_updates);
  $('gate').textContent=p.predictor_enabled
    ? representationFrozen?'Active · representation frozen':'Active · each action alongside representation'
    : 'Waiting for stable representation';
  const replay=p.predictor_replay;
  $('predictor-replay').textContent=replay
    ? `Replay: ${fmt(replay.episodes)}/${fmt(replay.capacity)} episodes · ${fmt(replay.transitions)} transitions · lote ${fmt(replay.actual_batch_size)}/${fmt(replay.batch_size)}`
    : c.predictor_replay_episodes>0 ? `Replay: up to ${fmt(c.predictor_replay_episodes)} episodes · target batch ${fmt(c.predictor_batch_size)}`
    : 'No replay · one transition per update';
  predictionSample(report.prediction_sample);
  $('tps').textContent=fmt(p.recent_tps,2);
  $('progress').textContent=p.phase==='evaluation'
    ? `${p.policy} · ${p.evaluation_game} / level ${fmt(p.evaluation_level)} · episode ${fmt(p.evaluation_episode)}/${fmt(p.evaluation_episodes)} · action ${fmt(p.evaluation_action)}`
    : `${p.game||c.game||'Preparing game'} / level ${fmt(p.level)} · subgoal ${fmt(p.goal_index)}/${fmt(p.goals)} · ${p.goal||''} · ${fmt(p.goal_actions)} actions (initial minimum ${fmt(p.minimum_actions)}) · exploration ${fmt(p.epsilon*100)} %`;
  if(historicalView){
    $('connection').textContent=report.status==='waiting'?'History unavailable':'History · read only';
    $('connection').className='';
    $('notice').hidden=false;
    $('notice').textContent=report.message||'Previous run: these curves and trajectories do not belong to the current sequence training run.';
    $('freshness').textContent=finite(report.updated_at)?`Last record: ${new Date(report.updated_at*1000).toLocaleString('en-US')}`:'No record timestamp';
    $('gpu-status').textContent='';
  }
  const fixed=fixedEvaluations(report), last=fixed.points.at(-1);
  const qualityPoints=key=>{let stage=null;const points=[];for(const r of fixed.points){if(stage!==null&&r.stage!==stage)points.push([r.actions,null]);points.push([r.actions,r[key],`${r.pairs} pairs · ${r.episodes} episodes`]);stage=r.stage;}return points;};
  const qualityThresholds=[{name:'Reconstruction',value:100*limits.representation_open,colorIndex:1},
    {name:'Prediction',value:100*limits.prediction_open,colorIndex:0},
    {name:'Drift',value:100*limits.latent_drift,colorIndex:3}];
  // Relative latent drift can exceed 100%; allow this chart to expand its axis.
  plot('fixed-delta',[{name:'Prediction · 1 step',points:qualityPoints('error')},
    {name:'Reconstruction',points:qualityPoints('reconstruction')},
    {name:'Prediction · 3 steps',points:qualityPoints('rollout')},
    {name:'Latent drift',points:qualityPoints('drift')}],{unit:'%',thresholds:qualityThresholds,percent:!fixed.points.some(r=>r.drift>100),responsive:true,empty:'Waiting for a complete fixed evaluation'});
  $('quality-cadence').textContent=finite(c.evaluate_every)
    ? `Quality, drift and activations every ${fmt(c.evaluate_every)} actions · at episode end.`+
      (c.evaluation_protocol==='current_level_cached_v1'?' Only the latest unlocked level · saved real trajectories; random episodes are not replayed.':''):'';
  $('fixed-summary').textContent=(last?`Latest measurement: action ${fmt(last.actions)} · ${fmt(last.pairs)} pairs · ${fmt(last.episodes)} episodes.`:'No complete fixed evaluation yet.')+
    (fixed.pending&&(!last||fixed.pending.actions>last.actions)?` In progress: ${fmt(fixed.pending.pairs)}/${fmt(fixed.pending.expected??fixed.expected)} pairs; the point will be added when complete.`:'');
  const latentPoints=key=>{
    const points=qualityPoints(key), first=points.findIndex(p=>finite(p[1]));
    return first<0?[]:points.slice(first); // Older reports have no latent diagnostics.
  };
  const latentLast=fixed.points.findLast(r=>r.latentSamples>0);
  $('latent-summary').textContent=latentLast
    ?`Latest measurement: action ${fmt(latentLast.actions)} · ${fmt(latentLast.latentSamples)} observations from ${fmt(latentLast.latentEpisodes)} episodes. Curves give each level equal weight; good reconstruction or high diversity do not guarantee good prediction.`
    :'Measurements will appear after a fixed evaluation with the new diagnostics. Historical gaps are not filled with zeros.';
  plot('latent-diversity',[{name:'Across observations',points:latentPoints('latentStd')},
    {name:'Within each episode',points:latentPoints('latentWithin')}],{digits:4,minimumRange:.0001});
  plot('latent-rank',[{name:'Effective rank',points:latentPoints('latentRank')},
    {name:'Observable maximum',muted:true,points:latentPoints('latentMaxRank')}]);
  plot('latent-inactive',[{name:'Nearly constant',points:latentPoints('latentInactive')}],{percent:true});
  plot('latent-sensitivity',[{name:'Changed blocks',points:latentPoints('latentChanged')},
    {name:'Unchanged blocks',points:latentPoints('latentUnchanged')}],{digits:4,minimumRange:.0001});
  const rows=report.curves||[], b=blocks(rows), t=trend(rows,b), series=(name,key)=>({name,points:rows.map(r=>[r.actions,r[key]])});
  const measured=rows.filter(r=>r.counts), unchanged=measured.filter(r=>r.counts.union===0).length;
  $('interaction-summary').textContent=rows.length?`Recent window: actions ${fmt(rows[0].actions)}–${fmt(rows.at(-1).actions)}. ${fmt(unchanged)} of ${fmt(measured.length)} measured actions without real or predicted changes.`:'Waiting for real actions from this run.';
  plot('delta',[{name:'Moving average · up to 512 actions',points:t.map(r=>[r.x,r.error])},
    {name:'Block of 32 actions',muted:true,points:b.map(r=>[r.x,r.error])}],
    {percent:true,markers:b.filter(r=>r.noChange).map(r=>({start:r.start,end:r.x}))});
  plot('changes',[
    ...deltaKinds.map(([name,key],colorIndex)=>({name,colorIndex,faint:true,legend:false,
      points:b.map(r=>[r.x,r.rates[key],'px/action · block of 32'])})),
    ...deltaKinds.map(([name,key],colorIndex)=>({name,colorIndex,
      points:t.map(r=>[r.x,r.rates[key],'px/action · moving average of up to 512 actions'])}))
  ],{unit:'px/action'});
  const objectiveStart=report.objective_changes?.at(-1)?.actions;
  const lossRows=finite(objectiveStart)?rows.filter(r=>r.actions>objectiveStart):rows;
  const lossTrend=trend(lossRows,blocks(lossRows));
  plot('loss',[{name:'Update average · up to 512 actions',points:lossTrend.map(r=>[r.x,r.loss])},
    {name:'Individual update',points:lossRows.map(r=>[r.actions,r.predictor_loss]),muted:true}]);
  plot('representation-loss',[{name:'Update average · up to 512 actions',points:t.map(r=>[r.x,r.representationLoss])},
    {...series('Individual update','representation_loss'),muted:true}]);
  lossBreakdown('representation-loss-terms',rows,'representation_terms');
  lossBreakdown('loss-terms',lossRows,'predictor_terms');
  const lossTerms=predictorLossKinds.map(([name,key],colorIndex)=>({name,colorIndex,
    points:lossRows.map(r=>[r.actions,r.predictor_terms?.[key]])}));
  plot('predictor-components',[
    ...lossTerms.map(s=>({...s,faint:true,legend:false})),
    ...lossTerms.map(s=>({...s,points:movingAverage(s.points)}))
  ],{digits:4,minimumRange:.001,empty:'No loss components recorded yet'});
  const lossMode=report.config?.prediction_delta_loss;
  $('predictor-objective').textContent=lossMode==='recall_focal_capped'
    ? 'Active NLL: 2× color error on real changes + focal penalty on static pixels (γ=2, T=1). Divided by max(32, real changes) to limit the weight of steps with few changes. No argmax or overlap; also applied to the 3-step rollout.'
    : lossMode==='recall_focal'
    ? 'Active NLL: 2× color error on real changes + focal penalty on static pixels (γ=2, T=1). Normalized by real changes. Penalizes omissions, wrong colors and invented changes; no argmax or overlap.'
    : lossMode==='focal'
    ? 'Active NLL: real changes and a soft focal penalty on static pixels (γ=2, T=1). Normalized by real changes, without an argmax mask.'
    : lossMode==='union'
    ? 'Active NLL: union of real and predicted changes. Correctly preserved background is excluded from this NLL.'
    : lossMode==='balanced' ? 'Active NLL: balanced groups of changed and static pixels.'
    : 'This report does not identify the NLL mode.';
  $('representation-regularization').textContent=representationFrozen
    ? 'Encoder and decoder frozen · reconstruction paused'
    : encoderFrozen
    ? 'Encoder frozen · encoder regularization disabled · decoder keeps learning'
    : p.sigreg_enabled===true
    ? 'SIGReg active · initial representation phase'
    : p.sigreg_enabled===false ? 'SIGReg disabled · reconstruction continues' : 'SIGReg status not recorded';
  plot('speed',[series('Actions/s · last 60 s','recent_tps')],{unit:'actions/s'});
  const successFirst=report.config?.path_objective_version>=2;
  const margin=report.config?.path_success_margin;
  $('path-objective').textContent=successFirst
    ? `Prioritize estimated success within budget; among routes within ${fmt(100*(finite(margin)?margin:.02))} percentage points of the best, prefer fewer actions. Ties are random. The estimate still needs to be learned; it does not certify success.`
    : 'Historical criterion: estimated duration plus a failure penalty.';
  if(finite(report.config?.path_exploration_warmup))$('path-objective').textContent+=
    ` Training exploration: from 100% to 20% over ${fmt(report.config.path_exploration_warmup)} planner actions per level. Planner evaluation does not add exploration.`;
  const search=report.config?.path_search;
  if(search)$('path-objective').textContent+=` Search: up to ${fmt(search.horizon)} steps; predictor and quality evaluation: 3 steps.`+
    (search.noop_cooldown>0?` Avoid repeating for ${fmt(search.noop_cooldown)} actions any action with no observed effect on the same goal-relevant board, when alternatives exist. This does not change legal actions.`:'');
  const change=report.objective_changes?.at(-1);
  const since=change?` Objective changed at action ${fmt(change.actions)}; earlier values use the previous function. Predictor charts show only the current objective; history remains in the report.`:'';
  const pathReplay=p.path_replay;
  const replayNote=pathReplay?` Mixed training: ${fmt(pathReplay.segments)} saved real attempts (${fmt(pathReplay.successes)} successes); up to ${fmt(pathReplay.batch_segments)} per update.`:'';
  const augmentationNote=pathReplay?.hindsight?' Also learns from goals actually reached within those episodes, without relabeling the original failures as wins.':'';
  const calibrationNote=pathReplay?.predicted_inputs?' Predictions are calibrated against the outcomes of the corresponding real actions.':'';
  const recovery=report.path_recovery;
  const recoveryNote=recovery?` Prior recovery: ${fmt(recovery.training_episodes)} episodes y ${fmt(recovery.training_actions)} additional real actions; counted separately from these curves.`:'';
  $('path-objective-loss').textContent=(successFirst
    ? 'Success/failure BCE + duration error only on successful trajectories. Interrupted attempts do not train these outputs.'
    : 'Error on the combined duration and failure cost.')+replayNote+augmentationNote+calibrationNote+recoveryNote+since;
  $('predictor-objective').textContent+=since;
  renderPlannerSuccess(report);
  plot('path-loss',[{name:'Update average · up to 512 actions',points:t.map(r=>[r.x,r.pathLoss])},
    {...series('Individual update','path_loss'),muted:true}]);
  const episodes=report.episodes||[], goals=episodes.flatMap(e=>e.segments.map(g=>({...g,x:e.end_action,game:e.game,level:e.level})));
  $('episodes').textContent=fmt(episodes.length);$('successes').textContent=`${fmt(episodes.filter(e=>e.success).length)} niveles resueltos · ventana reciente`;
  $('subgoals').textContent=`${fmt(goals.filter(g=>g.success).length)} / ${fmt(goals.length)}`;$('goal-label').textContent=p.goal||'';
  renderObservationContext();
  const efficiency=[['Real','actions'],['Minimum','minimum_actions']].map(([name,key],colorIndex)=>({name,colorIndex,points:goals.filter(g=>g.success).map(g=>[g.x,g[key]])}));
  plot('efficiency',[
    ...efficiency.map(s=>({...s,faint:true,legend:false})),
    ...efficiency.map(s=>({...s,name:`${s.name} · average over up to 512 actions`,points:movingAverage(s.points)}))
  ],{unit:'actions'});
  renderGoalOutcomes(goals);
  const rewards=goals.map(g=>[g.x,g.reward]);
  plot('reward',[{name:'Average per subgoal · up to 512 actions',points:movingAverage(rewards)},
    {name:'Reward individual',points:rewards,muted:true}]);
  const evaluations=report.evaluations||[];$('evaluation-empty').hidden=!!evaluations.length;
  $('evaluation-rows').replaceChildren(...evaluations.map(r=>{const tr=document.createElement('tr');
    const policy=r.fixed_baseline?'Random · fixed baseline':r.policy==='quality_probe'?'Quality · saved sample':r.policy;
    for(const value of [fmt(r.actions),r.split,`${r.game} / ${r.level}`,policy,
      finite(r.successes)?`${r.successes}/${r.episodes}`:'—',finite(r.subgoals)?`${r.subgoals}/${r.attempted_subgoals}`:'—',
      finite(r.delta_error)?`${fmt(r.delta_error*100,2)} %`:'—',
      r.policy==='quality_probe'?`0 new · ${fmt(r.analyzed_actions)} stored`:fmt(r.real_actions),fmt(r.decision_ms,2)]) {
        const td=document.createElement('td');td.textContent=value;tr.append(td);
      }return tr;}));
}
const tabs=['world','path','evaluation','curriculum'];
function selectTab(name) {
  for(const tab of tabs){$(tab).hidden=tab!==name;$('tab-'+tab).setAttribute('aria-selected',String(tab===name));$('tab-'+tab).tabIndex=tab===name?0:-1;}
  // A hidden SVG has no measured width; fit it as soon as its tab is visible.
  if(name==='world'&&displayedReport)render(displayedReport);
  if(name==='path'){
    if(displayedReport)renderPlannerSuccess(displayedReport);
    refreshTrajectory(displayedReport?.planner_trajectory);
  }
}
tabs.forEach((name,index)=>{
  $('tab-'+name).addEventListener('click',()=>selectTab(name));
  $('tab-'+name).addEventListener('keydown',event=>{
    const next=event.key==='ArrowRight'?(index+1)%tabs.length:event.key==='ArrowLeft'?(index+tabs.length-1)%tabs.length:event.key==='Home'?0:event.key==='End'?tabs.length-1:null;
    if(next===null)return;
    event.preventDefault();selectTab(tabs[next]);$('tab-'+tabs[next]).focus();
  });
});
let controlState=null, controlBusy=false, controlError='';
const phaseButton=$('advance-phase'), levelButton=$('advance-level');
const controlParking=$('curriculum-action-parking');
function renderControls(state){
  if(historicalView)state=null;
  controlState=state||null;
  const target=state?.target;
  const active=curriculumCells.find(row=>row.game===target?.game&&row.level===target?.level&&row.unlocked);
  // Level advance unlocks the next curriculum level, even when practicing an older one.
  const frontier=displayedReport?.curriculum?.ordered_stages?active:curriculumCells.filter(row=>row.game===target?.game&&row.unlocked)
    .reduce((last,row)=>!last||row.level>last.level?row:last,null);
  for(const [button,cell] of [[phaseButton,active?.phaseCell],[levelButton,frontier?.levelCell]]){
    const parent=cell||controlParking;
    if(button.parentNode!==parent)parent.append(button);
  }
  phaseButton.disabled=controlBusy||!state?.phase||!active;
  levelButton.disabled=controlBusy||!state?.level||!frontier;
  phaseButton.title=active?`Advance phase for ${active.game} / ${active.level}`:'';
  levelButton.title=frontier?`Open the level after ${frontier.game} / ${frontier.level}`:'';
  $('control-target').textContent=target?.game&&target?.level?`Training level: ${target.game} / ${target.level}`:'';
  const pending=state?.pending;
  const result=state?.result;
  let message=state?'The change takes effect when the current episode or evaluation finishes.':'Controls are not enabled on this dashboard.';
  if(state?.message)message=state.message;
  if(result?.status==='applied'){
    const event=result.event;
    const outcome={advanced:'Advance applied',already_planner:'This level was already in the planner phase',last_level:'The final level was already unlocked',budget_exhausted:'The training budget has been exhausted'};
    message=`${outcome[event.result]||event.result} · level ${event.level} · phase ${event.phase}.`;
  }
  if(result?.status==='error')message=result.message;
  if(pending)message=`Advancing ${pending.operation==='phase'?'phase':'level'} pending: waiting for the episode or evaluation to finish.`;
  if(controlBusy)message='Sending request…';
  if(controlError)message=controlError;
  $('control-status').textContent=historicalView?'Read-only history. Training controls are disabled.':message;
}
async function advanceCurriculum(operation){
  if(historicalView||controlBusy||!controlState?.[operation])return;
  const token=controlState.token;
  controlBusy=true;controlError='';renderControls(controlState);
  try{
    const response=await fetch('api/curriculum',{method:'POST',headers:{'Content-Type':'application/json','X-Curriculum-Token':token},body:JSON.stringify({operation}),signal:AbortSignal.timeout(10000)});
    if(!response.ok){let reason='Could not request advancement.';try{reason=(await response.json()).error||reason;}catch{}throw Error(reason);}
    controlState=await response.json();
  }catch(error){
    // Never retry an uncertain signal delivery. Polling recovers server state.
    controlError=`${error.message} Checking status; the request will not be resent automatically.`;
    controlState=null;
  }finally{controlBusy=false;renderControls(controlState);}
}
phaseButton.addEventListener('click',()=>advanceCurriculum('phase'));
levelButton.addEventListener('click',()=>advanceCurriculum('level'));
async function poll(){try{const response=await fetch(reportApi+'state',{cache:'no-store',signal:AbortSignal.timeout(10000)});if(!response.ok)throw Error('HTTP '+response.status);const data=await response.json();gpuStatus(data.gpu);if(!data.subgoals)throw Error('Waiting for subgoal data');render(data.subgoals);if(data.curriculum_control?.pending||data.curriculum_control?.result)controlError='';renderControls(data.curriculum_control);await refreshTrajectory(data.subgoals.planner_trajectory);}catch(error){gpuStatus(null);renderControls(null);$('connection').textContent='Disconnected · '+error.message;$('connection').className='warn';}finally{setTimeout(poll,2000);}}
const initialTab=globalThis.location?.hash?.slice(1);
selectTab(tabs.includes(initialTab)?initialTab:'world');
poll();
