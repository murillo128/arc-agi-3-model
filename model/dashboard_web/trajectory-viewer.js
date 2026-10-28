'use strict';
// Shared by the current sequence dashboard and the read-only legacy viewer.
globalThis.ArcTrajectory = {
  create({getReport, picture, goalCanvas='goal', reportApi='api/'}) {
    const $=id=>document.getElementById(id);
    const finite=x=>typeof x==='number'&&Number.isFinite(x);
    const fmt=(x,n=0)=>finite(x)?x.toLocaleString('en-US',{maximumFractionDigits:n}):'—';
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
  const names={0:'Reset',1:'↑',2:'↓',3:'←',4:'→',5:'Space',6:'Click',7:'↶ Undo'};
  return `A${action.id} ${names[action.id]||''}${action.id===6?` (${action.data.x}, ${action.data.y})`:''}`.trim();
}
function sameAction(a,b){
  return !!a&&!!b&&a.id===b.id&&Object.keys(a.data||{}).length===Object.keys(b.data||{}).length
    &&Object.entries(a.data||{}).every(([k,v])=>b.data?.[k]===v);
}
function renderObservationContext(){
  const player=trajectoryPlayer, sample=player.sample, frame=player.frames[player.position], p=getReport()?.progress||{};
  if(!sample){
    picture('observation',p.observation);picture(goalCanvas,p.goal_frame,p.goal_mask);
    $('observation-context').textContent='Current interaction';
    $('observation-detail').textContent=p.game?`${p.game} / level ${fmt(p.level)}`:'Waiting for an observation…';
    $('goal-context').textContent=p.goal||'Current goal';
    const curriculum=getReport()?.curriculum;
    $('goal-detail').textContent=curriculum?.ordered_stages?.[curriculum.index]?.objective==='world_model'
      ?'Stage without a subgoal teacher: dynamics are learned from random actions.'
      :'Teacher-provided image. Checkerboard: area excluded from the goal.';
    return;
  }
  picture('observation',frameImage(frame));picture(goalCanvas,frameImage(frame.goalImage));
  $('observation-context').textContent=`Selected sample · t = ${frame.step}`;
  $('observation-detail').textContent=`${sample.game} / level ${sample.level} · episode ${sample.episode_seed} · ${sample.phase==='training'?'Training':'Evaluation'} · frame ${player.position+1}/${player.frames.length}`;
  $('goal-context').textContent=frame.goal||'Sample subgoal';
  $('goal-detail').textContent=frame.goalImage?`${frame.goal_source==='learned'?'Learned':frame.goal_source==='teacher'?'Teacher-provided':'Selected'} goal for this action. Checkerboard: area excluded from the goal.`:'This older sample does not include a subgoal image.';
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
      const button=make('button','trajectory-state');button.type='button';button.setAttribute('aria-label',`View real state t = ${t} · ${actionLabel(frame.action)}`);
      button.append(thumb(frame,`Real state t = ${t}`),make('span','action-badge',actionLabel(frame.action)));
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
    button.type='button';button.setAttribute('aria-label',`View origin t = ${origin}`);
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
  const reasons={success:'Level solved',game_over:'Game over',budget:'Action limit',interrupted:'Interrupted',unreachable_subgoal:'Unreachable goal',level_budget:'Level window ended',session_budget:'Session window ended',sample_limit:'Sample limit',goals_reached:'Teacher subgoals reached'};
  const capture=Number.isInteger(sample.batch)?`training batch ${fmt(sample.batch)}`:`captured at action ${fmt(sample.training_action)}`;
  $('trajectory-info').textContent=`${sample.game} / level ${sample.level} · episode ${sample.episode_seed} · ${sample.phase==='training'?'Training':'Evaluation'} · ${sample.actions} actions · ${reasons[sample.reason]||sample.reason} · ${capture}${sample.truncated?' · Partial episode':''}`;
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
  for(const [t,entry] of [{frames:sample.initial,goal:steps[0].goal,goalImage:steps[0].goalImage,goal_source:steps[0].goal_source},...steps].entries()){
    for(const frame of unpack(entry.frames))frames.push({...entry,...frame,step:t,frameIndex:frames.length});
    states.push(frames.at(-1));
  }
  if(frames.length!==sample.frame_count)throw Error('Incomplete trajectory');
  return {frames,states,steps,extent};
}
async function refreshTrajectory(metadata){
  const player=trajectoryPlayer;
  $('trajectory-warning').textContent=getReport()?.trajectory_error||'';
  if(!metadata){
    stopTrajectory();player.sample=null;player.frames=[];player.states=[];player.steps=[];renderTrajectory();
    $('trajectory-info').textContent=getReport()?.trajectory_error||'Waiting for a completed trajectory with the planner active…';
    return;
  }
  if($('path').hidden||!$('trajectory-live').checked||player.fetching||player.sample?.id===metadata.id)return;
  player.fetching=true;
  try{
    const response=await fetch(reportApi+'trajectory',{cache:'no-store',signal:AbortSignal.timeout(10000)});
    if(!response.ok)throw Error('HTTP '+response.status);
    const sample=(await response.json()).trajectory;
    if(getReport()?.planner_trajectory?.id!==metadata.id||!$('trajectory-live').checked||$('path').hidden)return;
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
  stopTrajectory();renderTrajectory();refreshTrajectory(getReport()?.planner_trajectory);
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
    return {refresh:refreshTrajectory,renderContext:renderObservationContext,stop:stopTrajectory};
  }
};
