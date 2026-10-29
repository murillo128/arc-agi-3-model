'use strict';
// Instance masks remain independent: no label-map reconstruction or identity matching.
(() => {
  const $=id=>document.getElementById(id);
  const palette=['#ffffff','#cccccc','#999999','#666666','#333333','#000000','#e53aa3','#ff7bcc','#f93c31','#1e93ff','#88d8f1','#ffdc00','#ff851b','#921231','#4fcc30','#a356d6'];
  const names={current:'Current SAM · t',learned_current:'Current student · t',next:'Next SAM · t + 1',predicted:'Predicted · t + 1'};
  const student=set=>set==='predicted'||set==='learned_current';
  const state={latest:null,sample:null,key:null,decoded:null,error:'',set:'current',mode:'outlines',selected:0,pinned:false,opacity:.4,uncovered:false};
  const key=sample=>sample?JSON.stringify([sample.game,sample.batch,sample.step]):null;
  const count=value=>value.toLocaleString('en-US');
  const percent=value=>(100*value).toLocaleString('en-US',{maximumFractionDigits:1})+'%';
  const color=(id,set)=>`hsl(${(id*137.508+{current:155,learned_current:260,next:35,predicted:260}[set])%360} 85% 65%)`;
  const label=(mask,set)=>`${student(set)?'Slot':'Mask'} ${mask.id+1}`;

  function decode(sample){
    const data=sample.instances;
    if(data.version!==1||data.encoding!=='packbits-little-base64')throw Error('Unsupported instance format.');
    if(!Array.isArray(data.shape)||data.shape.length!==2)throw Error('Invalid mask dimensions.');
    const [height,width]=data.shape, size=height*width;
    if(![height,width].every(n=>Number.isInteger(n)&&n>0&&n<=64))throw Error('Invalid mask dimensions.');
    for(const frame of [sample.before,sample.actual]){
      if(!Array.isArray(frame)||frame.length!==height||frame.some(row=>!Array.isArray(row)||row.length!==width))throw Error('Real frame and mask dimensions differ.');
    }
    const sets={};
    for(const name of Object.keys(names)){
      const source=data.sets?.[name], ids=new Set(), union=new Uint8Array(size);
      if(name==='learned_current'&&source===undefined)continue;
      if(!Array.isArray(source)||source.length>64)throw Error('Missing or invalid mask set.');
      const masks=source.map(entry=>{
        if(!Number.isInteger(entry.id)||entry.id<0||entry.id>=64||ids.has(entry.id))throw Error('Invalid instance number.');
        ids.add(entry.id);
        if(typeof entry.bits!=='string')throw Error('Missing instance pixels.');
        const bytes=atob(entry.bits);
        if(bytes.length!==Math.ceil(size/8))throw Error('Incomplete instance pixels.');
        const pixels=new Uint8Array(size);let area=0;
        for(let i=0;i<size;i++){
          pixels[i]=(bytes.charCodeAt(i>>3)>>(i%8))&1;
          if(pixels[i]){area++;union[i]=1;}
        }
        if(entry.area!==area||!Number.isFinite(entry.confidence)||entry.confidence<0||entry.confidence>1)throw Error('Invalid instance area or confidence.');
        return {...entry,pixels};
      });
      sets[name]={masks,union,covered:union.reduce((sum,value)=>sum+value,0)};
    }
    return {height,width,size,sets};
  }

  function outline(ctx,mask,width,height,scale,tone){
    // Draw each true boundary, including holes; opaque interiors never hide another mask.
    for(const [thickness,paint] of [[Math.max(2,scale*.3),'#10171b'],[Math.max(1,scale*.15),tone]]){
      ctx.fillStyle=paint;
      for(let y=0;y<height;y++)for(let x=0;x<width;x++){
        const i=y*width+x;if(!mask.pixels[i])continue;
        if(y===0||!mask.pixels[i-width])ctx.fillRect(x*scale,y*scale,scale,thickness);
        if(y===height-1||!mask.pixels[i+width])ctx.fillRect(x*scale,(y+1)*scale-thickness,scale,thickness);
        if(x===0||!mask.pixels[i-1])ctx.fillRect(x*scale,y*scale,thickness,scale);
        if(x===width-1||!mask.pixels[i+1])ctx.fillRect((x+1)*scale-thickness,y*scale,thickness,scale);
      }
    }
  }

  function paint(){
    const {width,height}=state.decoded, {masks,union}=state.decoded.sets[state.set];
    const selected=masks[state.selected], frame=['current','learned_current'].includes(state.set)?state.sample.before:state.sample.actual;
    const canvas=$('objects-inspector-canvas'), ctx=canvas.getContext('2d');
    const scale=Math.max(2,Math.min(8,Math.floor(768/Math.max(width,height))));
    canvas.width=width*scale;canvas.height=height*scale;
    ctx.globalAlpha=1;
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      ctx.fillStyle=palette[frame[y][x]]||'#787878';ctx.fillRect(x*scale,y*scale,scale,scale);
    }
    if(state.mode!=='frame'&&state.uncovered){
      ctx.fillStyle='#ff9e38';ctx.globalAlpha=.6;
      for(let y=0;y<height;y++)for(let x=0;x<width;x++)if(!union[y*width+x]){
        // A sparse diagonal hatch leaves the underlying real colors visible.
        for(let k=0;k<scale;k++)ctx.fillRect(x*scale+k,y*scale+(k+x+y)%scale,1,1);
      }
      ctx.globalAlpha=1;
    }
    if(state.mode==='single'&&selected){
      ctx.fillStyle=color(selected.id,state.set);ctx.globalAlpha=state.opacity;
      for(let y=0;y<height;y++)for(let x=0;x<width;x++)if(selected.pixels[y*width+x])ctx.fillRect(x*scale,y*scale,scale,scale);
      ctx.globalAlpha=1;outline(ctx,selected,width,height,scale,color(selected.id,state.set));
    }else if(state.mode==='outlines'){
      [...masks].sort((a,b)=>b.area-a.area).forEach(mask=>outline(ctx,mask,width,height,scale,color(mask.id,state.set)));
    }
    const mode=state.mode==='frame'?'real frame only':state.mode==='single'?(selected?`${label(selected,state.set)} over the real frame`:'no mask selected'):'all mask outlines over the real frame';
    const caption=`${names[state.set]} · ${mode}${state.uncovered&&state.mode!=='frame'?' · uncovered pixels hatched':''}. Colors are local to this set, not persistent object IDs.`;
    $('objects-inspector-caption').textContent=caption;canvas.setAttribute('aria-label',caption);
  }

  function update(){
    const sample=state.sample, available=!!state.decoded;
    $('objects-pin').checked=state.pinned;$('objects-pin').disabled=!sample;
    $('objects-inspector-controls').hidden=!available;$('objects-inspector-content').hidden=!available;
    const conditioning=sample?.conditioning==='sam'?'SAM masks':sample?.conditioning==='student'?'student masks':'not recorded';
    const action=sample?.action?` · action ${sample.action.id}${Object.keys(sample.action.data||{}).length?' '+JSON.stringify(sample.action.data):''}`:'';
    $('objects-sample-info').textContent=sample?`${sample.game} · step ${sample.step} · ${sample.instances?.sets?.learned_current?'before auxiliary fitting':'first auxiliary fit pass'}, batch ${sample.batch}${action} · prediction input: ${conditioning}`:'Waiting for an object sample.';
    const inspection=state.decoded?.sets.learned_current
      ?'Compare Current student with Current SAM to inspect extraction.'
      :'Current student masks were not recorded in this older sample.';
    const pinned=state.pinned?`Pinned sample.${key(state.latest)!==state.key?' A newer sample is available; unpin to follow it.':''} `:'';
    $('objects-inspector-status').textContent=pinned+(available?`Independent masks retain overlaps. ${inspection} SAM proposals may miss or merge meaningful game objects.`
      :state.error?`Instance inspection unavailable: ${state.error} The flattened overview remains below.`
      :sample?'Legacy sample: only flattened masks were saved. Overlap, individual area, confidence and coverage cannot be recovered; use the overview below.'
      :'Individual masks appear when a new object sample is saved.');
    if(!available){$('objects-instance').replaceChildren();return;}
    if(!state.decoded.sets[state.set])state.set='current';
    $('objects-set').replaceChildren(...Object.keys(state.decoded.sets).map(name=>{
      const option=document.createElement('option');option.value=name;option.textContent=names[name];return option;
    }));
    const {masks,covered}=state.decoded.sets[state.set], {size}=state.decoded;
    if(state.selected>=masks.length)state.selected=0;
    const selected=masks[state.selected];
    $('objects-set').value=state.set;$('objects-mode').value=state.mode;
    $('objects-opacity').value=String(state.opacity*100);$('objects-opacity').disabled=state.mode!=='single';
    $('objects-opacity-value').textContent=percent(state.opacity);
    $('objects-uncovered').checked=state.uncovered;$('objects-uncovered').disabled=state.mode==='frame';
    $('objects-set-summary').textContent=`${count(masks.length)} ${student(state.set)?'student masks':'SAM masks'} · covered ${count(covered)} / ${count(size)} pixels (${percent(covered/size)}) · uncovered ${count(size-covered)} (${percent(1-covered/size)})`;
    $('objects-instance').replaceChildren(...masks.map((mask,index)=>{
      const option=document.createElement('option');option.value=String(index);
      option.textContent=`${label(mask,state.set)} · ${count(mask.area)} px · ${percent(mask.area/size)}`;return option;
    }));
    $('objects-instance').value=String(state.selected);$('objects-instance').disabled=!masks.length;
    $('objects-prev').disabled=!masks.length||state.selected===0;$('objects-next').disabled=!masks.length||state.selected===masks.length-1;
    $('objects-instance-detail').textContent=selected?`${label(selected,state.set)} · ${count(selected.area)} pixels (${percent(selected.area/size)}) · ${student(state.set)?'presence':'SAM confidence'} ${percent(selected.confidence)}`:'No masks in this set. The real frame remains visible; all pixels are uncovered.';
    $('objects-instance-detail').style.borderColor=selected?color(selected.id,state.set):'transparent';
    $('objects-confidence-help').textContent=student(state.set)?'Presence is the model’s probability for this slot, not semantic correctness. Saved student masks use presence ≥ 50% and mask probability > 50%; empty masks are omitted.'
      :'SAM confidence is the lower of its predicted IoU and stability scores, not the probability of a correct game object. Overlapping instance areas are counted separately; coverage counts their union once.';
    paint();
  }

  function render(sample){
    state.latest=sample||null;
    if(!state.pinned||!state.sample){
      const nextKey=key(sample);
      if(state.key!==nextKey){state.set='current';state.mode='outlines';state.selected=0;state.uncovered=false;}
      state.sample=sample||null;state.key=nextKey;
      state.decoded=null;state.error='';
      if(sample?.instances){try{state.decoded=decode(sample);}catch(error){state.error=error.message;}}
    }
    update();return state.sample;
  }
  $('objects-set').addEventListener('change',()=>{state.set=$('objects-set').value;state.selected=0;update();});
  $('objects-mode').addEventListener('change',()=>{state.mode=$('objects-mode').value;update();});
  $('objects-instance').addEventListener('change',()=>{state.selected=Number($('objects-instance').value);state.mode='single';update();});
  for(const [id,delta] of [['objects-prev',-1],['objects-next',1]])$(id).addEventListener('click',()=>{
    state.selected=Math.max(0,Math.min(state.decoded.sets[state.set].masks.length-1,state.selected+delta));state.mode='single';update();
  });
  $('objects-opacity').addEventListener('input',()=>{state.opacity=Number($('objects-opacity').value)/100;update();});
  $('objects-uncovered').addEventListener('change',()=>{state.uncovered=$('objects-uncovered').checked;update();});
  $('objects-pin').addEventListener('change',()=>{state.pinned=$('objects-pin').checked;render(state.latest);});
  globalThis.renderObjectInspector=render;
})();
