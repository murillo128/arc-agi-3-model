// Interaction/data contract in Node; no browser or pixel assertions.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assets = path.join(__dirname, '../dashboard_web');
const html = fs.readFileSync(path.join(assets,'optimization.html'),'utf8');
function element(tag='div') {
  return {tag, textContent:'', hidden:false, dataset:{}, style:{}, children:[], listeners:{}, options:[], checked:true, value:'',
    append(...items){this.children.push(...items);}, replaceChildren(...items){this.children=items;if(tag==='select')this.options=items;},
    addEventListener(k,fn){this.listeners[k]=fn;},setAttribute(k,v){this[k]=v;}, focus(){},
    getBoundingClientRect(){return {width:700,height:260,left:0,top:0};},
    getContext(){return new Proxy({}, {get:(target,key)=>target[key]||(()=>{})});},
    querySelector(selector){return this[selector]??=(element(selector));}};
}
const nodes=new Map([...html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"[^>]*>/g)].map(m=>[m[2],element(m[1])]));
for(const m of html.matchAll(/<select[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)){
  const e=nodes.get(m[1]);e.options=[...m[2].matchAll(/<option value="([^"]+)"/g)].map(o=>({value:o[1]}));e.value=e.options[0]?.value||'';
}
const name='dynamics.cell.gates.weight';
let report={status:'ready',identity:'a',updated_at:1,batch:4,running:true,session_count:4,sample_per_tensor:256,every:8,
  catalog:[{name,module:'dynamics',size:4,shape:[2,2]},{name:'encoder.<unsafe>',module:'encoder',size:1,shape:[1,1]}],
  selected:name,age_seconds:1,control:{token:'test-token',enabled:true},
  events:[{id:1,batch:1,phase:'representation',clip_scale:1,matrices:{}},
    {id:2,batch:4,phase:'predictor',clip_scale:.5,matrices:{[name]:{status:'measured',gradient_rms:0,relative_update:.002,lr:.00003,alignment:.7}},detail:{name,gradient:[[0,0],[0,0]],update:[[.01,.02],[.03,.04]]}}],
  points:[1,2,3,4].map(batch=>({batch,reference_loss:5-batch,event_id:batch===4?2:1})),
  planes:{dynamics:{basis_id:1,projections:[0,1,2,3].map(x=>({x,y:x*.2,coverage:x?.25:null,angle:x?90:null,step_norm:x?1:0}))}},
  probe:{status:'ready',session:0,game:'<game>',batch:4,session_steps:512,context_start:480,start:496,stop:512,boundaries:[496],steps:[{t:480,rms:.001},{t:511,rms:.1}],loss:2,cleared_loss:3,seconds:.1}};
let polling, posts=[], interval, responseStatus=200, offline=false;
const context=vm.createContext({console,window:{devicePixelRatio:1,addEventListener(){}},
  document:{getElementById:id=>nodes.get(id),createElement:element,createDocumentFragment:()=>element('fragment')},
  setTimeout(fn){polling=fn;},setInterval(fn){interval=fn;return 1;},clearInterval(){interval=null;},
  fetch:async(url,options)=>{if(options?.method==='POST'){posts.push(options);return {ok:true,json:async()=>({id:'request-1',message:'Pendiente'})};}if(offline)throw Error('offline');return {ok:true,status:responseStatus,json:async()=>structuredClone(report)};}});
const text=id=>nodes.get(id).textContent;
const click=id=>nodes.get(id).listeners.click();
const change=(id,value)=>{nodes.get(id).value=value;nodes.get(id).listeners.change({target:nodes.get(id)});};
const input=(id,value)=>{nodes.get(id).value=value;nodes.get(id).listeners.input({target:nodes.get(id)});};
const flush=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
  vm.runInContext(fs.readFileSync(path.join(assets,'optimization.js'),'utf8'),context);await flush();
  assert.equal(text('gradient-value'),'0');assert.equal(text('matrix-state'),'Aprende');assert.equal(text('update-value'),'0,2 %');
  assert.match(text('event-label'),/Actualización 2/);
  offline=true;await polling();assert.equal(text('optimization-status'),'Sin conexión');
  offline=false;responseStatus=304;await polling();assert.equal(text('optimization-status'),'Midiendo entrenamiento');responseStatus=200;
  input('event-scrubber',0);assert.equal(text('matrix-state'),'Congelada en este paso');assert.equal(text('gradient-value'),'—');
  report.updated_at=2;report.events.push({...report.events[1],id:3});await polling();
  assert.match(text('event-label'),/Actualización 1/,'pinned snapshot does not drift as live history advances');
  nodes.get('live').checked=true;nodes.get('live').listeners.change();assert.match(text('event-label'),/Actualización 3/);
  click('tab-path');assert.equal(nodes.get('pulse').hidden,true);assert.equal(text('reference-value'),'1');assert.equal(text('coverage-value'),'25 %');
  input('point-scrubber',0);assert.equal(text('reference-value'),'4');assert.equal(text('coverage-value'),'—');
  click('view-2d');assert.equal(nodes.get('view-2d')['aria-pressed'],'true');
  click('tab-memory');assert.equal(text('probe-game'),'<game> / 1');assert.match(text('probe-window'),/480–511 · 16 \+ 16/);assert.match(text('probe-comparison'),/reduce la pérdida/);
  nodes.get('probe-session').value='2';click('capture-probe');await flush();assert.deepEqual(JSON.parse(posts[0].body),{operation:'probe',session:2});assert.equal(posts[0].headers['X-Optimization-Token'],'test-token');
  click('tab-pulse');change('module','encoder');assert.equal(text('matrix-title'),'<unsafe>');assert.equal(text('gradient-value'),'—');
  click('capture-matrix');await flush();assert.deepEqual(JSON.parse(posts[1].body),{operation:'matrix',matrix:'encoder.<unsafe>'});
  report={status:'waiting',message:'No hay registro'};await polling();assert.equal(nodes.get('pulse').hidden,true);assert.equal(nodes.get('capture-probe').disabled,true);assert.equal(text('event-label'),'Sin capturas');
  click('tab-path');assert.equal(nodes.get('path').hidden,true,'missing run must not reveal stale metrics on tab selection');
})().catch(error=>{console.error(error);process.exitCode=1;});
