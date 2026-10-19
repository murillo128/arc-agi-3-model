// Interaction/data contract in Node; no browser or pixel assertions.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const assets = path.join(__dirname, '../dashboard_web');
const html = fs.readFileSync(path.join(assets,'optimization.html'),'utf8');
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname,'optimization_snapshot.json'),'utf8'));
function element(tag='div') {
  return {tag, textContent:'', hidden:false, dataset:{}, style:{}, children:[], listeners:{}, options:[], checked:true, value:'', draws:[],
    set width(value){this._width=value;this.draws=[];}, get width(){return this._width;},
    append(...items){this.children.push(...items);}, replaceChildren(...items){this.children=items;if(tag==='select')this.options=items;},
    addEventListener(k,fn){this.listeners[k]=fn;},setAttribute(k,v){this[k]=v;}, focus(){},
    getBoundingClientRect(){return {width:700,height:260,left:0,top:0};},
    getContext(){return new Proxy({}, {get:(target,key)=>target[key]||((...args)=>this.draws.push([key,...args]))});},
    querySelector(selector){return this[selector]??=(element(selector));}};
}
async function dashboard() {
  const nodes=new Map([...html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"[^>]*>/g)].map(m=>[m[2],element(m[1])]));
  for(const m of html.matchAll(/<select[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)){
    const e=nodes.get(m[1]);e.options=[...m[2].matchAll(/<option value="([^"]+)"/g)].map(o=>({value:o[1]}));e.value=e.options[0]?.value||'';
  }
  const state={report:{...structuredClone(fixture),status:'ready',running:true,session_count:4,
    age_seconds:1,control:{token:'test-token',enabled:true}},responseStatus:200,offline:false,invalidJSON:false,etag:'"good"',posts:[],gets:[]};
  const context=vm.createContext({console,window:{devicePixelRatio:1,addEventListener(){}},
    document:{getElementById:id=>nodes.get(id),createElement:element,createDocumentFragment:()=>element('fragment')},
    setTimeout(fn){state.polling=fn;},setInterval(fn){state.interval=fn;return 1;},clearInterval(){state.interval=null;},
    fetch:async(url,options)=>{if(options?.method==='POST'){state.posts.push(options);return {ok:true,json:async()=>({id:'request-1',message:'Pendiente'})};}
      state.gets.push(options);if(state.offline)throw Error('offline');return {ok:true,status:state.responseStatus,headers:{get:()=>state.etag},
        json:async()=>{if(state.invalidJSON)throw SyntaxError('invalid JSON');return structuredClone(state.report);}};}});
  const text=id=>nodes.get(id).textContent;
  const click=id=>nodes.get(id).listeners.click();
  const change=(id,value)=>{nodes.get(id).value=value;nodes.get(id).listeners.change({target:nodes.get(id)});};
  const input=(id,value)=>{nodes.get(id).value=value;nodes.get(id).listeners.input({target:nodes.get(id)});};
  const flush=()=>new Promise(resolve=>setImmediate(resolve));
  vm.runInContext(fs.readFileSync(path.join(assets,'optimization.js'),'utf8'),context);await flush();
  return {nodes,state,text,click,change,input,flush};
}

test('live, pin, selectors, controls and missing data',async()=>{
  const {nodes,state,text,click,change,input,flush}=await dashboard();
  assert.equal(text('gradient-value'),'0');assert.equal(text('matrix-state'),'Aprende');assert.equal(text('update-value'),'0,2 %');
  assert.match(text('event-label'),/Actualización 4/);
  state.offline=true;await state.polling();assert.equal(text('optimization-status'),'Sin conexión');
  state.offline=false;state.responseStatus=304;await state.polling();assert.equal(text('optimization-status'),'Midiendo entrenamiento');state.responseStatus=200;
  input('event-scrubber',0);assert.equal(text('matrix-state'),'Congelada en este paso');assert.equal(text('gradient-value'),'—');
  state.report.updated_at=2;state.report.tick=5;state.report.events.push({...state.report.events[1],id:5});await state.polling();
  assert.match(text('event-label'),/Actualización 1/,'pinned snapshot does not drift as live history advances');
  nodes.get('live').checked=true;nodes.get('live').listeners.change();assert.match(text('event-label'),/Actualización 5/);
  click('tab-path');assert.equal(nodes.get('pulse').hidden,true);assert.equal(text('reference-value'),'1');assert.equal(text('coverage-value'),'25 %');
  input('point-scrubber',0);assert.equal(text('reference-value'),'4');assert.equal(text('coverage-value'),'—');
  click('view-2d');assert.equal(nodes.get('view-2d')['aria-pressed'],'true');
  click('tab-memory');assert.equal(text('probe-game'),'<game> / 1');assert.match(text('probe-window'),/480–511 · 16 \+ 16/);assert.match(text('probe-comparison'),/reduce la pérdida/);
  nodes.get('probe-session').value='2';click('capture-probe');await flush();assert.deepEqual(JSON.parse(state.posts[0].body),{operation:'probe',session:2});assert.equal(state.posts[0].headers['X-Optimization-Token'],'test-token');
  click('tab-pulse');change('module','encoder');assert.equal(text('matrix-title'),'<unsafe>');assert.equal(text('gradient-value'),'—');
  click('capture-matrix');await flush();assert.deepEqual(JSON.parse(state.posts[1].body),{operation:'matrix',matrix:'encoder.<unsafe>'});
  click('tab-memory');assert.equal(text('probe-loss'),'2');
  state.report.updated_at=3;state.report.probe={status:'unavailable',reason:'No supervised transition'};await state.polling();
  assert.equal(text('probe-loss'),'—');assert.equal(text('probe-comparison'),'');
  state.report={status:'waiting',message:'No hay registro'};await state.polling();assert.equal(nodes.get('pulse').hidden,true);assert.equal(nodes.get('capture-probe').disabled,true);assert.equal(text('event-label'),'Sin capturas');
  click('tab-path');assert.equal(nodes.get('path').hidden,true,'missing run must not reveal stale metrics on tab selection');
});

test('a retained trajectory point never borrows an evicted optimizer capture',async()=>{
  const {nodes,state,text,click,change,input}=await dashboard();
  click('tab-path');input('point-scrubber',1); // batch 2 has a point, but no retained optimizer event.
  assert.equal(text('point-title'),'Batch 2');
  assert.match(text('event-label'),/Batch 2.*captura no disponible/);
  click('tab-pulse');
  assert.equal(text('matrix-state'),'Captura no disponible');
  for(const id of ['gradient-value','update-value','learning-rate','clip-value','alignment-value'])assert.equal(text(id),'—');
  for(const id of ['gradient-map','update-map','update-histogram']){
    assert(nodes.get(id).draws.some(([op,label])=>op==='fillText' && /no capturad[oa]/.test(label)),`${id} must explicitly lack detail`);
    assert(!nodes.get(id).draws.some(([op])=>op==='fillRect'),`${id} must not show another event's measurements`);
  }
  change('module','encoder');assert.equal(text('matrix-state'),'Captura no disponible');
  change('module','dynamics');state.report.updated_at=2;await state.polling();
  assert.match(text('event-label'),/Batch 2.*captura no disponible/,'pin survives a live refresh');
  click('tab-path');input('point-scrubber',3);click('tab-pulse');
  assert.match(text('event-label'),/Actualización 4.*batch 4/);assert.equal(text('gradient-value'),'0');
  click('tab-path');input('point-scrubber',1);nodes.get('live').checked=true;nodes.get('live').listeners.change();
  click('tab-pulse');assert.equal(text('matrix-state'),'Aprende');
});

test('malformed responses clear all live/pinned state and recover without caching the bad revision',async()=>{
  const {nodes,state,text,click,change,input}=await dashboard();
  const valid=structuredClone(state.report);
  const corruptions=[
    s=>{s.catalog=[null];},
    s=>{s.events[1].detail.gradient=[null];},
    s=>{s.planes.dynamics.projections.pop();},
    s=>{s.probe.steps=[null];},
    s=>{s.events[1].matrices=null;},
    s=>{s.points[1]=null;},
    s=>{s.catalog[0].shape='2x2';},
    s=>{s.events[1].detail.histogram.edges=null;},
  ];
  for(const corrupt of [...corruptions,null]){
    state.report=structuredClone(valid);state.invalidJSON=false;state.etag='"good"';await state.polling();
    change('module','dynamics');click('tab-pulse');input('event-scrubber',1);click('playback');
    assert.equal(text('gradient-value'),'0');
    if(corrupt)corrupt(state.report);else state.invalidJSON=true;
    state.etag='"bad"';await state.polling();
    assert.equal(text('optimization-status'),'Datos no válidos');assert.equal(text('event-label'),'Sin capturas');
    assert.equal(text('gradient-value'),'—');assert.equal(state.interval,null,'discard invalidated playback');
    assert.equal(nodes.get('capture-probe').disabled,true);
    for(const id of ['pulse','path','memory']){click(`tab-${id}`);assert.equal(nodes.get(id).hidden,true);}
    change('module','encoder'); // Must remain safe after a broken catalog.
    state.report=structuredClone(valid);state.invalidJSON=false;state.etag='"good"';await state.polling();
    assert.equal(state.gets.at(-1).headers['If-None-Match'],undefined,'invalid data must not retain an ETag');
    change('module','dynamics');click('tab-pulse');
    assert.equal(text('optimization-status'),'Midiendo entrenamiento');assert.equal(text('gradient-value'),'0');
    assert.equal(nodes.get('pulse').hidden,false);assert.equal(nodes.get('capture-probe').disabled,false);
  }
});
