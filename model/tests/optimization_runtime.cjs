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
    getContext(){return new Proxy({}, {get:(target,key)=>target[key]||((...args)=>this.draws.push([key,...args])),
      set:(target,key,value)=>{target[key]=value;this.draws.push(['set',key,value]);return true;}});},
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
    fetch:async(url,options)=>{if(options?.method==='POST'){state.posts.push(options);return {ok:true,json:async()=>({id:'request-1',message:'Pending'})};}
      state.gets.push(options);if(state.offline)throw Error('offline');return {ok:true,status:state.responseStatus,headers:{get:()=>state.etag},
        json:async()=>{if(state.invalidJSON)throw SyntaxError('invalid JSON');return structuredClone(state.report);}};}});
  const text=id=>nodes.get(id).textContent;
  const click=id=>nodes.get(id).listeners.click();
  const change=(id,value)=>{nodes.get(id).value=value;nodes.get(id).listeners.change({target:nodes.get(id)});};
  const input=(id,value)=>{nodes.get(id).value=value;nodes.get(id).listeners.input({target:nodes.get(id)});};
  const flush=()=>new Promise(resolve=>setImmediate(resolve));
  vm.runInContext(fs.readFileSync(path.join(assets,'chart-tools.js'),'utf8'),context);
  vm.runInContext(fs.readFileSync(path.join(assets,'optimization.js'),'utf8'),context);await flush();
  return {nodes,state,text,click,change,input,flush};
}

test('live, pin, selectors, controls and missing data',async()=>{
  const {nodes,state,text,click,change,input,flush}=await dashboard();
  assert.equal(text('gradient-value'),'0');assert.equal(text('matrix-state'),'Learning');assert.equal(text('update-value'),'0.2 %');
  for(const id of ['gradient-map','update-map']) {
    assert(nodes.get(id).draws.some(([op,label])=>op==='fillText' && label==='2 × 2 blocks'),
      `${id} labels the measured block dimensions in English`);
  }
  assert.match(text('event-label'),/Update 4/);
  state.offline=true;await state.polling();assert.equal(text('optimization-status'),'Disconnected');
  state.offline=false;state.responseStatus=304;await state.polling();assert.equal(text('optimization-status'),'Measuring training');state.responseStatus=200;
  input('event-scrubber',0);assert.equal(text('matrix-state'),'Frozen at this step');assert.equal(text('gradient-value'),'—');
  state.report.updated_at=2;state.report.tick=5;state.report.events.push({...state.report.events[1],id:5});await state.polling();
  assert.match(text('event-label'),/Update 1/,'pinned snapshot does not drift as live history advances');
  nodes.get('live').checked=true;nodes.get('live').listeners.change();assert.match(text('event-label'),/Update 5/);
  click('tab-path');assert.equal(nodes.get('pulse').hidden,true);assert.equal(text('reference-value'),'1');assert.equal(text('coverage-value'),'25 %');
  input('point-scrubber',0);assert.equal(text('reference-value'),'4');assert.equal(text('coverage-value'),'—');
  click('view-2d');assert.equal(nodes.get('view-2d')['aria-pressed'],'true');
  click('tab-memory');assert.equal(text('probe-game'),'<game> / 1');assert.match(text('probe-window'),/480–511 · 16 \+ 16/);assert.match(text('probe-comparison'),/reduces the loss/);
  nodes.get('probe-session').value='2';click('capture-probe');await flush();assert.deepEqual(JSON.parse(state.posts[0].body),{operation:'probe',session:2});assert.equal(state.posts[0].headers['X-Optimization-Token'],'test-token');
  click('tab-pulse');change('module','encoder');assert.equal(text('matrix-title'),'<unsafe>');assert.equal(text('gradient-value'),'—');
  click('capture-matrix');await flush();assert.deepEqual(JSON.parse(state.posts[1].body),{operation:'matrix',matrix:'encoder.<unsafe>'});
  click('tab-memory');assert.equal(text('probe-loss'),'2');
  state.report.updated_at=3;state.report.probe={status:'unavailable',reason:'No supervised transition'};await state.polling();
  assert.equal(text('probe-loss'),'—');assert.equal(text('probe-comparison'),'');
  state.report={status:'waiting',message:'No hay registro'};await state.polling();assert.equal(nodes.get('pulse').hidden,true);assert.equal(nodes.get('capture-probe').disabled,true);assert.equal(text('event-label'),'No captures');
  click('tab-path');assert.equal(nodes.get('path').hidden,true,'missing run must not reveal stale metrics on tab selection');
});

test('planner modules expose their matrices, projections and own losses without replacing legacy NLL',async()=>{
  const {nodes,state,text,click,change,input,flush}=await dashboard();
  assert(!nodes.get('module').options.some(o=>o.value==='goal_planner'),'legacy runs have no planner modules');
  const captured=structuredClone(state.report.events.at(-1).matrices[state.report.selected]);
  for(const name of ['goal_planner','path_planner']){
    state.report.catalog.push({name:name+'.output.weight',module:name,shape:[2,2],size:4});
    state.report.planes[name]=structuredClone(state.report.planes.dynamics);
    state.report.planes[name].projections.forEach((p,i)=>{p.z=i*.1;p.coverage_3d=i?.5:null;});
  }
  state.report.events.push({...structuredClone(state.report.events.at(-1)),id:5,phase:'goal_planner',detail:null,
    matrices:{'goal_planner.output.weight':captured}});
  state.report.tick=5;state.report.updated_at++;
  state.report.points.forEach((p,i)=>{p.module_losses={goal_planner:i===3?null:2+i,path_planner:10+i};});
  await state.polling();click('tab-path');
  for(const [name,label,expected] of [['goal_planner','Goal Planner','4'],['path_planner','Path Planner','12']]){
    change('module',name);input('point-scrubber',2);
    assert.equal(text('reference-value'),expected);
    assert.equal(text('reference-label'),`${label} loss · batch mean`);
    assert.equal(text('path-loss-label'),`Point color · ${label} loss`);
    assert.match(text('path-loss-note'),/without head updates have no loss/);
    assert.equal(nodes.get('matrix').value,name+'.output.weight');
    click('view-2d');assert.equal(nodes.get('view-2d')['aria-pressed'],'true');
    click('view-3d');assert.equal(nodes.get('view-3d')['aria-pressed'],'true');
  }
  change('module','goal_planner');input('point-scrubber',3);
  assert.equal(text('reference-value'),'—');assert.equal(text('loss-change'),'—');
  click('tab-pulse');click('capture-matrix');await flush();
  assert.deepEqual(JSON.parse(state.posts.at(-1).body),{operation:'matrix',matrix:'goal_planner.output.weight'});
  change('module','dynamics');click('tab-path');
  assert.equal(text('reference-value'),'1');assert.equal(text('reference-label'),'Reference visual NLL');
});

test('a retained trajectory point never borrows an evicted optimizer capture',async()=>{
  const {nodes,state,text,click,change,input}=await dashboard();
  click('tab-path');input('point-scrubber',1); // batch 2 has a point, but no retained optimizer event.
  assert.equal(text('point-title'),'Batch 2');
  assert.match(text('event-label'),/Batch 2.*capture unavailable/);
  click('tab-pulse');
  assert.equal(text('matrix-state'),'Capture unavailable');
  for(const id of ['gradient-value','update-value','learning-rate','clip-value','alignment-value'])assert.equal(text(id),'—');
  for(const id of ['gradient-map','update-map','update-histogram']){
    assert(nodes.get(id).draws.some(([op,label])=>op==='fillText' && /not captured/.test(label)),`${id} must explicitly lack detail`);
    assert(!nodes.get(id).draws.some(([op])=>op==='fillRect'),`${id} must not show another event's measurements`);
  }
  change('module','encoder');assert.equal(text('matrix-state'),'Capture unavailable');
  change('module','dynamics');state.report.updated_at=2;await state.polling();
  assert.match(text('event-label'),/Batch 2.*capture unavailable/,'pin survives a live refresh');
  click('tab-path');input('point-scrubber',3);click('tab-pulse');
  assert.match(text('event-label'),/Update 4.*batch 4/);assert.equal(text('gradient-value'),'0');
  click('tab-path');input('point-scrubber',1);nodes.get('live').checked=true;nodes.get('live').listeners.change();
  click('tab-pulse');assert.equal(text('matrix-state'),'Learning');
});

test('training batch loss is labeled separately from fixed reference loss',async()=>{
  const {state,text,click}=await dashboard();
  click('tab-path');
  assert.equal(text('reference-label'),'Reference visual NLL');
  state.report.updated_at=99;
  state.report.reference_source='training_pre_update';
  await state.polling();
  assert.equal(text('reference-label'),'Visual NLL before learning');
  assert.match(text('reference-source'),/New data from each batch/);
  assert.match(text('reference-source'),/weights at each point are from after the update/);
});

test('trajectory radii distinguish relative residual motion on a fixed scale',async()=>{
  const {nodes,state,text,click,input}=await dashboard();
  const projections=state.report.planes.dynamics.projections;
  const cases=[
    {coverage:null,coverage_3d:null,step_norm:0},
    {coverage:.08,coverage_3d:.155,step_norm:.0488},
    {coverage:.005,coverage_3d:.011,step_norm:.0479},
    {coverage:1/26,coverage_3d:.5,step_norm:7},
  ];
  projections.forEach((p,i)=>Object.assign(p,cases[i],{z:i*.3}));
  state.report.updated_at++;
  await state.polling();click('tab-path');
  // The last arc is the selected point's outline, 4px outside its bubble.
  const radius=()=>nodes.get('parameter-path').draws.filter(d=>d[0]==='arc').at(-1)[3]-4;
  input('point-scrubber',1);const mostlyOutside=radius();
  input('point-scrubber',2);const almostEntirelyOutside=radius();
  assert(almostEntirelyOutside>mostlyOutside*1.5,'15.5% and 1.1% coverage remain visibly distinct on the log scale');
  assert(mostlyOutside>7.70&&mostlyOutside<7.72,'15.5% captured: about 7.707px');
  assert(almostEntirelyOutside>12.17&&almostEntirelyOutside<12.20,'1.1% captured: about 12.183px');
  assert.match(text('relative-step-value'),/^9\.482 ×$/);
  assert.equal(text('outside-step-value'),'0.0476','keep absolute residual length in the inspector');
  click('zoom-in');assert.equal(radius(),almostEntirelyOutside,'zoom does not rescale the encoding');
  input('point-scrubber',3);
  assert(Math.abs(radius()-5.708878)<1e-6,'50% coverage means outside/inside = 1');
  assert.equal(text('relative-step-value'),'1 ×');
  click('view-2d');
  assert(Math.abs(radius()-10.002348)<1e-6,'1/26 captured means outside/inside = 5 in 2D');
  assert.equal(text('relative-step-value'),'5 ×');
  assert.match(text('path-size-label'),/outside \/ inside.*2D/);
  assert.equal(text('path-size-ten'),'10×');assert.equal(text('path-size-hundred'),'100×');
  assert.equal(text('path-size-max'),'≥1,000×');
  assert.match(text('path-size-note'),/radius.*logarithmic.*1,000×.*30 px/i);
  click('view-3d');
  projections.forEach(p=>{p.step_norm*=100;});
  projections[3].coverage_3d=.999;projections[3].step_norm=10000;
  state.report.updated_at++;await state.polling();
  nodes.get('live').checked=true;nodes.get('live').listeners.change();
  input('point-scrubber',2);
  assert.equal(radius(),almostEntirelyOutside,'absolute norms and other points cannot change this radius');
  assert.equal(text('relative-step-value'),'9.482 ×');
  assert.equal(text('outside-step-value'),'4.7636');
});

test('relative bubble boundaries retain honest missing, zero, infinite and capped states',async()=>{
  const {nodes,state,text,click,input}=await dashboard();
  const projections=state.report.planes.dynamics.projections;
  click('tab-path'); // Original capture is legacy 2D with no PC3.
  assert.equal(nodes.get('view-3d').disabled,true);
  const radius=()=>nodes.get('parameter-path').draws.filter(d=>d[0]==='arc').at(-1)[3]-4;
  input('point-scrubber',0);assert.equal(radius(),3);assert.equal(text('relative-step-value'),'—');
  const cases=[
    {coverage:null,step_norm:0,radius:3,label:'—'},
    {coverage:null,step_norm:1,radius:3,label:'—'},
    {coverage:1,step_norm:1,radius:3,label:'0 ×'},
    {coverage:0,step_norm:1,radius:30,label:'∞ · no projected step'},
    {coverage:1/101,step_norm:1,radius:12.371178,label:'10 ×',legend:'ten'},
    {coverage:1/401,step_norm:1,radius:14.898252,label:'20 ×'},
    {coverage:1/10001,step_norm:1,radius:21.036283,label:'100 ×',legend:'hundred'},
    {coverage:1/1000001,step_norm:1,radius:30,label:'1,000 ×',legend:'max'},
    {coverage:1/4000001,step_norm:1,radius:30,label:'2,000 × · size capped at 1,000×'},
  ];
  for(const c of cases){
    Object.assign(projections[1],{coverage:c.coverage,step_norm:c.step_norm});
    state.report.updated_at++;await state.polling();
    nodes.get('live').checked=true;nodes.get('live').listeners.change();
    input('point-scrubber',1);
    assert(Math.abs(radius()-c.radius)<1e-6,`${c.label}: radius ${radius()} should be ${c.radius}`);
    assert.equal(text('relative-step-value'),c.label);
    if(c.legend){
      const dot=nodes.get(`path-size-${c.legend}-dot`);
      assert(Math.abs(parseFloat(dot.style.width)-2*radius())<1e-12,'legend diameter matches rendered radius');
      assert.equal(dot.style.width,dot.style.height);
    }
    assert(nodes.get('parameter-path').draws.filter(d=>d[0]==='arc').every(d=>Number.isFinite(d[3])));
  }
  state.invalidJSON=true;await state.polling();
  assert.equal(text('relative-step-value'),'—','invalid snapshots clear the new measurement');
  assert.equal(nodes.get('path-size-scale').hidden,true);
});

test('temporal optimization curves expose persistent EMA guides, including log-scale memory',async()=>{
  const {nodes,state,click}=await dashboard();
  click('tab-path');
  const canvas=nodes.get('reference-history'),button=nodes.get('reference-history-legend').children[0];
  assert(canvas.draws.some(d=>d[0]==='set'&&d[1]==='lineWidth'&&d[2]===1.8));
  button.listeners.click();
  assert(canvas.draws.some(d=>d[0]==='fillText'&&d[1]==='3.439'&&d[2]===8));
  state.report.updated_at=5;state.report.points.at(-1).reference_loss=0;
  await state.polling();
  assert.equal(nodes.get('reference-history-legend').children[0],button);
  assert.equal(button['aria-pressed'],'true');
  assert(canvas.draws.some(d=>d[0]==='fillText'&&d[1]==='3.339'&&d[2]===8));
  button.listeners.click();
  assert(!canvas.draws.some(d=>d[0]==='strokeRect'));
  click('tab-memory');
  const memory=nodes.get('memory-gradient'),memoryButton=nodes.get('memory-gradient-legend').children[0];
  assert(memory.draws.some(d=>d[0]==='set'&&d[1]==='lineWidth'&&d[2]===1.8));
  memoryButton.listeners.click();
  assert(memory.draws.some(d=>d[0]==='strokeRect'));
  assert(memory.draws.some(d=>d[0]==='fillText'&&d[1]==='0.0109'&&d[2]===8));
  state.report.updated_at=6;state.report.probe={status:'unavailable',reason:'No sample'};
  await state.polling();
  assert.equal(nodes.get('memory-gradient-legend').children.length,0);
  assert(!memory.draws.some(d=>d[0]==='strokeRect'));
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
    assert.equal(text('optimization-status'),'Invalid data');assert.equal(text('event-label'),'No captures');
    assert.equal(text('gradient-value'),'—');assert.equal(state.interval,null,'discard invalidated playback');
    assert.equal(nodes.get('capture-probe').disabled,true);
    for(const id of ['pulse','path','memory']){click(`tab-${id}`);assert.equal(nodes.get(id).hidden,true);}
    change('module','encoder'); // Must remain safe after a broken catalog.
    state.report=structuredClone(valid);state.invalidJSON=false;state.etag='"good"';await state.polling();
    assert.equal(state.gets.at(-1).headers['If-None-Match'],undefined,'invalid data must not retain an ETag');
    change('module','dynamics');click('tab-pulse');
    assert.equal(text('optimization-status'),'Measuring training');assert.equal(text('gradient-value'),'0');
    assert.equal(nodes.get('pulse').hidden,false);assert.equal(nodes.get('capture-probe').disabled,false);
  }
});
