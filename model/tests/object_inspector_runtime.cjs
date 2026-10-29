// Behavioral drawing/state contract; actual layout is verified separately in a browser.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const assets=path.join(__dirname,'../dashboard_web');
function element(){
  const context={operations:[],globalAlpha:1,fillRect(x,y,width,height){
    this.operations.push({x,y,width,height,color:this.fillStyle,alpha:this.globalAlpha});
  }};
  return {textContent:'',style:{},children:[],listeners:{},context,
    set width(value){this._width=value;context.operations=[];},get width(){return this._width;},
    set height(value){this._height=value;context.operations=[];},get height(){return this._height;},
    addEventListener(type,handler){this.listeners[type]=handler;},
    replaceChildren(...nodes){this.children=nodes;},setAttribute(name,value){this[name]=value;},
    getContext(){return context;}};
}
const nodes=new Map([...fs.readFileSync(path.join(assets,'sequences.html'),'utf8').matchAll(/id="([^"]+)"/g)].map(match=>[match[1],element()]));
const context=vm.createContext({document:{getElementById:id=>nodes.get(id),createElement:element},
  atob:value=>Buffer.from(value,'base64').toString('binary')});
vm.runInContext(fs.readFileSync(path.join(assets,'object-inspector.js'),'utf8'),context);
const render=sample=>context.renderObjectInspector(sample);
const node=id=>nodes.get('objects-'+id);
const change=(id,value)=>{node(id).value=value;node(id).listeners.change();};
const check=(id,value)=>{node(id).checked=value;node(id).listeners.change();};
const operations=()=>node('inspector-canvas').context.operations;
// Literal row-major, little-endian bits: full 3x3, center pixel, ninth pixel, second pixel.
// The center is fully contained in the large mask, so a flattened label image loses one.
const sample={game:'fixture',batch:7,step:2,conditioning:'student',action:{id:6,data:{x:2,y:1}},
  before:[[0,1,2],[3,4,5],[6,7,8]],actual:[[9,10,11],[12,13,14],[15,0,1]],
  instances:{version:1,encoding:'packbits-little-base64',shape:[3,3],sets:{
    current:[{id:0,bits:'/wE=',area:9,confidence:.9},{id:1,bits:'EAA=',area:1,confidence:.6}],
    next:[{id:0,bits:'AAE=',area:1,confidence:.8}],
    predicted:[{id:4,bits:'AgA=',area:1,confidence:.7}]}}};
render(sample);
assert.equal(node('inspector-content').hidden,false);
assert.match(node('sample-info').textContent,/fixture · step 2.*batch 7 · action 6.*prediction input: student masks/);
assert.match(node('set-summary').textContent,/2 SAM masks · covered 9 \/ 9 pixels \(100%\) · uncovered 0 \(0%\)/);
assert.equal(node('instance').children.length,2,'overlapping instances remain individually selectable');
assert.equal(node('instance').children[1].textContent,'Mask 2 · 1 px · 11.1%');
assert.deepEqual(operations().slice(0,9).map(op=>op.color),
  ['#ffffff','#cccccc','#999999','#666666','#333333','#000000','#e53aa3','#ff7bcc','#f93c31'],
  'inspection starts with the current real observation');
assert.ok(operations().slice(-8).every(op=>op.x>=8&&op.x<16&&op.y>=8&&op.y<16),
  'the one-pixel mask is outlined after the containing mask, retaining a visible boundary');
change('instance','1');
assert.equal(node('mode').value,'single');
assert.match(node('instance-detail').textContent,/Mask 2 · 1 pixels \(11.1%\) · SAM confidence 60%/);
assert.deepEqual(operations().filter(op=>op.alpha===.4).map(({x,y,width,height})=>({x,y,width,height})),
  [{x:8,y:8,width:8,height:8}],'single mode fills exactly the selected overlapping instance');
assert.ok(operations().slice(9).every(op=>op.x>=8&&op.x<16&&op.y>=8&&op.y<16),
  'single mode does not draw the large mask');
node('opacity').value='65';node('opacity').listeners.input();
assert.equal(node('opacity-value').textContent,'65%');
assert.equal(operations()[9].alpha,.65);
render(JSON.parse(JSON.stringify(sample)));
assert.equal(node('instance').value,'1','same-sample polling retains mask selection');
assert.equal(node('mode').value,'single');

change('set','next');
assert.deepEqual(operations().slice(0,9).map(op=>op.color),
  ['#1e93ff','#88d8f1','#ffdc00','#ff851b','#921231','#4fcc30','#a356d6','#ffffff','#cccccc'],
  'next SAM masks use the actual next observation, not the current frame');
assert.deepEqual(operations().filter(op=>op.alpha===.65).map(({x,y})=>({x,y})),[{x:16,y:16}],
  'ninth bit crosses the byte boundary and maps to the last row-major pixel');
assert.match(node('set-summary').textContent,/covered 1 \/ 9 pixels \(11.1%\) · uncovered 8 \(88.9%\)/);
check('uncovered',true);
assert.equal(operations().filter(op=>op.color==='#ff9e38').length,64,'hatching marks eight uncovered pixels, eight hatch points each');
assert.ok(operations().filter(op=>op.color==='#ff9e38').every(op=>op.x<16||op.y<16),
  'covered ninth pixel is excluded from uncovered hatching');
change('mode','frame');
assert.equal(operations().length,9,'real-only mode has neither mask nor coverage paint');
assert.equal(node('uncovered').disabled,true);

change('set','predicted');change('mode','single');
assert.match(node('instance-detail').textContent,/Slot 5 · 1 pixels \(11.1%\) · presence 70%/);
assert.match(node('confidence-help').textContent,/not semantic correctness/);
assert.deepEqual(operations().filter(op=>op.alpha===.65).map(({x,y})=>({x,y})),[{x:8,y:0}]);
assert.match(node('inspector-caption').textContent,/Predicted.*not persistent object IDs/);

check('pin',true);
const newer={...sample,batch:8};
assert.equal(render(newer).batch,7);
assert.equal(node('set').value,'predicted');
assert.match(node('inspector-status').textContent,/Pinned sample. A newer sample is available/);
check('pin',false);
assert.match(node('sample-info').textContent,/batch 8/);
assert.equal(node('set').value,'current');
assert.equal(node('instance').value,'0');
assert.equal(node('mode').value,'outlines');
assert.equal(node('uncovered').checked,false);
node('next').listeners.click();
assert.equal(node('instance').value,'1');assert.equal(node('next').disabled,true);
node('prev').listeners.click();
assert.equal(node('instance').value,'0');assert.equal(node('prev').disabled,true);

render({...sample,batch:9,instances:{...sample.instances,sets:{current:[],next:[],predicted:[]}}});
assert.equal(node('inspector-content').hidden,false);
assert.match(node('set-summary').textContent,/0 SAM masks.*uncovered 9 \(100%\)/);
assert.equal(node('instance').disabled,true);
assert.equal(node('prev').disabled,true);assert.equal(node('next').disabled,true);
assert.match(node('instance-detail').textContent,/No masks in this set/);
assert.equal(operations().length,9,'an empty set still shows the real frame');

const legacy={...sample,batch:10};delete legacy.instances;
render(legacy);
assert.equal(node('inspector-content').hidden,true);
assert.equal(node('inspector-controls').hidden,true);
assert.equal(node('instance').children.length,0,'legacy label colors never fabricate selectable instances');
assert.match(node('inspector-status').textContent,/Legacy sample: only flattened masks were saved/);

for(const [instances,reason] of [
  [{...sample.instances,version:2},/Unsupported instance format/],
  [{...sample.instances,shape:[3,65]},/Invalid mask dimensions/],
  [{...sample.instances,sets:{...sample.instances.sets,current:[{id:0,bits:'AA==',area:0,confidence:.9}]}},/Incomplete instance pixels/],
  [{...sample.instances,sets:{...sample.instances.sets,current:[{id:0,bits:'/wE=',area:8,confidence:.9}]}},/Invalid instance area/]
]){
  render({...sample,instances});
  assert.equal(node('inspector-content').hidden,true);
  assert.match(node('inspector-status').textContent,reason);
}
render(null);
assert.equal(node('pin').disabled,true);
assert.match(node('sample-info').textContent,/Waiting for an object sample/);
