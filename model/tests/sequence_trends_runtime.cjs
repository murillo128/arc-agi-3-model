// Numerical contracts and counterexamples, independent of the plot renderer.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const context = vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../dashboard_web/sequence-trends.js'),'utf8'),context);
const {compare,rolling} = context.SequenceTrends;
const q=(wrong,union,transitions=1)=>({wrong,union,transitions});
const row=(batch,a,b,extra={})=>({batch,quality:{...(a?{a}:{}),...(b?{b}:{})},...extra});
// The large easy game dominates pooled pixels; equal-game weighting must not.
const rows=[row(1,q(90,100),q(0,10)),row(2,q(810,900)),
  row(3,q(10,100),q(10,10)),row(4,q(90,900))];
let result=compare(rows,['a','b'],2);
assert.equal(result.previous.balanced,45); // (90% + 0%)/2, not mean of per-batch means.
assert.equal(result.current.balanced,55); // (10% + 100%)/2: worse despite better pooled error.
assert(Math.abs(result.previous.pooled-900/1010*100)<1e-10);
assert(Math.abs(result.current.pooled-110/1010*100)<1e-10);
assert.equal(result.current.games[0].batches,2);
assert.equal(result.current.games[1].batches,1);
assert.equal(result.current.start,3);assert.equal(result.previous.end,2);
let points=rolling(rows,['a','b'],2);
assert.equal(points[0].balanced,null);assert.equal(points[1].balanced,45);assert.equal(points[3].balanced,55);
// Missing games and invalid counts cannot masquerade as a perfect prediction.
assert.equal(compare(rows,['a','b','missing'],2).current.balanced,null);
assert.equal(compare([row(1,q(0,0)),row(2,q(0,0))],['a'],2).current.balanced,null);
assert.equal(compare([row(1,q(0,10)),row(2,q(0,0))],['a'],2).current.balanced,0);
assert.equal(compare([row(1,q(1,10)),row(2,{delta_error:0})],['a'],2).current.balanced,null);
assert.equal(compare([row(1,q(11,10)),row(2,q(0,10))],['a'],2).current.balanced,null);
// Neither comparison nor moving means may bridge a missing batch or a new regime.
for(const changed of [{batch:7},{measurement_source:'training_pre_update'},{phase:'predictor'},
  {learning_rates:{predictor:1e-4}},{exploration:{policy:'learned',identity:'different'}}]){
  const altered=rows.map(r=>({...r}));Object.assign(altered[2],changed);
  if(changed.batch)altered[3].batch=8;else Object.assign(altered[3],changed);
  const c=compare(altered,['a','b'],2);
  assert.equal(c.previous,null);assert.equal(c.current.balanced,55);
  const p=rolling(altered,['a','b'],2);
  assert.equal(p[2].balanced,null);assert.equal(p[3].balanced,55);
}
assert.equal(compare(rows.slice(0,1),['a','b'],2).current,null);
assert.equal(compare([],['a','b'],2).available,0);
// Removing a previously invalid row from a rolling window restores valid evidence.
assert.equal(rolling([row(1,{wrong:null,union:10}),row(2,q(2,10)),row(3,q(0,10))],['a'],2)[2].balanced,10);
console.log('Sequence trend statistics: passed');
