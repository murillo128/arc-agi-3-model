// Numerical and interaction contracts; no browser or screenshots.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const context = vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../dashboard_web/chart-tools.js'),'utf8'),context);
const {ema} = context.ChartTools;
// Hand-calculated EMA: 0, 1, 2.9. A missing observation starts a new segment.
const result=ema([[0,0],[1,10],[2,20],[3,null],[4,30],[5,0],[6,NaN],[7,-10],[8,Infinity],[9,5]]);
assert.deepEqual(Array.from(result,p=>p[0]),[0,1,2,3,4,5,6,7,8,9]);
const expected=[0,1,2.9,null,30,27,null,-10,null,5];
result.forEach((p,i)=>expected[i]===null?assert.equal(p[1],null):assert(Math.abs(p[1]-expected[i])<1e-12));
assert.equal(ema([]).length,0);
// Invalid time cannot connect measurements; true zero remains measured.
assert.deepEqual(Array.from(ema([[0,2],[NaN,10],[2,0]]),p=>p[1]),[2,null,0]);
console.log('EMA numeric contracts passed');
