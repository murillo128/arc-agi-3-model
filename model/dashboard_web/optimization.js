'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const C = {mint: '#8fd8bf', lilac: '#bcaafa', amber: '#e9b86c', text: '#e7efec', muted: '#9db2a8', grid: '#2b4038', dark: '#152321'};
  const fmt = (v, d = 3) => Number.isFinite(v) ? v.toLocaleString('en-US', {maximumFractionDigits: d}) : '—';
  const sci = v => !Number.isFinite(v) ? '—' : v === 0 ? '0' : Math.abs(v) < .001 ? v.toExponential(2) : fmt(v, 4);
  const moduleNames = {encoder: 'Visual encoder', decoder: 'Decoder', dynamics: 'Predictor', goal_planner: 'Goal Planner', path_planner: 'Path Planner'};
  const moduleName = m => moduleNames[m] || m;
  const shortName = n => n.split('.').slice(1).join('.');
  const defaultCamera = () => ({yaw: -.6, pitch: .45, roll: 0, zoom: 1});
  let camera = defaultCamera();
  let latest = null, pinned = null, eventIndex = 0, pointIndex = 0, tab = 'pulse', dimension = 3, playing = null, requestId = null, revision = null;
  let selection = 'dynamics.cell.gates.weight', module = 'dynamics', drawHits = [];
  const plannerModule = () => module === 'goal_planner' || module === 'path_planner';
  const pointLoss = p => plannerModule() ? p?.module_losses?.[module] ?? null : p?.reference_loss;
  const lossName = () => plannerModule() ? `${moduleName(module)} loss` : 'Visual NLL';
  const data = () => pinned || latest;
  const text = (id, value) => { $(id).textContent = value; };
  function node(tag, value, cls) { const e = document.createElement(tag); if (value != null) e.textContent = value; if (cls) e.className = cls; return e; }
  function validSnapshot(s) {
    // Check the fields rendering consumes before replacing either live or pinned
    // data. The server also validates the on-disk measurement contract.
    const object = v => v !== null && typeof v === 'object' && !Array.isArray(v);
    const integer = (v, min=0) => Number.isSafeInteger(v) && v >= min;
    const number = v => v === null || Number.isFinite(v);
    const coverage = v => v === null || (Number.isFinite(v) && v >= 0 && v <= 1);
    const spatialPair = v => {
      if (Object.hasOwn(v,'z') !== Object.hasOwn(v,'coverage_3d')) return false;
      if (!Object.hasOwn(v,'z')) return true;
      return Number.isFinite(v.z) && coverage(v.coverage_3d) && (v.step_norm === 0
        ? v.coverage === null && v.coverage_3d === null
        : v.coverage !== null && v.coverage_3d !== null && v.coverage_3d + 1e-9 >= v.coverage);
    };
    const list = (v, check, max=Infinity) => Array.isArray(v) && v.length <= max && v.every(check);
    const grid = v => Array.isArray(v) && v.length > 0 && v.length <= 16 &&
      list(v, row => Array.isArray(row) && row.length > 0 && row.length === v[0].length && list(row, Number.isFinite, 16));
    try {
      if (!object(s) || s.kind !== 'optimization' || s.version !== 1 || typeof s.identity !== 'string' || !s.identity ||
          !integer(s.batch) || !Number.isFinite(s.updated_at) || !integer(s.every,1) || !integer(s.sample_per_tensor,1) ||
          !integer(s.session_count,1) || !Number.isFinite(s.age_seconds) || typeof s.running !== 'boolean' ||
          !object(s.control) || typeof s.control.enabled !== 'boolean' || typeof s.control.token !== 'string') return false;
      if (!list(s.catalog, m => object(m) && typeof m.name === 'string' && m.name.length > 0 &&
          Object.hasOwn(moduleNames,m.module) && list(m.shape, n => integer(n,1)) && integer(m.size,1) &&
          m.shape.reduce((a,b)=>a*b,1) === m.size) || !s.catalog.length) return false;
      const names = new Set(s.catalog.map(m=>m.name));
      if (names.size !== s.catalog.length || !names.has(s.selected)) return false;
      if (!list(s.events, (e,i) => object(e) && integer(e.id,1) && (!i || e.id > s.events[i-1].id) &&
          integer(e.batch,1) && e.batch <= s.batch && Number.isFinite(e.clip_scale) && object(e.matrices) &&
          Object.entries(e.matrices).every(([name,row]) => names.has(name) && object(row) &&
            ['measured','no_gradient','frozen'].includes(row.status) && (row.status !== 'measured' ||
              ['gradient_rms','relative_update','lr','alignment'].every(k=>number(row[k])))) &&
          (e.detail == null || (object(e.detail) && names.has(e.detail.name) && e.matrices[e.detail.name]?.status === 'measured' &&
            grid(e.detail.gradient) && grid(e.detail.update) && object(e.detail.histogram) &&
            list(e.detail.histogram.counts,n=>integer(n),16) && e.detail.histogram.counts.length > 0 &&
            list(e.detail.histogram.edges,Number.isFinite) && e.detail.histogram.edges.length === e.detail.histogram.counts.length+1)),64)) return false;
      if (!list(s.points, (p,i) => object(p) && integer(p.batch,1) && p.batch <= s.batch &&
          (!i || p.batch > s.points[i-1].batch) && integer(p.event_id) && number(p.reference_loss) &&
          (p.module_losses === undefined || (object(p.module_losses) && Object.entries(p.module_losses).every(([name,value]) =>
            s.catalog.some(m=>m.module===name) && (value === null || (Number.isFinite(value) && value >= 0))))) &&
          !s.events.some(e=>e.id === p.event_id && e.batch > p.batch),96) || !object(s.planes)) return false;
      if (!Object.entries(s.planes).every(([name,p]) => s.catalog.some(m=>m.module===name) && object(p) && integer(p.basis_id) &&
          list(p.projections, v => object(v) && Number.isFinite(v.x) && Number.isFinite(v.y) &&
            coverage(v.coverage) && number(v.angle) && Number.isFinite(v.step_norm) && v.step_norm >= 0 && spatialPair(v)) &&
          new Set(p.projections.map(v=>Object.hasOwn(v,'z'))).size <= 1 &&
          p.projections.length === (p.basis_id ? s.points.length : 0))) return false;
      const p=s.probe;
      if (p == null) return true;
      if (!object(p)) return false;
      if (p.status === 'unavailable') return typeof p.reason === 'string';
      return p.status === 'ready' && integer(p.session) && p.session < s.session_count && typeof p.game === 'string' &&
        integer(p.batch,1) && p.batch <= s.batch && ['session_steps','context_start','start','stop'].every(k=>integer(p[k])) &&
        p.context_start <= p.start && p.start < p.stop && p.stop <= p.session_steps &&
        list(p.boundaries,t=>integer(t,p.context_start) && t < p.stop) &&
        list(p.steps,(v,i)=>object(v) && integer(v.t,p.context_start) && v.t < p.stop && (!i || v.t > p.steps[i-1].t) && number(v.rms)) &&
        p.steps.length > 0 && ['loss','cleared_loss','seconds'].every(k=>Number.isFinite(p[k]));
    } catch { return false; }
  }
  function color(value, low = -8, high = -1) {
    if (!Number.isFinite(value)) return C.dark;
    const t = value > 0 ? Math.max(0, Math.min(1, (Math.log10(value) - low) / (high-low))) : 0;
    const a = [31, 43, 50], b = [188, 170, 250], c = [143, 238, 207];
    const mix = t < .55 ? t/.55 : (t-.55)/.45, start = t < .55 ? a : b, end = t < .55 ? b : c;
    return `rgb(${start.map((v, i) => Math.round(v+(end[i]-v)*mix)).join(',')})`;
  }
  function canvas(id) {
    const el = $(id), ratio = Math.min(2, window.devicePixelRatio || 1), box = el.getBoundingClientRect();
    const w = Math.max(200, box.width || 700), h = Math.max(70, box.height || 240);
    el.width = Math.round(w*ratio); el.height = Math.round(h*ratio);
    const g = el.getContext('2d'); g.scale(ratio, ratio); g.font = '11px system-ui'; g.lineWidth = 1; g.fillStyle = C.muted;
    return {g, w, h, el};
  }
  function empty(id, message) { const {g, w, h} = canvas(id); g.fillStyle = C.muted; g.textAlign = 'center'; g.fillText(message, w/2, h/2); }
  function line(id, rows, key, {label = '', color: stroke = C.mint, selected = -1, log = false, ceiling = null} = {}) {
    const points = rows.map((row,i)=>[i,key(row)]), last = points.findLast(ChartTools.valid);
    const values = points.map(p=>p[1]).filter(Number.isFinite);
    ChartTools.legend($(`${id}-legend`),id,[{key:label,text:`${label}: ${sci(last?.[1])}`,color:stroke,disabled:!last}],
      ()=>line(id,rows,key,{label,color:stroke,selected,log,ceiling}));
    if (!values.length) { empty(id, 'No measurements in this interval'); return; }
    const {g, w, h} = canvas(id), left = 60, right = w-20, top = 27, bottom = h-32;
    const transform = v => log ? Math.log10(Math.max(1e-12, v)) : v;
    const minimum = log ? Math.min(-1, ...values.map(transform)) - .5 : Math.min(0, ...values);
    const maximum = ceiling ?? (log ? Math.max(-1, ...values.map(transform)) + .5 : Math.max(1e-9, ...values)*1.15);
    const x = i => left + i*(right-left)/Math.max(1, rows.length-1), y = v => bottom - (transform(v)-minimum)/(maximum-minimum)*(bottom-top);
    g.fillStyle = C.muted; g.fillText(label, left, 12);
    for (let i=0; i<=4; i++) { const yy = bottom-i*(bottom-top)/4, v = minimum+(maximum-minimum)*i/4;
      g.strokeStyle = C.grid; g.beginPath(); g.moveTo(left, yy); g.lineTo(right, yy); g.stroke(); g.textAlign='right';
      g.fillStyle=C.muted; g.fillText(log ? `10^${fmt(v, 1)}` : sci(v), left-10, yy+4);
    }
    if (selected >= 0 && selected < rows.length) { g.fillStyle='#8fd8bf12'; g.fillRect(x(selected)-8, top, 16, bottom-top); }
    const style=ChartTools.focusStyle(ChartTools.selected(id),label);
    g.save(); g.globalAlpha=style.rawOpacity;
    g.strokeStyle=stroke; g.lineWidth=style.rawWidth; g.beginPath(); let started=false;
    rows.forEach((row, i) => { const value=key(row); if (!Number.isFinite(value)) { started=false; return; } if (!started) g.moveTo(x(i), y(value)); else g.lineTo(x(i), y(value)); started=true; }); g.stroke();
    rows.forEach((row,i) => { const value=key(row); if (!Number.isFinite(value)) return; g.beginPath(); g.arc(x(i),y(value),i===selected?4:2.4,0,2*Math.PI); g.fillStyle=stroke; g.fill(); });
    g.globalAlpha=style.emaOpacity;
    ChartTools.canvasEMA(g,points,x,y,stroke,style.emaWidth);
    g.restore();
    g.fillStyle=C.muted; g.textAlign='left'; g.fillText(String(rows[0].batch ?? rows[0].id ?? rows[0].t),left,h-10);
    g.textAlign='right'; g.fillText(String(rows.at(-1).batch ?? rows.at(-1).id ?? rows.at(-1).t),right,h-10);
    const endpoint=ChartTools.latestEMA(points);
    if(ChartTools.selected(id)===label&&endpoint)ChartTools.canvasGuide(g,{left,right,y:y(endpoint[1]),lastX:x(endpoint[0]),color:stroke,text:sci(endpoint[1])});
  }
  function heatmap(id, grid) {
    if (!Array.isArray(grid) || !grid.length) { empty(id, 'Detail not captured for this update'); return; }
    const {g,w,h}=canvas(id), vals=grid.flat().filter(Number.isFinite), positive=vals.filter(v=>v>0);
    const lo=positive.length ? Math.log10(Math.min(...positive)) : -8, hi=positive.length ? Math.max(lo+1,Math.log10(Math.max(...positive))) : -1;
    const left=38, top=14, width=w-68, height=h-49, cellW=width/grid[0].length, cellH=height/grid.length;
    grid.forEach((row,i)=>row.forEach((v,j)=>{g.fillStyle=color(v,lo,hi); g.fillRect(left+j*cellW,top+i*cellH,Math.max(1,cellW-1),Math.max(1,cellH-1));}));
    g.fillStyle=C.muted; g.textAlign='left'; g.fillText('1',left-20,top+10); g.fillText(String(grid.length),left-25,top+height);
    g.fillText(`RMS  ${sci(Math.min(...vals))} — ${sci(Math.max(...vals))}`,left,h-9);
    g.textAlign='right'; g.fillText(`${grid.length} × ${grid[0].length} blocks`,w-25,h-9);
  }
  function freeze() { if (!pinned && latest) pinned=latest; $('live').checked=false; }
  function chooseEvent(i) {
    const s=data(); if (!s?.events?.length) return;
    freeze(); eventIndex=Math.max(0,Math.min(i,s.events.length-1));
    const batch=s.events[eventIndex].batch, found=s.points.findIndex(p=>p.batch===batch);
    pointIndex=found;
    render();
  }
  function choosePoint(i) {
    const s=data(); if(!s?.points?.length) return;
    freeze(); pointIndex=Math.max(0,Math.min(i,s.points.length-1));
    const batch=s.points[pointIndex].batch;
    const found=s.events.map((e,i)=>({e,i})).filter(x=>x.e.batch===batch).at(-1);
    eventIndex=found ? found.i : -1;
    render();
  }
  function populateSelects(s) {
    const modules=Object.keys(moduleNames).filter(name=>s.catalog.some(m=>m.module===name));
    if(!modules.includes(module))module=modules.includes('dynamics')?'dynamics':modules[0];
    if(Array.from($('module').options).map(o=>o.value).join('|')!==modules.join('|')){
      $('module').replaceChildren(...modules.map(name=>{const o=node('option',moduleName(name));o.value=name;return o;}));
    }
    $('module').value=module;
    const names=s.catalog.filter(m=>m.module===module);
    if(!names.some(m=>m.name===selection)) selection=names.find(m=>m.name===s.selected)?.name || names.find(m=>m.shape.length>1)?.name || names[0]?.name || '';
    const current=Array.from($('matrix').options).map(o=>o.value).join('|');
    if(current!==names.map(n=>n.name).join('|')) {
      $('matrix').replaceChildren(...names.map(m=>{const o=node('option',shortName(m.name));o.value=m.name;return o;}));
    }
    $('matrix').value=selection;
    if(!pinned && tab==='pulse') {
      const matching=s.events.map((e,i)=>({e,i})).filter(x=>x.e.matrices?.[selection]?.status==='measured').at(-1);
      if(matching) eventIndex=matching.i;
    }
    if($('probe-session').options.length!==s.session_count && Number.isInteger(s.session_count)) {
      $('probe-session').replaceChildren(...Array.from({length:s.session_count},(_,i)=>{const o=node('option',`Session ${i+1}`);o.value=i;return o;}));
    }
  }
  function renderPulse(s) {
    const event=s.events[eventIndex], item=event?.matrices?.[selection], meta=s.catalog.find(m=>m.name===selection);
    text('matrix-title',shortName(selection));text('matrix-shape',meta?`${meta.shape.join(' × ') || 'Scalar'} · ${fmt(meta.size,0)} parameters`:'');
    const status=event ? item?.status || 'frozen' : 'unavailable';
    text('matrix-state',({measured:'Learning',no_gradient:'No gradient',frozen:'Frozen at this step',unavailable:'Capture unavailable'})[status] || 'No data');
    text('gradient-value',sci(item?.gradient_rms));text('update-value',Number.isFinite(item?.relative_update)?`${sci(item.relative_update*100)} %`:'—');
    text('learning-rate',sci(item?.lr));text('clip-value',event?fmt(event.clip_scale,4):'—');text('alignment-value',Number.isFinite(item?.alignment)?fmt(item.alignment,3):'—');
    const detail=event?.detail?.name===selection ? event.detail : null;
    text('capture-note',!event?'No optimization capture is retained for the selected batch.':detail?'These maps correspond to this matrix and update. Each cell summarizes an entire block.':'Matrix detail was not saved at this point. Request a capture for a future update.');
    const measure=$('measure').value, rows=s.catalog.filter(m=>m.module===module && m.shape.length>1);
    const table=$('activity'), head=node('tr');head.append(node('th','Matrix / update'));
    s.events.forEach((e,i)=>{const th=node('th'),b=node('button',String(e.id));b.dataset.event=i;b.title=`Batch ${e.batch} · ${{predictor:'Predictor',representation:'Representation'}[e.phase]||moduleName(e.phase)}`;th.append(b);head.append(th);});
    table.querySelector('thead').replaceChildren(head);
    const fragment=document.createDocumentFragment();
    rows.forEach(meta=>{const tr=node('tr'),th=node('th'),b=node('button',shortName(meta.name));b.dataset.matrix=meta.name;b.title=meta.name;th.append(b);tr.append(th);tr.dataset.selected=String(meta.name===selection);
      s.events.forEach((e,i)=>{const row=e.matrices[meta.name],td=node('td');td.dataset.event=i;td.dataset.matrix=meta.name;td.dataset.status=row?.status||'frozen';
        const value=row?.[measure];td.title=`${meta.name} · update ${e.id}: ${Number.isFinite(value)?sci(value):row?.status==='no_gradient'?'No gradient':'Frozen'}`;
        if(Number.isFinite(value)) td.style.backgroundColor=color(value);else td.textContent='—';
        td.dataset.selected=String(i===eventIndex&&meta.name===selection);tr.append(td);
      });fragment.append(tr);
    });table.querySelector('tbody').replaceChildren(fragment);
    text('activity-scale','Fixed logarithmic scale: 10⁻⁸ to 10⁻¹. Colors saturate at the extremes; the inspector retains the value.');
    const g=$('heat-scale').getContext('2d');for(let x=0;x<160;x++){g.fillStyle=color(10**(-8+7*x/159));g.fillRect(x,0,1,12);}
    heatmap('gradient-map',detail?.gradient);heatmap('update-map',detail?.update);
    if(detail?.histogram){const {g,w,h}=canvas('update-histogram'),{counts,edges}=detail.histogram,max=Math.max(1,...counts),width=(w-30)/counts.length;
      counts.forEach((v,i)=>{g.fillStyle=(edges[i]+edges[i+1])<0?C.lilac:C.mint;g.fillRect(15+i*width,h-22-v/max*(h-47),Math.max(1,width-2),v/max*(h-47));});
      g.fillStyle=C.muted;g.fillText('Signed ΔW · fixed sample',15,11);g.fillText(sci(edges[0]),15,h-5);g.textAlign='right';g.fillText(sci(edges.at(-1)),w-15,h-5);
    }else empty('update-histogram','Distribution not captured');
    line('matrix-history',s.events,e=>Number.isFinite(e.matrices?.[selection]?.relative_update)?e.matrices[selection].relative_update*100:null,{label:'Weight change (%) · norm on a fixed sample',selected:eventIndex});
  }
  function hasThirdAxis(s) {
    const projections=s?.planes[module]?.projections;
    return projections?.length > 0 && projections.every(p=>Number.isFinite(p.z));
  }
  const displayedDimension = s => dimension===3 && hasThirdAxis(s) ? 3 : 2;
  const movementCoverage = (projection, dims) => dims===3 ? projection?.coverage_3d : projection?.coverage;
  function outsideStep(projection, dims=2) {
    if (!Number.isFinite(projection?.step_norm) || projection.step_norm < 0) return null;
    if (projection.step_norm === 0) return 0;
    const coverage=movementCoverage(projection,dims);
    if (!Number.isFinite(coverage)) return null;
    // Coverage is a ratio of squared norms, so recover length with a square root.
    return projection.step_norm * Math.sqrt(1 - Math.max(0, Math.min(1, coverage)));
  }
  function relativeStep(projection, dims=2) {
    const coverage=movementCoverage(projection,dims);
    if (!(projection?.step_norm>0) || !Number.isFinite(coverage)) return null;
    // Compare lengths, not squared energies. Zero captured motion has an
    // infinite ratio; the display caps its radius but retains this measurement.
    return coverage===0 ? Infinity : Math.sqrt((1-coverage)/coverage);
  }
  const bubbleRatioLimit = 1000;
  const bubbleRadius = ratio => ratio==null ? 3 : 3+27*Math.log1p(Math.min(bubbleRatioLimit,ratio))/Math.log1p(bubbleRatioLimit);
  function sizeScale(hasSteps, dims=2) {
    const space=dims===3?'3D subspace':'2D plane';
    text('path-size-label',`Step length outside / inside the ${space}`);
    $('path-size-scale').hidden=!hasSteps;
    for(const [id,ratio,label] of [['zero',0,'0×'],['ten',10,'10×'],['hundred',100,'100×'],['max',bubbleRatioLimit,'≥1,000×']]){
      text(`path-size-${id}`,label);
      const dot=$(`path-size-${id}-dot`);
      dot.style.width=dot.style.height=`${2*bubbleRadius(ratio)}px`;
    }
    text('path-size-note',hasSteps
      ? 'Bubble radius uses a fixed logarithmic scale for outside / projected step length. Larger bubbles mean more motion is hidden by the projection. Size caps at 1,000× (30 px); the inspector keeps the full ratio. The scale is independent of history and zoom. First, stationary or unavailable steps show position markers only.'
      : 'No relative step measurements available. First, stationary or unavailable steps show position markers only.');
  }
  function lossColor(value, low, high) {
    if (!Number.isFinite(value)) return C.muted;
    const t=high>low?Math.max(0,Math.min(1,(value-low)/(high-low))):.5;
    return '#'+[143,216,191].map((v,i)=>Math.round(v+([233,184,108][i]-v)*t).toString(16).padStart(2,'0')).join('');
  }
  function lossScale(points) {
    const values=points.map(pointLoss).filter(Number.isFinite), unit=plannerModule()?'loss':'NLL';
    const low=values.length?Math.min(...values):null, high=values.length?Math.max(...values):null;
    text('path-loss-low',values.length?`Lower ${unit} · ${sci(low)}`:`${lossName()} unavailable`);
    text('path-loss-high',values.length?(high>low?`Higher ${unit} · ${sci(high)}`:`Constant ${unit}`):'');
    $('path-loss-scale').style.background=high>low
      ? `linear-gradient(to right, ${lossColor(low,low,high)}, ${lossColor(high,low,high)})`
      : lossColor(low,low,high);
    $('path-loss-scale').title=values.length?`${lossName()}: ${low} to ${high}`:`${lossName()} unavailable`;
    return value=>lossColor(value,low,high);
  }
  function rotate(x,y,z) {
    const a=x*Math.cos(camera.yaw)+z*Math.sin(camera.yaw), b=-x*Math.sin(camera.yaw)+z*Math.cos(camera.yaw);
    const c=y*Math.cos(camera.pitch)-b*Math.sin(camera.pitch), depth=y*Math.sin(camera.pitch)+b*Math.cos(camera.pitch);
    return [a*Math.cos(camera.roll)-c*Math.sin(camera.roll),a*Math.sin(camera.roll)+c*Math.cos(camera.roll),depth];
  }
  function trajectory(s) {
    const plane=s.planes[module], projections=plane?.projections || [], points=s.points;
    const dims=displayedDimension(s), pointColor=lossScale(points);
    const ratios=projections.map((p,i)=>i?relativeStep(p,dims):null);
    sizeScale(ratios.some(r=>r!==null),dims);
    text('camera-zoom',`${fmt(camera.zoom*100,0)} %`);
    $('zoom-out').disabled=camera.zoom<=.5;$('zoom-in').disabled=camera.zoom>=4;
    if(!projections.length) { empty('parameter-path','Calibrating the plane with four measured batches…');drawHits=[];return; }
    const {g,w,h}=canvas('parameter-path');
    const coordinates=projections.map(p=>[p.x,p.y,dims===3?p.z:0]);
    const center=[0,1,2].map(axis=>{const v=coordinates.map(p=>p[axis]);return Math.min(...v)/2+Math.max(...v)/2;});
    const centered=coordinates.map(p=>p.map((v,i)=>v-center[i]));
    const extent=Math.max(...centered.map(p=>Math.hypot(...p))) || 1;
    const left=16,right=w-16,top=16,bottom=h-34;
    // One scale for every weight coordinate; orthographic camera rotations
    // preserve distance before the intentional projection onto the screen.
    const scale=Math.min(right-left,bottom-top)*.43*camera.zoom;
    function project(x,y,z){const p=dims===3?rotate(x,y,z):[x,y,0];
      return [(left+right)/2+p[0]*scale,(top+bottom)/2-p[1]*scale,p[2]]; }
    function segment(a,b,color=C.grid){g.strokeStyle=color;g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(b[0],b[1]);g.stroke();}
    g.save();g.beginPath();g.rect(left,top,right-left,bottom-top);g.clip();
    for(let i=0;i<=4;i++){const k=-1+i*.5;segment(project(-1,k,0),project(1,k,0));segment(project(k,-1,0),project(k,1,0));}
    [[1,0,0],[0,1,0],[0,0,1]].slice(0,dims).forEach((axis,i)=>{
      const end=project(...axis);segment(project(0,0,0),end,C.muted);
      g.fillStyle=C.text;g.textAlign='center';g.fillText(`PC${i+1}`,end[0],end[1]-8);
    });
    // Radius encodes a fixed log scale of the outside/inside length ratio,
    // independent of absolute update size, window extrema and camera depth/zoom.
    drawHits=centered.map((p,i)=>{const position=project(...p.map(v=>v/extent));return {
      position:position.slice(0,2),depth:position[2],
      visible:position[0]>=left&&position[0]<=right&&position[1]>=top&&position[1]<=bottom,
      radius:bubbleRadius(ratios[i]),
    };});
    g.lineWidth=2;
    drawHits.forEach((hit,i)=>{if(i)segment(drawHits[i-1].position,hit.position,C.muted);});
    function bubble(hit,i) {
      const p=hit.position, radius=hit.radius, color=pointColor(pointLoss(points[i]));
      if(radius>3){g.fillStyle=`${color}24`;g.strokeStyle=`${color}70`;g.lineWidth=1;g.beginPath();g.arc(...p,radius,0,2*Math.PI);g.fill();g.stroke();}
      g.fillStyle=color;g.beginPath();g.arc(...p,3,0,2*Math.PI);g.fill();
      if(i===pointIndex){g.lineWidth=2;g.strokeStyle=C.text;g.beginPath();g.arc(...p,radius+4,0,2*Math.PI);g.stroke();g.textAlign='left';g.fillStyle=C.text;g.fillText(`Batch ${points[i].batch}`,Math.min(w-100,p[0]+radius+8),Math.max(18,p[1]-radius-8));}
    }
    drawHits.map((hit,i)=>({hit,i})).sort((a,b)=>a.hit.depth-b.hit.depth).forEach(({hit,i})=>{if(i!==pointIndex)bubble(hit,i);});
    if(drawHits[pointIndex])bubble(drawHits[pointIndex],pointIndex);
    g.restore();g.fillStyle=C.muted;g.textAlign='center';g.fillText(`PC1 / PC2${dims===3?' / PC3':''} · common parameter scale`,w/2,h-9);
  }
  function renderPath(s) {
    const plane=s.planes[module], p=s.points[pointIndex], projection=plane?.projections?.[pointIndex],previous=s.points[pointIndex-1];
    const dims=displayedDimension(s),space=dims===3?'3D subspace':'2D plane',coverage=movementCoverage(projection,dims);
    const legacy=plane?.projections?.length>0&&!hasThirdAxis(s);
    $('view-2d').setAttribute('aria-pressed',String(dims===2));$('view-3d').setAttribute('aria-pressed',String(dims===3));
    $('view-3d').disabled=!hasThirdAxis(s);
    text('basis-note',plane?.basis_id?`Fixed basis ${plane.basis_id} · ${moduleName(module)} · ${s.points.length} measured batches · module-wide fixed sample.${legacy?' 2D fallback: this capture has no PC3. Follow training for updated telemetry.':''}`:'Waiting for four batches to establish the PCA basis.');
    const loss=pointLoss(p), previousLoss=pointLoss(previous);
    text('point-title',p?`Batch ${p.batch}`:'Waiting for batches');text('reference-value',Number.isFinite(loss)?String(loss):'—');
    $('reference-value').title=Number.isFinite(loss)?String(loss):`${lossName()} unavailable`;
    const change=Number.isFinite(loss)&&Number.isFinite(previousLoss)?loss-previousLoss:null;
    text('loss-change',Number.isFinite(change)?`${change>0?'+':''}${fmt(change,4)}`:'—');
    text('coverage-label',`Squared movement captured by the ${space}`);text('outside-step-label',`Step outside the ${space}`);
    $('coverage').setAttribute('aria-label',`Fraction of squared step length captured by the ${space}`);
    $('coverage').value=coverage ?? 0; text('coverage-value',Number.isFinite(coverage)?`${fmt(coverage*100,1)} %`:'—');
    text('outside-plane',projection&&pointIndex===0?'First retained batch: no previous step in this window.'
      : Number.isFinite(coverage)?`${fmt((1-coverage)*100,1)} % of squared step length lies outside the ${space}. Measured on sampled coefficients.`
      : projection?.step_norm===0?'No measurable change between these sampled weights.':'No measurable displacement to calculate coverage.');
    text('turn-value',Number.isFinite(projection?.angle)?`${fmt(projection.angle,1)}°`:'—');text('step-value',pointIndex>0?sci(projection?.step_norm):'—');
    text('outside-step-value',pointIndex>0?sci(outsideStep(projection,dims)):'—');
    const ratio=pointIndex>0?relativeStep(projection,dims):null;
    text('relative-step-value',ratio==null?'—':ratio===Infinity?'∞ · no projected step'
      : `${fmt(ratio)} ×${ratio>bubbleRatioLimit?' · size capped at 1,000×':''}`);
    trajectory(s);line('reference-history',s.points,pointLoss,{label:plannerModule()?`${lossName()} · batch mean`:s.reference_source==='training_pre_update'?'Visual NLL · batch before learning':'Visual NLL · fixed reference',color:C.lilac,selected:pointIndex});
    line('turn-history',s.points,(p)=>plane?.projections?.[s.points.indexOf(p)]?.angle,{label:'Angle between updates (°)',selected:pointIndex,ceiling:180});
    $('point-scrubber').max=Math.max(0,s.points.length-1);$('point-scrubber').value=Math.max(0,pointIndex);
  }
  function renderMemory(s) {
    const p=s.probe;
    if(!p || p.status!=='ready'){
      text('probe-description',p?.reason || 'Request a probe to measure the temporal signal of a loss.');
      for(const id of ['probe-loss','probe-cleared','probe-game','probe-batch','probe-window','probe-cost'])text(id,'—');
      text('probe-comparison','');empty('memory-window','No probe recorded');empty('memory-gradient','No probe recorded');
      ChartTools.legend($('memory-gradient-legend'),'memory-gradient',[],()=>{});return;
    }
    text('probe-description',`Probe from batch ${p.batch} · session ${p.session+1} · ${p.game}. Derivative of the visual loss at step ${p.stop-1} with respect to earlier memory states.`);
    {const {g,w,h}=canvas('memory-window'),left=22,width=w-44,y=37,x=t=>left+t/p.session_steps*width;
      g.fillStyle=C.dark;g.fillRect(left,y,width,22);g.fillStyle='#71618e';g.fillRect(x(p.context_start),y,x(p.start)-x(p.context_start),22);g.fillStyle=C.mint;g.fillRect(x(p.start),y,x(p.stop)-x(p.start),22);
      g.fillStyle=C.muted;g.fillText('Complete session',left,17);g.fillText('0',left,y+43);g.textAlign='right';g.fillText(String(p.session_steps),w-22,y+43);
      g.textAlign='center';g.fillStyle=C.lilac;g.fillText(`Truncation at ${p.context_start}`,Math.max(75,Math.min(w-90,x(p.context_start))),y-8);
    }
    const {g,w,h}=canvas('memory-gradient'),left=64,right=w-30,top=28,bottom=h-45;
    const start=p.context_start,stop=p.stop-1,x=t=>left+(t-start)*(right-left)/Math.max(1,stop-start);
    const positive=p.steps.map(v=>v.rms).filter(v=>Number.isFinite(v)&&v>0),lo=positive.length?Math.floor(Math.log10(Math.min(...positive)))-1:-8,hi=positive.length?Math.ceil(Math.log10(Math.max(...positive)))+1:0;
    const y=v=>bottom-(Math.log10(Math.max(10**lo,v))-lo)/(hi-lo)*(bottom-top);
    g.fillStyle='#bcaafa10';g.fillRect(left,top,Math.max(0,x(p.start)-left),bottom-top);g.fillStyle='#8fd8bf0a';g.fillRect(x(p.start),top,right-x(p.start),bottom-top);
    g.setLineDash([4,4]);g.strokeStyle=C.lilac;g.beginPath();g.moveTo(left,top);g.lineTo(left,bottom);g.stroke();
    for(const t of p.boundaries){g.strokeStyle=C.amber;g.beginPath();g.moveTo(x(t),top);g.lineTo(x(t),bottom);g.stroke();}g.setLineDash([]);
    for(let i=0;i<=4;i++){const exponent=lo+(hi-lo)*i/4,yy=bottom-i*(bottom-top)/4;g.strokeStyle=C.grid;g.beginPath();g.moveTo(left,yy);g.lineTo(right,yy);g.stroke();g.textAlign='right';g.fillStyle=C.muted;g.fillText(`10^${fmt(exponent,1)}`,left-10,yy+4);}
    const points=p.steps.map(row=>[row.t,row.rms]),last=points.findLast(ChartTools.valid);
    ChartTools.legend($('memory-gradient-legend'),'memory-gradient',[{key:'rms',text:`RMS ∂L / ∂h: ${sci(last?.[1])}`,color:C.mint,disabled:!last}],()=>renderMemory(s));
    const style=ChartTools.focusStyle(ChartTools.selected('memory-gradient'),'rms');
    g.save();g.globalAlpha=style.rawOpacity;
    g.lineWidth=style.rawWidth;g.strokeStyle=C.mint;g.beginPath();let active=false;
    p.steps.forEach(row=>{if(!Number.isFinite(row.rms)){active=false;return;}if(active)g.lineTo(x(row.t),y(row.rms));else g.moveTo(x(row.t),y(row.rms));active=true;});g.stroke();
    p.steps.forEach(row=>{if(!Number.isFinite(row.rms))return;g.fillStyle=row.rms===0?C.amber:C.mint;g.beginPath();g.arc(x(row.t),y(row.rms),3,0,2*Math.PI);g.fill();});
    g.globalAlpha=style.emaOpacity;
    ChartTools.canvasEMA(g,points,x,y,C.mint,style.emaWidth);
    g.restore();
    g.fillStyle=C.muted;g.textAlign='left';g.fillText(`Truncation · t=${start}`,left,bottom+23);g.textAlign='right';g.fillText(`Loss · t=${stop}`,right,bottom+23);g.textAlign='left';g.fillText('Final RMS ∂L / ∂h · log scale · zeros in amber at the floor',left,14);
    const endpoint=ChartTools.latestEMA(points);
    if(ChartTools.selected('memory-gradient')&&endpoint)ChartTools.canvasGuide(g,{left,right,y:y(endpoint[1]),lastX:x(endpoint[0]),color:C.mint,text:sci(endpoint[1])});
    text('probe-loss',fmt(p.loss,4));text('probe-cleared',fmt(p.cleared_loss,4));
    text('probe-comparison',p.loss<p.cleared_loss?'History reduces the loss at this step.':p.loss>p.cleared_loss?'Clearing history reduces the loss at this step.':'No loss difference in this probe.');
    text('probe-game',`${p.game} / ${p.session+1}`);text('probe-batch',`${p.batch} / t = ${stop}`);text('probe-window',`${p.context_start}–${stop} · ${p.start-p.context_start} + ${p.stop-p.start} steps`);text('probe-cost',`${fmt(p.seconds,3)} s`);
  }
  function render() {
    const s=data();if(!s || s.status!=='ready') return;
    eventIndex=Math.min(eventIndex,Math.max(0,s.events.length-1));pointIndex=Math.min(pointIndex,Math.max(0,s.points.length-1));
    populateSelects(s);
    text('reference-label',plannerModule()?`${lossName()} · batch mean`:s.reference_source==='training_pre_update'?'Visual NLL before learning':'Reference visual NLL');
    text('path-loss-label',`Point color · ${lossName()}`);
    text('path-loss-unavailable',plannerModule()?'Unavailable loss':'Unavailable NLL');
    $('parameter-path').setAttribute('aria-label',`Weight PCA trajectory: PC1, PC2 and PC3; color shows ${lossName()} and bubble radius shows outside versus projected step length`);
    $('reference-history').setAttribute('aria-label',`${lossName()} by batch`);
    const lossSource=plannerModule()
      ? `${moduleName(module)} training loss, averaged across this batch's head updates. Weights are measured after the batch. Batches without head updates have no loss value.`
      :s.reference_source==='training_pre_update'
      ? 'New data from each batch: visual NLL is measured before learning; weights at each point are from after the update.'
      : 'Visual NLL uses the same real reference observations across batches.';
    text('reference-source',lossSource);text('path-loss-note',`${lossSource} Colors span the retained window; they do not describe a loss landscape.`);
    text('sample-note',`Up to ${s.sample_per_tensor} coefficients / tensor · fixed sample`);
    text('optimization-status',pinned?'Pinned instant':s.running?'Measuring training':'Training stopped');
    text('optimization-summary',`${moduleName(module)} · batch ${s.batch} · ${s.catalog.length} tensors · capture every ${s.every} predictor/planner updates`);
    const enabled=latest?.control?.enabled === true;
    for(const id of ['capture-matrix','capture-probe','recalibrate'])$(id).disabled=!enabled;
    $('event-scrubber').max=Math.max(0,s.events.length-1);$('event-scrubber').value=Math.max(0,eventIndex);
    const event=s.events[eventIndex], point=s.points[pointIndex];
    const label=event?`Update ${event.id} · batch ${event.batch}`:point?`Batch ${point.batch} · capture unavailable`:'No captures';
    text('event-label',label);$('event-scrubber').setAttribute('aria-valuetext',label);
    if(tab==='pulse')renderPulse(s);if(tab==='path')renderPath(s);if(tab==='memory')renderMemory(s);
  }
  function setTab(name) {
    tab=name;for(const id of ['pulse','path','memory']){const active=id===name;$(id).hidden=!active||!data();$(`tab-${id}`).setAttribute('aria-selected',String(active));$(`tab-${id}`).tabIndex=active?0:-1;}render();
  }
  async function request(payload) {
    try { const response=await fetch('api/optimization',{method:'POST',headers:{'Content-Type':'application/json','X-Optimization-Token':latest?.control?.token || ''},body:JSON.stringify(payload)});const result=await response.json();if(!response.ok)throw Error(result.error || 'Could not submit request');requestId=result.id;text('request-status',result.message);$('optimization-notice').hidden=false;text('optimization-notice',result.message);
      pinned=null;$('live').checked=true;
    } catch(error){$('optimization-notice').hidden=false;text('optimization-notice',error.message);}
  }
  function stopPlayback(){clearInterval(playing);playing=null;text('playback','Play');}
  function clearData(status,message) {
    latest=null;pinned=null;revision=null;requestId=null;eventIndex=-1;pointIndex=-1;drawHits=[];drag=null;stopPlayback();sizeScale(null);
    $('live').checked=true;text('optimization-status',status);text('optimization-summary',message);
    $('optimization-notice').hidden=false;text('optimization-notice',message);text('event-label','No captures');
    text('request-status','');text('sample-note','No measurements');$('matrix').replaceChildren();
    for(const id of ['capture-matrix','capture-probe','recalibrate'])$(id).disabled=true;
    for(const id of ['pulse','path','memory'])$(id).hidden=true;
    for(const id of ['gradient-value','update-value','learning-rate','clip-value','alignment-value','matrix-state',
      'reference-value','loss-change','coverage-value','turn-value','step-value','outside-step-value','relative-step-value','probe-loss','probe-cleared',
      'probe-game','probe-batch','probe-window','probe-cost'])text(id,'—');
  }
  $('module').addEventListener('change',e=>{module=e.target.value;render();});
  $('matrix').addEventListener('change',e=>{selection=e.target.value;render();});
  $('measure').addEventListener('change',render);
  $('live').addEventListener('change',()=>{stopPlayback();if($('live').checked){pinned=null;eventIndex=Math.max(0,(latest?.events?.length||1)-1);pointIndex=Math.max(0,(latest?.points?.length||1)-1);}else freeze();render();});
  $('activity').addEventListener('click',e=>{const target=e.target.closest('[data-event], [data-matrix]');if(!target)return;if(target.dataset.matrix)selection=target.dataset.matrix;if(target.dataset.event!=null)chooseEvent(Number(target.dataset.event));else render();});
  $('event-scrubber').addEventListener('input',e=>chooseEvent(Number(e.target.value)));
  $('point-scrubber').addEventListener('input',e=>choosePoint(Number(e.target.value)));
  $('capture-matrix').addEventListener('click',()=>request({operation:'matrix',matrix:selection}));
  $('capture-probe').addEventListener('click',()=>request({operation:'probe',session:Number($('probe-session').value)}));
  $('recalibrate').addEventListener('click',()=>request({operation:'recalibrate'}));
  $('view-2d').addEventListener('click',()=>{dimension=2;render();});
  $('view-3d').addEventListener('click',()=>{dimension=3;render();});
  ['pulse','path','memory'].forEach((id,i)=>{const button=$(`tab-${id}`);button.addEventListener('click',()=>setTab(id));button.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const names=['pulse','path','memory'];const j=e.key==='Home'?0:e.key==='End'?2:(i+(e.key==='ArrowRight'?1:2))%3;setTab(names[j]);$(`tab-${names[j]}`).focus();});});
  $('playback').addEventListener('click',()=>{if(playing){stopPlayback();return;}freeze();if(!data()?.events?.length)return;text('playback','Pause');if(eventIndex>=data().events.length-1)eventIndex=-1;playing=setInterval(()=>{if(eventIndex>=data().events.length-1){stopPlayback();return;}chooseEvent(eventIndex+1);},900);});
  let drag=null;
  const pathCanvas=$('parameter-path');
  const redrawCamera=()=>{if(data())trajectory(data());};
  const zoomBy=factor=>{camera.zoom=Math.max(.5,Math.min(4,camera.zoom*factor));redrawCamera();};
  const resetCamera=()=>{camera=defaultCamera();redrawCamera();};
  $('zoom-out').addEventListener('click',()=>zoomBy(1/1.2));
  $('zoom-in').addEventListener('click',()=>zoomBy(1.2));
  $('reset-view').addEventListener('click',resetCamera);
  pathCanvas.addEventListener('pointerdown',e=>{
    if(e.button!==0 || e.isPrimary===false || !data())return;
    drag={id:e.pointerId,x:e.clientX,y:e.clientY,camera:{...camera},roll:e.shiftKey,moved:false};
    pathCanvas.focus();pathCanvas.setPointerCapture(e.pointerId);
  });
  pathCanvas.addEventListener('pointermove',e=>{
    if(!drag || drag.id!==e.pointerId)return;
    const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
    drag.moved ||= Math.hypot(dx,dy)>=5;
    if(!drag.moved || !data() || displayedDimension(data())!==3)return;
    if(drag.roll)camera.roll=drag.camera.roll+(dx+dy)/180;
    else {camera.yaw=drag.camera.yaw+dx/180;camera.pitch=drag.camera.pitch+dy/180;}
    redrawCamera();
  });
  pathCanvas.addEventListener('pointerup',e=>{
    if(!drag || drag.id!==e.pointerId)return;
    const click=!drag.moved&&Math.hypot(e.clientX-drag.x,e.clientY-drag.y)<5;
    drag=null;
    if(pathCanvas.hasPointerCapture(e.pointerId))pathCanvas.releasePointerCapture(e.pointerId);
    if(!click)return;
    const box=pathCanvas.getBoundingClientRect(),x=e.clientX-box.left,y=e.clientY-box.top;
    if(x<16||x>box.width-16||y<16||y>box.height-34)return;
    let closest=-1,distance=Infinity;
    drawHits.forEach((hit,i)=>{const d=Math.hypot(hit.position[0]-x,hit.position[1]-y);
      if(hit.visible&&d<=Math.max(24,hit.radius+4)&&d<distance){distance=d;closest=i;}});
    if(closest>=0)choosePoint(closest);
  });
  for(const event of ['pointercancel','lostpointercapture'])pathCanvas.addEventListener(event,()=>{drag=null;});
  pathCanvas.addEventListener('wheel',e=>{
    e.preventDefault();
    const unit=e.deltaMode===1?16:e.deltaMode===2?pathCanvas.getBoundingClientRect().height:1;
    zoomBy(Math.exp(Math.max(-1,Math.min(1,-e.deltaY*unit*.002))));
  },{passive:false});
  pathCanvas.addEventListener('keydown',e=>{
    if(e.ctrlKey||e.metaKey||e.altKey)return;
    const key=e.key.toLowerCase(),rotation={a:['yaw',-.12],d:['yaw',.12],w:['pitch',-.12],s:['pitch',.12],q:['roll',-.12],e:['roll',.12]}[key];
    if(key==='arrowleft'||key==='arrowright'){e.preventDefault();choosePoint(pointIndex+(key==='arrowright'?1:-1));}
    else if(key==='+'||key==='='||key==='-'){e.preventDefault();zoomBy(key==='-'?1/1.2:1.2);}
    else if(key==='0'||key==='home'){e.preventDefault();resetCamera();}
    else if(rotation&&data()&&displayedDimension(data())===3){e.preventDefault();camera[rotation[0]]+=rotation[1];redrawCamera();}
  });
  window.addEventListener('resize',render);
  async function refresh(){
    let received=false;
    try{const response=await fetch('api/optimization',{cache:'no-store',headers:revision?{'If-None-Match':revision}:{}});if(response.status===304){if(data())text('optimization-status',pinned?'Pinned instant':latest.running?'Measuring training':'Training stopped');return;}if(!response.ok)throw Error('Could not read diagnostics');received=true;const next=await response.json();
      if(next?.status==='waiting' || next?.status==='unavailable'){
        if(typeof next.message!=='string')throw Error('Invalid diagnostic message');
        clearData(next.status==='unavailable'?'Not instrumented':'Waiting for captures',next.message);
      }
      else {if(next?.status!=='ready' || !validSnapshot(next))throw Error('Invalid diagnostic snapshot');
        const changed=!latest||latest.identity!==next.identity||latest.updated_at!==next.updated_at||latest.running!==next.running||latest.control?.enabled!==next.control?.enabled;if(latest&&latest.identity!==next.identity){pinned=null;$('live').checked=true;}latest=next;for(const id of ['pulse','path','memory'])$(id).hidden=id!==tab;
        if(requestId&&next.last_request===requestId){text('request-status','Capture completed.');requestId=null;}
        if(!requestId){$('optimization-notice').hidden=next.age_seconds<180;text('optimization-notice','The latest capture is over three minutes old. Check training status.');}
        if(!pinned){eventIndex=next.events.length-1;pointIndex=next.points.length-1;}if(changed)render();
        revision=response.headers?.get('ETag') || null;
      }
    }catch{if(received)clearData('Invalid data','The optimization record is invalid. Waiting for a compatible capture.');else text('optimization-status','Disconnected');}
    finally{setTimeout(refresh,2000);}
  }
  refresh();
})();
