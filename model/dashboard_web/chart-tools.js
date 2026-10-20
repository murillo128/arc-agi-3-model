'use strict';
// Shared presentation only: never changes recorded measurements or phase gates.
globalThis.ChartTools = (() => {
  const alpha = .1, hint = 'Thin line: EMA (α = 0.1). Select a legend label to pin the latest raw value.';
  const states = new Map();
  const valid = p => Number.isFinite(p?.[0]) && Number.isFinite(p?.[1]);
  function ema(points) {
    let previous = null;
    return points.map(p => {
      if (!valid(p)) { previous = null; return [p?.[0], null]; }
      previous = previous === null ? p[1] : alpha*p[1] + (1-alpha)*previous;
      return [p[0], previous];
    });
  }
  function svgNode(tag, attrs, text='') {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    node.textContent = text; return node;
  }
  function svgEMA(points, x, y, color, opacity=1) {
    const nodes = []; let segment = [];
    const flush = () => {
      if (segment.length) {
        const node = svgNode('polyline', {points: segment.join(' '), fill: 'none', stroke: color,
          'stroke-width': .75, opacity, class: 'series-ema', 'pointer-events': 'none'});
        node.append(svgNode('title', {}, 'EMA (α = 0.1)')); nodes.push(node);
      }
      segment = [];
    };
    for (const p of ema(points)) {
      if (!valid(p)) { flush(); continue; }
      segment.push(`${x(p[0])},${y(p[1])}`);
    }
    flush(); return nodes;
  }
  function canvasEMA(ctx, points, x, y, color) {
    ctx.strokeStyle = color; ctx.lineWidth = .75; ctx.beginPath();
    let started = false;
    for (const p of ema(points)) {
      if (!valid(p)) { started = false; continue; }
      if (started) ctx.lineTo(x(p[0]), y(p[1])); else ctx.moveTo(x(p[0]), y(p[1]));
      started = true;
    }
    ctx.stroke();
  }
  function svgGuide(layer, {left, right, y, color, text, lastX}) {
    layer.replaceChildren();
    const width = Math.max(30, text.length*6.5 + 12);
    layer.append(svgNode('line', {x1:left, x2:right, y1:y, y2:y, stroke:color,
      'stroke-width':1, 'stroke-dasharray':'2 4', class:'last-value-line'}),
    svgNode('circle', {cx:lastX, cy:y, r:3.5, fill:'#172122', stroke:color, 'stroke-width':1.5}),
    svgNode('rect', {x:2, y:y-10, width, height:20, rx:4, fill:'#172122', stroke:color,
      'stroke-width':1, class:'last-value-badge'}),
    svgNode('text', {x:8, y:y+3.5, fill:color, 'font-size':11}, text));
  }
  function canvasGuide(ctx, {left, right, y, color, text, lastX}) {
    ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([2,4]);
    ctx.beginPath(); ctx.moveTo(left,y); ctx.lineTo(right,y); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle='#172122'; ctx.beginPath(); ctx.arc(lastX,y,3.5,0,Math.PI*2); ctx.fill(); ctx.stroke();
    ctx.font='11px system-ui'; ctx.textAlign='left';
    const width=Math.max(30, text.length*6.5+12);
    ctx.fillRect(2,y-10,width,20); ctx.strokeRect(2,y-10,width,20);
    ctx.fillStyle=color; ctx.fillText(text,8,y+3.5);
  }
  function legend(host, id, items, redraw) {
    if (!states.has(id)) states.set(id, {selected:null});
    const state=states.get(id); state.redraw=redraw;
    const signature=JSON.stringify(items.map(i=>[i.key,i.passive===true]));
    if (state.signature!==signature) {
      state.signature=signature;
      host.replaceChildren(...items.map(item=>{
        const node=document.createElement(item.passive?'span':'button');
        if (!item.passive) {
          node.type='button'; node.setAttribute('aria-controls',id);
          node.addEventListener('click',()=>{
            state.selected=state.selected===item.key?null:item.key;
            state.sync(); state.redraw();
          });
        }
        return node;
      }));
    }
    if (!items.some(i=>i.key===state.selected&&!i.disabled&&!i.passive)) state.selected=null;
    state.sync=()=>items.forEach((item,i)=>{
      const node=host.children[i]; node.textContent=item.text; node.style.color=item.color;
      node.className=item.className||''; node.title=[item.title,!item.passive&&hint].filter(Boolean).join(' · ');
      if (!item.passive) {
        node.disabled=!!item.disabled;
        node.setAttribute('aria-pressed',String(state.selected===item.key));
        node.setAttribute('data-highlighted',String(state.selected===item.key));
      }
    });
    state.sync();
  }
  return {alpha, hint, ema, valid, svgEMA, canvasEMA, svgGuide, canvasGuide, legend,
    selected:id=>states.get(id)?.selected??null};
})();
