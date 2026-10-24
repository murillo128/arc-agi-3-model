'use strict';
// Window statistics use recorded counts, never the displayed EMA or sampled points.
globalThis.SequenceTrends = (() => {
  const finite = Number.isFinite;
  const regime = row => JSON.stringify([row.measurement_source ?? 'fixed_evaluation', row.phase,
    row.exploration?.policy ?? 'uniform', row.exploration?.identity,
    ...['predictor', 'encoder', 'decoder'].map(k => row.learning_rates?.[k])]);
  const connected = (a, b) => a && b.batch === a.batch + 1 && regime(a) === regime(b);
  function suffix(rows) {
    let start = rows.length - 1;
    while (start > 0 && connected(rows[start - 1], rows[start])) start--;
    return rows.slice(Math.max(0, start));
  }
  function accumulator(games) {
    const totals = new Map(games.map(game => [game, {wrong:0, union:0, transitions:0, missingSamples:0, batches:0, invalid:0}]));
    function add(row, sign=1) {
      for (const [game, total] of totals) {
        const q = row.quality?.[game];
        if (!q) continue; // This game was not in this batch.
        if (!finite(q.wrong) || !finite(q.union) || q.wrong < 0 || q.union < q.wrong) {
          total.invalid += sign; continue;
        }
        total.wrong += sign*q.wrong; total.union += sign*q.union; total.batches += sign;
        if(finite(q.transitions)&&q.transitions>=0)total.transitions+=sign*q.transitions;
        else total.missingSamples+=sign;
      }
    }
    function result(detail=false) {
      const errors = [], byGame = [];
      let wrong = 0, union = 0;
      for (const [game, t] of totals) {
        const error = t.union > 0 && !t.invalid ? 100*t.wrong/t.union : null;
        if (error !== null) { errors.push(error); wrong += t.wrong; union += t.union; }
        if (detail) byGame.push({game, error,
          transitions:t.batches&&!t.missingSamples&&!t.invalid?t.transitions:null, batches:t.batches});
      }
      const complete = errors.length === games.length && games.length > 0;
      return {balanced:complete ? errors.reduce((a,b)=>a+b,0)/games.length : null,
        pooled:complete ? 100*wrong/union : null, coverage:errors.length, games:byGame};
    }
    return {add, result};
  }
  function windowSummary(rows, games, size) {
    if (rows.length !== size) return null;
    const a = accumulator(games);
    rows.forEach(row => a.add(row));
    return {...a.result(true), start:rows[0].batch, end:rows.at(-1).batch};
  }
  function compare(rows, games, size) {
    const tail = suffix(rows);
    return {available:tail.length, current:windowSummary(tail.slice(-size), games, size),
      previous:tail.length >= 2*size ? windowSummary(tail.slice(-2*size, -size), games, size) : null};
  }
  function rolling(rows, games, size) {
    let a = accumulator(games), start = 0;
    return rows.map((row, index) => {
      if (index && !connected(rows[index-1], row)) { a = accumulator(games); start = index; }
      a.add(row);
      if (index-start >= size) a.add(rows[index-size], -1);
      const value = index-start+1 >= size ? a.result() : {balanced:null, pooled:null, coverage:0};
      return {batch:row.batch, actions:row.actions, measurement_source:row.measurement_source,
        phase:row.phase, exploration:row.exploration, learning_rates:row.learning_rates, ...value};
    });
  }
  return {regime, connected, suffix, compare, rolling};
})();
