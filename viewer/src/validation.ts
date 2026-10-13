import { fail } from './errors';
import { WireFloat, isMap, type WireValue, type WireMap } from './messagepack';

type Input = WireValue | undefined;
function map(value: Input, path: string, required: string[] = []): WireMap {
  if (!isMap(value)) fail(path, 'expected map');
  for (const key of required) if (!Object.hasOwn(value, key) || value[key] === null) fail(`${path}.${key}`, 'missing required field');
  return value;
}
function array(value: Input, path: string): WireValue[] {
  if (!Array.isArray(value)) fail(path, 'expected array');
  return value;
}
function integer(value: Input, path: string, min = -Number.MAX_SAFE_INTEGER, max = Number.MAX_SAFE_INTEGER): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < min || value > max) fail(path, `expected integer in [${min}, ${max}]`);
  return value;
}
function number(value: Input, path: string, min = -Infinity): number {
  const n = value instanceof WireFloat ? value.value : value;
  if (typeof n !== 'number' || !Number.isFinite(n) || n < min) fail(path, 'expected finite number in range');
  return n;
}
function text(value: Input, path: string, identifier = false): string {
  if (typeof value !== 'string' || (identifier && value.length === 0)) fail(path, identifier ? 'expected nonempty STR' : 'expected STR');
  return value;
}
function choice(value: Input, path: string, choices: string[]): string {
  const s = text(value, path);
  if (!choices.includes(s)) fail(path, `expected ${choices.join(', ')}`);
  return s;
}
function jsonConfig(value: WireValue, path: string): void {
  if (value instanceof Uint8Array) fail(path, 'BIN is prohibited in JSON config');
  if (Array.isArray(value)) value.forEach(v => jsonConfig(v, path));
  else if (isMap(value)) Object.values(value).forEach(v => jsonConfig(v, path));
}

/** DataView reads unaligned, explicitly little-endian tensor elements. */
export function tensorElement(dtype: 'u8' | 'f16' | 'f32', data: Uint8Array, index: number): number {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  if (dtype === 'u8') return view.getUint8(index);
  if (dtype === 'f32') return view.getFloat32(index * 4, true);
  const bits = view.getUint16(index * 2, true);
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >> 10) & 31;
  const mantissa = bits & 1023;
  if (exp === 31) return mantissa ? NaN : sign * Infinity;
  return sign * (exp === 0 ? mantissa * 2 ** -24 : (1 + mantissa / 1024) * 2 ** (exp - 15));
}

function tensor(value: Input, path: string, cap: number, rank?: number, observed = false): WireMap {
  const t = map(value, path, ['dtype', 'shape', 'data']);
  const dtype = choice(t.dtype, `${path}.dtype`, observed ? ['u8'] : ['u8', 'f16', 'f32']) as 'u8' | 'f16' | 'f32';
  const shape = array(t.shape, `${path}.shape`);
  if (shape.length < 1 || shape.length > 8 || (rank !== undefined && shape.length !== rank)) fail(`${path}.shape`, 'invalid tensor rank');
  const dims = shape.map(d => integer(d, `${path}.shape dimension`, 0, cap));
  if (!(t.data instanceof Uint8Array)) fail(`${path}.data`, 'expected BIN');
  if (observed && dims.length === 3 && dims.every(d => d === 0)) {
    if (t.data.length !== 0) fail(`${path}.data`, 'empty observation requires empty BIN');
    return t;
  }
  const itemSize = dtype === 'u8' ? 1 : dtype === 'f16' ? 2 : 4;
  let bytes = itemSize;
  for (const dim of dims) {
    if (dim === 0 || dim > Math.floor(cap / bytes)) fail(`${path}.shape`, 'nonpositive dimension or tensor byte product exceeds cap');
    bytes *= dim;
  }
  if (bytes !== t.data.length) fail(`${path}.data`, 'BIN length differs from dtype/shape byte count');
  if (dtype !== 'u8') {
    for (let i = 0; i < bytes / itemSize; i++) if (!Number.isFinite(tensorElement(dtype, t.data, i))) fail(`${path}.data`, 'nonfinite tensor element');
  }
  return t;
}

function observation(value: Input, path: string, cap: number): WireMap {
  const o = map(value, path, ['frames', 'state', 'levels_completed', 'win_levels', 'available_actions']);
  tensor(o.frames, `${path}.frames`, cap, 3, true);
  text(o.state, `${path}.state`, true);
  for (const key of ['levels_completed', 'win_levels']) integer(o[key], `${path}.${key}`, 0);
  array(o.available_actions, `${path}.available_actions`).forEach(a => integer(a, `${path}.available_actions entry`, 0, 7));
  return o;
}
function action(value: Input, path: string, available: WireValue[]): void {
  const a = map(value, path, ['id', 'data']);
  const id = integer(a.id, `${path}.id`, 1, 7);
  const data = map(a.data, `${path}.data`);
  for (const [key, param] of Object.entries(data)) integer(param, `${path}.data.${key}`);
  if (id === 6) for (const key of ['x', 'y']) integer(data[key], `${path}.data.${key}`, 0, 63);
  if (!available.includes(id)) fail(`${path}.id`, 'action unavailable in pre-action observation');
}
function sameShape(a: WireMap | undefined, b: WireMap | undefined): boolean {
  const x = a?.shape as number[] | undefined, y = b?.shape as number[] | undefined;
  return !!x && !!y && x.length === y.length && x.every((d, i) => d === y[i]);
}
function stateTensor(value: Input, path: string, cap: number): WireMap {
  const t = tensor(value, path, cap, 1);
  if (t.dtype !== 'f32') fail(`${path}.dtype`, 'state requires f32');
  return t;
}
function representation(value: Input, path: string, cap: number): WireMap {
  const r = map(value, path, ['z', 'h']);
  for (const key of ['z', 'h']) stateTensor(r[key], `${path}.${key}`, cap);
  return r;
}
function extensionDiagnostics(step: WireMap, path: string, cap: number, observations: WireMap[]): void {
  if (Object.hasOwn(step, 'representation')) {
    representation(step.representation, `${path}.representation`, cap);
    if (!(((step.observation as WireMap).frames as WireMap).shape as number[])[0]) fail(`${path}.representation`, 'real representation requires a visual result');
  }
  if (Object.hasOwn(step, 'policy_mode')) choice(step.policy_mode, `${path}.policy_mode`, ['random', 'learned']);
  const p = (step.prediction ?? {}) as WireMap, dest = `${path}.prediction`;
  if (Object.hasOwn(p, 'dynamics')) {
    const d = map(p.dynamics, `${dest}.dynamics`, ['z', 'h', 'delta_z', 'h_next']);
    if (p.target !== 'post_action_last_frame' || !Object.hasOwn(p, 'latent')) fail(`${dest}.dynamics`, 'requires latent and last-frame target');
    for (const key of ['z', 'h', 'delta_z', 'h_next']) stateTensor(d[key], `${dest}.dynamics.${key}`, cap);
    stateTensor(p.latent, `${dest}.latent`, cap);
    if (!sameShape(d.z as WireMap, d.delta_z as WireMap) || !sameShape(d.z as WireMap, p.latent as WireMap)) fail(`${dest}.dynamics`, 'latent dimensions must match');
    if (!sameShape(d.h as WireMap, d.h_next as WireMap)) fail(`${dest}.dynamics`, 'memory dimensions must match');
  }
  const errors = (p.errors ?? {}) as WireMap, r = step.representation as WireMap | undefined;
  for (const [key, source] of [['latent_mse', p.latent], ['no_change_mse', (p.dynamics as WireMap | undefined)?.z]] as const) {
    if (Object.hasOwn(errors, key)) {
      number(errors[key], `${dest}.errors.${key}`, 0);
      if (p.target !== 'post_action_last_frame' || !sameShape(source as WireMap | undefined, r?.z as WireMap | undefined)) fail(`${dest}.errors.${key}`, 'latent error requires matched real/predicted tensors and last-frame target');
    }
  }
  if (Object.hasOwn(p, 'visual_delta')) {
    const location = `${dest}.visual_delta`, v = map(p.visual_delta, location, ['reference_position', 'reference_frame', 'delta']);
    if (p.target !== 'post_action_last_frame' || !Object.hasOwn(p, 'frames')) fail(location, 'requires absolute frames and last-frame target');
    const position = integer(v.reference_position, `${location}.reference_position`, 0, observations.length - 1);
    for (const [key, maximum] of [['reference_frame', 15], ['delta', 16]] as const) {
      const t = tensor(v[key], `${location}.${key}`, cap, 2);
      if (t.dtype !== 'u8' || (t.data as Uint8Array).some(value => value > maximum)) fail(`${location}.${key}`, 'invalid u8 palette/KEEP code');
    }
    const frames = p.frames as WireMap, reference = v.reference_frame as WireMap, delta = v.delta as WireMap;
    if (frames.dtype !== 'u8' || (frames.data as Uint8Array).some(value => value > 15)) fail(`${dest}.frames`, 'visual decoder requires u8 colors 0..15');
    if (!sameShape(frames, reference) || !sameShape(delta, reference)) fail(location, 'internal geometry must match');
    const observed = observations[position]!.frames as WireMap, shape = observed.shape as number[];
    const ref = reference.data as Uint8Array, codes = delta.data as Uint8Array, board = frames.data as Uint8Array, real = observed.data as Uint8Array;
    if (!shape[0] || (reference.shape as number[]).some((d, i) => d !== shape[i + 1]) || ref.some((v, i) => v !== real[real.length - ref.length + i])) fail(location, "reference must equal a nonempty causal observation's last frame");
    if (ref.some((v, i) => (codes[i] === 16 ? v : codes[i]) !== board[i])) fail(location, 'KEEP/replacement delta differs from absolute predicted board');
  }
  const l = step.learning as WireMap | undefined;
  if (l && Object.hasOwn(l, 'modules')) {
    const location = `${path}.learning.modules`, modules = map(l.modules, location);
    if (!Object.keys(modules).length) fail(location, 'expected nonempty module map');
    for (const key of ['world_model', 'decoder', 'actor', 'critic']) {
      if (!Object.hasOwn(modules, key)) continue;
      const dest = `${location}.${key}`, m = map(modules[key], dest);
      if (!['updates', 'loss', 'losses', 'reset_reason'].some(k => Object.hasOwn(m, k))) fail(dest, 'module requires actual reported data');
      if (Object.hasOwn(m, 'updates')) integer(m.updates, `${dest}.updates`, 0);
      if (Object.hasOwn(m, 'loss')) number(m.loss, `${dest}.loss`);
      if (Object.hasOwn(m, 'losses')) {
        const losses = map(m.losses, `${dest}.losses`);
        if (!Object.keys(losses).length) fail(`${dest}.losses`, 'expected nonempty loss map');
        for (const [name, loss] of Object.entries(losses)) { text(name, `${dest}.losses key`, true); number(loss, `${dest}.losses.${name}`); }
      }
      if (Object.hasOwn(m, 'reset_reason')) {
        if (!['actor', 'critic'].includes(key)) fail(`${dest}.reset_reason`, 'reset events are actor/critic only');
        text(m.reset_reason, `${dest}.reset_reason`, true);
      }
    }
  }
}

function diagnostics(step: WireMap, path: string, previous: WireMap, cap: number, minor: number, observations: WireMap[]): void {
  if (Object.hasOwn(step, 'decision')) {
    const dest = `${path}.decision`;
    const d = map(step.decision, dest, ['score_type', 'candidates']);
    const type = choice(d.score_type, `${dest}.score_type`, ['logit', 'probability', 'value']);
    const candidates = array(d.candidates, `${dest}.candidates`);
    if (!candidates.length) fail(`${dest}.candidates`, 'expected at least one candidate');
    candidates.forEach((value, i) => {
      const p = `${dest}.candidates[${i}]`;
      const c = map(value, p, ['action', 'score']);
      action(c.action, `${p}.action`, previous.available_actions as WireValue[]);
      const score = number(c.score, `${p}.score`);
      if (type === 'probability' && (score < 0 || score > 1)) fail(`${p}.score`, 'probability outside [0, 1]');
    });
    if (Object.hasOwn(d, 'selected_value')) number(d.selected_value, `${dest}.selected_value`);
    if (Object.hasOwn(d, 'action_entropy')) number(d.action_entropy, `${dest}.action_entropy`, 0);
  }
  if (Object.hasOwn(step, 'prediction')) {
    const dest = `${path}.prediction`;
    const p = map(step.prediction, dest, ['target']);
    const target = choice(p.target, `${dest}.target`, ['post_action_last_frame', 'post_action_sequence']);
    if (!['frames', 'latent', 'uncertainty'].some(key => Object.hasOwn(p, key))) fail(dest, 'prediction requires frames, latent or uncertainty');
    if (Object.hasOwn(p, 'frames')) tensor(p.frames, `${dest}.frames`, cap, target === 'post_action_last_frame' ? 2 : 3);
    if (Object.hasOwn(p, 'latent')) tensor(p.latent, `${dest}.latent`, cap);
    if (Object.hasOwn(p, 'uncertainty')) number(p.uncertainty, `${dest}.uncertainty`, 0);
    if (Object.hasOwn(p, 'errors')) {
      const errors = map(p.errors, `${dest}.errors`);
      for (const key of ['mae', 'mse']) if (Object.hasOwn(errors, key)) {
        number(errors[key], `${dest}.errors.${key}`, 0);
        const observed = (step.observation as WireMap).frames as WireMap;
        const shape = observed.shape as number[];
        const expected = target === 'post_action_last_frame' ? shape.slice(1) : shape;
        const predicted = p.frames as WireMap | undefined;
        const actual = predicted?.shape as number[] | undefined;
        if (!shape[0] || !actual || actual.length !== expected.length || actual.some((d, i) => d !== expected[i])) fail(`${dest}.errors.${key}`, 'pixel errors require comparable predicted/observed shapes');
      }
    }
  }
  if (Object.hasOwn(step, 'learning')) {
    const l = map(step.learning, `${path}.learning`);
    for (const key of ['updates', 'replay_size']) if (Object.hasOwn(l, key)) integer(l[key], `${path}.learning.${key}`, 0);
    if (Object.hasOwn(l, 'loss')) number(l.loss, `${path}.learning.loss`);
  }
  if (Object.hasOwn(step, 'timing')) {
    const t = map(step.timing, `${path}.timing`);
    for (const [key, value] of Object.entries(t)) number(value, `${path}.timing.${key}`, 0);
  }
  if (Object.hasOwn(step, 'notes')) array(step.notes, `${path}.notes`).forEach(note => text(note, `${path}.notes entry`));
  if (minor >= 1) extensionDiagnostics(step, path, cap, observations);
}

function timestamp(value: Input): void {
  const s = text(value, 'metadata.started_at');
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,9})?(?:Z|([+-])(\d{2}):(\d{2}))$/.exec(s);
  if (!m) fail('metadata.started_at', 'invalid timestamp profile');
  const [y, mo, d, h, mi, se] = m.slice(1, 7).map(Number) as [number, number, number, number, number, number];
  const leap = y % 4 === 0 && (y % 100 !== 0 || y % 400 === 0);
  const days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][mo - 1];
  if (y < 1 || !days || d < 1 || d > days || h > 23 || mi > 59 || se > 59 || (m[7] && (Number(m[8]) > 23 || Number(m[9]) > 59))) fail('metadata.started_at', 'invalid calendar date, time or offset');
}

export function validateAttempt(value: WireMap, cap: number, minor = 0): void {
  const a = map(value, 'attempt', ['metadata', 'initial_observation', 'steps', 'termination', 'summary']);
  const m = map(a.metadata, 'metadata', ['game_id', 'run_id', 'session_id', 'attempt_index', 'seed', 'sdk_version', 'source_split', 'started_at']);
  for (const key of ['game_id', 'run_id', 'session_id', 'sdk_version', 'model_id', 'checkpoint_id']) if (Object.hasOwn(m, key)) text(m[key], `metadata.${key}`, true);
  integer(m.attempt_index, 'metadata.attempt_index', 0);
  integer(m.seed, 'metadata.seed');
  choice(m.source_split, 'metadata.source_split', ['train', 'evaluation', 'unspecified']);
  timestamp(m.started_at);
  if (Object.hasOwn(m, 'config')) jsonConfig(m.config!, 'metadata.config');
  let previous = observation(a.initial_observation, 'initial_observation', cap);
  if (minor >= 1 && Object.hasOwn(a, 'initial_representation')) representation(a.initial_representation, 'initial_representation', cap);
  const observations = [previous];
  const steps = array(a.steps, 'steps');
  steps.forEach((value, i) => {
    const path = `steps[${i}]`;
    const s = map(value, path, ['index', 'action', 'observation']);
    if (integer(s.index, `${path}.index`, 0) !== i) fail(`${path}.index`, 'indices must be contiguous from zero');
    if (['WIN', 'GAME_OVER', 'NOT_PLAYED'].includes(previous.state as string)) fail(path, 'ordinary step after terminal or initialization state');
    action(s.action, `${path}.action`, previous.available_actions as WireValue[]);
    observation(s.observation, `${path}.observation`, cap);
    diagnostics(s, path, previous, cap, minor, observations);
    previous = s.observation as WireMap;
    observations.push(previous);
  });
  const t = map(a.termination, 'termination', ['reason', 'final_state']);
  const reason = choice(t.reason, 'termination.reason', ['win', 'game_over', 'reset', 'action_budget', 'timeout', 'error', 'interrupted', 'training_level_boundary']);
  text(t.final_state, 'termination.final_state', true);
  if (Object.hasOwn(t, 'detail')) text(t.detail, 'termination.detail');
  const reset = Number(Object.hasOwn(t, 'reset_action'));
  if (reset) {
    if (reason !== 'reset') fail('termination.reset_action', 'only permitted for reset termination');
    const r = map(t.reset_action, 'termination.reset_action', ['id', 'data']);
    integer(r.id, 'termination.reset_action.id', 0, 0);
    if (Object.keys(map(r.data, 'termination.reset_action.data')).length) fail('termination.reset_action.data', 'RESET data must be empty');
  }
  const s = map(a.summary, 'summary', ['real_actions', 'levels_completed', 'win_levels', 'wall_seconds']);
  for (const key of ['real_actions', 'levels_completed', 'win_levels']) integer(s[key], `summary.${key}`, 0);
  number(s.wall_seconds, 'summary.wall_seconds', 0);
  const count = steps.length + reset;
  const extra = Number(['training_level_boundary', 'error', 'timeout', 'interrupted'].includes(reason));
  if ((s.real_actions as number) < count || (s.real_actions as number) > count + extra) fail('summary.real_actions', 'inconsistent executed-action count');
  if (reason !== 'training_level_boundary') {
    if (t.final_state !== previous.state) fail('termination.final_state', 'differs from final persisted observation');
    for (const key of ['levels_completed', 'win_levels']) if (s[key] !== previous[key]) fail(`summary.${key}`, 'differs from final persisted observation');
  }
  const terminalReason = previous.state === 'WIN' ? 'win' : previous.state === 'GAME_OVER' ? 'game_over' : undefined;
  if (terminalReason && reason !== terminalReason) fail('termination.reason', 'known persisted terminal state requires corresponding reason');
  if (reason === 'win' || reason === 'game_over') {
    const expected = reason === 'win' ? 'WIN' : 'GAME_OVER';
    if (previous.state !== expected || t.final_state !== expected) fail('termination', 'terminal reason requires matching persisted terminal state');
  }
}
