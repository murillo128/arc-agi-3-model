import type { Attempt, ModuleName, Observation, Prediction, Representation, Step, Tensor } from './types';

// Diagnostic work is deliberately much smaller than the real-observation bitmap limit.
export const MAX_DIAGNOSTIC_ELEMENTS = 65_536;
export const MAX_CHART_STEPS = 2048;

/** Decode the frozen v1 scalar types without assuming BIN alignment. */
export function tensorValue(tensor: Tensor, index: number): number {
  const view = new DataView(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength);
  if (tensor.dtype === 'u8') return view.getUint8(index);
  if (tensor.dtype === 'f32') return view.getFloat32(index * 4, true);
  const bits = view.getUint16(index * 2, true), exponent = (bits >> 10) & 31, mantissa = bits & 1023;
  const sign = bits & 0x8000 ? -1 : 1;
  if (exponent === 31) return mantissa ? NaN : sign * Infinity;
  return sign * (exponent ? (1 + mantissa / 1024) * 2 ** (exponent - 15) : mantissa * 2 ** -24);
}

export interface VisualComparison {
  message: string;
  predicted?: Observation;
  difference?: Observation;
  disagreements?: number;
  total?: number;
  fraction?: number;
  values?: number[];
}

/** Only the selected post-action frame is compared. Sequence shape includes count. */
export function comparePrediction(prediction: Prediction | undefined, observed: Observation, frame: number): VisualComparison {
  if (!prediction) return { message: 'Not recorded for this position.' };
  const tensor = prediction.frames;
  if (!tensor) return { message: prediction.latent ? 'Not recorded: no decoded visual prediction. A model-specific latent decoder is unavailable.' : 'Not recorded: no decoded visual prediction.' };
  const [count = 0, height = 0, width = 0] = observed.frames.shape;
  if (!count) return { message: 'No comparable prediction: no real frames were returned.' };
  if (frame < 0 || frame >= count) return { message: 'No comparable prediction: frame is outside the real sequence.' };
  if (prediction.target === 'post_action_last_frame' && frame !== count - 1) {
    return { message: `Not recorded for real frame ${frame}. Last-frame target aligns only with real frame ${count - 1}.` };
  }
  const expected = prediction.target === 'post_action_last_frame' ? [height, width] : [count, height, width];
  if (tensor.shape.length !== expected.length || tensor.shape.some((d, i) => d !== expected[i])) {
    return { message: `No comparable prediction: predicted shape [${tensor.shape.join(', ')}] differs from target [${expected.join(', ')}].` };
  }
  const total = height * width;
  if (total > MAX_DIAGNOSTIC_ELEMENTS) return { message: `Visual diagnostics exceed the ${MAX_DIAGNOSTIC_ELEMENTS.toLocaleString('en-US')}-pixel per-frame limit.` };
  const pixels = new Uint8Array(total), difference = new Uint8Array(total), values: number[] = [];
  const predictedOffset = prediction.target === 'post_action_last_frame' ? 0 : frame * total;
  let disagreements = 0;
  for (let i = 0; i < total; i++) {
    const value = tensorValue(tensor, predictedOffset + i);
    // Float palette estimates are rounded for display only; comparison uses exact values.
    const display = Math.round(value);
    pixels[i] = display >= 0 && display <= 255 ? display : 255;
    const different = value !== observed.frames.data[frame * total + i];
    difference[i] = different ? 8 : 5; // red = disagreement, black = equal
    disagreements += Number(different);
    if (Math.floor(i / width) < 32 && i % width < 32) values.push(value);
  }
  const observation = (data: Uint8Array): Observation => ({ ...observed, frames: { dtype: 'u8', shape: [1, height, width], data } });
  return {
    message: `${prediction.target} · real frame ${frame} · ${tensor.dtype === 'u8' ? 'Recorded palette indices' : 'Float palette estimates rounded for display only; out-of-range values use cyan'}`,
    predicted: observation(pixels), difference: observation(difference), values,
    disagreements, total, fraction: disagreements / total,
  };
}

export function latentSummary(tensor: Tensor | undefined): string {
  if (!tensor) return 'Not recorded.';
  const count = tensor.shape.reduce((a, b) => a * b, 1);
  const dimensions = `${tensor.dtype} [${tensor.shape.join(', ')}] · ${count.toLocaleString('en-US')} elements`;
  if (count > MAX_DIAGNOSTIC_ELEMENTS) return `${dimensions}\nL2 norm not computed: exceeds ${MAX_DIAGNOSTIC_ELEMENTS.toLocaleString('en-US')}-element limit.`;
  // Scaled sum of squares avoids overflowing for large finite f32 values.
  let scale = 0, sum = 1;
  for (let i = 0; i < count; i++) {
    const value = Math.abs(tensorValue(tensor, i));
    if (!value) continue;
    if (value > scale) { sum = 1 + sum * (scale / value) ** 2; scale = value; }
    else sum += (value / scale) ** 2;
  }
  return `${dimensions}\nL2 norm: ${scale * Math.sqrt(sum)} (model-specific units)`;
}

/** Select the actual stored state, never borrow a prior step's cached encoding. */
export function representationAt(attempt: Attempt, position: number, minor = 0): Representation | undefined {
  if (minor < 1 || position < 0 || position > attempt.steps.length) return undefined;
  return position === 0 ? attempt.initial_representation : attempt.steps[position - 1]?.representation;
}

export function residualConsistency(prediction: Prediction | undefined, minor = 0): string {
  const d = minor >= 1 ? prediction?.dynamics : undefined, successor = prediction?.latent;
  if (!d || !successor) return 'Residual equation: Not recorded.';
  const count = d.z.shape[0]!;
  if (count > MAX_DIAGNOSTIC_ELEMENTS) return `Residual equation: not checked; exceeds ${MAX_DIAGNOSTIC_ELEMENTS}-element limit.`;
  let mismatches = 0;
  for (let i = 0; i < count; i++) {
    const recorded = tensorValue(successor, i), sum = tensorValue(d.z, i) + tensorValue(d.delta_z, i);
    mismatches += Number(Math.abs(recorded - sum) > 1e-6 + 1e-5 * Math.max(Math.abs(recorded), Math.abs(sum)));
  }
  return mismatches ? `Residual equation inconsistent at ${mismatches}/${count} elements. Recorded z_hat is retained; this is not prediction quality.` :
    'Residual equation consistent within f32 tolerance (derived check, not prediction quality).';
}

function stateSummary(label: string, tensor: Tensor | undefined): string {
  const preview = tensor && tensor.shape.reduce((a, b) => a * b, 1) <= MAX_DIAGNOSTIC_ELEMENTS ?
    `\nRecorded values (first 16): [${Array.from({ length: Math.min(16, tensor.shape[0]!) }, (_, i) => tensorValue(tensor, i)).join(', ')}]` : '';
  return `${label}\n${latentSummary(tensor)}${preview}`;
}
export function latentStateSummary(attempt: Attempt, position: number, minor = 0): string {
  const step = position > 0 && position <= attempt.steps.length ? attempt.steps[position - 1] : undefined;
  if (minor < 1) return latentSummary(step?.prediction?.latent);
  const actual = representationAt(attempt, position, minor), d = step?.prediction?.dynamics;
  return [
    ...(position === 0 ? [] : [stateSummary('Frozen input z_t', d?.z), stateSummary('Frozen input h_t', d?.h),
      stateSummary('Predicted residual d_hat', d?.delta_z), stateSummary('Recorded predicted successor z_hat', step?.prediction?.latent),
      stateSummary('Predicted memory h_next', d?.h_next), residualConsistency(step?.prediction, minor)]),
    stateSummary(position === 0 ? 'Initial actual z' : 'Real successor z (before learning)', actual?.z),
    stateSummary(position === 0 ? 'Initial actual h' : 'Real successor h (before learning)', actual?.h),
  ].join('\n\n');
}

export interface ChangeView { message: string; observation?: Observation; codes?: number[] }
function changeView(delta: Uint8Array, height: number, width: number, label: string, observation: Observation): ChangeView {
  return { message: `${label}\nKEEP=16 (cyan); 0..15 are replacement colors. This is not a palette-index board or subtraction.`,
    observation: { ...observation, frames: { dtype: 'u8', shape: [1, height, width], data: delta } },
    codes: Array.from(delta.subarray(0, Math.min(32, delta.length))),
  };
}
export function predictedVisualChange(prediction: Prediction | undefined, observed: Observation, frame: number, minor = 0): ChangeView {
  const bundle = minor >= 1 ? prediction?.visual_delta : undefined;
  if (!bundle) return { message: 'Not recorded: no predicted visual-change bundle.' };
  const [count, height, width] = observed.frames.shape;
  if (!count || frame !== count - 1 || bundle.delta.shape[0] !== height || bundle.delta.shape[1] !== width) return { message: 'Unavailable for this real frame: decoded final target is not comparable.' };
  if (height! * width! > MAX_DIAGNOSTIC_ELEMENTS) return { message: 'Predicted visual-change view exceeds the diagnostic pixel limit.' };
  return changeView(bundle.delta.data, height!, width!, `Predicted decoder visual change · immutable reference position ${bundle.reference_position}`, observed);
}

/** Derive changes only from consecutive real frames. No backward search past an empty response. */
export function observedAnimationChange(observed: Observation, frame: number, preAction?: Observation): ChangeView {
  const [count = 0, height = 0, width = 0] = observed.frames.shape, total = height * width;
  if (!count || frame < 0 || frame >= count) return { message: 'Unavailable: no selected real animation frame.' };
  if (total > MAX_DIAGNOSTIC_ELEMENTS) return { message: 'Observed change derivation exceeds the diagnostic pixel limit.' };
  const previous = frame > 0 ? observed : preAction;
  if (!previous?.frames.shape[0] || previous.frames.shape[1] !== height || previous.frames.shape[2] !== width) return { message: 'Unavailable: no same-size consecutive real predecessor.' };
  const offset = frame > 0 ? (frame - 1) * total : (previous.frames.shape[0] - 1) * total;
  const delta = new Uint8Array(total);
  for (let i = 0; i < total; i++) {
    const value = observed.frames.data[frame * total + i]!;
    // Real frames retain unknown palette bytes: do not reinterpret them as KEEP.
    if (value > 15 || previous.frames.data[offset + i]! > 15) return { message: 'Unavailable: real colors outside 0..15 cannot be represented as KEEP/color changes.' };
    delta[i] = value === previous.frames.data[offset + i] ? 16 : value;
  }
  return changeView(delta, height, width, `Observed / derived animation change · real frame ${frame}`, observed);
}

export const MODULE_NAMES: ModuleName[] = ['world_model', 'decoder', 'actor', 'critic'];
export function moduleSummary(step: Step | undefined, minor = 0): string {
  return MODULE_NAMES.map(name => {
    const m = minor >= 1 ? step?.learning?.modules?.[name] : undefined;
    if (!m) return `${name}: Not recorded.`;
    const fields = [m.updates === undefined ? '' : `optimizer steps: ${m.updates}`, m.loss === undefined ? '' : `reported loss: ${m.loss}`,
      ...Object.entries(m.losses ?? {}).slice(0, 32).map(([key, value]) => `${key.slice(0, 200)}: ${value}`),
      m.reset_reason === undefined ? '' : `Actual reset: ${m.reset_reason.slice(0, 2000)}`].filter(Boolean);
    return `${name}\n${fields.join('\n')}`;
  }).join('\n\n');
}

export interface Metric { key: string; label: string; read: (step: Step) => number | undefined }
const metrics: Metric[] = [
  { key: 'uncertainty', label: 'Prediction uncertainty (recorded units)', read: step => step.prediction?.uncertainty },
  { key: 'entropy', label: 'Action entropy (nats)', read: step => step.decision?.action_entropy },
  { key: 'loss', label: 'Learning loss (recorded objective)', read: step => step.learning?.loss },
  { key: 'updates', label: 'Optimizer updates (count)', read: step => step.learning?.updates },
  { key: 'replay_size', label: 'Replay size (entries)', read: step => step.learning?.replay_size },
  ...['decision_ms', 'environment_ms', 'learning_ms'].map(key => ({ key: `timing.${key}`, label: `${key} (ms)`, read: (step: Step) => step.timing?.[key] })),
];
export function diagnosticMetrics(attempt: Attempt, minor = 0): Metric[] {
  const available = [...metrics];
  if (minor >= 1) {
    for (const key of ['state_value', 'selected_score'] as const) available.push({ key: `planning.${key}`, label: `Planner ${key} (discounted value)`, read: step => step.planning?.[key] });
    for (const key of ['progress', 'game_over'] as const) available.push({ key: `planning.${key}`, label: `Goal ${key} (estimated probability)`, read: step => step.planning?.steps[0]?.[key] });
    for (const key of ['latent_mse', 'no_change_mse'] as const) available.push({ key: `prediction.${key}`, label: `Prediction ${key} (squared latent units)`, read: step => step.prediction?.errors?.[key] });
    for (const name of MODULE_NAMES) {
      for (const key of ['loss', 'updates'] as const) available.push({ key: `modules.${name}.${key}`, label: `${name} ${key} (reported per step)`, read: step => step.learning?.modules?.[name]?.[key] });
      if (name === 'actor' || name === 'critic') available.push({ key: `modules.${name}.reset`, label: `${name} reset event (derived presence marker)`, read: step => step.learning?.modules?.[name]?.reset_reason === undefined ? undefined : 1 });
      // Only supported component names get separate charts; arbitrary future
      // loss names remain literal in the bounded per-step module report.
      for (const key of name === 'world_model' ? ['latent_mse', 'sigreg'] : name === 'decoder' ? ['color_cross_entropy'] : []) {
        available.push({ key: `modules.${name}.losses.${key}`, label: `${name} ${key} (reported loss)`, read: step => step.learning?.modules?.[name]?.losses?.[key] });
      }
    }
  }
  return available.filter(metric => attempt.steps.some(step => metric.read(step) !== undefined));
}
export interface ChartPoint { step: number; value: number | undefined; resets?: string[] }
export function chartPoints(attempt: Attempt, metric: Metric, from: number, to: number, minor = 0): ChartPoint[] {
  const start = Math.max(0, Math.trunc(from)), end = Math.min(attempt.steps.length - 1, Math.trunc(to));
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start || end - start + 1 > MAX_CHART_STEPS) return [];
  return attempt.steps.slice(start, end + 1).map(step => {
    const resets = minor >= 1 ? (['actor', 'critic'] as const).flatMap(name => {
      const reason = step.learning?.modules?.[name]?.reset_reason;
      return reason === undefined ? [] : [`${name}: ${reason.slice(0, 2000)}`];
    }) : [];
    return { step: step.index, value: metric.read(step), ...(resets.length ? { resets } : {}) };
  });
}
