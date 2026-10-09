import type { Attempt, Observation, Prediction, Step, Tensor } from './types';

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

export interface Metric { key: string; label: string; read: (step: Step) => number | undefined }
const metrics: Metric[] = [
  { key: 'uncertainty', label: 'Prediction uncertainty (recorded units)', read: step => step.prediction?.uncertainty },
  { key: 'entropy', label: 'Action entropy (nats)', read: step => step.decision?.action_entropy },
  { key: 'loss', label: 'Learning loss (recorded objective)', read: step => step.learning?.loss },
  { key: 'updates', label: 'Optimizer updates (count)', read: step => step.learning?.updates },
  { key: 'replay_size', label: 'Replay size (entries)', read: step => step.learning?.replay_size },
  ...['decision_ms', 'environment_ms', 'learning_ms'].map(key => ({ key: `timing.${key}`, label: `${key} (ms)`, read: (step: Step) => step.timing?.[key] })),
];
export function diagnosticMetrics(attempt: Attempt): Metric[] {
  return metrics.filter(metric => attempt.steps.some(step => metric.read(step) !== undefined));
}
export interface ChartPoint { step: number; value: number | undefined }
export function chartPoints(attempt: Attempt, metric: Metric, from: number, to: number): ChartPoint[] {
  const start = Math.max(0, Math.trunc(from)), end = Math.min(attempt.steps.length - 1, Math.trunc(to));
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start || end - start + 1 > MAX_CHART_STEPS) return [];
  return attempt.steps.slice(start, end + 1).map(step => ({ step: step.index, value: metric.read(step) }));
}
