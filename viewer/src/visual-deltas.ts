import { MAX_DIAGNOSTIC_ELEMENTS, tensorValue, type ChangeView } from './diagnostics';
import type { Observation, Prediction } from './types';

export const KEEP = 16;
export interface DeltaComparison {
  real: ChangeView;
  predicted: ChangeView;
  difference: ChangeView;
  disagreements?: number;
  changedPixels?: number;
  fraction?: number;
}

/** Compare net action deltas against one shared reference, never adjacent animation frames. */
export function compareVisualDeltas(prediction: Prediction | undefined, observed: Observation,
  frame: number, preAction?: Observation, minor = 0): DeltaComparison {
  const unavailable = (message: string): DeltaComparison => ({
    real: { message }, predicted: { message }, difference: { message },
  });
  const [count = 0, height = 0, width = 0] = observed.frames.shape, total = height * width;
  if (!count || frame < 0 || frame >= count) return unavailable('Unavailable: no selected real action-result frame.');
  if (total > MAX_DIAGNOSTIC_ELEMENTS) return unavailable('Unavailable: action deltas exceed the diagnostic pixel limit.');
  const bundle = minor >= 1 ? prediction?.visual_delta : undefined;
  const sameSize = (shape: number[]) => shape.length === 2 && shape[0] === height && shape[1] === width;
  let reference: Uint8Array;
  let source: string;
  if (bundle) {
    if (!sameSize(bundle.reference_frame.shape)) return unavailable('Unavailable: recorded delta reference has a different size.');
    reference = bundle.reference_frame.data;
    source = `Recorded reference position ${bundle.reference_position}`;
  } else {
    const previous = preAction?.frames;
    if (!previous?.shape[0] || !sameSize(previous.shape.slice(1))) {
      return unavailable('Unavailable: no same-size pre-action reference frame.');
    }
    reference = previous.data.subarray((previous.shape[0] - 1) * total, previous.shape[0] * total);
    source = 'Pre-action final frame reference';
  }
  const view = (data: Uint8Array, label: string): ChangeView => ({
    message: `${label} · real frame ${frame}\n${source}.`,
    observation: { ...observed, frames: { dtype: 'u8', shape: [1, height, width], data } },
    codes: Array.from(data.subarray(0, 32)),
  });
  const real = new Uint8Array(total);
  for (let i = 0; i < total; i++) {
    const value = observed.frames.data[frame * total + i]!;
    if (value > 15 || reference[i]! > 15) return unavailable('Unavailable: real colors outside 0..15 cannot be represented as KEEP/color deltas.');
    real[i] = value === reference[i] ? KEEP : value;
  }
  const result = unavailable('Not recorded: no decoded visual prediction.');
  result.real = view(real, 'Real net action delta');
  const noPrediction = (message: string): DeltaComparison => ({ ...result, predicted: { message }, difference: { message } });
  if (!prediction) return result;
  if (prediction.target === 'post_action_last_frame' && frame !== count - 1) {
    return noPrediction(`Unavailable: final prediction aligns only with real frame ${count - 1}.`);
  }
  let predicted: Uint8Array;
  if (bundle) {
    if (!sameSize(bundle.delta.shape)) return noPrediction('Unavailable: predicted delta has a different size.');
    predicted = bundle.delta.data;
  } else {
    const tensor = prediction.frames;
    if (!tensor) return result;
    const expected = prediction.target === 'post_action_last_frame' ? [height, width] : [count, height, width];
    if (tensor.shape.length !== expected.length || tensor.shape.some((d, i) => d !== expected[i])) {
      return noPrediction('Unavailable: predicted frame shape differs from the real target.');
    }
    predicted = new Uint8Array(total);
    const offset = prediction.target === 'post_action_last_frame' ? 0 : frame * total;
    for (let i = 0; i < total; i++) {
      const value = tensorValue(tensor, offset + i);
      if (!Number.isInteger(value) || value < 0 || value > 15) {
        return noPrediction('Unavailable: predicted delta requires exact palette colors 0..15; fractional or out-of-range estimates are not rounded.');
      }
      predicted[i] = value === reference[i] ? KEEP : value;
    }
  }
  const difference = new Uint8Array(total);
  let disagreements = 0, changedPixels = 0;
  for (let i = 0; i < total; i++) {
    const different = real[i] !== predicted[i];
    difference[i] = different ? 8 : KEEP;
    disagreements += Number(different);
    changedPixels += Number(real[i] !== KEEP || predicted[i] !== KEEP);
  }
  const fraction = changedPixels ? disagreements / changedPixels : 0;
  result.predicted = view(predicted, bundle ? 'Recorded predicted delta' : 'Predicted delta derived from the absolute prediction');
  result.difference = {
    ...view(difference, 'Delta difference'),
    message: `Delta error: ${(fraction * 100).toFixed(2)}% (${disagreements}/${changedPixels} pixels changed in either delta).\n${source} · real frame ${frame}.`,
  };
  return { ...result, disagreements, changedPixels, fraction };
}
