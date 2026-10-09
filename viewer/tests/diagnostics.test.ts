import { describe, expect, it } from 'vitest';
import { chartPoints, comparePrediction, diagnosticMetrics, latentSummary, MAX_CHART_STEPS, MAX_DIAGNOSTIC_ELEMENTS, tensorValue } from '../src/diagnostics';
import type { Attempt, Observation, Prediction, Tensor } from '../src/types';

const tensor = (shape: number[], data: number[]): Tensor => ({ dtype: 'u8', shape, data: new Uint8Array(data) });
const observed: Observation = { frames: tensor([2, 2, 2], [0, 1, 2, 3, 3, 2, 1, 0]), state: 'NOT_FINISHED', levels_completed: 0, win_levels: 2, available_actions: [1] };

describe('prediction alignment and independent tiny-matrix pixel oracle', () => {
  it('aligns a last-frame target only to the final post-action frame', () => {
    const prediction: Prediction = { target: 'post_action_last_frame', frames: tensor([2, 2], [3, 2, 1, 0]) };
    expect(comparePrediction(prediction, observed, 0)).toEqual({ message: 'Not recorded for real frame 0. Last-frame target aligns only with real frame 1.' });
    const result = comparePrediction(prediction, observed, 1);
    expect(result.fraction).toBe(0); expect(result.disagreements).toBe(0); expect(result.total).toBe(4);
    expect([...result.difference!.frames.data]).toEqual([5, 5, 5, 5]);
  });
  it('pairs a full sequence by frame index and uses an explicit 2/4 oracle', () => {
    const prediction: Prediction = { target: 'post_action_sequence', frames: tensor([2, 2, 2], [0, 1, 2, 3, 3, 9, 1, 9]) };
    expect(comparePrediction(prediction, observed, 0).fraction).toBe(0);
    const result = comparePrediction(prediction, observed, 1);
    expect(result.fraction).toBe(0.5); expect(result.disagreements).toBe(2); expect(result.total).toBe(4);
    expect([...result.predicted!.frames.data]).toEqual([3, 9, 1, 9]);
    expect([...result.difference!.frames.data]).toEqual([5, 8, 5, 8]);
  });
  it.each([
    undefined,
    { target: 'post_action_last_frame', latent: tensor([4], [3, 2, 1, 0]) },
    { target: 'post_action_sequence', uncertainty: 0 },
    { target: 'post_action_last_frame', frames: tensor([1, 4], [3, 2, 1, 0]) },
    { target: 'post_action_sequence', frames: tensor([1, 2, 2], [3, 2, 1, 0]) },
  ] satisfies (Prediction | undefined)[])('does not compute errors from missing visuals, latents or mismatched shapes: %j', prediction => {
    const result = comparePrediction(prediction, observed, 1);
    expect(result.fraction).toBeUndefined(); expect(result.difference).toBeUndefined(); expect(result.predicted).toBeUndefined();
    expect(result.message).toMatch(/Not recorded|No comparable/);
  });
  it('does not borrow a frame for an empty observation or an invalid selected index', () => {
    const prediction: Prediction = { target: 'post_action_last_frame', frames: tensor([2, 2], [3, 2, 1, 0]) };
    expect(comparePrediction(prediction, { ...observed, frames: tensor([0, 0, 0], []) }, 0).message).toContain('no real frames');
    expect(comparePrediction(prediction, observed, 2).fraction).toBeUndefined();
  });
  it('compares fractional f32 values before display rounding, with unaligned bytes', () => {
    // Literal little-endian [3, 2.25, 1, 0], framed with unrelated bytes.
    const data = Uint8Array.from([255, 0, 0, 64, 64, 0, 0, 16, 64, 0, 0, 128, 63, 0, 0, 0, 0, 255]).subarray(1, 17);
    const result = comparePrediction({ target: 'post_action_last_frame', frames: { dtype: 'f32', shape: [2, 2], data } }, observed, 1);
    expect(result.values).toEqual([3, 2.25, 1, 0]); expect(result.fraction).toBe(0.25);
    expect([...result.predicted!.frames.data]).toEqual([3, 2, 1, 0]);
    expect([...result.difference!.frames.data]).toEqual([5, 8, 5, 5]);
  });
  it('decodes an f16 visual prediction as palette indices', () => {
    // Literal little-endian [3, 2, 1, 0], not values produced by tensorValue.
    const frames: Tensor = { dtype: 'f16', shape: [2, 2], data: Uint8Array.from([0, 66, 0, 64, 0, 60, 0, 0]) };
    const result = comparePrediction({ target: 'post_action_last_frame', frames }, observed, 1);
    expect(result.values).toEqual([3, 2, 1, 0]); expect(result.fraction).toBe(0);
  });
  it('bounds visual work at the declared per-frame limit', () => {
    const shape = [1, MAX_DIAGNOSTIC_ELEMENTS + 1];
    const frames = tensor(shape, Array(MAX_DIAGNOSTIC_ELEMENTS + 1).fill(0));
    expect(comparePrediction({ target: 'post_action_last_frame', frames }, { ...observed, frames: { ...frames, shape: [1, ...shape] } }, 0).message).toContain('limit');
  });
});

describe('latent dimensionality and norm, without inventing a projection', () => {
  it('decodes little-endian f16 signs, subnormals and zero independently', () => {
    const latent: Tensor = { dtype: 'f16', shape: [5], data: Uint8Array.from([0, 66, 0, 196, 1, 0, 0, 0, 0, 128]) };
    expect([0, 1, 2, 3, 4].map(i => tensorValue(latent, i))).toEqual([3, -4, 2 ** -24, 0, -0]);
    expect(latentSummary({ ...latent, shape: [2], data: latent.data.slice(0, 4) })).toBe('f16 [2] · 2 elements\nL2 norm: 5 (model-specific units)');
  });
  it('reports zero and missing tensors honestly, and refuses excessive norm work', () => {
    expect(latentSummary(undefined)).toBe('Not recorded.');
    expect(latentSummary(tensor([2], [0, 0]))).toContain('L2 norm: 0');
    expect(latentSummary(tensor([MAX_DIAGNOSTIC_ELEMENTS + 1], []))).toContain('not computed');
  });
});

describe('recorded metric selection and sparse step filtering', () => {
  const attempt = { steps: [
    { index: 0, prediction: { uncertainty: 0 }, learning: { loss: 0.25, updates: 0 }, timing: { decision_ms: 2.5 } },
    { index: 1, learning: { replay_size: 8 } },
    { index: 2, prediction: { uncertainty: 0.5 }, learning: { loss: -0.125 }, decision: { action_entropy: 1 } },
  ] } as Attempt;
  it('discovers available recorded fields, including zero; filters without filling gaps', () => {
    const metrics = diagnosticMetrics(attempt);
    expect(metrics.map(metric => metric.key)).toEqual(['uncertainty', 'entropy', 'loss', 'updates', 'replay_size', 'timing.decision_ms']);
    expect(chartPoints(attempt, metrics[0]!, 0, 2)).toEqual([{ step: 0, value: 0 }, { step: 1, value: undefined }, { step: 2, value: 0.5 }]);
    expect(chartPoints(attempt, metrics.find(metric => metric.key === 'loss')!, 1, 2)).toEqual([{ step: 1, value: undefined }, { step: 2, value: -0.125 }]);
    expect(diagnosticMetrics({ steps: [] } as unknown as Attempt)).toEqual([]);
  });
  it('refuses invalid and oversized chart ranges instead of silently sampling', () => {
    const metric = diagnosticMetrics(attempt)[0]!;
    expect(chartPoints(attempt, metric, 2, 1)).toEqual([]);
    expect(chartPoints(attempt, metric, NaN, 2)).toEqual([]);
    expect(chartPoints({ steps: Array(MAX_CHART_STEPS + 1).fill(attempt.steps[0]) } as Attempt, metric, 0, MAX_CHART_STEPS)).toEqual([]);
  });
});
