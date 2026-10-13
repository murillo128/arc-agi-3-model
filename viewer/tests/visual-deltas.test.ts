import { describe, expect, it } from 'vitest';
import { drawObservation } from '../src/canvas';
import { compareVisualDeltas, KEEP } from '../src/visual-deltas';
import { MAX_DIAGNOSTIC_ELEMENTS } from '../src/diagnostics';
import type { Observation, Prediction, Tensor } from '../src/types';

const tensor = (shape: number[], data: number[]): Tensor => ({ dtype: 'u8', shape, data: Uint8Array.from(data) });
const observation = (data: number[], shape = [1, 2, 3]): Observation => ({
  frames: tensor(shape, data), state: 'NOT_FINISHED', levels_completed: 0, win_levels: 1, available_actions: [1],
});
const before = observation([0, 1, 2, 3, 4, 5]);
const after = observation([0, 6, 2, 8, 9, 5]);
const prediction: Prediction = { target: 'post_action_last_frame', frames: tensor([2, 3], [0, 6, 7, 3, 10, 5]) };

describe('net action deltas against one common reference', () => {
  it('distinguishes correct, extra, missed and wrong-color changes with an independent 3/4 oracle', () => {
    const result = compareVisualDeltas(prediction, after, 0, before);
    expect([...result.real.observation!.frames.data]).toEqual([16, 6, 16, 8, 9, 16]);
    expect([...result.predicted.observation!.frames.data]).toEqual([16, 6, 7, 16, 10, 16]);
    expect([...result.difference.observation!.frames.data]).toEqual([16, 16, 8, 8, 8, 16]);
    expect(result.disagreements).toBe(3);
    expect(result.changedPixels).toBe(4);
    expect(result.fraction).toBe(0.75); // Unchanged background cannot dilute the error to 3/6.
    expect(result.difference.message).toContain('75.00% (3/4');
    expect([...before.frames.data]).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it('compares the net action change, not the last adjacent-animation delta', () => {
    const initial = observation([0], [1, 1, 1]), animated = observation([1, 1], [2, 1, 1]);
    const final: Prediction = { target: 'post_action_last_frame', frames: tensor([1, 1], [1]) };
    const intermediate = compareVisualDeltas(final, animated, 0, initial);
    expect(intermediate.real.codes).toEqual([1]);
    expect(intermediate.predicted.observation).toBeUndefined();
    expect(intermediate.difference.observation).toBeUndefined();
    expect(intermediate.predicted.message).toContain('only with real frame 1');
    const result = compareVisualDeltas(final, animated, 1, initial);
    expect(result.real.codes).toEqual([1]); // Adjacent frame subtraction would incorrectly say KEEP.
    expect(result.predicted.codes).toEqual([1]);
    expect(result.fraction).toBe(0);
    const sequence: Prediction = { target: 'post_action_sequence', frames: tensor([2, 1, 1], [1, 0]) };
    expect(compareVisualDeltas(sequence, animated, 0, initial).fraction).toBe(0);
    expect(compareVisualDeltas(sequence, animated, 1, initial).fraction).toBe(1);
  });

  it('uses the recorded immutable reference even when the immediate predecessor is empty', () => {
    const recorded: Prediction = { ...prediction, visual_delta: {
      reference_position: 0, reference_frame: tensor([2, 3], [0, 1, 2, 3, 4, 5]),
      delta: tensor([2, 3], [16, 6, 7, 16, 10, 16]),
    } };
    const empty = observation([], [0, 0, 0]);
    const result = compareVisualDeltas(recorded, after, 0, empty, 1);
    expect(result.fraction).toBe(0.75);
    expect(result.real.message).toContain('Recorded reference position 0');
    expect(result.predicted.message).toContain('Recorded predicted delta');
    // v1.0 must ignore fields whose names only became recognized in v1.1.
    expect(compareVisualDeltas(recorded, after, 0, empty, 0).real.observation).toBeUndefined();
    const differentPredecessor = observation([15, 15, 15, 15, 15, 15]);
    expect(compareVisualDeltas(recorded, after, 0, differentPredecessor, 1).real.codes).toEqual([16, 6, 16, 8, 9, 16]);
    expect(compareVisualDeltas(recorded, after, 0, differentPredecessor, 0).real.codes).toEqual([0, 6, 2, 8, 9, 5]);
  });

  it('shows a comparable no-change action as transparent with zero error, not missing data', () => {
    const unchanged: Prediction = { target: 'post_action_last_frame', frames: tensor([2, 3], [0, 1, 2, 3, 4, 5]) };
    const result = compareVisualDeltas(unchanged, before, 0, before);
    expect(result.fraction).toBe(0); expect(result.changedPixels).toBe(0);
    expect(result.real.codes).toEqual([16, 16, 16, 16, 16, 16]);
    expect(result.difference.codes).toEqual([16, 16, 16, 16, 16, 16]);
  });

  it.each([undefined, { target: 'post_action_last_frame', latent: tensor([2], [1, 2]) }] satisfies (Prediction | undefined)[])(
    'keeps the real delta without inventing a prediction or error: %j', missing => {
      const result = compareVisualDeltas(missing, after, 0, before);
      expect(result.real.codes).toEqual([16, 6, 16, 8, 9, 16]);
      expect(result.predicted.observation).toBeUndefined();
      expect(result.difference.observation).toBeUndefined(); expect(result.fraction).toBeUndefined();
    });

  it('refuses missing, empty, mismatched and excessive references or targets', () => {
    for (const previous of [undefined, observation([], [0, 0, 0]), observation([0], [1, 1, 1])]) {
      expect(compareVisualDeltas(prediction, after, 0, previous).real.observation).toBeUndefined();
    }
    expect(compareVisualDeltas(prediction, observation([], [0, 0, 0]), 0, before).fraction).toBeUndefined();
    expect(compareVisualDeltas(prediction, after, 1, before).real.observation).toBeUndefined();
    expect(compareVisualDeltas({ ...prediction, frames: tensor([1, 6], [0, 6, 7, 3, 10, 5]) }, after, 0, before).predicted.observation).toBeUndefined();
    const oversized = observation([], [1, 1, MAX_DIAGNOSTIC_ELEMENTS + 1]);
    expect(compareVisualDeltas(prediction, oversized, 0, oversized).real.message).toContain('limit');
    expect(compareVisualDeltas(prediction, observation([16, 6, 2, 8, 9, 5]), 0, before).real.observation).toBeUndefined();
  });

  it.each([0.25, 16, -1])('refuses non-palette predictions (%s) instead of rounding them into a delta', value => {
    const bytes = new Uint8Array(6 * 4), view = new DataView(bytes.buffer);
    [value, 6, 7, 3, 10, 5].forEach((n, i) => view.setFloat32(i * 4, n, true));
    const result = compareVisualDeltas({ ...prediction, frames: { dtype: 'f32', shape: [2, 3], data: bytes } }, after, 0, before);
    expect(result.real.observation).toBeDefined();
    expect(result.predicted.message).toContain('are not rounded');
    expect(result.predicted.observation).toBeUndefined(); expect(result.fraction).toBeUndefined();
  });
});

it('makes KEEP transparent while retaining real white and red; regular boards still use the cyan fallback', () => {
  let pixels: Uint8ClampedArray = new Uint8ClampedArray();
  const canvas = { width: 0, height: 0, hidden: true, style: {}, getContext: () => ({
    createImageData: (w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: (bitmap: { data: Uint8ClampedArray }) => { pixels = bitmap.data; },
  }) } as unknown as HTMLCanvasElement;
  const delta = observation([16, 0, 8], [1, 1, 3]);
  expect(drawObservation(canvas, delta, 0, KEEP)).toBe('');
  expect([...pixels]).toEqual([0, 0, 0, 0, 255, 255, 255, 255, 249, 60, 49, 255]);
  expect(drawObservation(canvas, delta, 0)).toContain('16 (cyan fallback');
  expect([...pixels.slice(0, 4)]).toEqual([0, 255, 255, 255]);
});
