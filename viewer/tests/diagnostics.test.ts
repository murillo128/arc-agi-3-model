import { readFileSync } from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import { chartPoints, comparePrediction, diagnosticMetrics, latentSummary, latentStateSummary, moduleSummary, observedAnimationChange, predictedVisualChange, representationAt, residualConsistency, MAX_CHART_STEPS, MAX_DIAGNOSTIC_ELEMENTS, tensorValue } from '../src/diagnostics';
import { DiagnosticsView } from '../src/diagnostics-view';
import { decodeAttempt } from '../src/decoder';
import { Replay } from '../src/replay';
import { ReplayView } from '../src/replay-view';
import type { Attempt, Observation, Prediction, Tensor } from '../src/types';
import { planningView } from '../src/planning';
import { PALETTE } from '../src/canvas';

const tensor = (shape: number[], data: number[]): Tensor => ({ dtype: 'u8', shape, data: new Uint8Array(data) });
const observed: Observation = { frames: tensor([2, 2, 2], [0, 1, 2, 3, 3, 2, 1, 0]), state: 'NOT_FINISHED', levels_completed: 0, win_levels: 2, available_actions: [1] };
const fixture = async (name: string) => decodeAttempt(new Uint8Array(readFileSync(new URL(`../.fixtures/${name}.arc3`, import.meta.url))));

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

describe('v1.1 recorded residual/state and separate decoder changes', () => {
  it('selects the actual initial/result states with data-driven dimensions, without filling gaps', async () => {
    const { attempt, minor } = await fixture('v11-module-timeline');
    expect(representationAt(attempt, 0, minor)?.z.shape).toEqual([2]);
    expect(representationAt(attempt, 0, minor)?.h.shape).toEqual([3]);
    expect(representationAt(attempt, 1, minor)?.z.data).toEqual(Uint8Array.from([0, 0, 0, 64, 0, 0, 128, 63]));
    expect(representationAt(attempt, 2, minor)).toBeUndefined();
    expect(representationAt(attempt, 4, minor)).toBeUndefined();
    const summary = latentStateSummary(attempt, 1, minor);
    expect(summary).toContain('Recorded predicted successor z_hat\nf32 [2]');
    expect(summary).toContain('Recorded values (first 16): [1.5, 1]');
    expect(summary).toContain('Predicted memory h_next\nf32 [3]');
    expect(summary).toContain('Real successor z (before learning)\nf32 [2]');
    expect(latentStateSummary(attempt, 0, minor)).toContain('Initial actual z');
    expect(latentStateSummary(attempt, 0, minor)).not.toContain('predicted successor');
    expect(latentStateSummary(attempt, 2, minor)).not.toContain('Recorded values');
  });
  it('checks residual consistency without repairing recorded z_hat or claiming accuracy', async () => {
    const { attempt, minor } = await fixture('v11-latent');
    const prediction = attempt.steps[0]!.prediction!;
    expect(residualConsistency(prediction, minor)).toContain('consistent within f32 tolerance');
    // Literal f32 [3, 1], deliberately inconsistent with [1, 2] + [.5, -1].
    prediction.latent!.data = Uint8Array.from([0, 0, 64, 64, 0, 0, 128, 63]);
    expect(residualConsistency(prediction, minor)).toContain('inconsistent at 1/2');
    expect(tensorValue(prediction.latent!, 0)).toBe(3);
    expect(latentStateSummary(attempt, 1, minor)).toContain('Recorded values (first 16): [3, 1]');
    prediction.dynamics!.z.shape = [MAX_DIAGNOSTIC_ELEMENTS + 1];
    expect(residualConsistency(prediction, minor)).toContain('not checked');
    expect(residualConsistency(prediction, 0)).toContain('Not recorded');
  });
  it('shows KEEP/replacement codes separately from the absolute board and the 1/4 disagreement', async () => {
    const { attempt, minor } = await fixture('v11-decoder');
    const step = attempt.steps[0]!;
    const delta = predictedVisualChange(step.prediction, step.observation, 1, minor);
    expect(delta.codes).toEqual([16, 4, 16, 0]);
    expect(delta.message).toContain('Predicted decoder');
    expect(delta.message).toContain('reference position 0');
    const visual = comparePrediction(step.prediction, step.observation, 1);
    expect([...visual.predicted!.frames.data]).toEqual([0, 4, 2, 0]);
    expect(visual.fraction).toBe(0.25);
    expect([...visual.difference!.frames.data]).toEqual([5, 5, 8, 5]);
    expect(predictedVisualChange(step.prediction, step.observation, 0, minor).observation).toBeUndefined();
    expect(predictedVisualChange(step.prediction, { ...observed, frames: tensor([0, 0, 0], []) }, 0, minor).observation).toBeUndefined();
    expect(predictedVisualChange(step.prediction, { ...observed, frames: tensor([1, 1, 4], [0, 4, 5, 0]) }, 0, minor).observation).toBeUndefined();
    const latent = (await fixture('v11-latent')).attempt.steps[0]!;
    expect(predictedVisualChange(latent.prediction, latent.observation, 0, minor).message).toContain('Not recorded');
    expect(comparePrediction(latent.prediction, latent.observation, 0).fraction).toBeUndefined();
  });
  it('derives bounded observed changes only from the consecutive real predecessor', async () => {
    const { attempt } = await fixture('v11-decoder');
    const real = attempt.steps[0]!.observation;
    expect(observedAnimationChange(real, 0, attempt.initial_observation).codes).toEqual([16, 16, 16, 16]);
    expect(observedAnimationChange(real, 1).codes).toEqual([16, 4, 5, 0]);
    expect(observedAnimationChange(real, 1).message).toContain('Observed / derived');
    expect(observedAnimationChange(real, 0).message).toContain('Unavailable');
    expect(observedAnimationChange(real, 0, { ...observed, frames: tensor([0, 0, 0], []) }).message).toContain('Unavailable');
    expect(observedAnimationChange(real, 0, { ...observed, frames: tensor([1, 1, 4], [0, 1, 2, 3]) }).message).toContain('same-size');
    expect(observedAnimationChange({ ...real, frames: tensor([2, 2, 2], [0, 1, 2, 3, 0, 16, 2, 3]) }, 1).message).toContain('outside 0..15');
    const big = { ...observed, frames: tensor([1, 1, MAX_DIAGNOSTIC_ELEMENTS + 1], []) };
    expect(observedAnimationChange(big, 0, big).message).toContain('limit');
    const prediction: Prediction = { target: 'post_action_last_frame', frames: tensor([1, MAX_DIAGNOSTIC_ELEMENTS + 1], []),
      visual_delta: { reference_position: 0, reference_frame: tensor([1, MAX_DIAGNOSTIC_ELEMENTS + 1], []), delta: tensor([1, MAX_DIAGNOSTIC_ELEMENTS + 1], []) } };
    expect(predictedVisualChange(prediction, big, 0, 1).message).toContain('limit');
  });
  it('does not interpret the new names in v1.0 or carry model availability from another file', async () => {
    const old = await fixture('v10-new-names-unknown');
    expect(representationAt(old.attempt, 0, old.minor)).toBeUndefined();
    expect(residualConsistency(old.attempt.steps[0]!.prediction, old.minor)).toContain('Not recorded');
    expect(predictedVisualChange(old.attempt.steps[0]!.prediction, observed, 1, old.minor).message).toContain('Not recorded');
    expect(diagnosticMetrics(old.attempt, old.minor)).toEqual([]);
    const baseline = await fixture('recorder-baseline-0');
    expect(diagnosticMetrics(baseline.attempt, baseline.minor)).toEqual([]);
    expect(moduleSummary(baseline.attempt.steps[0], baseline.minor)).toBe('world_model: Not recorded.\n\ndecoder: Not recorded.\n\nactor: Not recorded.\n\ncritic: Not recorded.');
  });
});

describe('component learning reports remain separate from prediction quality', () => {
  it('keeps random-with-model reports, sparse losses, zero updates, negative losses and actual reset markers', async () => {
    const { attempt, minor } = await fixture('v11-module-timeline');
    const metrics = diagnosticMetrics(attempt, minor);
    const byKey = (key: string) => metrics.find(metric => metric.key === key)!;
    expect(byKey('prediction.latent_mse').read(attempt.steps[0]!)).toBe(0.125);
    expect(byKey('prediction.no_change_mse').read(attempt.steps[0]!)).toBe(1);
    expect(byKey('modules.world_model.losses.latent_mse').read(attempt.steps[0]!)).toBe(0.125);
    expect(metrics.some(metric => metric.key === 'loss')).toBe(false); // no invented aggregate
    expect(chartPoints(attempt, byKey('modules.world_model.loss'), 0, 2, minor)).toEqual([
      { step: 0, value: 0.125 }, { step: 1, value: undefined },
      { step: 2, value: 0, resets: ['actor: Synthetic actual reset <script>literal</script>', 'critic: Synthetic actual reset'] },
    ]);
    expect(byKey('modules.actor.loss').read(attempt.steps[2]!)).toBe(-0.5);
    expect(byKey('modules.world_model.updates').read(attempt.steps[2]!)).toBe(0);
    expect(byKey('modules.critic.reset').read(attempt.steps[2]!)).toBe(1);
    expect(byKey('modules.critic.reset').read(attempt.steps[1]!)).toBeUndefined();
    expect(moduleSummary(attempt.steps[0], minor)).toContain('optimizer steps: 2');
    expect(moduleSummary(attempt.steps[0], minor)).toContain('actor: Not recorded.');
    expect(attempt.steps[0]!.policy_mode).toBe('random');
    expect(moduleSummary(attempt.steps[2], minor)).toContain('<script>literal</script>');
  });
});

// Small DOM double verifies diagnostic replacement/clearing without launching a
// browser; Canvas drawing is an existing boundary and manual acceptance owns it.
class TestElement {
  textContent = ''; children: TestElement[] = []; hidden = false; disabled = false;
  width = 0; height = 0; value = ''; max = ''; valueAsNumber = 0;
  style: Record<string, string> = {}; attributes: Record<string, string> = {};
  addEventListener() {}
  setAttribute(key: string, value: string) { this.attributes[key] = value; }
  toggleAttribute(key: string, force: boolean) { if (force) this.attributes[key] = ''; else delete this.attributes[key]; }
  classList = { add() {}, toggle() {} };
  replaceChildren() { this.children = []; }
  append(...values: (TestElement | string)[]) { for (const value of values) typeof value === 'string' ? this.textContent += value : this.children.push(value); }
  bitmap?: Uint8Array;
  getContext() { return { imageSmoothingEnabled: false, createImageData: (w: number, h: number) => ({ data: new Uint8Array(w * h * 4) }), putImageData: (b: {data: Uint8Array}) => { this.bitmap = b.data; } }; }
}

it('clears every populated diagnostic panel, canvas and chart when replacing a file', async () => {
  const elements = new Map<string, TestElement>();
  const get = (id: string) => { if (!elements.has(id)) elements.set(id, new TestElement()); return elements.get(id)!; };
  vi.stubGlobal('document', { getElementById: get, createElement: () => new TestElement(), createElementNS: () => new TestElement() });
  try {
    const view = new DiagnosticsView();
    const loaded = await fixture('v11-decoder'), replay = new Replay(loaded.attempt, loaded.minor);
    replay.select(1); replay.selectFrame(1); view.display(replay);
    get('metric-select').value = 'modules.world_model.loss'; view.render(replay);
    expect(get('latent-info').textContent).toContain('[1.5, 1]');
    expect(get('predicted-change-canvas').hidden).toBe(false);
    expect(get('observed-change-canvas').hidden).toBe(false);
    for (const id of ['real-delta-canvas', 'predicted-delta-canvas', 'delta-difference-canvas']) expect(get(id).hidden).toBe(false);
    expect(get('delta-difference-info').textContent).toContain('33.33% (1/3');
    // Moving only the animation frame must also clear final-only predictions.
    replay.selectFrame(0); view.render(replay);
    expect(get('real-delta-canvas').hidden).toBe(false);
    expect(get('predicted-delta-canvas').hidden).toBe(true);
    expect(get('delta-difference-canvas').hidden).toBe(true);
    replay.selectFrame(1); view.render(replay);
    expect(get('delta-difference-canvas').hidden).toBe(false);
    expect(get('module-learning-info').textContent).toContain('optimizer steps: 2');
    expect(get('diagnostic-chart').children.length).toBeGreaterThan(0);
    expect(get('chart-values').children.length).toBe(1);
    const planned = await fixture('v11-planning'), planReplay = new Replay(planned.attempt, planned.minor);
    planReplay.select(1); view.display(planReplay); view.render(planReplay);
    expect(get('goal-value').textContent).toBe('0.4000');
    expect(get('goal-progress').textContent).toBe('30.00%');
    expect(get('chosen-action').textContent).toBe('ACTION1 (↑) · {}');
    expect(get('goal-score-label').textContent).toBe('Selected SDK-success score');
    expect(get('plan-steps').children).toHaveLength(2);
    expect(get('plan-steps').children[0]!.children[0]!.textContent).toBe('1. ACTION1 (↑)');
    expect(get('plan-steps').children[1]!.children[1]!.textContent).toContain('Imagined only');
    expect(get('plan-outcome').textContent).toContain('No progress or terminal event observed');
    const goalFile = await fixture('v11-goal-planning'), goalReplay = new Replay(goalFile.attempt, goalFile.minor);
    goalReplay.select(1); view.display(goalReplay); view.render(goalReplay);
    expect(get('goal-info').textContent).toContain('Goal hypothesis');
    expect(get('goal-info').textContent).toContain('age 0');
    expect(get('goal-info').textContent).toContain('imagined proposal');
    expect(get('goal-info').textContent).toContain('Current goal distance (latent MSE): 2.00000');
    expect(get('goal-score-label').textContent).toBe('Selected goal similarity (0–1)');
    expect(get('goal-canvas').hidden).toBe(false);
    expect(get('goal-reference-canvas').hidden).toBe(false);
    const rgba = (color: number) => [...Buffer.from(PALETTE[color]!, 'hex'), 255];
    expect([...get('goal-canvas').bitmap!.slice(0, 4)]).toEqual(rgba(4));
    expect([...get('goal-reference-canvas').bitmap!.slice(0, 4)]).toEqual(rgba(0));
    const endpointCanvas = get('plan-steps').children[1]!.children[2]!.children[0]!;
    expect([...endpointCanvas.bitmap!.slice(0, 4)]).toEqual(rgba(2));
    expect(get('plan-steps').children[1]!.children.some(c => c.textContent.includes('0.500000'))).toBe(true);
    expect(get('plan-outcome').textContent).toContain('No progress or terminal event observed');
    view.display(planReplay); view.render(planReplay);
    expect(get('goal-canvas').hidden).toBe(true);
    expect(get('goal-canvas').width).toBe(1);
    expect(get('goal-image-info').textContent).toContain('Not recorded');
    planReplay.select(0); view.render(planReplay);
    expect(get('goal-values').hidden).toBe(true);
    expect(get('goal-canvas').hidden).toBe(true);
    expect(get('goal-reference-canvas').hidden).toBe(true);
    expect(get('plan-steps').children).toHaveLength(0);
    view.clear();
    for (const id of ['prediction-canvas', 'difference-canvas', 'predicted-change-canvas', 'observed-change-canvas',
      'real-delta-canvas', 'predicted-delta-canvas', 'delta-difference-canvas']) {
      expect(get(id).hidden).toBe(true); expect(get(id).width).toBe(1);
    }
    expect(get('latent-info').textContent).toBe('Not recorded.');
    expect(get('module-learning-info').textContent).toBe('Not recorded.');
    expect(get('metric-select').children).toEqual([]);
    expect(get('diagnostic-chart').children).toEqual([]);
    const old = await fixture('recorder-baseline-0'), baseline = new Replay(old.attempt, old.minor);
    baseline.select(1); view.display(baseline); view.render(baseline);
    expect(get('goal-values').hidden).toBe(true);
    expect(get('plan-info').textContent).toContain('Not recorded');
    expect(get('latent-info').textContent).toBe('Not recorded.');
    expect(get('predicted-change-info').textContent).toContain('Not recorded');
    expect(get('module-learning-info').textContent).not.toContain('optimizer steps:');
    expect(get('recorded-errors').textContent).not.toContain('0.125');
    expect(get('metric-select').disabled).toBe(true);
    expect(get('real-delta-canvas').hidden).toBe(false);
    expect(get('predicted-delta-canvas').hidden).toBe(true);
    expect(get('delta-difference-info').textContent).toContain('Not recorded');
    baseline.select(0); view.render(baseline);
    expect(get('real-delta-canvas').hidden).toBe(true);
    expect(get('real-delta-info').textContent).toContain('Unavailable');
  } finally { vi.unstubAllGlobals(); }
});

it('propagates shared labels and keeps help keys, palette warnings and recording replacement intact', async () => {
  const elements = new Map<string, TestElement>();
  const get = (id: string) => { if (!elements.has(id)) elements.set(id, new TestElement()); return elements.get(id)!; };
  const listeners: Record<string, (event: KeyboardEvent) => void> = {};
  vi.stubGlobal('document', { getElementById: get, createElement: () => new TestElement(), createElementNS: () => new TestElement(),
    addEventListener: (name: string, listener: (event: KeyboardEvent) => void) => { listeners[name] = listener; } });
  const key = (tag: string, name: string) => {
    const preventDefault = vi.fn();
    // Simulate the target's native closest() without simulating keyboard behavior.
    const target = { closest: (selector: string) => selector.split(', ').includes(tag) ? target : null };
    listeners.keydown!({ target, key: name, preventDefault } as unknown as KeyboardEvent);
    return preventDefault;
  };
  try {
    get('episode-speed').value = '1'; get('frame-speed').value = '6';
    const view = new ReplayView();
    const decoded = await fixture('multi-frame');
    view.display(decoded, { kind: 'local', name: 'synthetic.arc3', bytes: 100 });
    expect(get('action-history').children[0]!.children[0]!.children[0]!.textContent).toBe('#0  ACTION6 (63, 0)');
    // Help summaries retain every shortcut, including their native Space/Enter.
    for (const name of ['ArrowLeft', 'ArrowRight', 'Home', 'End', ' ', 'Enter']) {
      expect(key('summary', name)).not.toHaveBeenCalled();
      expect(get('timeline-position').textContent).toBe('Position 0 / 1');
      expect(get('play').attributes['aria-pressed']).toBe('false');
    }
    for (const tag of ['input', 'select', 'textarea']) expect(key(tag, 'ArrowRight')).not.toHaveBeenCalled();
    expect(key('div', 'ArrowRight')).toHaveBeenCalledOnce();
    expect(get('timeline-position').textContent).toBe('Position 1 / 1');
    expect(get('observation-info').textContent).toContain('ACTION6 (63, 0)');
    expect(get('action-info').textContent).toBe('ACTION6 (63, 0) · {"x":63,"y":0,"extra":-2}');
    expect(get('score-rows').children[0]!.children[0]!.textContent).toBe('ACTION1 (↑) · {}');
    expect(get('score-rows').children[1]!.children[0]!.textContent).toBe('ACTION6 (1, 0) · {"x":1,"y":0}');
    expect(get('score-heading').textContent).toBe('Probability (0–1)');
    expect(key('div', ' ')).toHaveBeenCalledOnce();
    expect(get('play').attributes['aria-pressed']).toBe('true');
    expect(key('div', ' ')).toHaveBeenCalledOnce();
    expect(get('play').attributes['aria-pressed']).toBe('false');
    expect(key('div', 'Home')).toHaveBeenCalledOnce();
    expect(get('timeline-position').textContent).toBe('Position 0 / 1');
    const planned = await fixture('v11-planning');
    view.display(planned, { kind: 'local', name: 'planned.arc3', bytes: 100 });
    key('div', 'ArrowRight');
    expect(get('action-history').children[0]!.children[0]!.children[0]!.textContent).toBe('#0  ACTION1 (↑)');
    expect(get('observation-info').textContent).toContain('ACTION1 (↑) {}');
    expect(get('action-info').textContent).toBe('ACTION1 (↑) · {}');
    // A recording with an unknown pre-action palette must keep its warning visible.
    decoded.attempt.initial_observation.frames.data[0] = 255;
    view.display(decoded, { kind: 'local', name: 'palette-warning.arc3', bytes: 100 });
    expect(get('canvas-message').hidden).toBe(false);
    expect(get('canvas-message').textContent).toContain('Unknown palette indices: 255');
    key('div', 'ArrowRight');
    expect(get('previous-palette-info').hidden).toBe(false);
    expect(get('previous-palette-info').textContent).toContain('Unknown palette indices: 255');
    const empty = await fixture('zero-step');
    view.display(empty, { kind: 'local', name: 'empty.arc3', bytes: 100 });
    expect(get('timeline-position').textContent).toBe('Position 0 / 0');
    expect(get('previous-palette-info').textContent).toBe('No pre-action observation for the initial view.');
    expect(get('previous-palette-info').textContent).not.toContain('255');
    expect(get('score-table').hidden).toBe(true);
    expect(get('score-rows').children).toHaveLength(0);
    view.clear();
    expect(key('div', 'ArrowRight')).not.toHaveBeenCalled();
  } finally { vi.unstubAllGlobals(); }
});

it('keeps pre-action goal beliefs separate from the real result and ignores v1.0 planning names', async () => {
  const {attempt, minor} = await fixture('v11-planning');
  const step = attempt.steps[0]!;
  const view = planningView(step, attempt.initial_observation, minor)!;
  expect(view.steps.map(s => s.label)).toEqual(['1. ACTION1 (↑)', '2. ACTION6 (3, 5)']);
  expect(view.steps[1]!.progress).toBe(.8);
  step.observation = {...step.observation, state: 'GAME_OVER'};
  expect(planningView(step, attempt.initial_observation, minor)!.outcome).toBe('GAME_OVER observed');
  expect(planningView(step, attempt.initial_observation, minor)!.plan.state_value).toBe(.4);
  expect(planningView(step, attempt.initial_observation, 0)).toBeUndefined();
  expect(planningView(undefined, undefined, minor)).toBeUndefined();
  const metrics = diagnosticMetrics(attempt, minor);
  expect(metrics.find(m => m.key === 'planning.progress')!.read(step)).toBe(.3);
});

it('keeps the goal, imagined endpoint and real observation as distinct recorded boards', async () => {
  const {attempt, minor} = await fixture('v11-goal-planning');
  const step = attempt.steps[0]!;
  const view = planningView(step, attempt.initial_observation, minor)!;
  expect([...view.goal!.frames.data]).toEqual([4, 4, 4, 4]);
  expect([...view.steps[1]!.image!.frames.data]).toEqual([2, 2, 2, 2]);
  expect([...step.observation.frames.data]).toEqual([3, 2, 1, 0]);
  expect(view.steps.map(s => s.goal_distance)).toEqual([1, .5]);
  expect(view.age).toBe(0);
  expect(planningView(step, attempt.initial_observation, 0)).toBeUndefined();
});
