import { drawObservation } from './canvas';
import { chartPoints, comparePrediction, diagnosticMetrics, latentSummary, MAX_CHART_STEPS, type Metric } from './diagnostics';
import { actionName, observationChanges, type Replay } from './replay';
import type { Attempt } from './types';

function element<T extends HTMLElement>(id: string): T { return document.getElementById(id) as T; }
function numeric(value: number | undefined): string { return value === undefined ? 'Not recorded' : String(value); }
function shortText(text: string): string { return text.length > 2000 ? `${text.slice(0, 2000)}… [preview truncated]` : text; }
const svgNS = 'http://www.w3.org/2000/svg';
function svgNode(tag: string, attributes: Record<string, string>): SVGElement {
  const node = document.createElementNS(svgNS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}

/** All diagnostics come from the opened trace. No model execution or network. */
export class DiagnosticsView {
  private replay: Replay | undefined;
  private metrics: Metric[] = [];
  private lastPosition = -1;
  private lastFrame = -1;
  constructor() {
    for (const id of ['metric-select', 'chart-from', 'chart-to']) element(id).addEventListener('change', () => this.renderChart());
  }
  clear(): void {
    this.replay = undefined; this.metrics = []; this.lastPosition = this.lastFrame = -1;
    for (const id of ['prediction-canvas', 'difference-canvas']) {
      const canvas = element<HTMLCanvasElement>(id); canvas.width = canvas.height = 1; canvas.hidden = true;
    }
    element('diagnostic-chart').replaceChildren(); element('chart-values').replaceChildren();
    element('score-rows').replaceChildren(); element('metric-select').replaceChildren();
  }
  display(replay: Replay): void {
    this.clear(); this.replay = replay; this.metrics = diagnosticMetrics(replay.attempt);
    const select = element<HTMLSelectElement>('metric-select');
    for (const metric of this.metrics) {
      const option = document.createElement('option'); option.value = metric.key; option.textContent = metric.label; select.append(option);
    }
    if (!this.metrics.length) {
      const option = document.createElement('option'); option.textContent = 'Not recorded'; select.append(option);
    }
    select.disabled = !this.metrics.length;
    for (const id of ['chart-from', 'chart-to']) {
      const input = element<HTMLInputElement>(id); input.disabled = !this.metrics.length;
      input.max = String(Math.max(0, replay.attempt.steps.length - 1));
      input.value = id === 'chart-from' ? '0' : String(Math.min(MAX_CHART_STEPS - 1, Math.max(0, replay.attempt.steps.length - 1)));
    }
  }
  render(replay: Replay): void {
    const positionChanged = replay.position !== this.lastPosition;
    if (!positionChanged && replay.frame === this.lastFrame) return;
    this.lastPosition = replay.position; this.lastFrame = replay.frame;
    const step = replay.step, prediction = step?.prediction;
    const visual = comparePrediction(prediction, replay.observation, replay.frame);
    const paletteMessage = drawObservation(element<HTMLCanvasElement>('prediction-canvas'), visual.predicted, 0);
    drawObservation(element<HTMLCanvasElement>('difference-canvas'), visual.difference, 0);
    element('prediction-info').textContent = visual.message + (visual.predicted && paletteMessage ? `\n${paletteMessage}` : '');
    element('prediction-canvas').setAttribute('aria-label', `Model prediction aligned to step ${step?.index}, real frame ${replay.frame}`);
    element('difference-canvas').setAttribute('aria-label', `Pixel disagreements for step ${step?.index}, real frame ${replay.frame}`);
    element('difference-info').textContent = visual.fraction === undefined ? 'Not available: no comparable decoded visual prediction.' :
      `Pixel disagreement fraction: ${visual.fraction} (${visual.disagreements}/${visual.total} pixels; ${(visual.fraction * 100).toFixed(2)}%)\nSelected real frame ${replay.frame} · exact palette-index values before display rounding. Red = disagreement; black = equal.`;
    element('prediction-values').textContent = visual.values ? JSON.stringify(visual.values) + '\nRow-major values, limited to the top-left 32 × 32.' : 'Not recorded for this frame.';
    if (!positionChanged) return;
    element('latent-info').textContent = latentSummary(prediction?.latent);
    const decision = step?.decision;
    element('chosen-action').textContent = step ? `${actionName(step.action)} · ${JSON.stringify(step.action.data)}` : replay.resetBoundary ? 'RESET · executed closing boundary; no result recorded' : 'No action precedes the initial observation.';
    element('selected-value').textContent = `Recorded selected-action value estimate: ${numeric(decision?.selected_value)}`;
    element('score-info').textContent = decision ? `Recorded ${decision.score_type} scores · writer order; candidates may be a subset.` : 'Not recorded for this position.';
    const rows = element('score-rows'); rows.replaceChildren();
    element('score-table').hidden = !decision;
    element('score-heading').textContent = decision ? `${decision.score_type[0]!.toUpperCase()}${decision.score_type.slice(1)} (recorded)` : 'Score';
    // Bound DOM work; preserve the writer's order and disclose a large-table preview.
    decision?.candidates.slice(0, 256).forEach(candidate => {
      const row = document.createElement('tr'), action = document.createElement('td'), score = document.createElement('td');
      action.textContent = shortText(`${actionName(candidate.action)} · ${JSON.stringify(candidate.action.data)}`);
      score.textContent = String(candidate.score); row.append(action, score); rows.append(row);
    });
    if (decision && decision.candidates.length > 256) element('score-info').append(` Showing first 256 of ${decision.candidates.length} candidates.`);
    element('uncertainty-info').textContent = `Prediction uncertainty: ${numeric(prediction?.uncertainty)} (recorded estimator units; public config below)\nAction entropy: ${numeric(decision?.action_entropy)} (nats)`;
    element('recorded-errors').textContent = `Recorded target MAE (palette-index units): ${numeric(prediction?.errors?.mae)}\nRecorded target MSE (squared palette-index units): ${numeric(prediction?.errors?.mse)}\nThese target-wide recorded errors are separate from the viewer's selected-frame disagreement.`;
    const o = replay.observation;
    element('level-progress-info').textContent = `${o.levels_completed} / ${o.win_levels} public levels completed` + (replay.preAction && replay.preAction.levels_completed !== o.levels_completed ? ` · counter ${replay.preAction.levels_completed} → ${o.levels_completed}` : '') + (o.win_levels === 0 ? ' · target unavailable (zero)' : '');
    const progress = element<HTMLProgressElement>('level-progress'); progress.hidden = o.win_levels === 0;
    progress.max = o.win_levels || 1; progress.value = o.levels_completed;
    element('learning-info').textContent = `Optimizer updates: ${numeric(step?.learning?.updates)}\nLearning loss (recorded objective): ${numeric(step?.learning?.loss)}\nReplay size (entries): ${numeric(step?.learning?.replay_size)}`;
    element('timing-info').textContent = step?.timing && Object.keys(step.timing).length ? shortText(Object.entries(step.timing).map(([key, value]) => `${key}: ${value} ms`).join('\n')) : 'Not recorded for this position.';
    element('rationale-info').textContent = step?.notes?.length ? `Recorded plain-text notes:\n${step.notes.slice(0, 20).map(shortText).join('\n')}${step.notes.length > 20 ? '\n[First 20 notes shown]' : ''}` : 'Recorded plain-text notes: Not recorded.';
    element('inferred-event').textContent = step && replay.preAction ? `Inferred by viewer from public observations: ${observationChanges(replay.preAction, step.observation)}` : 'Inferred by viewer: no recorded transition at this position.';
    this.renderChart();
  }
  private renderChart(): void {
    const replay = this.replay;
    if (!replay) return;
    const chart = document.getElementById('diagnostic-chart')!;
    chart.replaceChildren(); element('chart-values').replaceChildren();
    const metric = this.metrics.find(metric => metric.key === element<HTMLSelectElement>('metric-select').value);
    chart.setAttribute('aria-label', metric?.label ?? 'No recorded diagnostic chart');
    if (!metric) { element('chart-info').textContent = 'Not recorded: no uncertainty, entropy, learning or standard timing samples.'; return; }
    this.plot(replay.attempt, metric, chart);
  }
  private plot(attempt: Attempt, metric: Metric, chart: Element): void {
    const from = element<HTMLInputElement>('chart-from').valueAsNumber, to = element<HTMLInputElement>('chart-to').valueAsNumber;
    const points = chartPoints(attempt, metric, from, to);
    if (!points.length) { element('chart-info').textContent = `Choose an inclusive range of up to ${MAX_CHART_STEPS} recorded steps (zero-based).`; return; }
    const values = points.flatMap(point => point.value === undefined ? [] : [point.value]);
    const table = element('chart-values');
    points.forEach(point => {
      const row = document.createElement('tr'), step = document.createElement('td'), value = document.createElement('td');
      step.textContent = String(point.step); value.textContent = numeric(point.value); row.append(step, value); table.append(row);
    });
    element('chart-info').textContent = `${metric.label} · steps ${points[0]!.step}–${points.at(-1)!.step} · ${values.length} recorded samples; ${points.length - values.length} missing. Gaps remain missing; values are not interpolated. Selected step: ${this.replay?.step?.index ?? 'none'}.`;
    if (!values.length) return;
    const min = Math.min(...values), max = Math.max(...values), scale = Math.max(Math.abs(min), Math.abs(max)) || 1;
    const span = max / scale - min / scale;
    const x = (index: number): number => 50 + (points.length === 1 ? 240 : index / (points.length - 1) * 480);
    const y = (value: number): number => span === 0 ? 95 : 160 - (value / scale - min / scale) / span * 130;
    chart.append(svgNode('path', { d: 'M50 30 V160 H530', class: 'chart-axis' }));
    for (const [text, px, py] of [[String(max), 5, 23], [String(min), 5, 183], [`Step ${points[0]!.step}`, 50, 204], [`Step ${points.at(-1)!.step}`, 440, 204]] as const) {
      const label = svgNode('text', { x: String(px), y: String(py) }); label.textContent = text; chart.append(label);
    }
    let previous: { x: number; y: number } | undefined;
    points.forEach((point, i) => {
      if (point.value === undefined) { previous = undefined; return; }
      const px = x(i), py = y(point.value);
      if (previous) chart.append(svgNode('path', { d: `M${previous.x} ${previous.y} L${px} ${py}`, class: 'chart-line' }));
      const dot = svgNode('circle', { cx: String(px), cy: String(py), r: point.step === this.replay?.step?.index ? '5' : '3', class: point.step === this.replay?.step?.index ? 'chart-selected' : 'chart-point' });
      const title = svgNode('title', {}); title.textContent = `Step ${point.step}: ${point.value}`; dot.append(title); chart.append(dot);
      previous = { x: px, y: py };
    });
  }
}
