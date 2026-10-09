import { drawObservation, pixelPreview } from './canvas';
import { DiagnosticsView } from './diagnostics-view';
import { actionName, observationChanges, Replay } from './replay';
import type { DecodedAttempt } from './types';
import type { Source } from './loader';

function element<T extends HTMLElement>(id: string): T { return document.getElementById(id) as T; }
function previewText(value: string): string { return value.length > 2000 ? `${value.slice(0, 2000)}… [text preview truncated]` : value; }
function recorded(value: unknown): string {
  return JSON.stringify(value, (_key, item: unknown) => {
    if (item instanceof Uint8Array) return `[${item.length} recorded tensor bytes]`;
    return typeof item === 'string' ? previewText(item) : item;
  }, 2);
}
export class ReplayView {
  private diagnostics = new DiagnosticsView();
  private replay: Replay | undefined;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private rows: HTMLButtonElement[] = [];
  private markers: HTMLButtonElement[] = [];
  private renderedPosition = -1;
  constructor() {
    for (const id of ['observation-select', 'timeline']) element<HTMLInputElement>(id).addEventListener(id === 'timeline' ? 'input' : 'change', () => this.select(Number(element<HTMLInputElement>(id).value)));
    element('previous').addEventListener('click', () => this.select(this.replay!.position - 1));
    element('next').addEventListener('click', () => this.select(this.replay!.position + 1));
    element('restart').addEventListener('click', () => this.select(0));
    element('play').addEventListener('click', () => this.togglePlayback());
    element('animate').addEventListener('click', () => {
      const replay = this.replay!;
      replay.playing = false;
      if (!replay.animating && replay.frame === replay.observation.frames.shape[0]! - 1) replay.selectFrame(0);
      replay.animating = !replay.animating;
      this.render(); this.schedule();
    });
    element<HTMLInputElement>('frame-select').addEventListener('change', () => {
      this.replay!.playing = this.replay!.animating = false;
      this.replay!.selectFrame(Number(element<HTMLInputElement>('frame-select').value));
      this.render(); this.schedule();
    });
    for (const id of ['episode-speed', 'frame-speed']) element(id).addEventListener('change', () => this.schedule());
    document.addEventListener('keydown', event => {
      if (!this.replay || event.altKey || event.ctrlKey || event.metaKey) return;
      const target = event.target as HTMLElement;
      if (target.closest('input, select, textarea, [contenteditable], #drop-zone, #url-form')) return;
      if (event.key === ' ' && target.closest('button, a, summary')) return;
      if (event.repeat && event.key === ' ') return;
      if (!['ArrowLeft', 'ArrowRight', ' ', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      if (event.key === ' ') this.togglePlayback();
      else this.select(event.key === 'Home' ? 0 : event.key === 'End' ? this.replay.end : this.replay.position + (event.key === 'ArrowLeft' ? -1 : 1));
    });
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && this.replay) { this.replay.playing = this.replay.animating = false; this.render(); this.schedule(); }
    });
  }
  clear(): void {
    this.diagnostics.clear();
    clearTimeout(this.timer); this.timer = undefined; this.replay = undefined;
    this.rows = []; this.markers = []; this.renderedPosition = -1;
    element('result').hidden = true;
    element('action-history').replaceChildren(); element('timeline-markers').replaceChildren();
    for (const id of ['observation-canvas', 'previous-canvas']) {
      const canvas = element<HTMLCanvasElement>(id); canvas.width = canvas.height = 1; canvas.hidden = true;
    }
  }
  display(decoded: DecodedAttempt, source: Source): void {
    this.clear();
    const attempt = decoded.attempt;
    this.replay = new Replay(attempt);
    this.diagnostics.display(this.replay);
    element('public-config').textContent = attempt.metadata.config === undefined ? 'Not recorded.' : previewText(recorded(attempt.metadata.config));
    element('game-id').textContent = previewText(attempt.metadata.game_id);
    element('version').textContent = `ARC3 ${decoded.major}.${decoded.minor}`;
    element('termination').textContent = `Closed: ${attempt.termination.reason}`;
    element('attempt-identity').textContent = `Attempt ${attempt.metadata.attempt_index} · Session ${previewText(attempt.metadata.session_id)}`;
    const summary = element('attempt-summary'); summary.replaceChildren();
    for (const [label, value] of [
      ['RECORDED STEPS', String(attempt.steps.length)], ['REAL ACTIONS', String(attempt.summary.real_actions)],
      ['FINAL LEVEL COUNTER', `${attempt.summary.levels_completed} / ${attempt.summary.win_levels}`], ['WALL TIME', `${attempt.summary.wall_seconds} s`],
    ]) {
      const stat = document.createElement('div'); stat.className = 'summary-stat';
      const name = document.createElement('span'), content = document.createElement('strong');
      name.textContent = label!; content.textContent = value!; stat.append(name, content); summary.append(stat);
    }
    const metadata = element('metadata'); metadata.replaceChildren();
    const entries: [string, string][] = [
      ['Source', `${source.kind === 'local' ? 'Local file' : 'URL'} · ${source.name}`],
      ['Size', `${source.bytes.toLocaleString()} bytes · ${decoded.uncompressedBytes.toLocaleString()} bytes uncompressed`],
      ['Run / session', `${attempt.metadata.run_id} / ${attempt.metadata.session_id}`],
      ['Attempt / seed', `${attempt.metadata.attempt_index} / ${attempt.metadata.seed}`],
      ['SDK / split', `${attempt.metadata.sdk_version} / ${attempt.metadata.source_split}`],
      ['Started', attempt.metadata.started_at], ['Model', attempt.metadata.model_id ?? 'Unavailable (not recorded)'],
      ['Checkpoint', attempt.metadata.checkpoint_id ?? 'Unavailable (not recorded)'],
      ['Termination', `${attempt.termination.reason} · ${attempt.termination.final_state}`],
      ['Summary', `${attempt.summary.real_actions} real actions · ${attempt.summary.levels_completed}/${attempt.summary.win_levels} levels · ${attempt.summary.wall_seconds} seconds`],
    ];
    if (source.lastModified !== undefined) entries.push(['Local file modified', new Date(source.lastModified).toISOString()]);
    for (const [key, value] of entries) {
      const dt = document.createElement('dt'), dd = document.createElement('dd');
      dt.textContent = key; dd.textContent = previewText(value); metadata.append(dt, dd);
    }
    element('recorded-fields').textContent = recorded({ termination: attempt.termination, summary: attempt.summary });
    element('sequence-info').textContent = `Initial observation + ${attempt.steps.length} recorded transitions. Position 0 = initial; position n = result of step n − 1. Frames belong to one action, not extra steps.`;
    element('history-count').textContent = `${attempt.summary.real_actions} real actions`;
    element('history-empty').hidden = this.replay.end !== 0;
    const unrecorded = attempt.summary.real_actions - attempt.steps.length - (attempt.termination.reset_action ? 1 : 0);
    element('boundary-info').textContent = [
      `Attempt ended: ${attempt.termination.reason} · ${previewText(attempt.termination.final_state)}.`,
      attempt.termination.reset_action ? 'Closing RESET was executed. Its returned observation belongs to the next attempt; no result is stored here.' : attempt.termination.reason === 'reset' ? 'Reset requested; no executed RESET was recorded.' : '',
      unrecorded ? `${unrecorded} executed action has no persisted result. The final summary may differ from the last recorded observation.` : '',
      attempt.termination.detail ? previewText(attempt.termination.detail) : '',
    ].filter(Boolean).join(' ');
    this.buildNavigation();
    for (const id of ['observation-select', 'timeline']) element<HTMLInputElement>(id).max = String(this.replay.end);
    element('result').hidden = false;
    this.render();
  }
  private buildNavigation(): void {
    const replay = this.replay!, history = element('action-history'), markers = element('timeline-markers');
    history.replaceChildren(); markers.replaceChildren();
    for (let position = 0; position <= replay.end; position++) {
      const step = replay.attempt.steps[position - 1];
      const before = position > 1 ? replay.attempt.steps[position - 2]!.observation : replay.attempt.initial_observation;
      const marker = document.createElement('button'); marker.type = 'button'; marker.className = 'marker';
      const levelChange = step && before.levels_completed !== step.observation.levels_completed;
      const terminal = step && ['WIN', 'GAME_OVER'].includes(step.observation.state);
      const label = position === 0 ? 'Initial' : step ? `Step ${step.index}${levelChange ? ` · L${step.observation.levels_completed}` : ''}${terminal ? ` · ${step.observation.state}` : ''}` : 'RESET boundary';
      marker.textContent = label;
      marker.setAttribute('aria-label', `Jump to ${label}`);
      if (levelChange) marker.classList.add('level-marker');
      if (terminal || position === replay.end) marker.classList.add('terminal-marker');
      marker.addEventListener('click', () => this.select(position)); markers.append(marker); this.markers.push(marker);
      if (position === 0) continue;
      const li = document.createElement('li'), row = document.createElement('button'); row.type = 'button'; row.className = 'action-row';
      const title = document.createElement('strong'), detail = document.createElement('span');
      title.textContent = step ? `#${step.index}  ${actionName(step.action)}` : 'RESET · closing boundary';
      detail.textContent = step ? observationChanges(before, step.observation) : 'Executed · returned observation not in this attempt';
      row.append(title, detail); row.addEventListener('click', () => this.select(position));
      li.append(row); history.append(li); this.rows.push(row);
    }
  }
  private select(position: number): void { if (!this.replay) return; this.replay.select(position); this.render(); this.schedule(); }
  private togglePlayback(): void { this.replay!.togglePlayback(); this.render(); this.schedule(); }
  private schedule(): void {
    clearTimeout(this.timer); this.timer = undefined;
    const replay = this.replay;
    if (!replay || (!replay.playing && !replay.animating)) return;
    const framePending = replay.frame < replay.observation.frames.shape[0]! - 1;
    const delay = replay.animating || framePending ? 1000 / Number(element<HTMLSelectElement>('frame-speed').value) : 1000;
    const speed = replay.playing ? Number(element<HTMLSelectElement>('episode-speed').value) : 1;
    this.timer = setTimeout(() => {
      if (replay.playing) replay.advance(); else replay.advanceFrame();
      this.render(); this.schedule();
    }, delay / speed);
  }
  private render(): void {
    const replay = this.replay!, o = replay.observation, step = replay.step;
    const positionLabel = replay.resetBoundary ? 'Closing RESET · pre-reset observation' : step ? `Step ${step.index} result` : 'Initial observation';
    const frameCount = o.frames.shape[0]!;
    element('position-label').textContent = positionLabel;
    element('current-level').textContent = `Observed levels: ${o.levels_completed}/${o.win_levels}`;
    element('timeline-position').textContent = `Position ${replay.position} / ${replay.end}`;
    for (const id of ['observation-select', 'timeline']) element<HTMLInputElement>(id).value = String(replay.position);
    element<HTMLInputElement>('timeline').disabled = replay.end === 0;
    element('previous').toggleAttribute('disabled', replay.position === 0);
    element('next').toggleAttribute('disabled', replay.position === replay.end);
    element('play').toggleAttribute('disabled', replay.end === 0);
    element('play').textContent = replay.playing ? 'Ⅱ Pause episode' : '▶ Play episode';
    element('play').setAttribute('aria-pressed', String(replay.playing));
    const frames = element<HTMLInputElement>('frame-select');
    frames.value = String(replay.frame); frames.max = String(Math.max(0, frameCount - 1)); frames.disabled = frameCount === 0;
    element('frame-count').textContent = `${frameCount} returned ${frameCount === 1 ? 'frame' : 'frames'}`;
    element('animate').toggleAttribute('disabled', frameCount < 2 || replay.resetBoundary);
    element('animate').textContent = replay.animating ? 'Pause frames' : 'Play frames';
    element('animate').setAttribute('aria-pressed', String(replay.animating));
    const message = drawObservation(element<HTMLCanvasElement>('observation-canvas'), o, replay.frame);
    element('canvas-message').textContent = message; element('canvas-message').hidden = !message;
    element('observation-canvas').setAttribute('aria-label', `${positionLabel} · frame ${replay.frame} · ${o.frames.shape[2]} by ${o.frames.shape[1]} · ${previewText(o.state)}`);
    element('pixels').textContent = pixelPreview(o, replay.frame);
    element('observation-info').textContent = `${previewText(o.state)} · levels ${o.levels_completed}/${o.win_levels} · shape [${o.frames.shape.join(', ')}] · available actions [${o.available_actions.join(', ')}]` + (step ? ` · ACTION${step.action.id} ${JSON.stringify(step.action.data)}` : replay.resetBoundary ? ' · RESET result unavailable' : ' · Initial observation');
    this.diagnostics.render(replay);
    if (this.renderedPosition === replay.position) return;
    this.renderedPosition = replay.position;
    const pre = replay.preAction;
    const previousMessage = drawObservation(element<HTMLCanvasElement>('previous-canvas'), pre, Math.max(0, (pre?.frames.shape[0] ?? 0) - 1));
    const previousVisible = !element('previous-canvas').hidden;
    element('previous-message').textContent = pre ? 'No returned frame' : 'Initial view';
    element('previous-message').hidden = previousVisible;
    element('previous-palette-info').textContent = previousMessage;
    element('previous-palette-info').hidden = !previousMessage;
    element('previous-info').textContent = pre ? `Last frame · ${previewText(pre.state)} · levels ${pre.levels_completed}/${pre.win_levels}` : 'Initial view; no action precedes it.';
    element('action-info').textContent = step ? `${actionName(step.action)} · ${JSON.stringify(step.action.data)}` : replay.resetBoundary ? 'RESET executed · no returned frame in this file' : 'Select a step to inspect its action and result.';
    this.rows.forEach((row, i) => { row.setAttribute('aria-current', String(i + 1 === replay.position)); row.classList.toggle('future', i + 1 > replay.position); });
    this.markers.forEach((marker, i) => marker.setAttribute('aria-current', String(i === replay.position)));
  }
}
