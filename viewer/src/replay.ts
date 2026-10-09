import type { Action, Attempt, Observation, Step } from './types';

export function actionName(action: Action): string {
  const coordinates = 'x' in action.data && 'y' in action.data ? ` (${action.data.x}, ${action.data.y})` : '';
  return `${action.id === 0 ? 'RESET' : `ACTION${action.id}`}${coordinates}`;
}

export function observationChanges(before: Observation, after: Observation): string {
  const a = before.frames, b = after.frames;
  const changed = a.shape.join() !== b.shape.join() || a.data.length !== b.data.length || a.data.some((v, i) => v !== b.data[i]);
  const flags = [changed ? 'Frames changed' : 'Frames unchanged'];
  if (before.levels_completed !== after.levels_completed) flags.push(`Level counter ${before.levels_completed} → ${after.levels_completed}`);
  if (before.state !== after.state) flags.push(`State ${after.state}`);
  if (['WIN', 'GAME_OVER'].includes(after.state)) flags.push('Terminal');
  return flags.join(' · ');
}

/** Positions are initial (0), step results (index + 1), then an optional RESET boundary.
 * RESET has no returned observation in this file; selection retains its pre-reset truth.
 */
export class Replay {
  position = 0;
  frame = 0;
  playing = false;
  animating = false;
  constructor(readonly attempt: Attempt) {}
  get end(): number { return this.attempt.steps.length + (this.attempt.termination.reset_action ? 1 : 0); }
  get resetBoundary(): boolean { return this.position > this.attempt.steps.length; }
  get step(): Step | undefined { return this.resetBoundary ? undefined : this.attempt.steps[this.position - 1]; }
  get observation(): Observation {
    return this.attempt.steps[Math.min(this.position, this.attempt.steps.length) - 1]?.observation ?? this.attempt.initial_observation;
  }
  get preAction(): Observation | undefined {
    if (this.position === 0) return undefined;
    if (this.resetBoundary) return this.observation;
    return this.attempt.steps[this.position - 2]?.observation ?? this.attempt.initial_observation;
  }
  select(position: number): void {
    this.position = Math.min(this.end, Math.max(0, Math.trunc(position) || 0));
    this.frame = this.resetBoundary ? Math.max(0, this.observation.frames.shape[0]! - 1) : 0;
    this.playing = false;
    this.animating = false;
  }
  selectFrame(frame: number): void {
    this.frame = Math.min(Math.max(0, this.observation.frames.shape[0]! - 1), Math.max(0, Math.trunc(frame) || 0));
  }
  togglePlayback(): void {
    if (this.end === 0) return;
    if (!this.playing && this.position === this.end && (this.resetBoundary || this.frame >= this.observation.frames.shape[0]! - 1)) this.select(0);
    this.animating = false;
    this.playing = !this.playing;
  }
  advance(): void {
    if (this.frame < this.observation.frames.shape[0]! - 1) this.frame++;
    else if (this.position < this.end) {
      this.position++;
      this.frame = this.resetBoundary ? Math.max(0, this.observation.frames.shape[0]! - 1) : 0;
    }
    else this.playing = false;
  }
  advanceFrame(): void {
    if (this.frame < this.observation.frames.shape[0]! - 1) this.frame++;
    else this.animating = false;
  }
}
