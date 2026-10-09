import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { decodeAttempt } from '../src/decoder';
import { actionName, observationChanges, Replay } from '../src/replay';

async function replay(name: string): Promise<Replay> {
  const { attempt } = await decodeAttempt(new Uint8Array(readFileSync(new URL(`../.fixtures/${name}.arc3`, import.meta.url))));
  return new Replay(attempt);
}

describe('replay chronology from Python-encoded synthetic recordings', () => {
  it('zero steps retain the initial observation and cannot start an episode', async () => {
    const view = await replay('zero-step');
    view.togglePlayback(); view.select(99);
    expect(view.position).toBe(0); expect(view.end).toBe(0); expect(view.playing).toBe(false);
    expect(view.preAction).toBeUndefined(); expect(view.step).toBeUndefined();
    expect([...view.observation.frames.data]).toEqual([0, 1, 2, 3]);
  });
  it('step selection uses the actual result and the preceding observation', async () => {
    const view = await replay('multi-frame'); view.select(1);
    expect(view.step?.index).toBe(0); expect(view.frame).toBe(0);
    expect(actionName(view.step!.action)).toBe('ACTION6 (63, 0)');
    expect([...view.preAction!.frames.data]).toEqual([0, 1, 2, 3]);
    view.selectFrame(99); expect(view.frame).toBe(2);
    expect([...view.observation.frames.data.slice(8)]).toEqual([255, 16, 7, 0]);
    view.select(-1); expect(view.position).toBe(0); expect(view.frame).toBe(0);
  });
  it('episode playback visits every frame before the next action and stops at the end', async () => {
    const view = await replay('replay-levels'); view.togglePlayback();
    const visited = [[view.position, view.frame]];
    while (view.playing) { view.advance(); visited.push([view.position, view.frame]); }
    expect(visited).toEqual([[0, 0], [1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 0]]);
    expect(view.observation.state).toBe('WIN');
    view.togglePlayback(); expect(view.position).toBe(0); expect(view.playing).toBe(true);
    view.select(2); expect(view.playing).toBe(false); expect(view.observation.levels_completed).toBe(1);
  });
  it('frame playback stops without advancing to another action', async () => {
    const view = await replay('multi-frame'); view.select(1); view.animating = true;
    view.advanceFrame(); expect(view.frame).toBe(1);
    view.advanceFrame(); expect(view.frame).toBe(2);
    view.advanceFrame(); expect(view.animating).toBe(false); expect(view.position).toBe(1);
  });
  it('closing RESET is counted separately and displays pre-reset truth without inventing a result', async () => {
    const view = await replay('closing-reset'); view.select(2);
    expect(view.resetBoundary).toBe(true); expect(view.step).toBeUndefined(); expect(view.frame).toBe(2);
    expect(view.preAction).toBe(view.observation); expect(view.observation.state).toBe('NOT_FINISHED');
    expect(view.attempt.summary.real_actions).toBe(2); expect(view.attempt.steps).toHaveLength(1);
    expect(actionName(view.attempt.termination.reset_action!)).toBe('RESET');
  });
  it('reports changes in frames, public level counters, and terminal state', async () => {
    const view = await replay('replay-levels'); view.select(3);
    expect(observationChanges(view.preAction!, view.observation)).toBe('Frames changed · Level counter 1 → 2 · State WIN · Terminal');
    expect(observationChanges(view.observation, view.observation)).toBe('Frames unchanged · Terminal');
  });
});
