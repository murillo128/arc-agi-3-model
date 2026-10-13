import { actionName } from './replay';
import type { Observation, Step } from './types';

/** Recorded pre-action beliefs and separately observed SDK outcome. No inference. */
export function planningView(step: Step | undefined, previous: Observation | undefined, minor: number) {
  const plan = minor >= 1 ? step?.planning : undefined;
  if (!plan || !step) return undefined;
  const actual = step.observation;
  const outcome = actual.state === 'WIN' ? 'WIN observed' :
    previous && actual.levels_completed > previous.levels_completed ? 'Level progress observed' :
      actual.state === 'GAME_OVER' ? 'GAME_OVER observed' : 'No progress or terminal event observed';
  return { plan, outcome, steps: plan.steps.map((s, i) => ({...s,
    label: `${i + 1}. ${actionName(s.action)}`,
    status: i === 0 ? 'Executed in the real environment' : 'Imagined only · not executed from this plan'})) };
}
