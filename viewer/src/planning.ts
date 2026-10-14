import { actionName } from './replay';
import type { Observation, Step, Tensor } from './types';

export function boardObservation(frame: Tensor | undefined): Observation | undefined {
  return frame && { frames: { ...frame, shape: [1, ...frame.shape] }, state: 'IMAGINED',
    levels_completed: 0, win_levels: 0, available_actions: [] };
}

/** Recorded pre-action beliefs and separately observed SDK outcome. No inference. */
export function planningView(step: Step | undefined, previous: Observation | undefined, minor: number) {
  const plan = minor >= 1 ? step?.planning : undefined;
  if (!plan || !step) return undefined;
  const actual = step.observation;
  const outcome = actual.state === 'WIN' ? 'WIN observed' :
    previous && actual.levels_completed > previous.levels_completed ? 'Level progress observed' :
      actual.state === 'GAME_OVER' ? 'GAME_OVER observed' : 'No progress or terminal event observed';
  return { plan, outcome, reference: previous, goal: boardObservation(plan.goal?.frame),
    age: plan.goal ? step.index - plan.goal.created_position : undefined,
    steps: plan.steps.map((s, i) => ({...s, image: boardObservation(s.frame),
    label: `${i + 1}. ${actionName(s.action)}`,
    status: i === 0 ? 'Executed in the real environment' : 'Imagined only · not executed from this plan'})) };
}
