/** Known frozen v1 fields. Optional properties stay absent when not recorded. */
export type Value = null | boolean | number | string | Uint8Array | Value[] | { [key: string]: Value };
export type JSONValue = null | boolean | number | string | JSONValue[] | { [key: string]: JSONValue };
export interface Tensor { dtype: 'u8' | 'f16' | 'f32'; shape: number[]; data: Uint8Array }
export interface Observation {
  frames: Tensor; state: string; levels_completed: number; win_levels: number; available_actions: number[];
}
export interface Action { id: number; data: Record<string, number> }
export interface Metadata {
  game_id: string; run_id: string; session_id: string; attempt_index: number; seed: number;
  sdk_version: string; source_split: 'train' | 'evaluation' | 'unspecified'; started_at: string;
  model_id?: string; checkpoint_id?: string; config?: JSONValue;
}
export interface Decision {
  score_type: 'logit' | 'probability' | 'value'; candidates: { action: Action; score: number }[];
  selected_value?: number; action_entropy?: number;
}
export interface Prediction {
  target: 'post_action_last_frame' | 'post_action_sequence'; frames?: Tensor; latent?: Tensor;
  uncertainty?: number; errors?: { mae?: number; mse?: number; [key: string]: Value | undefined };
}
export interface Learning { updates?: number; loss?: number; replay_size?: number }
export interface Step {
  index: number; action: Action; observation: Observation; decision?: Decision; prediction?: Prediction;
  learning?: Learning; timing?: Record<string, number>; notes?: string[];
}
export interface Termination {
  reason: 'win' | 'game_over' | 'reset' | 'action_budget' | 'timeout' | 'error' | 'interrupted' | 'training_level_boundary';
  final_state: string; detail?: string; reset_action?: Action;
}
export interface Summary { real_actions: number; levels_completed: number; win_levels: number; wall_seconds: number }
export interface Attempt {
  metadata: Metadata; initial_observation: Observation; steps: Step[]; termination: Termination; summary: Summary;
}
export interface DecodedAttempt { major: 1; minor: number; uncompressedBytes: number; attempt: Attempt }
