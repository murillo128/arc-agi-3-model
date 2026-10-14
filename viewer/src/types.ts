/** Known v1 fields; 1.1 additions are interpreted only for wire minor >= 1. */
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
export interface Representation { z: Tensor; h: Tensor }
export interface Dynamics extends Representation { delta_z: Tensor; h_next: Tensor }
export interface VisualDelta { reference_position: number; reference_frame: Tensor; delta: Tensor }
export interface Prediction {
  target: 'post_action_last_frame' | 'post_action_sequence'; frames?: Tensor; latent?: Tensor;
  uncertainty?: number; errors?: { mae?: number; mse?: number; latent_mse?: number; no_change_mse?: number; [key: string]: Value | undefined };
  dynamics?: Dynamics; visual_delta?: VisualDelta;
}
export type ModuleName = 'world_model' | 'decoder' | 'actor' | 'critic';
export interface ModuleLearning { updates?: number; loss?: number; losses?: Record<string, number>; reset_reason?: string }
export interface Learning { updates?: number; loss?: number; replay_size?: number; modules?: Partial<Record<ModuleName, ModuleLearning>> }
export interface Planning {
  objective: 'sdk_level_progress' | 'latent_goal'; mode: 'q' | 'mpc' | 'goal_mpc'; horizon: number; beam: number;
  discount: number; state_value: number; selected_score: number;
  goal?: { source: 'imagined_proposal'; created_position: number; proposal_score: number; distance: number; frame: Tensor };
  steps: { action: Action; continue: number; progress: number; game_over: number; successor_value?: number; frame?: Tensor; goal_distance?: number }[];
}
export interface Step {
  index: number; action: Action; observation: Observation; decision?: Decision; prediction?: Prediction;
  learning?: Learning; timing?: Record<string, number>; notes?: string[];
  representation?: Representation; policy_mode?: 'random' | 'learned';
  planning?: Planning;
}
export interface Termination {
  reason: 'win' | 'game_over' | 'reset' | 'action_budget' | 'timeout' | 'error' | 'interrupted' | 'training_level_boundary';
  final_state: string; detail?: string; reset_action?: Action;
}
export interface Summary { real_actions: number; levels_completed: number; win_levels: number; wall_seconds: number }
export interface Attempt {
  metadata: Metadata; initial_observation: Observation; steps: Step[]; termination: Termination; summary: Summary;
  initial_representation?: Representation;
}
export interface DecodedAttempt { major: 1; minor: number; uncompressedBytes: number; attempt: Attempt }
