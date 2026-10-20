from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SequenceConfig:
    batch_games: int = 4
    session_steps: int = 512
    block_steps: int = 16
    context_steps: int = 16
    level_steps: int = 32
    loss_start_weight: float = 1.
    loss_warmup_steps: int = 32
    game_schedule: str = 'interleaved'
    representation_updates: int = 4

    def __post_init__(self):
        for name in ('batch_games', 'session_steps', 'block_steps', 'level_steps', 'loss_warmup_steps'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.block_steps > self.session_steps:
            raise ValueError('block_steps cannot exceed session_steps')
        if type(self.representation_updates) is not int or self.representation_updates < 0:
            raise ValueError('representation_updates must be a nonnegative integer')
        if type(self.context_steps) is not int or self.context_steps < 0:
            raise ValueError('context_steps must be a nonnegative integer')
        if (type(self.loss_start_weight) not in (int, float) or
                not math.isfinite(self.loss_start_weight) or not 0 < self.loss_start_weight <= 1):
            raise ValueError('loss_start_weight must be in (0, 1]')
        if self.game_schedule not in ('interleaved', 'blocked', 'shuffled'):
            raise ValueError('game_schedule must be interleaved, blocked or shuffled')


@dataclass(frozen=True)
class NetworkConfig:
    width: int = 768
    heads: int = 12
    encoder_depth: int = 12
    predictor_depth: int = 6
    decoder_depth: int = 8
    latent: int = 1024
    memory: int = 2048
    animation_memory: int = 1024
    action_embedding: int = 128
    patch: int = 4
    sigreg_slices: int = 1024
    residual_scale: float = 1.
    position_scale: float = 1.
    pixel_latent: bool = False
    broadcast_conditioning: bool = False
    local_dynamics: bool = False

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for k, v in asdict(self).items()
               if k not in {'residual_scale', 'position_scale', 'pixel_latent', 'broadcast_conditioning', 'local_dynamics'}):
            raise ValueError('Network sizes must be positive integers')
        if any(not math.isfinite(v) or not 0 < v <= 1 for v in (self.residual_scale, self.position_scale)):
            raise ValueError('Residual and position scales must lie in (0, 1]')
        if any(type(v) is not bool for v in (self.pixel_latent, self.broadcast_conditioning, self.local_dynamics)):
            raise ValueError('Network feature switches must be booleans')
        if self.local_dynamics and not self.pixel_latent:
            raise ValueError('Local dynamics requires pixel-structured latents')
        if self.pixel_latent and (self.latent % self.patch ** 2 or self.latent // self.patch ** 2 < 2):
            raise ValueError('Pixel latents require at least two channels per native pixel')
        if self.width % self.heads or self.width % 4:
            raise ValueError('Width must be divisible by heads and four')

    @classmethod
    def small(cls):
        return cls(width=32, heads=4, encoder_depth=1, predictor_depth=1,
                   decoder_depth=1, latent=16, memory=32, animation_memory=16,
                   action_embedding=8, sigreg_slices=32)


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 3e-5
    sigreg_weight: float = .002
    local_reconstruction_weight: float = 1.
    delta_overlap_weight: float = 1.
    false_change_weight: float = 1.
    prediction_delta_loss: str = 'balanced'
    freeze_representation: bool = False
    auto_representation: bool = False
    representation_unfreeze_error: float = .02
    representation_freeze_error: float = .01
    representation_gate_patience: int = 2
    variance_weight: float = 10.
    train_every: int = 1
    updates_per_train: int = 1
    frame_chunk: int = 4
    loss_steps: int = 1
    rollout_weight: float = 0.
    rollout_visual_weight: float = 0.
    decoder_readout: str = 'contextual'
    backward_mode: str = 'fragmented'
    rollout_diagnostics: bool = False
    retain_replay_across_levels: bool = False
    representation_joint_steps: int = 0
    burn_in_steps: int = 0
    cache_frozen_latents: bool = False
    replay_episodes: int = 1
    batch_episodes: int = 1
    replay_bytes: int = 128 * 1024 * 1024
    min_observations: int = 2  # Regularization only; never a predictive-loss gate.
    max_episode_steps: int = 256
    validate_every: int = 1000
    validation_episodes: int = 8
    validation_steps: int = 128
    min_validation_transitions: int = 128
    min_changed_transitions: int = 32
    delta_error: float = 0.01
    latent_delta_mse: float = 0.01
    min_latent_std: float = 0.1
    collapse_std: float = 0.01
    collapse_patience: int = 3
    collapse_after_updates: int = 256
    consecutive_passes: int = 3
    checkpoint_every: int = 1000
    max_level_steps: int = 100_000
    max_steps: int = 500_000

    def __post_init__(self):
        for k, v in asdict(self).items():
            if k in {'decoder_readout', 'backward_mode', 'prediction_delta_loss'}:
                allowed = {'decoder_readout': {'contextual', 'local'},
                           'backward_mode': {'fragmented', 'ordinary'},
                           'prediction_delta_loss': {'balanced', 'union', 'focal', 'recall_focal', 'recall_focal_capped'}}[k]
                if v not in allowed:
                    raise ValueError(f'Invalid {k}: {v}')
                continue
            if k in {'freeze_representation', 'auto_representation', 'rollout_diagnostics',
                     'retain_replay_across_levels', 'cache_frozen_latents'}:
                if type(v) is not bool:
                    raise ValueError(f'{k} must be boolean')
                continue
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f'Invalid {k}')
            if k in {'representation_freeze_error', 'delta_error', 'latent_delta_mse', 'min_latent_std', 'max_steps', 'max_level_steps', 'sigreg_weight', 'local_reconstruction_weight', 'delta_overlap_weight', 'variance_weight',
                     'rollout_weight', 'rollout_visual_weight', 'representation_joint_steps', 'burn_in_steps'}:
                if v < 0:
                    raise ValueError(f'{k} must be nonnegative')
            elif v <= 0:
                raise ValueError(f'{k} must be positive')
        if self.delta_error > 1 or self.min_observations < 2 or self.sigreg_weight > 1:
            raise ValueError('Invalid error threshold / representation sample count')
        if not 0 <= self.representation_freeze_error < self.representation_unfreeze_error <= 1:
            raise ValueError('Representation thresholds require 0 <= freeze < unfreeze <= 1')
        if type(self.representation_gate_patience) is not int:
            raise ValueError('representation_gate_patience must be a positive integer')
        if any(type(v) is not int for v in (self.representation_joint_steps, self.burn_in_steps)):
            raise ValueError('Representation schedule and burn-in steps must be integers')
        if self.auto_representation and self.representation_joint_steps:
            raise ValueError('Choose automatic representation gating or a fixed joint-learning schedule')
