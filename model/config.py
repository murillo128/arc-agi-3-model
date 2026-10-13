from dataclasses import asdict, dataclass
import math


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

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError('Network sizes must be positive integers')
        if self.width % self.heads or self.width % 4:
            raise ValueError('Width must be divisible by heads and four')

    @classmethod
    def small(cls):
        return cls(width=32, heads=4, encoder_depth=1, predictor_depth=1,
                   decoder_depth=1, latent=16, memory=32, animation_memory=16,
                   action_embedding=8, sigreg_slices=32)


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 3e-4
    train_every: int = 32
    updates_per_train: int = 1
    batch_episodes: int = 1
    frame_chunk: int = 8
    loss_steps: int = 16
    replay_episodes: int = 32
    replay_bytes: int = 128 * 1024 * 1024
    min_observations: int = 32
    max_episode_steps: int = 256
    validate_every: int = 1000
    validation_episodes: int = 8
    validation_steps: int = 128
    min_validation_transitions: int = 128
    min_changed_transitions: int = 32
    delta_error: float = 0.01
    latent_delta_mse: float = 0.01
    min_latent_std: float = 0.1
    consecutive_passes: int = 3
    checkpoint_every: int = 1000
    max_level_steps: int = 100_000
    max_steps: int = 500_000

    def __post_init__(self):
        for k, v in asdict(self).items():
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f'Invalid {k}')
            if k in {'delta_error', 'latent_delta_mse', 'min_latent_std'}:
                if v < 0:
                    raise ValueError(f'{k} must be nonnegative')
            elif v <= 0:
                raise ValueError(f'{k} must be positive')
        if self.delta_error > 1 or self.min_observations < 2:
            raise ValueError('Invalid error threshold / representation sample count')
