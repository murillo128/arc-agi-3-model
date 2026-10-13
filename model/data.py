"""Only public observation data and executed actions enter neural code."""
from dataclasses import dataclass, field
import random

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Action:
    id: int
    data: dict[str, int] = field(default_factory=dict)

    def features(self):
        if not 0 <= self.id <= 7:
            raise ValueError('SDK action ID outside 0..7')
        result = [float(i == self.id) for i in range(8)]
        result += [2 * self.data[k] / 63 - 1 if k in self.data else 0. for k in ('x', 'y')]
        result += [float(k in self.data) for k in ('x', 'y')]
        return result


@dataclass(frozen=True)
class Observation:
    frames: np.ndarray
    available_actions: tuple[int, ...]
    state: str
    levels_completed: int
    win_levels: int
    action_input: Action | None = None

    @classmethod
    def from_sdk(cls, raw, action=None):
        if raw is None:
            raise RuntimeError('SDK returned no confirmed observation')
        frames = np.asarray(raw.frame)
        if frames.size == 0:
            frames = np.empty((0, 0, 0), dtype=np.uint8)
        elif frames.ndim != 3 or min(frames.shape) < 1 or np.any((frames < 0) | (frames > 15)):
            raise ValueError('Expected native rectangular ARC palette frames [F,H,W]')
        frames = np.array(frames, dtype=np.uint8, copy=True)
        frames.flags.writeable = False
        echo = getattr(raw, 'action_input', None)
        if echo is not None:
            action = Action(int(echo.id.value), dict(echo.data or {}))
        return cls(frames, tuple(map(int, raw.available_actions)), raw.state.name,
                   int(raw.levels_completed), int(raw.win_levels), action)

    @property
    def final(self):
        return self.frames[-1] if len(self.frames) else None

    @property
    def terminal(self):
        return self.state in {'WIN', 'GAME_OVER'}


def random_action(observation: Observation, rng: random.Random) -> Action:
    legal = [i for i in observation.available_actions if 1 <= i <= 7]
    if not legal:
        raise RuntimeError('No ordinary SDK actions available')
    action = rng.choice(legal)
    # SDK coordinates, not inferred sprite coordinates or canvas-rescaled values.
    return Action(action, {'x': rng.randrange(64), 'y': rng.randrange(64)} if action == 6 else {})


@dataclass
class Episode:
    observations: list[Observation]
    actions: list[Action] = field(default_factory=list)

    def append(self, action, observation):
        self.actions.append(action)
        self.observations.append(observation)

    @property
    def nbytes(self):
        return sum(o.frames.nbytes for o in self.observations)


class Replay:
    """Bounded real episodes, including initialization; never persisted."""
    def __init__(self, episodes=32, byte_limit=128 * 1024 * 1024):
        self.episodes: list[Episode] = []
        self.capacity, self.byte_limit = episodes, byte_limit

    def add(self, episode):
        if not any(e is episode for e in self.episodes):
            self.episodes.append(episode)
        self.trim()

    def trim(self):
        while len(self.episodes) > self.capacity or sum(e.nbytes for e in self.episodes) > self.byte_limit:
            if len(self.episodes) == 1:
                raise MemoryError('Current episode exceeds replay byte budget; lower max_episode_steps')
            self.episodes.pop(0)

    def sample(self, rng, count, minimum):
        eligible = [e for e in self.episodes if len(e.actions)]
        rng.shuffle(eligible)
        chosen, observations = [], 0
        for episode in eligible:
            chosen.append(episode)
            observations += sum(bool(len(o.frames)) for o in episode.observations)
            if len(chosen) >= count and observations >= minimum:
                break
        if observations < minimum:
            return []
        # No stale latent cache: every chosen prefix starts at episode initialization.
        return [Episode(list(e.observations), list(e.actions)) for e in chosen]


def frame_features(observation: Observation, device, patch):
    """16 colors + 18 animation-delta symbols + one native-pixel validity channel."""
    pixels = torch.tensor(np.array(observation.frames, copy=True), device=device, dtype=torch.long)
    delta = torch.full_like(pixels, 17)  # first frame is absolute, not a predicted transition
    if len(pixels) > 1:
        delta[1:] = torch.where(pixels[1:] == pixels[:-1], 16, pixels[1:])
    color = F.one_hot(pixels, 16)
    changes = F.one_hot(delta, 18)
    x = torch.cat((color, changes, torch.ones_like(pixels[..., None])), -1).float().permute(0, 3, 1, 2)
    h, w = pixels.shape[-2:]
    return F.pad(x, (0, (-w) % patch, 0, (-h) % patch))
