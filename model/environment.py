"""Pinned SDK boundary. Level isolation is a local synthetic training capability."""
from importlib.metadata import version
from pathlib import Path

from arc3.envs.sdk import open_arcade
from arcengine import GameAction

from .data import Observation


class Environment:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.observation = Observation.from_sdk(wrapper.observation_space)

    def step(self, action):
        self.observation = Observation.from_sdk(
            self.wrapper.step(GameAction.from_id(action.id), data=action.data), action)
        return self.observation


class Factory:
    def __init__(self, game_id, environments_dir, mode='offline'):
        self.game_id = game_id
        self.environments_dir = Path(environments_dir).resolve()
        self.arcade = open_arcade(mode, str(self.environments_dir))

    def make(self, seed, level=None):
        wrapper = self.arcade.make(self.game_id, seed=seed, save_recording=False)
        if wrapper is None:
            raise RuntimeError(f'SDK could not load {self.game_id}')
        if level is not None:
            self._isolate_synthetic_level(wrapper, level)
        return Environment(wrapper)

    def _isolate_synthetic_level(self, wrapper, level):
        # No portable level-selection API exists in SDK 0.9.9. This explicit,
        # pinned capability is restricted to this repository's synthetic games.
        from arc_agi.local_wrapper import LocalEnvironmentWrapper

        counts = {'sc01-v1': 5, 'si01-v1': 3}
        root = Path(__file__).resolve().parents[1] / 'games'
        if (self.environments_dir != root or self.game_id not in counts or
                not isinstance(wrapper, LocalEnvironmentWrapper) or version('arcengine') != '0.9.3'):
            raise ValueError('Level isolation only supports bundled synthetic games with arcengine==0.9.3; omit --levels for other SDK games')
        if level not in range(1, counts[self.game_id] + 1):
            raise ValueError(f'Invalid level {level} for {self.game_id}')
        game = wrapper._game
        original_reset = game.full_reset

        def reset_selected_level():
            original_reset()
            game.set_level(level - 1)

        game.full_reset = reset_selected_level
        game.is_last_level = lambda: True
        game._win_score = 1
        game.full_reset()  # also correct when ONLY_RESET_LEVELS is set externally
        # Wrapper.reset remains responsible for constructing the real SDK
        # observation. Native winning frames stay on the selected level.
        wrapper.reset()
