"""Pinned SDK boundary. Level isolation is a local synthetic training capability."""
from importlib.metadata import version
from pathlib import Path
import random

from arc3.envs.sdk import open_arcade
from arcengine import GameAction

from .data import Observation
from .data import random_action

SYNTHETIC_LEVELS = {'sc01-v1': 5, 'si01-v1': 5}


def synthetic_levels():
    """Bundled catalog metadata; does not instantiate or solve any game."""
    from games.skill_core import SPECS
    from games.skill_common import LEVELS
    return SYNTHETIC_LEVELS | {spec.game_id + '-v1': LEVELS for spec in SPECS}


def training_boundary(factory, observation):
    """Keep the public final level out of local development learning."""
    if not getattr(factory, 'reserve_final_level', False):
        return False
    if observation.win_levels < 2:
        raise ValueError('SDK training needs at least two known public levels')
    return observation.levels_completed >= observation.win_levels - 1


class Environment:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.observation = Observation.from_sdk(wrapper.observation_space)

    def step(self, action):
        self.observation = Observation.from_sdk(
            self.wrapper.step(GameAction.from_id(action.id), data=action.data), action)
        return self.observation


class Factory:
    def __init__(self, game_id, environments_dir, mode='offline', *, reserve_final_level=False):
        self.game_id = game_id
        self.environments_dir = Path(environments_dir).resolve()
        self.arcade = open_arcade(mode, str(self.environments_dir))
        self.reserve_final_level = reserve_final_level

    def make(self, seed, level=None):
        wrapper = self.arcade.make(self.game_id, seed=seed, save_recording=False)
        if wrapper is None:
            raise RuntimeError(f'SDK could not load {self.game_id}')
        if level is not None:
            if self.reserve_final_level:
                raise ValueError('Public SDK training uses native levels only')
            self._isolate_synthetic_level(wrapper, level)
        return Environment(wrapper)

    def teacher(self, environment):
        """Privileged teacher capability, explicitly restricted to bundled games.

        The returned object is trainer-owned and never passed to Agent/Planner.
        Public/held-out SDK environments cannot opt in by matching a method name.
        """
        from arc_agi.local_wrapper import LocalEnvironmentWrapper
        root = Path(__file__).resolve().parents[1] / 'games'
        if (self.reserve_final_level or self.environments_dir != root or
                self.game_id not in SYNTHETIC_LEVELS or version('arcengine') != '0.9.3' or
                not isinstance(environment.wrapper, LocalEnvironmentWrapper)):
            raise ValueError('Synthetic teacher is only available for bundled local training games')
        return environment.wrapper._game

    def _isolate_synthetic_level(self, wrapper, level):
        # No portable level-selection API exists in SDK 0.9.9. This explicit,
        # pinned capability is restricted to this repository's synthetic games.
        from arc_agi.local_wrapper import LocalEnvironmentWrapper

        counts = synthetic_levels()
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


class SyntheticWorldFactory(Factory):
    """Bundled level isolation without a subgoal-teacher contract."""
    world_only = True

    def teacher(self, environment):
        raise ValueError('World-model stages do not use a subgoal teacher')


class NativeTrainingFactory(Factory):
    """Select a public native level by interaction, never by engine internals.

    Access observations are transient. Only the selected level's initial frame
    is returned to the learner; access cost is reported separately.
    """
    native_training = True
    world_only = True

    def __init__(self, game_id, environments_dir, mode='offline', *,
                 reserve_final_level=True, access_steps=512):
        super().__init__(game_id, environments_dir, mode, reserve_final_level=reserve_final_level)
        self.access_steps = access_steps
        self.access_actions = 0

    def make(self, seed, level=None):
        if type(level) is not int or level < 1:
            raise ValueError('Native training needs an explicit positive target level')
        env = super().make(seed)
        self.access_actions = 0
        count = env.observation.win_levels
        maximum = count - int(self.reserve_final_level)
        if level > maximum:
            raise ValueError(f'{self.game_id}/{level} is unknown or reserved (maximum training level {maximum})')
        rng = random.Random(seed)
        while env.observation.levels_completed + 1 < level:
            if env.observation.terminal or self.access_actions >= self.access_steps:
                raise ValueError(f'Could not reach {self.game_id}/{level} through native actions '
                                 f'within access budget {self.access_steps}; used {self.access_actions}')
            env.step(random_action(env.observation, rng))
            self.access_actions += 1
        if env.observation.terminal or env.observation.levels_completed + 1 != level:
            raise ValueError(f'Native progression did not enter requested level {level}')
        if self.access_actions:
            env.observation = env.observation.level_initial()
        return env

    def teacher(self, environment):
        raise ValueError('Public SDK training never has a synthetic teacher')
