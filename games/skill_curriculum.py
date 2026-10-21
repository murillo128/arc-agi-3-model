"""Thin ARCEngine adapter for the original synthetic micro-skill curricula."""
from __future__ import annotations

from copy import deepcopy

from arcengine import ARCBaseGame, Camera, Level

from games.palette import palette_for_seed
from games.skill_core import LEVELS, make_puzzle


class SkillGame(ARCBaseGame):
    """SDK entrypoints set ``skill_id``; all six levels retain the same context."""
    skill_id: str

    def __init__(self, seed: int = 0):
        self.instance_seed = seed
        self.palette = palette_for_seed(seed)
        self._last_transition = None
        super().__init__(
            game_id=self.skill_id, seed=seed, available_actions=[6, 7],
            levels=[Level(grid_size=(64, 64), data={'number': n})
                    for n in range(1, LEVELS + 1)],
            camera=Camera(width=64, height=64, background=5,
                          letter_box=5, interfaces=[self]),
        )
        # ARCEngine sets this *after* the first on_set_level callback.
        self._available_actions = self._actions()

    def _actions(self):
        return self.puzzle.available_actions()

    def on_set_level(self, level):
        self.number = level.get_data('number')
        self.puzzle = make_puzzle(self.skill_id, self.instance_seed, self.number)
        self._pending_frames = None
        self._display_frame = self.puzzle.render()
        self._available_actions = self._actions()

    def render_interface(self, frame):
        # Rendering must not advance rules, budgets, histories or animations.
        return self.palette[self._display_frame]

    def get_subgoals(self):
        return self.puzzle.get_subgoals()

    def get_training_info(self):
        result = self.puzzle.get_training_info()
        result['last_transition'] = deepcopy(self._last_transition)
        return result

    def step(self):
        action = self.action.id.value
        if action == 0:
            # RESET is handled by ARCBaseGame before this callback. A second
            # untouched RESET follows the SDK's normal full-reset semantics.
            self._last_transition = None
            self.complete_action()
            return
        if self._pending_frames is None:
            before = self.puzzle.get_subgoals()
            if action == 6:
                point = self.camera.display_to_grid(
                    self.action.data.get('x', -1), self.action.data.get('y', -1))
                if point is None:
                    self.puzzle.miss()
                else:
                    self.puzzle.click(*point)
            else:
                self.puzzle.input_action(action)
            self._available_actions = self._actions()
            self._last_transition = {
                'level': self.number, 'before': before,
                'after': self.puzzle.get_subgoals(),
                'outcome': 'level_complete' if self.puzzle.won else
                           'lost' if self.puzzle.lost else self.puzzle.event,
            }
            self._pending_frames = list(self.puzzle.transient_frames) + [self.puzzle.render()]
        if self._pending_frames:
            self._display_frame = self._pending_frames.pop(0)
            return
        self._pending_frames = None
        if self.puzzle.won:
            self.next_level()
        elif self.puzzle.lost:
            self.lose()
        self.complete_action()


def training_info(environment):
    """Explicit, synthetic-only teacher channel for a game or local SDK wrapper.

    Never call from an evaluated policy. `_game` is the pinned SDK's local
    wrapper boundary; official games and remote/competition wrappers are refused.
    No oracle fields are added to SDK FrameData or to the local play protocol.
    """
    game = environment if isinstance(environment, SkillGame) else getattr(environment, '_game', None)
    if not isinstance(game, SkillGame):
        raise ValueError('Teacher subgoals are available only for synthetic SkillGame environments')
    return game.get_training_info()
