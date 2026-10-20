"""Complete public-frame relabeling without changing synthetic game rules."""
from pathlib import Path
import unittest

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction


ROOT = Path(__file__).resolve().parents[1]
MAPPING = np.array([3, 10, 1, 8, 15, 6, 13, 4, 11, 2, 9, 0, 7, 14, 5, 12],
                   dtype=np.int8)


class SyntheticPaletteTests(unittest.TestCase):
    def setUp(self):
        self.arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                             environments_dir=str(ROOT / 'games'))
        self.addCleanup(self.arcade.close_scorecard)

    def test_default_backgrounds_include_previously_missing_grays(self):
        for game in ('bf01-v1', 'sy55-v1', 'si01-v1'):
            backgrounds = set()
            for seed in range(128):
                env = self.arcade.make(game, seed=seed, save_recording=False)
                pixels = env.observation_space.frame[-1]
                backgrounds.add(int(np.bincount(pixels.ravel(), minlength=16).argmax()))
            with self.subTest(game=game):
                self.assertEqual(backgrounds, set(range(16)))

    def test_literal_mapping_includes_every_skill_frame_pixel(self):
        env = self.arcade.make('sy55-v1', seed=42, save_recording=False)
        game = env._game
        game.palette = MAPPING.copy()
        # Includes all colors, not just the particular puzzle's object palette.
        game._display_frame = np.tile(np.arange(16, dtype=np.int8), (64, 4))
        actual = game.render_interface(np.zeros((64, 64), dtype=np.int8))
        np.testing.assert_array_equal(actual, np.tile(MAPPING, (64, 4)))

    def test_sdk_animation_levels_resets_and_visual_teacher_keep_one_mapping(self):
        for game_id in ('bf01-v1', 'sy55-v1', 'si01-v1'):
            with self.subTest(game=game_id):
                colored = self.arcade.make(game_id, seed=42, save_recording=False)
                canonical = self.arcade.make(game_id, seed=42, save_recording=False)
                colored._game.palette = MAPPING.copy()
                canonical._game.palette = np.arange(16, dtype=np.int8)
                for level in (0, 1, 3):
                    for env in (colored, canonical):
                        env._game.set_level(level)
                    # Direct level setup does not refresh the wrapper's cached FrameData.
                    previous = colored._game.camera.render(colored._game.current_level.get_sprites())
                    reference = canonical._game.camera.render(canonical._game.current_level.get_sprites())
                    np.testing.assert_array_equal(previous, MAPPING[reference])
                    if game_id == 'si01-v1':
                        players = canonical._game.players
                        points = [(p.x + 1, p.y + 1) for p in players]
                        # Goals must match observed colors including selection marks.
                        goals = colored._game.training_subgoals()
                        originals = canonical._game.training_subgoals()
                        for goal, original in zip(goals, originals, strict=True):
                            np.testing.assert_array_equal(goal.frame, MAPPING[original.frame])
                            np.testing.assert_array_equal(goal.mask, original.mask)
                    else:
                        rects = canonical._game.puzzle.button_rects()
                        points = [(x + w // 2, y + h // 2) for x, y, w, h in rects[:2]]
                    actions = [(6, {'x': x, 'y': y}) for x, y in points]
                    actions += [(7, {}), (0, {})]
                    for action, data in actions:
                        actual = colored.step(GameAction.from_id(action), data=data)
                        expected = canonical.step(GameAction.from_id(action), data=data)
                        self.assertEqual((actual.state, actual.levels_completed, actual.available_actions),
                                         (expected.state, expected.levels_completed, expected.available_actions))
                        for pixels, original in zip(actual.frame, expected.frame, strict=True):
                            np.testing.assert_array_equal(pixels, MAPPING[original])
                            np.testing.assert_array_equal(pixels != previous, original != reference)
                            previous, reference = pixels.copy(), original.copy()
                        np.testing.assert_array_equal(colored._game.palette, MAPPING)


if __name__ == '__main__':
    unittest.main()
