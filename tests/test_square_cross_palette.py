"""Observed palette diversity and seeded lifecycle through the real SDK."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

from games.square_cross.v1 import sc01
from square_cross_support import act, frame


class PaletteTests(unittest.TestCase):
    def test_literal_relabeling_covers_hud_and_every_color(self):
        mapping = np.array([3, 10, 1, 8, 15, 6, 13, 4, 11, 2, 9, 0, 7, 14, 5, 12], dtype=np.int8)
        with patch.object(sc01, 'palette_for_seed', return_value=mapping):
            game = sc01.Sc01(42)
        game.scene = game.scene._replace(edge='top')
        game.remaining = game.scene.budget
        pixels = np.tile(np.arange(16, dtype=np.int8), (64, 4))
        actual = game.render_interface(pixels.copy())
        # Top HUD is three rows x sixty columns, with a one-pixel separator.
        expected = np.tile(mapping, (64, 4))
        expected[:4] = 6  # Canonical background 5 -> displayed 6.
        expected[:3, 2:62] = 3  # Canonical fill 0 -> displayed 3.
        np.testing.assert_array_equal(actual, expected)

    def test_palette_preserves_all_level_dynamics_and_delta_masks(self):
        game = sc01.Sc01(42)
        with patch.object(sc01, 'palette_for_seed', return_value=np.arange(16, dtype=np.int8)):
            canonical = sc01.Sc01(42)
        mapping = game.palette.copy()
        self.assertEqual(sorted(mapping.tolist()), list(range(16)))
        for level in range(5):
            game.set_level(level)
            canonical.set_level(level)
            before, canonical_before = frame(game).copy(), frame(canonical).copy()
            np.testing.assert_array_equal(before, mapping[canonical_before])
            for action in (1, 2, 3, 4):
                actual, reference = act(game, action), act(canonical, action)
                self.assertEqual((actual.state, actual.levels_completed),
                                 (reference.state, reference.levels_completed))
                self.assertEqual(len(actual.frame), len(reference.frame))
                for image, original in zip(actual.frame, reference.frame, strict=True):
                    np.testing.assert_array_equal(image, mapping[original])
                    np.testing.assert_array_equal(image != before, original != canonical_before)
                    before, canonical_before = image.copy(), original.copy()
            np.testing.assert_array_equal(game.palette, mapping)

    def test_sdk_background_varies_and_reset_replays_exact_pixels(self):
        arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                        environments_dir=str(Path(__file__).resolve().parents[1] / 'games'))
        self.addCleanup(arcade.close_scorecard)
        backgrounds = set()
        for seed in range(32):
            env = arcade.make('sc01-v1', seed=seed, save_recording=False)
            initial = env.observation_space.frame[-1].copy()
            # Background dominates a sparse scene; no game-internal role lookup.
            backgrounds.add(int(np.bincount(initial.ravel(), minlength=16).argmax()))
            env.step(GameAction.ACTION1)
            np.testing.assert_array_equal(env.reset().frame[-1], initial)
            duplicate = arcade.make('sc01-v1', seed=seed, save_recording=False)
            np.testing.assert_array_equal(duplicate.observation_space.frame[-1], initial)
        # A broad deterministic seed sample must exercise most palette entries,
        # including the extremes; this catches the old always-5 background.
        self.assertGreaterEqual(len(backgrounds), 12)
        self.assertTrue({0, 15} <= backgrounds)


if __name__ == '__main__':
    unittest.main()
