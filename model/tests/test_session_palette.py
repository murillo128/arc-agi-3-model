"""Session appearance lifetime at the real bundled SDK boundary."""
from pathlib import Path
import unittest

import numpy as np

from model.config import SequenceConfig
from model.environment import Factory
from model.game_sequences import collect_session


ROOT = Path(__file__).resolve().parents[2]


class RecordingFactory(Factory):
    def make(self, seed, level=None, *, palette_seed=None):
        env = super().make(seed, level, palette_seed=palette_seed)
        self.records.append((seed, level, env.wrapper._game.palette.copy()))
        return env


class SessionPaletteTests(unittest.TestCase):
    def test_real_sessions_keep_one_palette_across_level_and_retry_boundaries(self):
        config = SequenceConfig(session_steps=8, block_steps=2, level_steps=2)
        for game in ('sc01-v1', 'si01-v1', 'mc01-v1', 'bf01-v1', 'sy55-v1'):
            with self.subTest(game=game):
                factory = RecordingFactory(game, ROOT / 'games')
                self.addCleanup(factory.arcade.close_scorecard)
                factory.records = []
                first = collect_session(factory, [1, 2], 42, config)
                records = factory.records[:]
                self.assertEqual([r[1] for r in records], [1, 2, 1, 2])
                self.assertEqual(len({r[0] for r in records}), 4)  # scene seeds still vary
                palette = records[0][2]
                self.assertEqual(sorted(palette.tolist()), list(range(16)))
                for _, _, current in records:
                    np.testing.assert_array_equal(current, palette)
                factory.records.clear()
                repeat = collect_session(factory, [1, 2], 42, config)
                for a, b in zip(first.transitions, repeat.transitions, strict=True):
                    self.assertEqual(a.action, b.action)
                    np.testing.assert_array_equal(a.before.frames, b.before.frames)
                    np.testing.assert_array_equal(a.after.frames, b.after.frames)
                factory.records.clear()
                collect_session(factory, [1, 2], 43, config)
                self.assertFalse(np.array_equal(factory.records[0][2], palette))

    def test_native_and_external_environments_reject_palette_override_before_loading(self):
        for game, directory, reserved in (
            ('sc01-v1', ROOT / 'games', True),
            ('sc01-v1', ROOT / 'external-games', False),
            ('official-v1', ROOT / 'games', False),
        ):
            with self.subTest(game=game, directory=directory, reserved=reserved):
                factory = Factory.__new__(Factory)
                factory.game_id = game
                factory.environments_dir = directory
                factory.reserve_final_level = reserved
                # No arcade client: rejection must precede any environment load.
                with self.assertRaisesRegex(ValueError, 'bundled synthetic'):
                    factory.make(0, palette_seed=42)

    def test_initial_observation_and_visual_goal_share_overridden_palette(self):
        factory = Factory('si01-v1', ROOT / 'games')
        self.addCleanup(factory.arcade.close_scorecard)
        for level in (None, 2):
            with self.subTest(level=level):
                env = factory.make(0, level, palette_seed=42)
                game = env.wrapper._game
                native = game.camera.render(game.current_level.get_sprites())
                np.testing.assert_array_equal(env.observation.final, native)
                palette = game.palette.copy()
                goal = factory.teacher(env).training_subgoals()[0]
                # A selection marker is the canonical color 0; the target
                # must use the same mapped color as a real successful click.
                player = game.players[0]
                from model.data import Action
                observed = env.step(Action(6, {'x': player.x + 2, 'y': player.y + 2}))
                self.assertTrue(any(np.array_equal(frame[goal.mask], goal.frame[goal.mask])
                                    for frame in observed.frames))
                self.assertIn(int(palette[0]), goal.frame[goal.mask])


if __name__ == '__main__':
    unittest.main()
