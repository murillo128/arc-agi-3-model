"""Real ARCEngine boundaries, generation properties and offline SDK lifecycle."""

from importlib.metadata import version
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from games.square_cross_interactions.v1 import si01
from square_cross_reference import cross_pixels, reference_actions, square_pixels
from square_cross_support import act, assert_seed_baseline, center, fixture, frame

ROOT = Path(__file__).resolve().parents[1]


class InteractionTests(unittest.TestCase):
    def game(self, level, **changes):
        values = dict(target_size=5)
        if level in (1, 3):
            values.update(target_color=9, other_target=(35, 45), other_color=11)
        if level == 3:
            values.update(other_player=(20, 45))
        values.update(changes)
        self.enterContext(patch.object(si01, 'generate_scene', return_value=fixture(**values)))
        game = si01.Si01(seed=42)
        act(game)
        game.set_level(level - 1)
        return game

    def click(self, game, x=20, y=30):
        before = game.remaining
        result = act(game, 6, {'x': x, 'y': y})
        self.assertEqual(game.remaining, before)
        return result

    def marker(self, game, expected=None):
        # Literal fixtures keep the HUD at the top, so these are only overlay pixels.
        pixels = {(x, y) for y, x in zip(*np.where(frame(game) == 0)) if y >= 4}
        self.assertEqual(pixels, set() if expected is None else {
            (expected[0] - 1, expected[1] - 1), (expected[0], expected[1] - 1),
            (expected[0] - 1, expected[1]), (expected[0], expected[1])})

    def test_actions_and_ignored_clicks_on_level_one(self):
        game = si01.Si01(seed=42)
        result = act(game)
        self.assertEqual(result.available_actions, [1, 2, 3, 4])
        for level in range(1, 4):
            game.set_level(level - 1)
            before = frame(game).copy()
            remaining = game.remaining
            result = act(game, 6, dict(zip(('x', 'y'), game.scene.player)))
            self.assertEqual(result.available_actions, [1, 2, 3, 4] + ([6] if level >= 2 else []))
            self.assertEqual(game.remaining, remaining)
            if level == 1:
                np.testing.assert_array_equal(result.frame[-1], before)
            reset = act(game)
            self.assertEqual(reset.available_actions, result.available_actions)
            np.testing.assert_array_equal(reset.frame[-1], before)

    def test_one_matching_endpoint_and_swept_wrong_cross_pixels(self):
        game = self.game(1, budget=3)
        for i in range(3):
            result = act(game, 4)
            self.assertEqual(result.levels_completed, int(i == 2))
        self.assertEqual(game.number, 2)
        self.assertEqual(result.available_actions, [1, 2, 3, 4, 6])
        self.assertEqual(game.remaining, 3)
        # Touch only the vertical arm at y=32: the square never reaches the
        # wrong centre (25,34), and the centre line never touches any cross pixel.
        for level in (1, 3):
            for decoy, stride in (((25, 34), 5), ((25, 30), 10)):
                with self.subTest(level=level, decoy=decoy, stride=stride):
                    game = self.game(level, other_target=decoy, stride=stride, target=(50, 30))
                    initial = frame(game).copy()
                    if level == 3:
                        self.click(game)
                    result = act(game, 4)
                    self.assertEqual(result.state, GameState.GAME_OVER)
                    self.assertEqual(center(game), (20 + stride, 30))  # Atomic stride even on loss.
                    self.assertEqual(game.remaining, 23)
                    self.assertEqual(np.count_nonzero(result.frame[-1][:4] == 0), 173)
                    self.marker(game)
                    np.testing.assert_array_equal(act(game).frame[-1], initial)
        # Cross bounding-box corner is transparent; touching only it is safe.
        game = self.game(1, other_target=(28, 33))
        self.assertEqual(act(game, 4).state, GameState.NOT_FINISHED)
        self.assertEqual(center(game), (25, 30))
        # Crossing a matching centre between endpoints never delivers.
        game = self.game(1, target=(24, 30), stride=7)
        self.assertEqual(act(game, 4).levels_completed, 0)
        self.assertEqual(center(game), (27, 30))

    def test_blocked_sweep_does_not_touch_decoy(self):
        for level in (1, 3):
            with self.subTest(level=level):
                game = self.game(level, stride=10, other_target=(25, 30), obstacle=(29, 28, 1, 5))
                if level == 3:
                    self.click(game)
                result = act(game, 4)
                self.assertEqual(result.state, GameState.NOT_FINISHED)
                self.assertEqual(center(game), (20, 30))
                self.assertEqual(game.remaining, 23)

    def test_two_hit_geometry_free_toggle_outside_and_marker_follows(self):
        game = self.game(2)
        initial = frame(game).copy()
        self.marker(game)
        act(game, 4)
        self.assertEqual(center(game), (20, 30))
        self.assertEqual(game.remaining, 23)
        # Every occupied pixel (including marker pixels) is a geometry hit.
        for y in range(28, 33):
            for x in range(18, 23):
                self.click(game, x, y)
                self.marker(game, (20, 30))
                self.click(game, x, y)
                self.marker(game)
        for x, y in ((0, 0), (35, 30), (17, 30), (23, 30), (20, 27), (20, 33), (63, 63)):
            self.click(game)
            self.click(game, x, y)
            self.marker(game)
            self.assertEqual(center(game), (20, 30))
        self.click(game)
        act(game, 4)
        self.marker(game, (25, 30))
        self.assertEqual(game.remaining, 22)
        self.click(game, 24, 29)  # Click the white marker itself at its new location.
        self.marker(game)
        self.click(game, 25, 30)
        np.testing.assert_array_equal(act(game).frame[-1], initial)
        self.marker(game)

    def test_marker_above_overlapping_cross_and_square(self):
        for level in (2, 3):
            with self.subTest(level=level):
                game = self.game(level, target=(20, 30), target_color=11)
                self.assertTrue(np.all(frame(game)[28:33, 18:23] == 9))
                self.click(game)
                self.marker(game, (20, 30))
                self.assertEqual(np.count_nonzero(frame(game)[28:33, 18:23] == 9), 21)

    def test_two_last_move_advances_and_loss_clears_selection(self):
        game = self.game(2, budget=3)
        self.click(game)
        for _ in range(3):
            result = act(game, 4)
        self.assertEqual(game.number, 3)
        self.assertEqual(result.state, GameState.NOT_FINISHED)
        self.assertEqual(result.available_actions, [1, 2, 3, 4, 6])
        self.marker(game)
        game = self.game(2, budget=1)
        initial = frame(game).copy()
        self.click(game)
        self.assertEqual(act(game, 1).state, GameState.GAME_OVER)
        self.marker(game)
        np.testing.assert_array_equal(act(game).frame[-1], initial)

    def test_three_atomic_switch_solid_collision_and_locking(self):
        game = self.game(3)
        initial = frame(game).copy()
        act(game, 4)
        self.assertEqual(center(game), (20, 30))
        self.assertEqual(game.remaining, 23)
        self.click(game)
        self.click(game, 20, 45)
        self.marker(game, (20, 45))
        act(game, 4)
        self.assertEqual(center(game), (20, 30))
        self.assertEqual((game.players[1].x, game.players[1].y), (23, 43))
        self.marker(game, (25, 45))
        self.click(game, 25, 45)
        self.marker(game)
        self.click(game, 25, 45)
        self.click(game, 0, 0)
        self.marker(game)
        self.click(game)
        for _ in range(3):
            result = act(game, 4)
        self.assertEqual(result.state, GameState.NOT_FINISHED)
        self.assertEqual(result.levels_completed, 0)
        self.assertEqual(game.delivered, {0})
        self.marker(game)
        self.assertTrue(np.all(frame(game)[28:33, 33:38] == 9))
        self.click(game, 35, 30)
        self.marker(game)
        act(game, 3)
        self.assertEqual(center(game), (35, 30))
        np.testing.assert_array_equal(act(game).frame[-1], initial)
        self.assertEqual(game.delivered, set())
        # Full-sweep collision with the other square, even with clear endpoints.
        for locked in (False, True):
            game = self.game(3, stride=15, target=(50, 30), other_player=(28, 30), other_target=(28, 30))
            if locked:
                # Deliver the second square by leaving and returning to its cross.
                self.click(game, 28, 30)
                act(game, 2)
                act(game, 1)
                self.assertEqual(game.delivered, {1})
            self.click(game)
            before = game.remaining
            self.assertEqual(act(game, 4).state, GameState.NOT_FINISHED)
            self.assertEqual(center(game), (20, 30))
            self.assertEqual(game.remaining, before - 1)
            self.assertEqual((game.players[1].x, game.players[1].y), (26, 28))

    def test_three_second_delivery_last_budget_wins_and_full_reset(self):
        game = self.game(3, budget=6)
        self.click(game)
        for _ in range(3):
            result = act(game, 4)
        self.assertEqual(result.state, GameState.NOT_FINISHED)
        self.assertEqual(game.remaining, 3)
        self.click(game, 20, 45)
        for _ in range(3):
            result = act(game, 4)
        self.assertEqual(result.state, GameState.WIN)
        self.assertEqual(game.remaining, 0)
        self.assertEqual(game.delivered, {0, 1})
        self.marker(game)
        self.assertEqual(act(game, 4).frame, [])
        reset = act(game)
        self.assertTrue(reset.full_reset)
        self.assertEqual(reset.levels_completed, 0)
        self.assertEqual(reset.available_actions, [1, 2, 3, 4])
        self.assertEqual(game.remaining, 6)
        self.assertEqual(game.delivered, set())

    def test_three_footprint_collision_and_loss_reset_restores_locked_pair(self):
        game = self.game(3, other_player=(29, 30))
        self.click(game)
        act(game, 4)  # Rightmost pixel x=27 would hit the other's leftmost pixel.
        self.assertEqual(center(game), (20, 30))
        self.assertEqual(game.remaining, 23)
        game = self.game(3, budget=4)
        initial = frame(game).copy()
        self.click(game)
        for _ in range(3):
            act(game, 4)
        self.assertEqual(game.delivered, {0})
        self.click(game, 20, 45)
        result = act(game, 2)
        self.assertEqual(result.state, GameState.GAME_OVER)
        self.marker(game)
        reset = act(game)
        self.assertFalse(reset.full_reset)
        np.testing.assert_array_equal(reset.frame[-1], initial)
        self.assertEqual(game.delivered, set())
        self.marker(game)


class GenerationTests(unittest.TestCase):
    def test_seed_matrix_safe_shortest_routes_in_both_delivery_orders(self):
        variations = {n: set() for n in (1, 2, 3)}
        for seed in range(64):
            game, repeated = si01.Si01(seed), si01.Si01(seed)
            act(game)
            for level in (1, 2, 3):
                for order in ((0, 1), (1, 0)) if level == 3 else ((0, 1),):
                    with self.subTest(seed=seed, level=level, order=order):
                        act(game)  # Clear terminal WIN before entering the next isolated case.
                        game.set_level(level - 1)
                        repeated.set_level(level - 1)
                        scene = game.scene
                        self.assertEqual(scene, repeated.scene)
                        image = frame(game).copy()
                        np.testing.assert_array_equal(image, frame(repeated))
                        self.assertEqual(image.shape, (64, 64))
                        self.assertTrue(np.all((0 <= image) & (image <= 15)))
                        self.assertEqual(np.count_nonzero(image == 0), 180)
                        self.assertEqual((scene.player_size, scene.target_size, scene.stride), (5, 5, 5))
                        self.assertIsNone(scene.obstacle)
                        self.assertNotIn(4, image)
                        self.assertEqual(len(scene.players), 2 if level == 3 else 1)
                        self.assertEqual(len(scene.targets), 1 if level == 2 else 2)
                        if level in (1, 3):
                            self.assertEqual(scene.player_color, scene.target_color)
                            self.assertNotEqual(scene.player_color, scene.other_color)
                        if level == 3:
                            self.assertEqual(scene.players[1][1], scene.targets[1][1])
                        objects = [(square_pixels(p), c) for p, c in scene.players]
                        objects += [(cross_pixels(p), c) for p, c in scene.targets]
                        occupied = set()
                        for pixels, color in objects:
                            self.assertFalse(occupied & pixels)
                            occupied |= pixels
                            for x, y in pixels:
                                self.assertTrue(0 <= x < 64 and 0 <= y < 64)
                                self.assertEqual(image[y, x], color)
                        dx, dy = scene.target[0] - scene.player[0], scene.target[1] - scene.player[1]
                        self.assertNotEqual(dx == 0, dy == 0)
                        variations[level].add((scene.edge, scene.player_color, dx, dy, scene.other_target))
                        actions = reference_actions(scene, level, order)
                        distance = sum(a != 6 for a, _ in actions)
                        self.assertEqual(scene.distance, distance)
                        self.assertLessEqual(distance, 24)
                        self.assertEqual(scene.budget, max(24, 5 * distance + 8))
                        self.assertLessEqual(scene.budget, 128)
                        # The independent safe search meets the Manhattan lower
                        # bound, proving optimality; full swept lane envelopes
                        # cannot intersect, even with a locked square at either end.
                        corridors = []
                        lower_bound = 0
                        for (p, _), (t, _) in zip(scene.players, scene.targets):
                            self.assertEqual((t[0] - p[0]) % 5, 0)
                            self.assertEqual((t[1] - p[1]) % 5, 0)
                            lower_bound += (abs(t[0] - p[0]) + abs(t[1] - p[1])) // 5
                            corridors.append({(x, y) for x in range(min(p[0], t[0])-2, max(p[0], t[0])+3)
                                              for y in range(min(p[1], t[1])-2, max(p[1], t[1])+3)})
                        self.assertEqual(distance, lower_bound)
                        if level == 3:
                            self.assertFalse(corridors[0] & corridors[1])
                        act(game, 1)
                        np.testing.assert_array_equal(act(game).frame[-1], image)
                        for i, (action, data) in enumerate(actions):
                            remaining = game.remaining
                            result = act(game, action, data)
                            self.assertNotEqual(result.state, GameState.GAME_OVER)
                            if i < len(actions) - 1:
                                self.assertEqual(game.remaining, remaining - (action != 6))
                            for pixels in result.frame:
                                self.assertEqual(pixels.shape, (64, 64))
                                self.assertTrue(np.all((0 <= pixels) & (pixels <= 15)))
                        self.assertEqual(result.state, GameState.WIN if level == 3 else GameState.NOT_FINISHED)
                        if level < 3:
                            self.assertEqual(game.number, level + 1)
        for cases in variations.values():
            self.assertGreater(len(cases), 40)

    def test_former_scenes_and_pixels_unchanged(self):
        assert_seed_baseline(self, si01.Si01, range(6, 9))

    def test_exactly_three_local_levels(self):
        game = si01.Si01(42)
        with self.assertRaises(IndexError):
            game.set_level(3)
        for level in (0, 4, 6, 7, 8):
            with self.assertRaises(ValueError):
                si01.generate_scene(42, level)


class SDKTests(unittest.TestCase):
    def test_real_discovery_two_instances_all_levels_terminal_and_reset(self):
        self.assertEqual(version("arc-agi"), "0.9.9")
        arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / "games"))
        self.addCleanup(arcade.close_scorecard)
        self.assertIn("si01-v1", [info.game_id for info in arcade.get_environments()])
        envs = [arcade.make("si01-v1", seed=42) for _ in range(2)]
        self.assertTrue(all(env is not None for env in envs))
        initial = envs[0].observation_space.frame[-1].copy()
        for level in range(3):
            # Privileged inspection is confined to this test-only reference solver.
            scene = envs[0]._game.scene
            self.assertEqual(scene, envs[1]._game.scene)
            level_frame = envs[0].observation_space.frame[-1].copy()
            for env in envs:
                self.assertEqual(env.observation_space.win_levels, 3)
                self.assertEqual(env.observation_space.available_actions, [1, 2, 3, 4] + ([6] if level >= 1 else []))
            if level:
                envs[0].step(GameAction.ACTION1)
                reset = envs[0].reset()
                self.assertFalse(reset.full_reset)
                self.assertEqual(reset.levels_completed, level)
                np.testing.assert_array_equal(reset.frame[-1], level_frame)
            for action, data in reference_actions(scene, level + 1):
                results = [env.step(GameAction.from_id(action), data=data) for env in envs]
                for result in results:
                    self.assertIsNotNone(result)
                    for pixels in result.frame:
                        self.assertEqual(pixels.shape, (64, 64))
                np.testing.assert_array_equal(results[0].frame, results[1].frame)
            self.assertEqual(results[0].levels_completed, level + 1)
            self.assertEqual(results[0].state, GameState.WIN if level == 2 else GameState.NOT_FINISHED)
        terminal = envs[0].step(GameAction.ACTION4)
        self.assertEqual(terminal.state, GameState.WIN)
        self.assertEqual(terminal.frame, [])
        reset = envs[0].reset()
        self.assertTrue(reset.full_reset)
        self.assertEqual(reset.levels_completed, 0)
        self.assertEqual(reset.available_actions, [1, 2, 3, 4])
        np.testing.assert_array_equal(reset.frame[-1], initial)


if __name__ == "__main__":
    unittest.main()
