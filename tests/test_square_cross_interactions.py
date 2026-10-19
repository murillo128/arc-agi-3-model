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


class ClickTests(unittest.TestCase):
    def game(self, level=1):
        generate = si01.generate_scene
        scenes = {
            1: si01.ClickScene((((20, 30), 9),), 'top'),
            2: si01.ClickScene((((20, 30), 9), ((45, 40), 11)), 'top'),
        }
        self.enterContext(patch.object(si01, 'generate_scene',
                                      side_effect=lambda seed, n: scenes[n] if n <= 2 else generate(seed, n)))
        game = si01.Si01(42)
        act(game)
        game.set_level(level - 1)
        return game

    def whites(self, pixels):
        return {(x, y) for y, x in zip(*np.where(pixels == 0)) if y >= 4}

    def test_one_solid_square_misses_directions_and_every_occupied_pixel(self):
        for y in range(28, 33):
            for x in range(18, 23):
                with self.subTest(x=x, y=y):
                    game = self.game()
                    initial = frame(game).copy()
                    self.assertEqual(set(np.unique(initial)), {0, 5, 9})
                    self.assertEqual(np.count_nonzero(initial == 9), 25)
                    self.assertEqual(self.whites(initial), set())
                    for action, data in [(a, None) for a in range(1, 5)] + [
                        (6, {'x': xx, 'y': yy}) for xx, yy in (
                            (0, 0), (63, 63), (17, 30), (23, 30), (20, 27), (20, 33), (-1, 30), (64, 30))]:
                        miss = act(game, action, data)
                        self.assertEqual((miss.state, miss.levels_completed, miss.available_actions),
                                         (GameState.NOT_FINISHED, 0, [6]))
                        self.assertEqual(game.remaining, 24)
                        np.testing.assert_array_equal(miss.frame[-1], initial)
                    result = act(game, 6, {'x': x, 'y': y})
                    self.assertEqual((result.state, result.levels_completed, result.win_levels),
                                     (GameState.NOT_FINISHED, 1, 5))
                    self.assertEqual(result.available_actions, [6])
                    self.assertEqual(game.number, 2)
                    completed = initial.copy()
                    completed[29:31, 19:21] = 0
                    self.assertEqual(len(result.frame), 2)
                    np.testing.assert_array_equal(result.frame[0], completed)
                    self.assertEqual(self.whites(result.frame[-1]), set())

    def test_two_independent_toggles_marker_hits_empty_clear_and_transition(self):
        game = self.game(2)
        initial = frame(game).copy()
        first = {(19, 29), (20, 29), (19, 30), (20, 30)}
        second = {(44, 39), (45, 39), (44, 40), (45, 40)}
        for point, marker in (((20, 30), first), ((45, 40), second)):
            for x, y in square_pixels(point):
                selected = act(game, 6, {'x': x, 'y': y})
                self.assertEqual(self.whites(selected.frame[-1]), marker)
                self.assertEqual(selected.levels_completed, 0)
                self.assertEqual(game.remaining, 24)
                toggled = act(game, 6, {'x': x, 'y': y})  # Includes marker pixels.
                np.testing.assert_array_equal(toggled.frame[-1], initial)
        for x, y in ((0, 0), (63, 63), (17, 30), (23, 30), (20, 27), (20, 33), (42, 40), (48, 40)):
            act(game, 6, {'x': 20, 'y': 30})
            miss = act(game, 6, {'x': x, 'y': y})
            self.assertEqual(miss.levels_completed, 0)
            self.assertEqual(game.remaining, 24)
            np.testing.assert_array_equal(miss.frame[-1], initial)
        act(game, 6, {'x': 20, 'y': 30})
        np.testing.assert_array_equal(act(game).frame[-1], initial)
        for action in range(1, 5):
            ignored = act(game, action)
            self.assertEqual(ignored.available_actions, [6])
            self.assertEqual(game.remaining, 24)
            np.testing.assert_array_equal(ignored.frame[-1], initial)
        act(game, 6, {'x': 19, 'y': 29})
        result = act(game, 6, {'x': 47, 'y': 42})  # Outer corner of second footprint.
        self.assertEqual((game.number, result.state, result.levels_completed), (3, GameState.NOT_FINISHED, 1))
        self.assertEqual(result.available_actions, [1, 2, 3, 4])
        self.assertEqual(len(result.frame), 2)
        self.assertEqual(self.whites(result.frame[0]), first | second)
        self.assertEqual(np.count_nonzero(result.frame[-1] == 0), 180)
        self.assertEqual(np.count_nonzero(result.frame[0][:4] == 0), 180)
        # The next level really is the former same-colour/decoy scene.
        self.assertEqual(len(game.scene.targets), 2)
        self.assertEqual(game.scene.player_color, game.scene.target_color)
        self.assertNotEqual(game.scene.player_color, game.scene.other_color)


class InteractionTests(unittest.TestCase):
    def game(self, level, **changes):
        values = dict(target_size=5)
        if level in (3, 5):
            values.update(target_color=9, other_target=(35, 45), other_color=11)
        if level == 5:
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

    def test_actions_and_ignored_clicks_on_level_three(self):
        game = si01.Si01(seed=42)
        self.assertEqual(act(game).available_actions, [6])
        for level in (3, 4, 5):
            game.set_level(level - 1)
            before = frame(game).copy()
            remaining = game.remaining
            result = act(game, 6, dict(zip(('x', 'y'), game.scene.player)))
            self.assertEqual(result.available_actions, [1, 2, 3, 4] + ([6] if level >= 4 else []))
            self.assertEqual(game.remaining, remaining)
            if level == 3:
                np.testing.assert_array_equal(result.frame[-1], before)
            reset = act(game)
            self.assertEqual(reset.available_actions, result.available_actions)
            np.testing.assert_array_equal(reset.frame[-1], before)

    def test_three_matching_endpoint_and_swept_wrong_cross_pixels(self):
        game = self.game(3, budget=3)
        for i in range(3):
            result = act(game, 4)
            self.assertEqual(result.levels_completed, int(i == 2))
        self.assertEqual(game.number, 4)
        self.assertEqual(result.available_actions, [1, 2, 3, 4, 6])
        self.assertEqual(game.remaining, 3)
        # Touch only the vertical arm at y=32: the square never reaches the
        # wrong centre (25,34), and the centre line never touches any cross pixel.
        for level in (3, 5):
            for decoy, stride in (((25, 34), 5), ((25, 30), 10)):
                with self.subTest(level=level, decoy=decoy, stride=stride):
                    game = self.game(level, other_target=decoy, stride=stride, target=(50, 30))
                    initial = frame(game).copy()
                    if level == 5:
                        self.click(game)
                    result = act(game, 4)
                    self.assertEqual(result.state, GameState.GAME_OVER)
                    self.assertEqual(result.available_actions, [1, 2, 3, 4] + ([6] if level == 5 else []))
                    self.assertEqual(center(game), (20 + stride, 30))  # Atomic stride even on loss.
                    self.assertEqual(game.remaining, 23)
                    self.assertEqual(np.count_nonzero(result.frame[-1][:4] == 0), 173)
                    self.marker(game)
                    np.testing.assert_array_equal(act(game).frame[-1], initial)
        # Cross bounding-box corner is transparent; touching only it is safe.
        game = self.game(3, other_target=(28, 33))
        self.assertEqual(act(game, 4).state, GameState.NOT_FINISHED)
        self.assertEqual(center(game), (25, 30))
        # Crossing a matching centre between endpoints never delivers.
        game = self.game(3, target=(24, 30), stride=7)
        self.assertEqual(act(game, 4).levels_completed, 0)
        self.assertEqual(center(game), (27, 30))

    def test_blocked_sweep_does_not_touch_decoy(self):
        for level in (3, 5):
            with self.subTest(level=level):
                game = self.game(level, stride=10, other_target=(25, 30), obstacle=(29, 28, 1, 5))
                if level == 5:
                    self.click(game)
                result = act(game, 4)
                self.assertEqual(result.state, GameState.NOT_FINISHED)
                self.assertEqual(center(game), (20, 30))
                self.assertEqual(game.remaining, 23)

    def test_four_hit_geometry_free_toggle_outside_and_marker_follows(self):
        game = self.game(4)
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
        for level in (4, 5):
            with self.subTest(level=level):
                game = self.game(level, target=(20, 30), target_color=11)
                self.assertTrue(np.all(frame(game)[28:33, 18:23] == 9))
                self.click(game)
                self.marker(game, (20, 30))
                self.assertEqual(np.count_nonzero(frame(game)[28:33, 18:23] == 9), 21)

    def test_four_last_move_advances_and_loss_clears_selection(self):
        game = self.game(4, budget=3)
        self.click(game)
        for _ in range(3):
            result = act(game, 4)
        self.assertEqual(game.number, 5)
        self.assertEqual(result.state, GameState.NOT_FINISHED)
        self.assertEqual(result.available_actions, [1, 2, 3, 4, 6])
        self.marker(game)
        game = self.game(4, budget=1)
        initial = frame(game).copy()
        self.click(game)
        loss = act(game, 1)
        self.assertEqual(loss.state, GameState.GAME_OVER)
        self.assertEqual(loss.available_actions, [1, 2, 3, 4, 6])
        self.marker(game)
        np.testing.assert_array_equal(act(game).frame[-1], initial)

    def test_five_atomic_switch_solid_collision_and_locking(self):
        game = self.game(5)
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
            game = self.game(5, stride=15, target=(50, 30), other_player=(28, 30), other_target=(28, 30))
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

    def test_five_second_delivery_last_budget_wins_and_full_reset(self):
        game = self.game(5, budget=6)
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
        self.assertEqual(reset.available_actions, [6])
        self.assertEqual(game.remaining, 6)
        self.assertEqual(game.delivered, set())

    def test_five_footprint_collision_and_loss_reset_restores_locked_pair(self):
        game = self.game(5, other_player=(29, 30))
        self.click(game)
        act(game, 4)  # Rightmost pixel x=27 would hit the other's leftmost pixel.
        self.assertEqual(center(game), (20, 30))
        self.assertEqual(game.remaining, 23)
        game = self.game(5, budget=4)
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
    def test_click_seed_matrix_solid_separated_varied_and_reset(self):
        variations = {1: set(), 2: set()}
        edges = {1: set(), 2: set()}
        for seed in range(64):
            for level in (1, 2):
                with self.subTest(seed=seed, level=level):
                    game, repeated = si01.Si01(seed), si01.Si01(seed)
                    act(game)
                    act(repeated)
                    game.set_level(level - 1)
                    repeated.set_level(level - 1)
                    scene = game.scene
                    image = frame(game).copy()
                    self.assertEqual(scene, repeated.scene)
                    np.testing.assert_array_equal(image, frame(repeated))
                    self.assertEqual(image.shape, (64, 64))
                    self.assertEqual(np.count_nonzero(image == 0), 180)
                    self.assertEqual((scene.player_size, scene.budget), (5, 24))
                    self.assertEqual(len(scene.players), level)
                    self.assertEqual(len({color for _, color in scene.players}), level)
                    limits = {'top': (0, 4, 63, 63), 'bottom': (0, 0, 63, 59),
                              'left': (4, 0, 63, 63), 'right': (0, 0, 59, 63)}
                    left, top, right, bottom = limits[scene.edge]
                    occupied = set()
                    for point, color in scene.players:
                        self.assertIn(color, (6, 7, 8, 9, 10, 11, 12, 14, 15))
                        pixels = square_pixels(point)
                        self.assertFalse(pixels & occupied)
                        occupied |= pixels
                        self.assertEqual(np.count_nonzero(image == color), 25)
                        for x, y in pixels:
                            self.assertTrue(left <= x <= right and top <= y <= bottom)
                            self.assertEqual(image[y, x], color)
                    self.assertEqual(set(np.unique(image)), {0, 5} | {c for _, c in scene.players})
                    self.assertEqual(np.count_nonzero(image != 5), 180 + 25 * level)
                    if level == 2:
                        a, b = [p for p, _ in scene.players]
                        self.assertGreaterEqual(max(abs(a[0]-b[0]), abs(a[1]-b[1])), 11)
                        x, y = a
                        selected = act(game, 6, {'x': x, 'y': y})
                        self.assertEqual(selected.available_actions, [6])
                        self.assertEqual(np.count_nonzero(selected.frame[-1] == 0), 184)
                    reset = act(game)
                    self.assertEqual(reset.available_actions, [6])
                    self.assertEqual(game.remaining, 24)
                    np.testing.assert_array_equal(reset.frame[-1], image)
                    variations[level].add(scene.players)
                    edges[level].add(scene.edge)
        for level in (1, 2):
            self.assertGreater(len(variations[level]), 40)
            self.assertEqual(edges[level], {'top', 'bottom', 'left', 'right'})

    def test_seed_matrix_safe_shortest_routes_in_both_delivery_orders(self):
        variations = {n: set() for n in (3, 4, 5)}
        for seed in range(64):
            game, repeated = si01.Si01(seed), si01.Si01(seed)
            act(game)
            for level in (3, 4, 5):
                for order in ((0, 1), (1, 0)) if level == 5 else ((0, 1),):
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
                        self.assertEqual(len(scene.players), 2 if level == 5 else 1)
                        self.assertEqual(len(scene.targets), 1 if level == 4 else 2)
                        if level in (3, 5):
                            self.assertEqual(scene.player_color, scene.target_color)
                            self.assertNotEqual(scene.player_color, scene.other_color)
                        if level == 5:
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
                        if level == 5:
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
                        self.assertEqual(result.state, GameState.WIN if level == 5 else GameState.NOT_FINISHED)
                        if level < 5:
                            self.assertEqual(game.number, level + 1)
        for cases in variations.values():
            self.assertGreater(len(cases), 40)

    def test_former_scenes_and_pixels_unchanged(self):
        assert_seed_baseline(self, si01.Si01, range(6, 9), local_start=3)

    def test_exactly_five_local_levels(self):
        game = si01.Si01(42)
        with self.assertRaises(IndexError):
            game.set_level(5)
        for level in (0, 6, 7, 8):
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
        for level in range(5):
            # Privileged inspection is confined to this test-only reference solver.
            scene = envs[0]._game.scene
            self.assertEqual(scene, envs[1]._game.scene)
            level_frame = envs[0].observation_space.frame[-1].copy()
            expected_actions = [6] if level < 2 else [1, 2, 3, 4] + ([6] if level >= 3 else [])
            for env in envs:
                self.assertEqual(env.observation_space.win_levels, 5)
                self.assertEqual(env.observation_space.available_actions, expected_actions)
            if level:
                if level == 1:
                    x, y = scene.players[0][0]
                    envs[0].step(GameAction.ACTION6, data={"x": x, "y": y})
                else:
                    envs[0].step(GameAction.ACTION1)
                reset = envs[0].reset()
                self.assertFalse(reset.full_reset)
                self.assertEqual(reset.levels_completed, level)
                self.assertEqual(reset.available_actions, expected_actions)
                self.assertEqual(np.count_nonzero(reset.frame[-1] == 0), 180)
                np.testing.assert_array_equal(reset.frame[-1], level_frame)
            for action, data in reference_actions(scene, level + 1):
                results = [env.step(GameAction.from_id(action), data=data) for env in envs]
                for result in results:
                    self.assertIsNotNone(result)
                    for pixels in result.frame:
                        self.assertEqual(pixels.shape, (64, 64))
                np.testing.assert_array_equal(results[0].frame, results[1].frame)
            self.assertEqual(results[0].levels_completed, level + 1)
            self.assertEqual(results[0].state, GameState.WIN if level == 4 else GameState.NOT_FINISHED)
        terminal = envs[0].step(GameAction.ACTION4)
        self.assertEqual(terminal.state, GameState.WIN)
        self.assertEqual(terminal.frame, [])
        reset = envs[0].reset()
        self.assertTrue(reset.full_reset)
        self.assertEqual(reset.levels_completed, 0)
        self.assertEqual(reset.available_actions, [6])
        np.testing.assert_array_equal(reset.frame[-1], initial)


if __name__ == "__main__":
    unittest.main()
