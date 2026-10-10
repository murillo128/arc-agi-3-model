"""Real ARCEngine boundaries, generation properties and offline SDK lifecycle."""

from importlib.metadata import version
from math import ceil
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import ActionInput, GameAction, GameState

from games.square_cross.v1 import sc01
from square_cross_reference import cross_pixels, reference_actions, reference_path, square_pixels

ROOT = Path(__file__).resolve().parents[1]


def act(game, action=0, data=None):
    return game.perform_action(ActionInput(id=GameAction.from_id(action), data=data or {}), raw=True)


def frame(game):
    return game.camera.render(game.current_level.get_sprites())


def center(game):
    r = game.scene.player_size // 2
    return game.player.x + r, game.player.y + r


def fixture(**changes):
    values = dict(player=(20, 30), target=(35, 30), player_size=5, target_size=3, stride=5,
                  edge="top", obstacle=None, distance=3, budget=24, player_color=9, target_color=11)
    values.update(changes)
    return sc01.Scene(**values)


class EngineTests(unittest.TestCase):
    def game(self, **changes):
        self.enterContext(patch.object(sc01, "generate_scene", return_value=fixture(**changes)))
        game = sc01.Sc01(seed=42)
        act(game)
        return game

    def test_directions_redraw_and_visible_bar(self):
        game = self.game()
        for action, expected in [(1, (20, 25)), (2, (20, 30)), (3, (15, 30)), (4, (20, 30))]:
            before = frame(game)
            np.testing.assert_array_equal(before, frame(game))
            remaining = game.remaining
            result = act(game, action)
            self.assertEqual(center(game), expected)
            self.assertEqual(game.remaining, remaining - 1)
            self.assertEqual(np.count_nonzero(result.frame[-1] == 0), ceil(180 * game.remaining / 24))
            self.assertGreater(np.count_nonzero(before == 0), np.count_nonzero(result.frame[-1] == 0))
        self.assertEqual(game.remaining, 20)
        act(game, 5)  # Not advertised; cannot consume a directional move.
        self.assertEqual(game.remaining, 20)

    def test_square_occludes_cross_and_reveals_it_after_moving(self):
        game = self.game(target=(20, 30), target_size=5)
        self.assertTrue(np.all(frame(game)[28:33, 18:23] == 9))
        result = act(game, 3)
        self.assertEqual(result.levels_completed, 0)
        self.assertEqual(np.count_nonzero(frame(game) == 11), 9)
        self.assertEqual(np.count_nonzero(frame(game) == 0), ceil(180 * 23 / 24))

    def test_full_footprint_obstacle_detour_and_blocked_command(self):
        game = self.game(target=(30, 30), stride=1, obstacle=(24, 28, 3, 5), distance=20, budget=108)
        self.assertEqual(len(reference_path(game.scene)), 20)  # 10 across + 5 up + 5 down.
        self.assertEqual(len(reference_path(game.scene, center_only=True)), 16)
        act(game, 4)
        before = frame(game)
        act(game, 4)  # Centre x=22 is clear, but right edge x=24 intersects the block.
        self.assertEqual(center(game), (21, 30))
        self.assertEqual(game.remaining, 106)
        self.assertGreater(np.count_nonzero(before == 0), np.count_nonzero(frame(game) == 0))
        self.assertTrue(np.all(frame(game)[28:33, 24:27] == 2))
        self.assertEqual(game.current_level.get_sprites_by_name("obstacle")[0].x, 24)

    def test_mid_sweep_obstacle_rejects_whole_command(self):
        for size, obstacle in [(3, (23, 30, 1, 1)), (5, (24, 32, 1, 1))]:
            with self.subTest(size=size):
                game = self.game(player_size=size, stride=7, obstacle=obstacle)
                # Both endpoints are clear; the second case only hits an outer pixel.
                self.assertTrue(sc01.fits((27, 30), size, sc01.BOUNDS['top'], obstacle))
                act(game, 4)
                self.assertEqual(center(game), (20, 30))
                self.assertEqual(game.remaining, 23)
                self.assertGreater(len(reference_path(game.scene._replace(target=(34, 30)))), 2)

    def test_stride_cannot_stop_partway_at_frame_border(self):
        game = self.game(player=(4, 30), player_size=3, stride=5)
        act(game, 3)  # Some intermediate pixels fit, but the full command does not.
        self.assertEqual(center(game), (4, 30))
        self.assertEqual(game.remaining, 23)

    def test_all_hud_edges_exclude_footprint_and_bar_decreases_every_action(self):
        cases = [("top", (20, 9), 1), ("bottom", (20, 54), 2),
                 ("left", (9, 20), 3), ("right", (54, 20), 4)]
        for edge, player, action in cases:
            for budget in (24, 38, 73, 128):
                with self.subTest(edge=edge, budget=budget), patch.object(sc01, "generate_scene", return_value=fixture(
                        edge=edge, player=player, player_size=3, budget=budget)):
                    game = sc01.Sc01()
                    act(game)
                    # Longitudinal order is stable, with thickness filled first.
                    rows = range(3) if edge in ('top', 'left') else range(61, 64)
                    cells = [(thickness, offset) if edge in ('top', 'bottom') else (offset, thickness)
                             for offset in range(2, 62) for thickness in rows]
                    previous = 181
                    for remaining in range(budget, -1, -1):
                        image = frame(game)
                        count = ceil(180 * remaining / budget)
                        self.assertEqual(set(zip(*np.where(image == 0))), set(cells[:count]))
                        self.assertLess(count, previous)
                        self.assertEqual(game.remaining, remaining)
                        self.assertEqual(center(game), player)  # No partial stride into the HUD.
                        previous = count
                        if remaining:
                            result = act(game, action)
                    self.assertEqual(result.state, GameState.GAME_OVER)

    def test_centre_goal_last_move_and_new_level_budget(self):
        game = self.game(player_size=3, target_size=7, target=(26, 30), stride=2, distance=3, budget=3)
        result = act(game, 4)
        self.assertEqual(result.levels_completed, 0)  # Already touching the cross's left arm.
        for _ in range(2):
            result = act(game, 4)
        self.assertEqual(result.state, GameState.NOT_FINISHED)
        self.assertEqual(result.levels_completed, 1)
        self.assertEqual(game.remaining, 3)
        self.assertEqual(len(result.frame), 2)  # Solved scene then new level, no double consumption.
        self.assertEqual(np.count_nonzero(result.frame[0] == 0), 0)
        self.assertEqual(np.count_nonzero(result.frame[1] == 0), 180)
        for _ in range(4 * 3):
            result = act(game, 4)
        self.assertEqual(result.state, GameState.NOT_FINISHED)
        self.assertEqual(result.levels_completed, 5)
        self.assertEqual(game.remaining, 3)

    def test_literal_five_pixel_arrivals_and_intermediate_goal_does_not_win(self):
        game = self.game(budget=3)
        for expected in ((25, 30), (30, 30)):
            result = act(game, 4)
            self.assertEqual(center(game), expected)
            self.assertEqual(result.levels_completed, 0)
        result = act(game, 4)
        self.assertEqual(result.levels_completed, 1)  # Final centre (35,30) advances.
        crossing = self.game(player_size=3, target=(24, 30), stride=7)
        result = act(crossing, 4)
        self.assertEqual(center(crossing), (27, 30))
        self.assertEqual(result.levels_completed, 0)

    def test_exhaustion_and_current_level_reset(self):
        game = self.game(budget=2)
        initial = frame(game).copy()
        act(game, 1)
        result = act(game, 1)
        self.assertEqual(result.state, GameState.GAME_OVER)
        self.assertEqual(np.count_nonzero(result.frame[-1] == 0), 0)
        self.assertEqual(game.remaining, 0)
        act(game, 1)
        self.assertEqual(game.remaining, 0)
        result = act(game)
        self.assertFalse(result.full_reset)
        np.testing.assert_array_equal(result.frame[-1], initial)

    def test_static_noise_is_under_objects_and_does_not_collide(self):
        noisy = sc01.Sc01(seed=42)
        plain = sc01.Sc01(seed=42)
        noisy.set_level(4)
        plain.set_level(4)
        noise = plain.current_level.get_sprites_by_name("noise")[0]
        plain.current_level.remove_sprite(noise)
        before = frame(noisy).copy()
        np.testing.assert_array_equal(before, frame(noisy))
        changed = before != frame(plain)
        self.assertGreater(np.count_nonzero(changed), 20)
        self.assertTrue(np.all(before[changed] == 4))
        self.assertTrue(np.all(frame(plain)[changed] == 5))
        self.assertFalse(noise.is_collidable)
        # Identical commands (including paths through speckles) have identical physical outcomes.
        for action in [1, 2, 3, 4] * 5:
            a, b = act(noisy, action), act(plain, action)
            self.assertEqual((center(noisy), noisy.remaining, a.state), (center(plain), plain.remaining, b.state))


class LaterLevelTests(unittest.TestCase):
    def game(self, level, **changes):
        values = dict(target_size=5)
        if level in (6, 8):
            values.update(target_color=9, other_target=(35, 45), other_color=11)
        if level == 8:
            values.update(other_player=(20, 45))
        values.update(changes)
        self.enterContext(patch.object(sc01, 'generate_scene', return_value=fixture(**values)))
        game = sc01.Sc01(seed=42)
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

    def test_actions_and_ignored_clicks_on_originals_and_level_six(self):
        game = sc01.Sc01(seed=42)
        result = act(game)
        self.assertEqual(result.available_actions, [1, 2, 3, 4])
        for level in range(1, 9):
            game.set_level(level - 1)
            before = frame(game).copy()
            remaining = game.remaining
            result = act(game, 6, dict(zip(('x', 'y'), game.scene.player)))
            self.assertEqual(result.available_actions, [1, 2, 3, 4] + ([6] if level >= 7 else []))
            self.assertEqual(game.remaining, remaining)
            if level <= 6:
                np.testing.assert_array_equal(result.frame[-1], before)
            reset = act(game)
            self.assertEqual(reset.available_actions, result.available_actions)
            np.testing.assert_array_equal(reset.frame[-1], before)

    def test_six_matching_endpoint_and_swept_wrong_cross_pixels(self):
        game = self.game(6, budget=3)
        for i in range(3):
            result = act(game, 4)
            self.assertEqual(result.levels_completed, int(i == 2))
        self.assertEqual(game.number, 7)
        self.assertEqual(result.available_actions, [1, 2, 3, 4, 6])
        self.assertEqual(game.remaining, 3)
        # Touch only the vertical arm at y=32: the square never reaches the
        # wrong centre (25,34), and the centre line never touches any cross pixel.
        for level in (6, 8):
            for decoy, stride in (((25, 34), 5), ((25, 30), 10)):
                with self.subTest(level=level, decoy=decoy, stride=stride):
                    game = self.game(level, other_target=decoy, stride=stride, target=(50, 30))
                    initial = frame(game).copy()
                    if level == 8:
                        self.click(game)
                    result = act(game, 4)
                    self.assertEqual(result.state, GameState.GAME_OVER)
                    self.assertEqual(center(game), (20 + stride, 30))  # Atomic stride even on loss.
                    self.assertEqual(game.remaining, 23)
                    self.assertEqual(np.count_nonzero(result.frame[-1][:4] == 0), 173)
                    self.marker(game)
                    np.testing.assert_array_equal(act(game).frame[-1], initial)
        # Cross bounding-box corner is transparent; touching only it is safe.
        game = self.game(6, other_target=(28, 33))
        self.assertEqual(act(game, 4).state, GameState.NOT_FINISHED)
        self.assertEqual(center(game), (25, 30))
        # Crossing a matching centre between endpoints never delivers.
        game = self.game(6, target=(24, 30), stride=7)
        self.assertEqual(act(game, 4).levels_completed, 0)
        self.assertEqual(center(game), (27, 30))

    def test_blocked_sweep_does_not_touch_decoy(self):
        for level in (6, 8):
            with self.subTest(level=level):
                game = self.game(level, stride=10, other_target=(25, 30), obstacle=(29, 28, 1, 5))
                if level == 8:
                    self.click(game)
                result = act(game, 4)
                self.assertEqual(result.state, GameState.NOT_FINISHED)
                self.assertEqual(center(game), (20, 30))
                self.assertEqual(game.remaining, 23)

    def test_seven_hit_geometry_free_toggle_outside_and_marker_follows(self):
        game = self.game(7)
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
        for level in (7, 8):
            with self.subTest(level=level):
                game = self.game(level, target=(20, 30), target_color=11)
                self.assertTrue(np.all(frame(game)[28:33, 18:23] == 9))
                self.click(game)
                self.marker(game, (20, 30))
                self.assertEqual(np.count_nonzero(frame(game)[28:33, 18:23] == 9), 21)

    def test_seven_last_move_advances_and_loss_clears_selection(self):
        game = self.game(7, budget=3)
        self.click(game)
        for _ in range(3):
            result = act(game, 4)
        self.assertEqual(game.number, 8)
        self.assertEqual(result.state, GameState.NOT_FINISHED)
        self.assertEqual(result.available_actions, [1, 2, 3, 4, 6])
        self.marker(game)
        game = self.game(7, budget=1)
        initial = frame(game).copy()
        self.click(game)
        self.assertEqual(act(game, 1).state, GameState.GAME_OVER)
        self.marker(game)
        np.testing.assert_array_equal(act(game).frame[-1], initial)

    def test_eight_atomic_switch_solid_collision_and_locking(self):
        game = self.game(8)
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
            game = self.game(8, stride=15, target=(50, 30), other_player=(28, 30), other_target=(28, 30))
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

    def test_eight_second_delivery_last_budget_wins_and_full_reset(self):
        game = self.game(8, budget=6)
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

    def test_eight_footprint_collision_and_loss_reset_restores_locked_pair(self):
        game = self.game(8, other_player=(29, 30))
        self.click(game)
        act(game, 4)  # Rightmost pixel x=27 would hit the other's leftmost pixel.
        self.assertEqual(center(game), (20, 30))
        self.assertEqual(game.remaining, 23)
        game = self.game(8, budget=4)
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
    def test_alignment_is_necessary_but_not_sufficient(self):
        aligned = fixture(player_size=3, stride=4, target=(32, 38))
        self.assertEqual(len(reference_path(aligned)), 5)  # (12,8) / 4.
        self.assertEqual(sc01.shortest_distance(aligned.player, aligned.target, 4, 3,
                                               sc01.BOUNDS['top'], None), 5)
        cases = [aligned._replace(target=(30, 38)),  # (10,8) is not aligned.
                 fixture(player_size=3, stride=7, target=(34, 30), obstacle=(23, 4, 1, 60)),
                 fixture(player=(20, 9), target=(20, 2), player_size=3, stride=7)]
        for scene in cases:
            with self.subTest(scene=scene):
                with self.assertRaisesRegex(AssertionError, 'no solution'):
                    reference_path(scene)
                self.assertIsNone(sc01.shortest_distance(scene.player, scene.target, scene.stride,
                                                        scene.player_size, sc01.BOUNDS[scene.edge], scene.obstacle))

    def test_seed_matrix_geometry_visuals_and_independent_solvability(self):
        edges, pairs, colors, directions, placements, distances = set(), set(), set(), set(), set(), set()
        strides = {level: set() for level in range(3, 6)}
        independent_sizes = {level: set() for level in range(3, 6)}
        for seed in range(64):
            game = sc01.Sc01(seed)
            for level in range(1, 6):
                with self.subTest(seed=seed, level=level):
                    game.set_level(level - 1)
                    scene = game.scene
                    image = frame(game).copy()
                    self.assertEqual(game.current_level.grid_size, (64, 64))
                    self.assertEqual((game.camera.width, game.camera.height), (64, 64))
                    self.assertEqual(image.shape, (64, 64))
                    self.assertTrue(np.issubdtype(image.dtype, np.integer))
                    self.assertTrue(np.all((0 <= image) & (image <= 15)))
                    self.assertEqual(np.count_nonzero(image == 0), 180)
                    # Every object pixel is in-frame, with no initial object occlusion.
                    self.assertEqual(np.count_nonzero(image == scene.player_color), scene.player_size ** 2)
                    self.assertEqual(np.count_nonzero(image == scene.target_color), 2 * scene.target_size - 1)
                    path = reference_path(scene)
                    self.assertGreater(len(path), 0)
                    self.assertLessEqual(len(path), 24)
                    self.assertEqual(scene.budget, max(24, 5 * len(path) + 8))
                    self.assertLessEqual(scene.budget, 128)
                    self.assertEqual(scene.distance, len(path))
                    dx, dy = scene.target[0] - scene.player[0], scene.target[1] - scene.player[1]
                    self.assertEqual(abs(dx) % scene.stride, 0)
                    self.assertEqual(abs(dy) % scene.stride, 0)
                    if level <= 2:
                        self.assertEqual(scene.stride, scene.player_size)
                    else:
                        self.assertIn(scene.stride, range(1, 8))
                        strides[level].add(scene.stride)
                        independent_sizes[level].add((scene.player_size, scene.stride))
                    if level <= 2:
                        self.assertNotEqual(dx == 0, dy == 0)
                        self.assertLessEqual(abs(scene.player[0] - 32), 3)
                        self.assertLessEqual(abs(scene.player[1] - 32), 3)
                        directions.add((int(np.sign(dx)), int(np.sign(dy))))
                        distances.add(len(path))
                    if level == 1:
                        self.assertEqual((scene.player_size, scene.target_size), (5, 5))
                    if level == 2:
                        pairs.add((scene.player_size, scene.target_size))
                    if level == 3:
                        self.assertTrue(dx != 0 and dy != 0)
                        placements.add((dx != 0 and dy != 0, scene.player))
                    self.assertEqual(scene.obstacle is not None, level >= 4)
                    if scene.obstacle:
                        x, y, w, h = scene.obstacle
                        self.assertEqual(np.count_nonzero(image == 2), w * h)
                        self.assertGreater(len(path), (abs(dx) + abs(dy)) // scene.stride)
                    if level < 5:
                        self.assertNotIn(4, image)
                    edges.add(scene.edge)
                    colors.add((scene.player_color, scene.target_color))
                    act(game, 2 if dy < 0 else 1)  # Move away: some levels now solve in one stride.
                    reset = act(game)
                    self.assertEqual(game.scene, scene)
                    np.testing.assert_array_equal(reset.frame[-1], image)
        self.assertEqual(edges, {"top", "bottom", "left", "right"})
        self.assertEqual(pairs, {(a, b) for a in (3, 5, 7) for b in (3, 5, 7)})
        self.assertEqual(len(directions), 4)
        self.assertGreater(len(distances), 5)
        self.assertGreater(len(colors), 10)
        self.assertGreater(sum(non_cardinal for non_cardinal, _ in placements), 20)
        self.assertGreater(len({position for _, position in placements}), 20)
        for level in range(3, 6):
            self.assertEqual(strides[level], set(range(1, 8)))
            self.assertTrue(any(size != stride for size, stride in independent_sizes[level]))
            self.assertEqual({size for size, _ in independent_sizes[level]}, {3, 5, 7})

    def test_later_seed_matrix_safe_shortest_routes_in_both_delivery_orders(self):
        variations = {n: set() for n in (6, 7, 8)}
        for seed in range(64):
            game, repeated = sc01.Sc01(seed), sc01.Sc01(seed)
            act(game)
            for level in (6, 7, 8):
                for order in ((0, 1), (1, 0)) if level == 8 else ((0, 1),):
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
                        self.assertEqual(len(scene.players), 2 if level == 8 else 1)
                        self.assertEqual(len(scene.targets), 1 if level == 7 else 2)
                        if level in (6, 8):
                            self.assertEqual(scene.player_color, scene.target_color)
                            self.assertNotEqual(scene.player_color, scene.other_color)
                        if level == 8:
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
                        if level == 8:
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
                        self.assertEqual(result.state, GameState.WIN if level == 8 else GameState.NOT_FINISHED)
                        if level < 8:
                            self.assertEqual(game.number, level + 1)
        for cases in variations.values():
            self.assertGreater(len(cases), 40)


class SDKTests(unittest.TestCase):
    def test_real_discovery_two_instances_all_levels_terminal_and_reset(self):
        self.assertEqual(version("arc-agi"), "0.9.9")
        arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / "games"))
        self.addCleanup(arcade.close_scorecard)
        self.assertIn("sc01-v1", [info.game_id for info in arcade.get_environments()])
        envs = [arcade.make("sc01-v1", seed=42) for _ in range(2)]
        self.assertTrue(all(env is not None for env in envs))
        initial = envs[0].observation_space.frame[-1].copy()
        for level in range(8):
            # Privileged inspection is confined to this test-only reference solver.
            scene = envs[0]._game.scene
            self.assertEqual(scene, envs[1]._game.scene)
            level_frame = envs[0].observation_space.frame[-1].copy()
            for env in envs:
                self.assertEqual(env.observation_space.win_levels, 8)
                self.assertEqual(env.observation_space.available_actions, [1, 2, 3, 4] + ([6] if level >= 6 else []))
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
            self.assertEqual(results[0].state, GameState.WIN if level == 7 else GameState.NOT_FINISHED)
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
