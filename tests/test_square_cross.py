"""Real ARCEngine boundaries, generation properties and offline SDK lifecycle."""

from importlib.metadata import version
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import ActionInput, GameAction, GameState

from games.square_cross.v1 import sc01
from square_cross_reference import reference_path

ROOT = Path(__file__).resolve().parents[1]


def act(game, action=0):
    return game.perform_action(ActionInput(id=GameAction.from_id(action)), raw=True)


def frame(game):
    return game.camera.render(game.current_level.get_sprites())


def center(game):
    r = game.scene.player_size // 2
    return game.player.x + r, game.player.y + r


def fixture(**changes):
    values = dict(player=(20, 30), target=(30, 30), player_size=5, target_size=3,
                  edge="top", obstacle=None, distance=10, budget=15, player_color=9, target_color=11)
    values.update(changes)
    return sc01.Scene(**values)


class EngineTests(unittest.TestCase):
    def game(self, **changes):
        self.enterContext(patch.object(sc01, "generate_scene", return_value=fixture(**changes)))
        game = sc01.Sc01(seed=42)
        act(game)
        return game

    def test_directions_redraw_and_visible_dots(self):
        game = self.game()
        for action, expected in [(1, (20, 29)), (2, (20, 30)), (3, (19, 30)), (4, (20, 30))]:
            before = frame(game)
            np.testing.assert_array_equal(before, frame(game))
            result = act(game, action)
            self.assertEqual(center(game), expected)
            self.assertEqual(np.count_nonzero(before == 0) - np.count_nonzero(result.frame[-1] == 0), 1)
        self.assertEqual(game.remaining, 11)
        act(game, 5)  # Not advertised; cannot consume a directional move.
        self.assertEqual(game.remaining, 11)

    def test_full_footprint_obstacle_detour_and_blocked_dot(self):
        game = self.game(obstacle=(24, 28, 3, 5), distance=20, budget=25)
        self.assertEqual(len(reference_path(game.scene)), 20)  # 10 across + 5 up + 5 down.
        self.assertEqual(len(reference_path(game.scene, center_only=True)), 16)
        act(game, 4)
        before = frame(game)
        act(game, 4)  # Centre x=22 is clear, but right edge x=24 intersects the block.
        self.assertEqual(center(game), (21, 30))
        self.assertEqual(np.count_nonzero(before == 0) - np.count_nonzero(frame(game) == 0), 1)
        self.assertTrue(np.all(frame(game)[28:33, 24:27] == 2))
        self.assertEqual(game.current_level.get_sprites_by_name("obstacle")[0].x, 24)

    def test_all_hud_edges_exclude_player_footprint(self):
        cases = [("top", (20, 5), 1), ("bottom", (20, 58), 2),
                 ("left", (5, 20), 3), ("right", (58, 20), 4)]
        for edge, player, action in cases:
            with self.subTest(edge=edge), patch.object(sc01, "generate_scene", return_value=fixture(
                    edge=edge, player=player, player_size=3, budget=30)):
                game = sc01.Sc01()
                act(game)
                before = frame(game)
                act(game, action)
                self.assertEqual(center(game), player)
                expected = {(1 if edge == "top" else 62, x) for x in range(2, 61, 2)} if edge in ("top", "bottom") else {
                    (y, 1 if edge == "left" else 62) for y in range(2, 61, 2)}
                self.assertEqual(set(zip(*np.where(before == 0))), expected)
                self.assertEqual(np.count_nonzero(frame(game) == 0), 29)

    def test_centre_goal_last_move_and_new_level_budget(self):
        game = self.game(player_size=3, target_size=7, target=(26, 30), distance=6, budget=6)
        for _ in range(3):
            result = act(game, 4)
        self.assertEqual(result.levels_completed, 0)  # Already touching the cross's left arm.
        for _ in range(3):
            result = act(game, 4)
        self.assertEqual(result.state, GameState.NOT_FINISHED)
        self.assertEqual(result.levels_completed, 1)
        self.assertEqual(game.remaining, 6)
        self.assertEqual(len(result.frame), 2)  # Solved scene then new level, no double consumption.
        self.assertEqual(np.count_nonzero(result.frame[0] == 0), 0)
        self.assertEqual(np.count_nonzero(result.frame[1] == 0), 6)
        for _ in range(4 * 6):
            result = act(game, 4)
        self.assertEqual(result.state, GameState.WIN)
        self.assertEqual(result.levels_completed, 5)
        self.assertEqual(game.remaining, 0)
        self.assertEqual(act(game, 4).frame, [])  # Native terminal response; no extra movement.
        self.assertEqual(game.remaining, 0)
        reset = act(game)
        self.assertTrue(reset.full_reset)
        self.assertEqual(reset.levels_completed, 0)
        self.assertEqual(game.remaining, 6)

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


class GenerationTests(unittest.TestCase):
    def test_seed_matrix_geometry_visuals_and_independent_solvability(self):
        edges, pairs, colors, directions, placements, distances = set(), set(), set(), set(), set(), set()
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
                    self.assertEqual(np.count_nonzero(image == 0), scene.budget)
                    # Every object pixel is in-frame, with no initial object occlusion.
                    self.assertEqual(np.count_nonzero(image == scene.player_color), scene.player_size ** 2)
                    self.assertEqual(np.count_nonzero(image == scene.target_color), 2 * scene.target_size - 1)
                    path = reference_path(scene)
                    self.assertGreater(len(path), 0)
                    self.assertLessEqual(len(path), 24)
                    self.assertIn(scene.budget - len(path), (4, 5, 6))
                    self.assertLessEqual(scene.budget, 30)
                    self.assertEqual(scene.distance, len(path))
                    dx, dy = scene.target[0] - scene.player[0], scene.target[1] - scene.player[1]
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
                        placements.add((dx != 0 and dy != 0, scene.player))
                    self.assertEqual(scene.obstacle is not None, level >= 4)
                    if scene.obstacle:
                        x, y, w, h = scene.obstacle
                        self.assertEqual(np.count_nonzero(image == 2), w * h)
                        self.assertGreater(len(path), abs(dx) + abs(dy))
                    if level < 5:
                        self.assertNotIn(4, image)
                    edges.add(scene.edge)
                    colors.add((scene.player_color, scene.target_color))
                    act(game, 1)
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


class SDKTests(unittest.TestCase):
    def test_real_discovery_two_instances_all_levels_terminal_and_reset(self):
        self.assertEqual(version("arc-agi"), "0.9.9")
        arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / "games"))
        self.addCleanup(arcade.close_scorecard)
        self.assertIn("sc01-v1", [info.game_id for info in arcade.get_environments()])
        envs = [arcade.make("sc01-v1", seed=42) for _ in range(2)]
        self.assertTrue(all(env is not None for env in envs))
        initial = envs[0].observation_space.frame[-1].copy()
        for level in range(5):
            # Privileged inspection is confined to this test-only reference solver.
            scene = envs[0]._game.scene
            self.assertEqual(scene, envs[1]._game.scene)
            level_frame = envs[0].observation_space.frame[-1].copy()
            if level:
                envs[0].step(GameAction.ACTION1)
                reset = envs[0].reset()
                self.assertFalse(reset.full_reset)
                self.assertEqual(reset.levels_completed, level)
                np.testing.assert_array_equal(reset.frame[-1], level_frame)
            for action in reference_path(scene):
                results = [env.step(GameAction.from_id(action)) for env in envs]
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
        np.testing.assert_array_equal(reset.frame[-1], initial)


if __name__ == "__main__":
    unittest.main()
