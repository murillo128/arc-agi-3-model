"""Independent pixel-sweep and graph oracles for the new native maze game."""
from collections import deque
from itertools import pairwise
from math import ceil
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from games.maze_cross.v1 import mc01
from games.pixel_grid import Scene
from square_cross_support import act, center, frame

ROOT = Path(__file__).resolve().parents[1]
STEPS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def visible_floor(game):
    image = frame(game)
    floor = np.isin(image, game.palette[[5, 9, 11]])
    # Literal HUD exclusion, independent of the production movement predicate.
    edge = game.scene.edge
    if edge == 'top': floor[:4] = False
    elif edge == 'bottom': floor[60:] = False
    elif edge == 'left': floor[:, :4] = False
    else: floor[:, 60:] = False
    return floor


def reference_path(game):
    """BFS using full swept rectangles read from rendered pixels, not game rules."""
    allowed = visible_floor(game)
    start, goal = center(game), game.scene.target
    queue, seen = deque([(start, ())]), {start}
    while queue:
        p, actions = queue.popleft()
        if p == goal:
            return actions
        for action, (dx, dy) in STEPS.items():
            q = p[0]+5*dx, p[1]+5*dy
            x0, x1 = min(p[0], q[0])-1, max(p[0], q[0])+1
            y0, y1 = min(p[1], q[1])-1, max(p[1], q[1])+1
            if q not in seen and 0 <= x0 <= x1 < 64 and 0 <= y0 <= y1 < 64 and allowed[y0:y1+1, x0:x1+1].all():
                seen.add(q)
                queue.append((q, actions+(action,)))
    raise AssertionError('No full-footprint path in the rendered maze')


def component(cells, start, excluded=frozenset()):
    found, queue = {start}, deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in STEPS.values():
            q = x+dx, y+dy
            if q in cells and q not in found and {(x, y), q} != excluded:
                found.add(q)
                queue.append(q)
    return found


class MazeGenerationTests(unittest.TestCase):
    def test_seed_matrix_topology_solvability_colours_and_clearance(self):
        variants = {level: set() for level in range(1, 7)}
        palettes, edges = set(), set()
        for seed in range(64):
            game = mc01.Mc01(seed)
            palette = game.palette.copy()
            for level in range(1, 7):
                with self.subTest(seed=seed, level=level):
                    game.set_level(level-1)
                    maze, scene = game.maze, game.scene
                    image = frame(game)
                    self.assertEqual(image.shape, (64, 64))
                    self.assertTrue(np.all((image >= 0) & (image < 16)))
                    self.assertEqual(np.unique(game.palette).size, 16)
                    np.testing.assert_array_equal(game.palette, palette)
                    self.assertEqual(np.count_nonzero(image == palette[9]), 9)
                    self.assertEqual(np.count_nonzero(image == palette[11]), 5)
                    self.assertEqual(np.count_nonzero(image == palette[0]), 180)
                    self.assertEqual(np.count_nonzero(visible_floor(game)), len(maze.cells)*25)
                    path = reference_path(game)
                    self.assertGreater(len(path), 0)
                    self.assertEqual(scene.distance, len(path))
                    self.assertLessEqual(len(path), scene.budget)
                    self.assertLessEqual(scene.budget, 128)
                    self.assertEqual(component(maze.cells, next(iter(maze.cells))), set(maze.cells))
                    degrees = {p: sum((p[0]+dx, p[1]+dy) in maze.cells for dx, dy in STEPS.values())
                               for p in maze.cells}
                    cycles = sum(degrees.values())//2-len(maze.cells)+1
                    if level <= 4:
                        self.assertEqual(cycles, 0)
                        self.assertEqual(sum(a != b for a, b in pairwise(path)), max(0, level-2))
                        self.assertEqual(sum(d == 1 for d in degrees.values()), 2)
                        if level == 1: self.assertTrue(set(path) <= {3, 4})
                        if level == 2: self.assertTrue(set(path) <= {1, 2})
                    elif level == 5:
                        self.assertEqual(cycles, 0)
                        self.assertGreaterEqual(sum(d == 1 for d in degrees.values()), 3)
                        self.assertTrue(any(d >= 3 for d in degrees.values()))
                    else:
                        self.assertEqual(cycles, 2)
                        x, y = ((scene.player[i]-maze.origin[i]-2)//5 for i in (0, 1))
                        points = [(x, y)]
                        for action in path:
                            dx, dy = STEPS[action]
                            x, y = x+dx, y+dy
                            points.append((x, y))
                        self.assertTrue(any(points[-1] in component(maze.cells, points[0], frozenset(edge))
                                            for edge in pairwise(points)), 'Loop must offer a route to the goal')
                    variants[level].add((maze.cells, scene.player, scene.target))
                    palettes.add(tuple(palette))
                    edges.add(scene.edge)
        self.assertTrue(all(len(layouts) >= 32 for layouts in variants.values()))
        self.assertGreaterEqual(len(palettes), 32)
        self.assertEqual(edges, {'top', 'bottom', 'left', 'right'})

    def test_seeded_generation_and_bounded_branching_fallback(self):
        for level in range(1, 7):
            self.assertEqual(mc01.generate_maze(123, level), mc01.generate_maze(123, level))
        # An unlucky unbranched generator must still yield a proper maze.
        with patch.object(mc01, '_tree', return_value={(x, 0) for x in range(9)}):
            maze = mc01.generate_maze(5, 5)
        degrees = [sum((x+dx, y+dy) in maze.cells for dx, dy in STEPS.values()) for x, y in maze.cells]
        self.assertGreaterEqual(degrees.count(1), 3)
        self.assertEqual(component(maze.cells, next(iter(maze.cells))), set(maze.cells))
        for level in (0, 7):
            with self.assertRaises(ValueError): mc01.generate_maze(0, level)


class MazeEngineTests(unittest.TestCase):
    def test_literal_turn_blocked_cost_static_walls_and_last_move_arrival(self):
        scene = Scene((12, 12), (17, 17), 3, 3, 5, 'top', None, 2, 4, 9, 11)
        maze = mc01.Maze(frozenset(((0, 0), (1, 0), (1, 1))), (10, 10), scene)
        with patch.object(mc01, 'generate_maze', return_value=maze), patch.object(mc01, 'palette_for_seed', return_value=np.arange(16)):
            game = mc01.Mc01(0)
            act(game)
            self.assertEqual(reference_path(game), (4, 2))
            walls = frame(game) == 2
            for action, expected, remaining in ((1, (12, 12), 3), (4, (17, 12), 2), (4, (17, 12), 1)):
                result = act(game, action)
                self.assertEqual(center(game), expected)
                self.assertEqual(game.remaining, remaining)
                self.assertEqual(result.levels_completed, 0)
                np.testing.assert_array_equal(frame(game) == 2, walls)
                self.assertEqual(np.count_nonzero(frame(game) == 0), ceil(180*remaining/4))
            result = act(game, 2)
            self.assertEqual(result.levels_completed, 1)
            self.assertEqual(result.state, GameState.NOT_FINISHED)
            self.assertEqual(game.remaining, 4)
            self.assertEqual(len(result.frame), 2)
            self.assertEqual(np.count_nonzero(result.frame[0] == 0), 0)
            self.assertEqual(np.count_nonzero(result.frame[-1] == 0), 180)

    def test_mid_sweep_and_full_footprint_cannot_cross_walls(self):
        floor = np.zeros((64, 64), dtype=bool)
        floor[10:15, 10:20] = True
        self.assertTrue(mc01.sweep_clear(floor, (12, 12), (1, 0), 5, 3))
        # Endpoints and the swept centre line are clear; only an outer pixel hits.
        floor[11, 15] = False
        self.assertFalse(mc01.sweep_clear(floor, (12, 12), (1, 0), 5, 3))
        self.assertFalse(mc01.sweep_clear(floor, (12, 12), (-1, 0), 5, 3))
        self.assertFalse(mc01.sweep_clear(np.ones((64, 64), dtype=bool), (2, 12), (-1, 0), 5, 3))


class MazeSDKTests(unittest.TestCase):
    def setUp(self):
        self.arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT/'games'))
        self.card = self.arcade.open_scorecard()
        self.addCleanup(self.arcade.close_scorecard, self.card)

    def test_real_sdk_six_level_completion_loss_retry_and_win_reset(self):
        self.assertIn('mc01-v1', {g.game_id for g in self.arcade.get_environments()})
        for seed in (0, 1, 42, 123):
            with self.subTest(seed=seed):
                env = self.arcade.make('mc01-v1', seed=seed, scorecard_id=self.card, save_recording=False)
                initial = env.observation_space.frame[-1].copy()
                self.assertEqual(env.observation_space.available_actions, [1, 2, 3, 4])
                self.assertEqual(env.observation_space.win_levels, 6)
                # Horizontal level: up is always blocked, consumes budget, and loses.
                for _ in range(env._game.remaining): result = env.step(GameAction.ACTION1)
                self.assertEqual(result.state, GameState.GAME_OVER)
                np.testing.assert_array_equal(env.reset().frame[-1], initial)
                for level in range(1, 7):
                    game = env._game
                    level_initial = env.observation_space.frame[-1].copy()
                    path = reference_path(game)
                    # Retry a moved later level through the native reset protocol.
                    if level == 3:
                        env.step(GameAction.from_id(path[0]))
                        retry = env.reset()
                        self.assertEqual(retry.levels_completed, 2)
                        np.testing.assert_array_equal(retry.frame[-1], level_initial)
                    for action in path:
                        result = env.step(GameAction.from_id(action))
                        self.assertNotEqual(result.state, GameState.GAME_OVER)
                        self.assertFalse({'maze', 'floor', 'route', 'distance', 'solution'} & vars(result).keys())
                    self.assertEqual(result.levels_completed, level)
                    self.assertEqual(result.state, GameState.WIN if level == 6 else GameState.NOT_FINISHED)
                reset = env.reset()
                self.assertEqual(reset.levels_completed, 0)
                np.testing.assert_array_equal(reset.frame[-1], initial)


if __name__ == '__main__':
    unittest.main()
