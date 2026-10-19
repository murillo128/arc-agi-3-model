"""Seeded square-to-cross mazes; route verification stays environment-side."""

from collections import deque
from itertools import combinations
import random
from typing import NamedTuple

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

from games.pixel_grid import BOUNDS, DIRECTIONS, Scene, draw_scene, render_hud

LEVELS = 6
CELL = 5
WALL, FLOOR = 2, 5


class Maze(NamedTuple):
    cells: frozenset[tuple[int, int]]
    origin: tuple[int, int]
    scene: Scene


def palette_for_seed(seed):
    rng = random.Random(f"maze-cross-palette:{seed}")
    return np.asarray(rng.sample(range(16), 16), dtype=np.int8)


def _route(cells, start, goal, blocked=None):
    """Generation-only BFS; also checks whether a route has a genuine alternative."""
    queue, previous = deque([start]), {start: None}
    while queue:
        p = queue.popleft()
        if p == goal:
            route = []
            while p is not None:
                route.append(p)
                p = previous[p]
            return tuple(reversed(route))
        for dx, dy in DIRECTIONS.values():
            q = p[0] + dx, p[1] + dy
            if q in cells and q not in previous and frozenset((p, q)) != blocked:
                previous[q] = p
                queue.append(q)
    return None


def _corridor(waypoints):
    cells = {waypoints[0]}
    for (x, y), (tx, ty) in zip(waypoints, waypoints[1:]):
        dx, dy = (tx > x) - (tx < x), (ty > y) - (ty < y)
        while (x, y) != (tx, ty):
            x, y = x + dx, y + dy
            cells.add((x, y))
    return cells, waypoints[0], waypoints[-1]


def _tree(rng):
    # Logical rooms are two cells apart; intervening cells are their corridors.
    nodes = [(x, y) for y in range(0, 9, 2) for x in range(0, 9, 2)]
    start = rng.choice(nodes)
    cells, visited, stack = {start}, {start}, [start]
    while stack:
        x, y = stack[-1]
        choices = [(x + 2*dx, y + 2*dy) for dx, dy in DIRECTIONS.values()
                   if 0 <= x + 2*dx <= 8 and 0 <= y + 2*dy <= 8
                   and (x + 2*dx, y + 2*dy) not in visited]
        if not choices:
            stack.pop()
            continue
        q = rng.choice(choices)
        cells.update((q, ((x + q[0])//2, (y + q[1])//2)))
        visited.add(q)
        stack.append(q)
    return cells


def _branching_maze(rng, loops):
    for attempt in range(16):
        # The final, branched comb is a bounded fallback for an unusually long
        # run of unbranched trees. It retains the level's topology contract.
        cells = (_tree(rng) if attempt < 15 else
                 {(x, y) for y in range(9) for x in range(9) if x == 0 or y % 2 == 0})
        nodes = sorted(p for p in cells if p[0] % 2 == p[1] % 2 == 0)
        if loops:
            walls = [(x + dx, y + dy) for x, y in nodes for dx, dy in ((1, 0), (0, 1))
                     if x + 2*dx <= 8 and y + 2*dy <= 8 and (x + dx, y + dy) not in cells]
            cells.update(rng.sample(walls, 2))
            endpoints = nodes
        else:
            endpoints = [p for p in nodes if sum((p[0]+dx, p[1]+dy) in cells
                         for dx, dy in DIRECTIONS.values()) == 1]
            if len(endpoints) < 3:
                continue
        candidates = list(combinations(endpoints, 2))
        rng.shuffle(candidates)
        for start, goal in candidates:
            route = _route(cells, start, goal)
            if len(route) < 9:
                continue
            if loops and not any(_route(cells, start, goal, frozenset(edge))
                                 for edge in zip(route, route[1:])):
                continue
            return cells, start, goal
    raise RuntimeError('Could not construct a branching maze')


def generate_maze(seed, level):
    if level not in range(1, LEVELS + 1):
        raise ValueError(f"Invalid maze-cross level: seed={seed}, level={level}")
    rng = random.Random(f"maze-cross-v1:{seed}:{level}")
    if level == 1:
        cells, start, goal = _corridor(((0, 0), (rng.randint(3, 8), 0)))
    elif level == 2:
        cells, start, goal = _corridor(((0, 0), (0, rng.randint(3, 8))))
    elif level == 3:
        w, h = rng.randint(2, 7), rng.randint(2, 7)
        cells, start, goal = _corridor(((0, 0), (w, 0), (w, h)))
    elif level == 4:
        w, h = rng.randint(3, 7), rng.randint(3, 7)
        cells, start, goal = _corridor(((0, 0), (w, 0), (w, h), (0, h)))
    else:
        cells, start, goal = _branching_maze(rng, loops=level == 6)
    flip_x, flip_y = rng.choice((-1, 1)), rng.choice((-1, 1))
    transpose = level > 2 and rng.choice((False, True))
    def orient(p):
        x, y = p[0]*flip_x, p[1]*flip_y
        return (y, x) if transpose else (x, y)
    oriented = [orient(p) for p in cells]
    min_x, min_y = min(p[0] for p in oriented), min(p[1] for p in oriented)
    def transform(p):
        x, y = orient(p)
        return x - min_x, y - min_y
    cells = frozenset(transform(p) for p in cells)
    start, goal = transform(start), transform(goal)
    edge = rng.choice(tuple(BOUNDS))
    left, top, right, bottom = BOUNDS[edge]
    width, height = CELL*(max(p[0] for p in cells)+1), CELL*(max(p[1] for p in cells)+1)
    origin = rng.randint(left+1, right-width), rng.randint(top+1, bottom-height)
    def center(p):
        return origin[0]+CELL*p[0]+CELL//2, origin[1]+CELL*p[1]+CELL//2
    route = _route(cells, start, goal)
    if route is None:
        raise RuntimeError(f"Unsolvable maze-cross scene: seed={seed}, level={level}")
    distance = len(route)-1
    budget = min(128, max(24, 5*distance+8))
    scene = Scene(center(start), center(goal), 3, 3, CELL, edge, None,
                  distance, budget, 9, 11)
    return Maze(cells, origin, scene)


def floor_mask(maze):
    floor = np.zeros((64, 64), dtype=bool)
    ox, oy = maze.origin
    for x, y in maze.cells:
        floor[oy+CELL*y:oy+CELL*(y+1), ox+CELL*x:ox+CELL*(x+1)] = True
    return floor


def sweep_clear(floor, center, direction, stride, size):
    x, y = center
    dx, dy = direction
    radius = size//2
    for step in range(1, stride+1):
        cx, cy = x+dx*step, y+dy*step
        if not (radius <= cx < 64-radius and radius <= cy < 64-radius):
            return False
        if not floor[cy-radius:cy+radius+1, cx-radius:cx+radius+1].all():
            return False
    return True


class Mc01(ARCBaseGame):
    def __init__(self, seed=0):
        self.instance_seed = seed
        self.palette = palette_for_seed(seed)
        super().__init__(game_id="mc01", seed=seed, available_actions=[1, 2, 3, 4],
            levels=[Level(grid_size=(64, 64), data={"number": n}) for n in range(1, LEVELS+1)],
            camera=Camera(width=64, height=64, background=FLOOR, letter_box=FLOOR, interfaces=[self]))

    def on_set_level(self, level):
        self.number = level.get_data("number")
        self.maze = generate_maze(self.instance_seed, self.number)
        self.scene = self.maze.scene
        self.floor = floor_mask(self.maze)
        self.remaining = self.scene.budget
        level.remove_all_sprites()
        level.add_sprite(Sprite(pixels=np.where(self.floor, FLOOR, WALL).astype(np.int8),
                                name="maze", layer=0, collidable=False))
        self.player = draw_scene(level, self.scene)[0]

    def render_interface(self, frame):
        return self.palette[render_hud(frame, self.scene, self.remaining)]

    def step(self):
        direction = DIRECTIONS.get(self.action.id.value)
        if direction is not None:
            self.remaining = max(0, self.remaining-1)
            radius = self.scene.player_size//2
            center = self.player.x+radius, self.player.y+radius
            if sweep_clear(self.floor, center, direction, self.scene.stride, self.scene.player_size):
                self.player.move(direction[0]*self.scene.stride, direction[1]*self.scene.stride)
                if (self.player.x+radius, self.player.y+radius) == self.scene.target:
                    self.next_level()
                    self.complete_action()
                    return
            if self.remaining == 0:
                self.lose()
        self.complete_action()
