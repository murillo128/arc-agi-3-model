"""Original square/cross curriculum. Geometry and verification stay environment-side."""

from collections import deque
import random
from typing import NamedTuple

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

DIRECTIONS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
SIZES = (3, 5, 7)
# Bounds are inclusive; the four-pixel HUD strip is never playable.
BOUNDS = {
    "top": (0, 4, 63, 63), "bottom": (0, 0, 63, 59),
    "left": (4, 0, 63, 63), "right": (0, 0, 59, 63),
}


class Scene(NamedTuple):
    player: tuple[int, int]
    target: tuple[int, int]
    player_size: int
    target_size: int
    stride: int
    edge: str
    obstacle: tuple[int, int, int, int] | None
    distance: int
    budget: int
    player_color: int
    target_color: int


def fits(center, size, bounds, obstacle=None):
    """A square's full footprint must fit bounds and avoid the solid rectangle."""
    x, y = center
    r = size // 2
    left, top, right, bottom = bounds
    if not (left <= x - r and x + r <= right and top <= y - r and y + r <= bottom):
        return False
    if obstacle is not None:
        ox, oy, ow, oh = obstacle
        if x + r >= ox and x - r < ox + ow and y + r >= oy and y - r < oy + oh:
            return False
    return True


def sweep_fits(start, direction, stride, size, bounds, obstacle):
    """A command is atomic: every intermediate footprint must be clear."""
    x, y = start
    dx, dy = direction
    return all(fits((x + dx * step, y + dy * step), size, bounds, obstacle)
               for step in range(1, stride + 1))


def shortest_distance(start, goal, stride, size, bounds, obstacle):
    """Bounded generation check; never exported as policy input."""
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (x, y), distance = queue.popleft()
        if (x, y) == goal:
            return distance
        if distance == 24:
            continue
        for dx, dy in DIRECTIONS.values():
            point = (x + dx * stride, y + dy * stride)
            if point not in seen and sweep_fits((x, y), (dx, dy), stride, size, bounds, obstacle):
                seen.add(point)
                queue.append((point, distance + 1))
    return None


def generate_scene(seed, level):
    # A fresh level-specific stream also makes level resets independent of history.
    if level not in range(1, 6):
        raise ValueError(f"Invalid square-cross level: seed={seed}, level={level}")
    rng = random.Random(f"square-cross-v1:{seed}:{level}")
    edge = rng.choice(tuple(BOUNDS))
    bounds = BOUNDS[edge]
    ps = 5 if level == 1 else rng.choice(SIZES)
    ts = 5 if level == 1 else rng.choice(SIZES)
    stride = ps if level <= 2 else rng.randint(1, 7)
    colors = rng.sample((6, 7, 8, 9, 10, 11, 12, 14, 15), 2)
    left, top, right, bottom = bounds
    obstacle = None
    if level <= 2:
        player = ((left + right) // 2, (top + bottom) // 2)
        # Enumerate only aligned, visible, disjoint cardinal destinations.
        targets = [(player[0] + dx * stride * steps, player[1] + dy * stride * steps)
                   for dx, dy in DIRECTIONS.values() for steps in range(1, 25)
                   if stride * steps > ps // 2 + ts // 2
                   and fits((player[0] + dx * stride * steps, player[1] + dy * stride * steps),
                            max(ps, ts), bounds)]
        target = rng.choice(targets)
    else:
        r, tr = ps // 2, max(ps, ts) // 2
        if level == 3:
            # Both offsets are nonzero multiples of stride; translation keeps the
            # entire rectangle between the two footprints inside the play area.
            minimum = (r + ts // 2) // stride + 1
            across = rng.randint(minimum, max(minimum, 20 // stride))
            dx = stride * across
            dy = stride * rng.randint(1, min(24 - across, max(1, 20 // stride)))
            player, target = (0, 0), (dx, dy)
            extent = (-tr, -tr, dx + tr, dy + tr)
        else:
            # Construct an aligned crossing with a guaranteed route around either
            # end of the block. Even at stride 1 / size 7 this takes <=24 actions.
            w, h = rng.choice((3, 5)), rng.choice((3, 5))
            player = (-r - 1, 0)
            across = (w + tr - player[0] + stride - 1) // stride
            target = (player[0] + across * stride, 0)
            obstacle = (0, -(h // 2), w, h)
            detour = ((r + h // 2) // stride + 1) * stride
            extent = (player[0] - r, -detour - tr, target[0] + tr, detour + tr)
        # Reflect/transpose the whole layout, then place its clearance envelope.
        if rng.choice((False, True)):
            player, target = (-player[0], player[1]), (-target[0], target[1])
            x0, y0, x1, y1 = extent
            extent = (-x1, y0, -x0, y1)
            if obstacle:
                x, y, w, h = obstacle
                obstacle = (-x - w + 1, y, w, h)
        if rng.choice((False, True)):
            player, target = player[::-1], target[::-1]
            x0, y0, x1, y1 = extent
            extent = (y0, x0, y1, x1)
            if obstacle:
                x, y, w, h = obstacle
                obstacle = (y, x, h, w)
        x0, y0, x1, y1 = extent
        ox = rng.randint(left - x0, right - x1)
        oy = rng.randint(top - y0, bottom - y1)
        player = (player[0] + ox, player[1] + oy)
        target = (target[0] + ox, target[1] + oy)
        if obstacle:
            x, y, w, h = obstacle
            obstacle = (x + ox, y + oy, w, h)
    distance = shortest_distance(player, target, stride, ps, bounds, obstacle)
    manhattan = (abs(player[0] - target[0]) + abs(player[1] - target[1])) // stride
    if distance is None or (level >= 4 and distance <= manhattan):
        raise RuntimeError(f"Invalid square-cross construction: seed={seed}, level={level}")
    return Scene(player, target, ps, ts, stride, edge, obstacle, distance,
                 max(24, 5 * distance + 8), *colors)


class Sc01(ARCBaseGame):
    def __init__(self, seed=0):
        self.instance_seed = seed  # Base constructor calls on_set_level before setting _seed.
        super().__init__(
            game_id="sc01", seed=seed, available_actions=[1, 2, 3, 4],
            levels=[Level(grid_size=(64, 64), data={"number": n}) for n in range(1, 6)],
            camera=Camera(width=64, height=64, background=5, letter_box=5, interfaces=[self]),
        )

    def on_set_level(self, level):
        number = level.get_data("number")
        self.scene = scene = generate_scene(self.instance_seed, number)
        self.remaining = scene.budget
        level.remove_all_sprites()
        if number == 5:
            rng = random.Random(f"square-cross-noise:{self.instance_seed}")
            pixels = np.full((64, 64), -1, dtype=np.int8)
            for x, y in rng.sample([(x, y) for y in range(64) for x in range(64)], 100):
                pixels[y, x] = 4  # Low-contrast, static and non-colliding.
            level.add_sprite(Sprite(pixels=pixels, name="noise", layer=0, collidable=False))
        if scene.obstacle:
            x, y, w, h = scene.obstacle
            level.add_sprite(Sprite(pixels=[[2] * w for _ in range(h)],
                                    x=x, y=y, name="obstacle", layer=1))
        r = scene.player_size // 2
        self.player = Sprite(pixels=[[scene.player_color] * scene.player_size for _ in range(scene.player_size)],
                             x=scene.player[0] - r, y=scene.player[1] - r, name="player", layer=2)
        level.add_sprite(self.player)
        r = scene.target_size // 2
        cross = [[scene.target_color if x == r or y == r else -1
                  for x in range(scene.target_size)] for y in range(scene.target_size)]
        level.add_sprite(Sprite(pixels=cross, x=scene.target[0] - r, y=scene.target[1] - r,
                                name="target", layer=3, collidable=False))

    def render_interface(self, frame):
        # Pure rendering: redraws never consume moves, and the strip masks noise.
        edge = self.scene.edge
        if edge == "top":
            frame[:4, :] = 5
        elif edge == "bottom":
            frame[60:, :] = 5
        elif edge == "left":
            frame[:, :4] = 5
        else:
            frame[:, 60:] = 5
        filled = (180 * self.remaining + self.scene.budget - 1) // self.scene.budget
        for i in range(filled):
            offset, thickness = 2 + i // 3, i % 3
            if edge in ("top", "bottom"):
                frame[thickness if edge == "top" else 61 + thickness, offset] = 0
            else:
                frame[offset, thickness if edge == "left" else 61 + thickness] = 0
        return frame

    def step(self):
        direction = DIRECTIONS.get(self.action.id.value)
        if direction is not None:
            dx, dy = direction
            r = self.scene.player_size // 2
            center = (self.player.x + r, self.player.y + r)
            if sweep_fits(center, direction, self.scene.stride, self.scene.player_size,
                          BOUNDS[self.scene.edge], self.scene.obstacle):
                self.player.move(dx * self.scene.stride, dy * self.scene.stride)
            self.remaining = max(0, self.remaining - 1)
            if (self.player.x + r, self.player.y + r) == self.scene.target:
                self.next_level()
            elif self.remaining == 0:
                self.lose()
        self.complete_action()
