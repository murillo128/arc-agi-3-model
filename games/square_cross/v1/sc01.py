"""Five-level square/cross curriculum. Verification stays environment-side."""

import random

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

from games.pixel_grid import (BOUNDS, DIRECTIONS, Scene, draw_scene, fits,
                              render_hud, shortest_distance, sweep_fits)

SIZES = (3, 5, 7)


def palette_for_seed(seed):
    """Independent of geometry: every visual role may use any ARC color."""
    rng = random.Random(f"square-cross-palette:{seed}")
    return np.asarray(rng.sample(range(16), 16), dtype=np.int8)


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
        # Keep the sampled cardinal route but remove the fixed central start.
        # Independent RNG leaves route length, colors and later levels intact.
        placement = random.Random(f"square-cross-placement:{seed}:{level}")
        pr, tr = ps // 2, max(ps, ts) // 2
        x0, x1 = min(player[0] - pr, target[0] - tr), max(player[0] + pr, target[0] + tr)
        y0, y1 = min(player[1] - pr, target[1] - tr), max(player[1] + pr, target[1] + tr)
        ox = placement.randint(left - x0, right - x1)
        oy = placement.randint(top - y0, bottom - y1)
        player = player[0] + ox, player[1] + oy
        target = target[0] + ox, target[1] + oy
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
        self.palette = palette_for_seed(seed)
        super().__init__(
            game_id="sc01", seed=seed, available_actions=[1, 2, 3, 4],
            levels=[Level(grid_size=(64, 64), data={"number": n}) for n in range(1, 6)],
            camera=Camera(width=64, height=64, background=5, letter_box=5, interfaces=[self]),
        )

    def on_set_level(self, level):
        self.number = number = level.get_data("number")
        self.scene = scene = generate_scene(self.instance_seed, number)
        self.remaining = scene.budget
        level.remove_all_sprites()
        if number == 5:
            rng = random.Random(f"square-cross-noise:{self.instance_seed}")
            pixels = np.full((64, 64), -1, dtype=np.int8)
            for x, y in rng.sample([(x, y) for y in range(64) for x in range(64)], 100):
                pixels[y, x] = 4  # Static and non-colliding; final palette sets its color.
            level.add_sprite(Sprite(pixels=pixels, name="noise", layer=0, collidable=False))
        self.player = draw_scene(level, scene)[0]

    def render_interface(self, frame):
        # Relabel the finished render so background, HUD and sprites all share
        # one bijection. Never permute per action or change collision/rule data.
        return self.palette[render_hud(frame, self.scene, self.remaining)]

    def step(self):
        direction = DIRECTIONS.get(self.action.id.value)
        if direction is not None:
            self.remaining = max(0, self.remaining - 1)
            size, stride = self.scene.player_size, self.scene.stride
            r = size // 2
            center = self.player.x + r, self.player.y + r
            if sweep_fits(center, direction, stride, size, BOUNDS[self.scene.edge], self.scene.obstacle):
                dx, dy = direction
                self.player.move(dx * stride, dy * stride)
                if (self.player.x + r, self.player.y + r) == self.scene.target:
                    self.next_level()
                    self.complete_action()
                    return
            if self.remaining == 0:
                self.lose()
        self.complete_action()
