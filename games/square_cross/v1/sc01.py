"""Original square/cross curriculum. Geometry and verification stay environment-side."""

from collections import deque
import random
from typing import NamedTuple

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

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


def shortest_distance(start, goal, size, bounds, obstacle):
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
            point = (x + dx, y + dy)
            if point not in seen and fits(point, size, bounds, obstacle):
                seen.add(point)
                queue.append((point, distance + 1))
    return None


def generate_scene(seed, level):
    # A fresh level-specific stream also makes level resets independent of history.
    rng = random.Random(f"square-cross-v1:{seed}:{level}")
    edge = rng.choice(tuple(BOUNDS))
    bounds = BOUNDS[edge]
    ps = 5 if level == 1 else rng.choice(SIZES)
    ts = 5 if level == 1 else rng.choice(SIZES)
    colors = rng.sample((6, 7, 8, 9, 10, 11, 12, 14, 15), 2)
    for _ in range(128):
        obstacle = None
        if level <= 2:
            left, top, right, bottom = bounds
            player = ((left + right) // 2, (top + bottom) // 2)
            dx, dy = rng.choice(tuple(DIRECTIONS.values()))
            gap = rng.randint(ps // 2 + ts // 2 + 2, 20)
            target = (player[0] + dx * gap, player[1] + dy * gap)
        else:
            player = (rng.randint(4, 59), rng.randint(4, 59))
            if level == 3:
                target = (player[0] + rng.randint(-16, 16), player[1] + rng.randint(-16, 16))
            else:
                # Short crossing of one rectangle, with room to go around its end.
                dx = rng.choice((-1, 1)) * rng.randint(9, 16)
                dy = rng.randint(-3, 3)
                if rng.choice((False, True)):
                    dx, dy = dy, dx
                target = (player[0] + dx, player[1] + dy)
                w, h = rng.choice((3, 5)), rng.choice((3, 5))
                obstacle = ((player[0] + target[0]) // 2 - w // 2,
                            (player[1] + target[1]) // 2 - h // 2, w, h)
        if not fits(player, ps, bounds, obstacle) or not fits(target, max(ps, ts), bounds, obstacle):
            continue
        if obstacle:
            x, y, w, h = obstacle
            left, top, right, bottom = bounds
            if not (left <= x and top <= y and x + w - 1 <= right and y + h - 1 <= bottom):
                continue
        # Disjoint bounding boxes ensure no initial square/cross overlap.
        if max(abs(player[0] - target[0]), abs(player[1] - target[1])) <= ps // 2 + ts // 2:
            continue
        distance = shortest_distance(player, target, ps, bounds, obstacle)
        if distance is None:
            continue
        manhattan = abs(player[0] - target[0]) + abs(player[1] - target[1])
        if level >= 4 and distance <= manhattan:
            continue
        return Scene(player, target, ps, ts, edge, obstacle, distance,
                     distance + rng.randint(4, 6), *colors)
    raise RuntimeError(f"No square-cross layout after 128 candidates: seed={seed}, level={level}")


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
        for i in range(self.remaining):
            offset = 2 + 2 * i
            if edge in ("top", "bottom"):
                frame[1 if edge == "top" else 62, offset] = 0
            else:
                frame[offset, 1 if edge == "left" else 62] = 0
        return frame

    def step(self):
        direction = DIRECTIONS.get(self.action.id.value)
        if direction is not None:
            dx, dy = direction
            r = self.scene.player_size // 2
            center = (self.player.x + r + dx, self.player.y + r + dy)
            if fits(center, self.scene.player_size, BOUNDS[self.scene.edge], self.scene.obstacle):
                self.player.move(dx, dy)
            self.remaining = max(0, self.remaining - 1)
            if (self.player.x + r, self.player.y + r) == self.scene.target:
                self.next_level()
            elif self.remaining == 0:
                self.lose()
        self.complete_action()
