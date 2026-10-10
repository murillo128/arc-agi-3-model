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
    other_target: tuple[int, int] | None = None
    other_color: int | None = None
    other_player: tuple[int, int] | None = None

    @property
    def players(self):
        return [(self.player, self.player_color)] + (
            [(self.other_player, self.other_color)] if self.other_player is not None else [])

    @property
    def targets(self):
        return [(self.target, self.target_color)] + (
            [(self.other_target, self.other_color)] if self.other_target is not None else [])


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
    if level not in range(1, 9):
        raise ValueError(f"Invalid square-cross level: seed={seed}, level={level}")
    rng = random.Random(f"square-cross-v1:{seed}:{level}")
    if level >= 6:
        return generate_later_scene(rng, level)
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


def generate_later_scene(rng, level):
    edge = rng.choice(tuple(BOUNDS))
    left, top, right, bottom = BOUNDS[edge]
    a, b = rng.sample((6, 7, 8, 9, 10, 11, 12, 14, 15), 2)
    other_player = other_target = other_color = None
    if level == 8:
        # Two parallel, disjoint clearance corridors. Each can be solved in
        # either order even after the other square is locked at its destination.
        pairs = []
        for lane in (12, 42):
            start = rng.randint(8, 18)
            end = start + 5 * rng.randint(2, 6)
            pair = [(start, lane), (end, lane)]
            if rng.choice((False, True)):
                pair.reverse()
            pairs.append(pair)
        if rng.choice((False, True)):
            pairs = [[(y, x) for x, y in pair] for pair in pairs]
        rng.shuffle(pairs)
        (player, target), (other_player, other_target) = [
            [(x + left, y + top) for x, y in pair] for pair in pairs]
        other_color = b
        target_color = a
        distance = sum((abs(p[0] - t[0]) + abs(p[1] - t[1])) // 5
                       for p, t in ((player, target), (other_player, other_target)))
    else:
        player = ((left + right) // 2, (top + bottom) // 2)
        direction = rng.choice(tuple(DIRECTIONS.values()))

        def destination(dx, dy):
            return rng.choice([(player[0] + dx * 5 * n, player[1] + dy * 5 * n)
                               for n in range(2, 6)
                               if fits((player[0] + dx * 5 * n, player[1] + dy * 5 * n),
                                       5, BOUNDS[edge])])

        target = destination(*direction)
        target_color = b
        if level == 6:
            # A different cardinal ray keeps the entire correct sweep clear
            # of the lethal cross, including its arms and the square's edges.
            other_target = destination(*rng.choice([d for d in DIRECTIONS.values() if d != direction]))
            target_color, other_color = a, b
        distance = (abs(player[0] - target[0]) + abs(player[1] - target[1])) // 5
    return Scene(player, target, 5, 5, 5, edge, None, distance,
                 max(24, 5 * distance + 8), a, target_color,
                 other_target, other_color, other_player)


def touches_cross(center, size, target, target_size):
    """Filled square against the two visible one-pixel-wide arms (not their box)."""
    x, y = center
    tx, ty = target
    r, tr = size // 2, target_size // 2
    return ((abs(x - tx) <= r and abs(y - ty) <= r + tr)
            or (abs(y - ty) <= r and abs(x - tx) <= r + tr))


class Sc01(ARCBaseGame):
    def __init__(self, seed=0):
        self.instance_seed = seed  # Base constructor calls on_set_level before setting _seed.
        super().__init__(
            game_id="sc01", seed=seed, available_actions=[1, 2, 3, 4],
            levels=[Level(grid_size=(64, 64), data={"number": n}) for n in range(1, 9)],
            camera=Camera(width=64, height=64, background=5, letter_box=5, interfaces=[self]),
        )

    def on_set_level(self, level):
        self.number = number = level.get_data("number")
        # ARCEngine 0.9.3 has no action setter; FrameData reads this list. Its
        # constructor assigns the same four directions after the first callback.
        self._available_actions = [1, 2, 3, 4] + ([6] if number >= 7 else [])
        self.scene = scene = generate_scene(self.instance_seed, number)
        self.remaining = scene.budget
        self.selected = None if number >= 7 else 0
        self.delivered = set()
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
        self.players = []
        for (x, y), color in scene.players:
            r = scene.player_size // 2
            player = Sprite(pixels=[[color] * scene.player_size for _ in range(scene.player_size)],
                            x=x - r, y=y - r, name="player", layer=3)
            self.players.append(player)
            level.add_sprite(player)
        self.player = self.players[0]
        r = scene.target_size // 2
        for (tx, ty), color in scene.targets:
            cross = [[color if x == r or y == r else -1
                      for x in range(scene.target_size)] for y in range(scene.target_size)]
            level.add_sprite(Sprite(pixels=cross, x=tx - r, y=ty - r,
                                    name="target", layer=2, collidable=False))

    def render_interface(self, frame):
        # Pure rendering: redraws never consume moves, and the strip masks noise.
        if self.number >= 7 and self.selected is not None:
            player = self.players[self.selected]
            r = self.scene.player_size // 2
            frame[player.y + r - 1:player.y + r + 1, player.x + r - 1:player.x + r + 1] = 0
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
        if self.action.id.value == 6 and self.number >= 7:
            point = self.camera.display_to_grid(self.action.data.get("x", -1), self.action.data.get("y", -1))
            hit = None
            if point is not None:
                x, y = point
                for i, player in enumerate(self.players):
                    if (i not in self.delivered and player.x <= x < player.x + self.scene.player_size
                            and player.y <= y < player.y + self.scene.player_size):
                        hit = i
                        break
            self.selected = None if hit == self.selected else hit
        direction = DIRECTIONS.get(self.action.id.value)
        if direction is not None:
            self.remaining = max(0, self.remaining - 1)
            lost = False
            if self.selected is not None:
                lost = self.move_selected(direction)
            if lost:
                self.selected = None
                self.lose()
            elif len(self.delivered) == len(self.players):
                self.next_level()
            elif self.remaining == 0:
                self.selected = None
                self.lose()
        self.complete_action()

    def move_selected(self, direction):
        index = self.selected
        player = self.players[index]
        size, stride = self.scene.player_size, self.scene.stride
        r = size // 2
        center = player.x + r, player.y + r
        obstacles = [self.scene.obstacle] + [
            (other.x, other.y, size, size) for other in self.players if other is not player]
        if not all(sweep_fits(center, direction, stride, size, BOUNDS[self.scene.edge], obstacle)
                   for obstacle in obstacles):
            return False
        dx, dy = direction
        color = self.scene.players[index][1]
        for step in range(1, stride + 1):
            point = center[0] + dx * step, center[1] + dy * step
            if self.number in (6, 8) and any(
                    other_color != color and touches_cross(point, size, target, self.scene.target_size)
                    for target, other_color in self.scene.targets):
                # The stride is atomic, but swept contact takes precedence over
                # any endpoint delivery. Blocked commands never reach this check.
                player.move(dx * stride, dy * stride)
                return True
        player.move(dx * stride, dy * stride)
        if any((player.x + r, player.y + r) == target
               and (self.number not in (6, 8) or color == target_color)
               for target, target_color in self.scene.targets):
            self.delivered.add(index)
            self.selected = None
        return False
