"""Common pixel geometry and drawing; no curriculum, game rules or SDK lifecycle."""

from collections import deque
from typing import NamedTuple

from arcengine import Sprite

DIRECTIONS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
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


def draw_scene(level, scene):
    """Add solid squares over non-solid crosses and return the square sprites."""
    if scene.obstacle:
        x, y, w, h = scene.obstacle
        level.add_sprite(Sprite(pixels=[[2] * w for _ in range(h)],
                                x=x, y=y, name="obstacle", layer=1))
    players = []
    for (x, y), color in scene.players:
        r = scene.player_size // 2
        player = Sprite(pixels=[[color] * scene.player_size for _ in range(scene.player_size)],
                        x=x - r, y=y - r, name="player", layer=3)
        players.append(player)
        level.add_sprite(player)
    r = scene.target_size // 2
    for (tx, ty), color in scene.targets:
        cross = [[color if x == r or y == r else -1
                  for x in range(scene.target_size)] for y in range(scene.target_size)]
        level.add_sprite(Sprite(pixels=cross, x=tx - r, y=ty - r,
                                name="target", layer=2, collidable=False))
    return players


def render_hud(frame, scene, remaining):
    """Pure rendering of the reserved four-pixel strip."""
    edge = scene.edge
    if edge == "top":
        frame[:4, :] = 5
    elif edge == "bottom":
        frame[60:, :] = 5
    elif edge == "left":
        frame[:, :4] = 5
    else:
        frame[:, 60:] = 5
    filled = (180 * remaining + scene.budget - 1) // scene.budget
    for i in range(filled):
        offset, thickness = 2 + i // 3, i % 3
        if edge in ("top", "bottom"):
            frame[thickness if edge == "top" else 61 + thickness, offset] = 0
        else:
            frame[offset, thickness if edge == "left" else 61 + thickness] = 0
    return frame
