"""Five interaction levels, with independent SDK identity and local numbering."""

import random
from typing import NamedTuple

from arcengine import ARCBaseGame, Camera, Level, Sprite

from games.pixel_grid import (BOUNDS, DIRECTIONS, Scene, draw_scene, fits,
                              render_hud, sweep_fits)
from games.subgoals import SyntheticSubgoals


class ClickScene(NamedTuple):
    """Only squares and the HUD; introductory levels have no navigation targets."""

    players: tuple[tuple[tuple[int, int], int], ...]
    edge: str
    player_size: int = 5
    budget: int = 24


def generate_scene(seed, level):
    if level not in range(1, 6):
        raise ValueError(f"Invalid square-cross-interactions level: seed={seed}, level={level}")
    if level <= 2:
        rng = random.Random(f"square-cross-interactions-v1:{seed}:click-{level}")
        edge = rng.choice(tuple(BOUNDS))
        left, top, right, bottom = BOUNDS[edge]
        colors = rng.sample((6, 7, 8, 9, 10, 11, 12, 14, 15), level)
        # Disjoint halves guarantee separated footprints without retries.
        points = [(rng.randint(left + 6, left + 24), rng.randint(top + 6, bottom - 6))]
        if level == 2:
            points.append((rng.randint(right - 24, right - 6), rng.randint(top + 6, bottom - 6)))
        if rng.choice((False, True)):
            # Transpose within the playable region, including its HUD offset.
            points = [(left + y - top, top + x - left) for x, y in points]
        return ClickScene(tuple(zip(points, colors)), edge)
    level -= 2
    # Preserve former scene geometry. Historical numbers belong only to this
    # RNG namespace; navigation levels 3–5 map to former local levels 1–3.
    rng = random.Random(f"square-cross-v1:{seed}:{level + 5}")
    edge = rng.choice(tuple(BOUNDS))
    left, top, right, bottom = BOUNDS[edge]
    a, b = rng.sample((6, 7, 8, 9, 10, 11, 12, 14, 15), 2)
    other_player = other_target = other_color = None
    if level == 3:
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
        if level == 1:
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


class Si01(SyntheticSubgoals, ARCBaseGame):
    def __init__(self, seed=0):
        self.instance_seed = seed  # Base constructor calls on_set_level before setting _seed.
        super().__init__(
            game_id="si01", seed=seed, available_actions=[6],
            levels=[Level(grid_size=(64, 64), data={"number": n}) for n in range(1, 6)],
            camera=Camera(width=64, height=64, background=5, letter_box=5, interfaces=[self]),
        )

    def on_set_level(self, level):
        self._teacher_level = None
        self._teacher_clicked = False
        self.number = number = level.get_data("number")
        # ARCEngine 0.9.3 has no action setter; FrameData reads this list. Its
        # constructor overwrites it after the first callback, so it also gets [6].
        self._available_actions = ([6] if number <= 2 else
                                   [1, 2, 3, 4] + ([6] if number >= 4 else []))
        self.scene = scene = generate_scene(self.instance_seed, number)
        self.remaining = scene.budget
        self.selected = 0 if number == 3 else None
        self.click_selected = set()
        self.delivered = set()
        level.remove_all_sprites()
        if number <= 2:
            self.players = []
            for (x, y), color in scene.players:
                player = Sprite(pixels=[[color] * 5 for _ in range(5)],
                                x=x - 2, y=y - 2, name="player", layer=3)
                level.add_sprite(player)
                self.players.append(player)
        else:
            self.players = draw_scene(level, scene)
        self.player = self.players[0]

    def render_interface(self, frame):
        # Pure rendering: redraws never consume moves.
        indices = self.click_selected if self.number == 2 else (
            [self.selected] if self.number >= 4 and self.selected is not None else [])
        for index in indices:
            player = self.players[index]
            r = self.scene.player_size // 2
            frame[player.y + r - 1:player.y + r + 1, player.x + r - 1:player.x + r + 1] = 0
        return render_hud(frame, self.scene, self.remaining)

    def step(self):
        if self.action.id.value == 6 and self.number != 3:
            point = self.camera.display_to_grid(self.action.data.get("x", -1), self.action.data.get("y", -1))
            hit = None
            if point is not None:
                x, y = point
                for i, player in enumerate(self.players):
                    if (i not in self.delivered and player.x <= x < player.x + self.scene.player_size
                            and player.y <= y < player.y + self.scene.player_size):
                        hit = i
                        break
            if self.number == 1:
                if hit is not None:
                    self._teacher_clicked = True
                    self.next_level()
            elif self.number == 2:
                if hit is None:
                    self.click_selected.clear()
                elif hit in self.click_selected:
                    self.click_selected.remove(hit)
                else:
                    self.click_selected.add(hit)
                if len(self.click_selected) == 2:
                    self.next_level()
            else:
                self.selected = None if hit == self.selected else hit
        direction = DIRECTIONS.get(self.action.id.value)
        if direction is not None and self.number >= 3:
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
            if self.number in (3, 5) and any(
                    other_color != color and touches_cross(point, size, target, self.scene.target_size)
                    for target, other_color in self.scene.targets):
                # The stride is atomic, but swept contact takes precedence over
                # any endpoint delivery. Blocked commands never reach this check.
                player.move(dx * stride, dy * stride)
                return True
        player.move(dx * stride, dy * stride)
        if any((player.x + r, player.y + r) == target
               and (self.number not in (3, 5) or color == target_color)
               for target, target_color in self.scene.targets):
            self.delivered.add(index)
            self.selected = None
        return False
