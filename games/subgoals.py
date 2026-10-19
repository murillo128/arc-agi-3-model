"""Optional teacher API for the bundled synthetic games, outside the SDK protocol.

Only a goal image/mask crosses into the agent. Rules, distances and completion
predicates stay on the environment/trainer side. No action sequence is exported.
"""
from copy import deepcopy
from dataclasses import dataclass
import heapq
from itertools import count

import numpy as np

from games.pixel_grid import BOUNDS, DIRECTIONS, sweep_fits


@dataclass(frozen=True)
class Subgoal:
    id: str
    label: str
    frame: np.ndarray
    mask: np.ndarray
    piece: int
    position: tuple[int, int] | None
    selected: tuple[int, ...] = ()


def center(game, index=0):
    player = getattr(game, 'players', [game.player])[index]
    r = game.scene.player_size // 2
    return player.x + r, player.y + r


def route(game, start, target, piece=0):
    """Shortest swept-footprint route; among ties, minimize direction changes.

Other pieces are fixed. At the exported goal activation boundaries they are
either in disjoint initial lanes or already delivered/locked. This is not an
arbitrary two-piece rearrangement solver.
"""
    scene = game.scene
    players = getattr(game, 'players', [game.player])
    obstacles = [getattr(scene, 'obstacle', None)] + [
        (p.x, p.y, scene.player_size, scene.player_size) for i, p in enumerate(players) if i != piece]
    dangerous = []
    if hasattr(game, 'delivered') and game.number in (3, 5):
        color = scene.players[piece][1]
        dangerous = [t for t, c in scene.targets if c != color]
    counter = count()
    queue = [(0, 0, next(counter), start, 0, ())]
    best = {}
    while queue:
        steps, turns, _, point, direction_id, path = heapq.heappop(queue)
        key = point, direction_id
        if key in best:
            continue
        best[key] = steps, turns
        if point == target:
            return path
        for action, (dx, dy) in DIRECTIONS.items():
            if not all(sweep_fits(point, (dx, dy), scene.stride, scene.player_size,
                                 BOUNDS[scene.edge], obstacle) for obstacle in obstacles):
                continue
            radius, cross = scene.player_size // 2, scene.target_size // 2
            if any((abs(point[0] + dx * k - tx) <= radius and abs(point[1] + dy * k - ty) <= radius + cross)
                   or (abs(point[1] + dy * k - ty) <= radius and abs(point[0] + dx * k - tx) <= radius + cross)
                   for k in range(1, scene.stride + 1) for tx, ty in dangerous):
                continue
            following = point[0] + dx * scene.stride, point[1] + dy * scene.stride
            if (following, action) not in best:
                heapq.heappush(queue, (steps + 1, turns + int(direction_id not in (0, action)),
                    next(counter), following, action, path + ((action, following),)))
    return None


class SyntheticSubgoals:
    """Explicit opt-in capability: never included in FrameData or available_actions."""

    def training_subgoals(self):
        if getattr(self, '_teacher_level', None) == (self.instance_seed, self.number):
            return self._teacher_goals
        specs = []
        interactions = hasattr(self, 'delivered')
        if interactions and self.number <= 2:
            for i in range(len(self.players)):
                specs.append((i, None, tuple(range(i + 1)), f'Seleccionar pieza {i + 1}'))
        elif interactions:
            for i, (origin, color) in enumerate(self.scene.players):
                target = next(t for t, c in self.scene.targets if self.number == 4 or color == c)
                specs.append((i, target, (), f'Llevar pieza {i + 1} a su cruz'))
        else:
            path = route(self, center(self), self.scene.target)
            if path is None:
                raise ValueError('Synthetic teacher found an unreachable generated level')
            for i, (action, point) in enumerate(path):
                if i == len(path) - 1 or path[i + 1][0] != action:
                    specs.append((0, point, (), 'Llegar a la cruz' if point == self.scene.target else 'Rodear / alinear la pieza'))
        imagined = deepcopy(self)
        goals = []
        for index, (piece, position, selected, label) in enumerate(specs):
            if position is not None:
                player = getattr(imagined, 'players', [imagined.player])[piece]
                x, y = center(imagined, piece)
                player.move(position[0] - x, position[1] - y)
                if interactions:
                    imagined.selected = None
            else:
                imagined.click_selected = set(selected)
            frame = imagined.camera.render(imagined.current_level.get_sprites()).copy()
            mask = np.zeros((64, 64), dtype=bool)
            left, top, right, bottom = BOUNDS[self.scene.edge]
            mask[top:bottom + 1, left:right + 1] = True
            # Constrain only the objects this subgoal talks about. In particular,
            # selecting/moving a different piece must not make a goal image
            # contradict its native completion predicate.
            if interactions and self.number <= 2:
                mask[:] = False
                for i in selected:
                    p = self.players[i]
                    mask[p.y:p.y + 5, p.x:p.x + 5] = True
            elif interactions and len(self.players) > 1:
                mask[:] = False
                x, y = position
                r = self.scene.player_size // 2
                mask[y-r:y+r+1, x-r:x+r+1] = True
            frame.flags.writeable = mask.flags.writeable = False
            goals.append(Subgoal(f'{self.number}:{index}', label, frame, mask, piece, position, selected))
        self._teacher_level = self.instance_seed, self.number
        self._teacher_goals = tuple(goals)
        return self._teacher_goals

    def training_goal_reached(self, goal):
        if not any(goal is item for item in self.training_subgoals()):
            raise ValueError('Goal does not belong to this randomized level')
        if goal.position is not None:
            return center(self, goal.piece) == goal.position
        if self.number == 1:
            return getattr(self, '_teacher_clicked', False)
        return set(goal.selected).issubset(self.click_selected)

    def training_goal_distance(self, goal):
        """Minimum SDK actions at activation; clicks count even when budget-free."""
        if self.training_goal_reached(goal):
            return 0
        if goal.position is None:
            return 1 if self.number == 1 else len(set(goal.selected) - self.click_selected)
        path = route(self, center(self, goal.piece), goal.position, goal.piece)
        if path is None:
            return None
        return len(path) + int(hasattr(self, 'selected') and self.selected != goal.piece)
