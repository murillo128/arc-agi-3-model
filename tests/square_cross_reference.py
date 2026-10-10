"""Test-only reference search. Never imported by the launcher or a policy."""

from collections import deque


def reference_path(scene, center_only=False, *, color_rule=False):
    # Enumerate occupied pixels independently of production rectangle arithmetic.
    limits = {"top": (0, 4, 63, 63), "bottom": (0, 0, 63, 59),
              "left": (4, 0, 63, 63), "right": (0, 0, 59, 63)}
    left, top, right, bottom = limits[scene.edge]
    solid = set()
    if scene.obstacle:
        x, y, w, h = scene.obstacle
        solid = {(xx, yy) for xx in range(x, x + w) for yy in range(y, y + h)}
    if scene.other_player is not None:
        solid |= square_pixels(scene.other_player, scene.player_size)
    if color_rule:
        for point, color in scene.targets:
            if color != scene.player_color:
                solid |= cross_pixels(point, scene.target_size)
    radius = 0 if center_only else scene.player_size // 2
    forbidden = set()
    for x in range(64):
        for y in range(64):
            if any(not (left <= xx <= right and top <= yy <= bottom) or (xx, yy) in solid
                   for xx in range(x - radius, x + radius + 1)
                   for yy in range(y - radius, y + radius + 1)):
                forbidden.add((x, y))
    queue = deque([scene.player])
    parents = {scene.player: None}
    steps = ((1, 0, -1), (2, 0, 1), (3, -1, 0), (4, 1, 0))
    while queue:
        point = queue.popleft()
        if point == scene.target:
            path = []
            while parents[point] is not None:
                point, action = parents[point]
                path.append(action)
            return path[::-1]
        x, y = point
        for action, dx, dy in steps:
            sweep = [(x + dx * step, y + dy * step) for step in range(1, scene.stride + 1)]
            nxt = sweep[-1]
            if nxt not in parents and all(0 <= xx < 64 and 0 <= yy < 64 and (xx, yy) not in forbidden
                                          for xx, yy in sweep):
                parents[nxt] = (point, action)
                queue.append(nxt)
    raise AssertionError("Reference search found no solution")


def square_pixels(point, size=5):
    x, y = point
    r = size // 2
    return {(xx, yy) for xx in range(x - r, x + r + 1) for yy in range(y - r, y + r + 1)}


def cross_pixels(point, size=5):
    x, y = point
    r = size // 2
    return {(x + d, y) for d in range(-r, r + 1)} | {(x, y + d) for d in range(-r, r + 1)}


def reference_actions(scene, level, order=(0, 1)):
    """SDK actions with free selections; only tests may inspect seeded geometry.

    Search one piece at a time with the other solid at its current/locked
    position. The matrix also proves the lanes disjoint and the total length
    equal to the Manhattan lower bound, so a joint two-piece BFS is unnecessary.
    """
    if level < 8:
        clicks = [(6, dict(zip(('x', 'y'), scene.player)))] if level == 7 else []
        return clicks + [(action, None) for action in reference_path(scene, color_rule=level == 6)]
    positions = [point for point, _ in scene.players]
    targets = [point for point, _ in scene.targets]
    colors = [color for _, color in scene.players]
    actions = []
    for i in order:
        single = scene._replace(player=positions[i], target=targets[i], player_color=colors[i],
                                target_color=colors[i], other_player=positions[1-i],
                                other_target=targets[1-i], other_color=colors[1-i])
        actions.append((6, dict(zip(('x', 'y'), positions[i]))))
        actions.extend((action, None) for action in reference_path(single, color_rule=True))
        positions[i] = targets[i]
    return actions
