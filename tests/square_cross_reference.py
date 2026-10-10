"""Test-only reference search. Never imported by the launcher or a policy."""

from collections import deque


def reference_path(scene, center_only=False):
    # Enumerate occupied pixels independently of production rectangle arithmetic.
    limits = {"top": (0, 4, 63, 63), "bottom": (0, 0, 63, 59),
              "left": (4, 0, 63, 63), "right": (0, 0, 59, 63)}
    left, top, right, bottom = limits[scene.edge]
    solid = set()
    if scene.obstacle:
        x, y, w, h = scene.obstacle
        solid = {(xx, yy) for xx in range(x, x + w) for yy in range(y, y + h)}
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
