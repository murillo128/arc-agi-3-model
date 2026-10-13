"""Test-only literal fixtures and pre-split fingerprints for both game boundaries."""

import hashlib
import json
from pathlib import Path

from arcengine import ActionInput, GameAction
from games.pixel_grid import Scene

ROOT = Path(__file__).resolve().parents[1]


def act(game, action=0, data=None):
    return game.perform_action(ActionInput(id=GameAction.from_id(action), data=data or {}), raw=True)


def frame(game):
    return game.camera.render(game.current_level.get_sprites())


def center(game):
    r = game.scene.player_size // 2
    return game.player.x + r, game.player.y + r


def fixture(**changes):
    values = dict(player=(20, 30), target=(35, 30), player_size=5, target_size=3, stride=5,
                  edge="top", obstacle=None, distance=3, budget=24, player_color=9, target_color=11)
    values.update(changes)
    return Scene(**values)


def assert_seed_baseline(test, game_class, historical_levels, *, local_levels=None):
    baseline = json.loads((ROOT / 'tests/fixtures/square_cross_seed_baseline.json').read_text())
    if local_levels is None:
        local_levels = range(1, len(historical_levels) + 1)
    for level, historical in zip(local_levels, historical_levels, strict=True):
        digest = hashlib.sha256()
        for seed in range(baseline['seeds']):
            game = game_class(seed)
            game.set_level(level - 1)
            digest.update(json.dumps(list(game.scene), separators=(',', ':')).encode())
            digest.update(frame(game).tobytes())
        test.assertEqual(digest.hexdigest(), baseline['levels'][str(historical)],
                         f'Pre-split scene/frame compatibility at local level {level}')
