"""Real SDK/Flask protocol tests. The alternate local fixture is not an original game."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from flask import Flask

from scripts.play_game import ROOT, PlaySession, main, web_routes
from arc3.envs.sdk import open_arcade
from square_cross_reference import reference_actions

FIXTURES = ROOT / 'tests/fixtures/play_game'


class LauncherTests(unittest.TestCase):
    def command(self, *args, env=None):
        with tempfile.TemporaryDirectory() as cwd:
            return subprocess.run([sys.executable, str(ROOT / 'scripts/play_game.py'), *args],
                                  cwd=cwd, env=env, capture_output=True, text=True, timeout=20)

    def test_sdk_finite_and_cwd_independent(self):
        for limit in (0, 100):
            with self.subTest(limit=limit):
                result = self.command('--mode', 'sdk', '--seed', '42', '--max-actions', str(limit))
                self.assertEqual(result.returncode, 0, result.stderr)
                summary = json.loads(result.stdout)
                self.assertLessEqual(summary['actions'], limit)
                if limit == 0:
                    self.assertEqual(summary['actions'], 0)
                    self.assertEqual(summary['state'], 'NOT_FINISHED')
                    self.assertEqual(summary['levels_completed'], 0)
                else:
                    self.assertGreater(summary['actions'], 0)
                if summary['stop'] == 'max-actions':
                    self.assertEqual(summary['actions'], limit)
                    self.assertEqual(summary['state'], 'NOT_FINISHED')
                else:
                    self.assertEqual(summary['stop'], 'terminal')
                    self.assertIn(summary['state'], ('WIN', 'GAME_OVER'))

    def test_cli_discovery_modes_and_failures(self):
        missing = self.command('--mode', 'sdk', '--game', 'zz00-missing')
        self.assertEqual(missing.returncode, 1)
        self.assertIn('SDK could not make', missing.stderr)
        fixture = self.command('--mode', 'sdk', '--originals-dir', str(FIXTURES),
                               '--game', 'tp01', '--max-actions', '2')
        self.assertEqual(fixture.returncode, 0, fixture.stderr)
        self.assertEqual(json.loads(fixture.stdout)['game'], 'tp01-v1')
        help_text = self.command('--help').stdout
        self.assertIn('--originals-mode {offline,normal}', help_text)
        self.assertIn('network access', help_text)
        for mode in ('online', 'competition'):
            result = self.command('--mode', 'sdk', env=dict(os.environ, OPERATION_MODE=mode))
            self.assertEqual(result.returncode, 1)
            self.assertIn('override', result.stderr)
            self.assertNotEqual(self.command('--originals-mode', mode).returncode, 0)

    def test_default_remains_offline_despite_normal_environment_override(self):
        with patch.dict(os.environ, {'OPERATION_MODE': 'normal'}):
            with patch('socket.socket.connect', side_effect=AssertionError('Unexpected network access')):
                with patch('builtins.print'):
                    self.assertEqual(main(['--mode', 'sdk', '--max-actions', '1']), 0)

    def test_cleanup_and_network_opt_in(self):
        # Real instances/cards; inject only setup/cleanup failures and the remote boundary.
        created = []
        def provider(mode, path):
            arcade = open_arcade('offline', path)
            created.append((mode, arcade))
            return arcade
        with patch('scripts.play_game.open_arcade', side_effect=provider), patch('builtins.print'):
            self.assertEqual(main(['--mode', 'sdk', '--max-actions', '0', '--originals-mode', 'normal']), 0)
        self.assertEqual([mode for mode, _ in created], ['offline', 'normal'])
        for _, arcade in created:
            self.assertEqual(arcade.scorecard_manager.scorecards, {})

        first = open_arcade('offline', str(ROOT / 'games'))
        with patch.object(first, 'close_scorecard', wraps=first.close_scorecard) as close:
            with patch('scripts.play_game.open_arcade', side_effect=[first, RuntimeError('second setup failed')]), patch('builtins.print'):
                self.assertEqual(main(['--mode', 'sdk']), 1)
            close.assert_called_once()
        # Failure after both scorecards exist must release both providers.
        a = open_arcade('offline', str(ROOT / 'games'))
        b = open_arcade('offline', str(FIXTURES))
        with patch.object(b, 'get_environments', side_effect=RuntimeError('catalog failed')):
            with patch('scripts.play_game.open_arcade', side_effect=[a, b]), patch('builtins.print'):
                self.assertEqual(main(['--mode', 'sdk']), 1)
        self.assertEqual(a.scorecard_manager.scorecards, {})
        self.assertEqual(b.scorecard_manager.scorecards, {})
        # ExitStack still closes the first card if the second close raises.
        a = open_arcade('offline', str(ROOT / 'games'))
        b = open_arcade('offline', str(FIXTURES))
        with patch.object(a, 'close_scorecard', wraps=a.close_scorecard) as close_a:
            with patch.object(b, 'close_scorecard', side_effect=RuntimeError('close failed')):
                with patch('scripts.play_game.open_arcade', side_effect=[a, b]), patch('builtins.print'):
                    self.assertEqual(main(['--mode', 'sdk', '--max-actions', '0']), 1)
            close_a.assert_called_once()
        for card_id in list(b.scorecard_manager.scorecards):
            b.close_scorecard(card_id)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.synthetic = open_arcade('offline', str(ROOT / 'games'))
        self.original = open_arcade('offline', str(FIXTURES))
        sources = {}
        for name, arcade in (('synthetic', self.synthetic), ('original', self.original)):
            card = arcade.open_scorecard()
            self.addCleanup(arcade.close_scorecard, card)
            sources[name] = (arcade, card)
        self.session = PlaySession(sources, 'sc01-v1', 42)
        self.app = Flask(__name__)
        self.app.testing = True
        web_routes(self.session)(self.synthetic, self.app)
        self.client = self.app.test_client()

    def state(self):
        return self.client.get('/play/state').get_json()

    def action(self, action, **extra):
        return self.client.post('/play/action', json=dict(action=action, epoch=self.state()['epoch'], **extra))

    def select(self, game='tp01-v1', source='original', **extra):
        return self.client.post('/play/select', json=dict(game=game, source=source, epoch=self.state()['epoch'], **extra))

    def test_catalog_and_real_cross_source_switch(self):
        self.assertIsNot(self.synthetic, self.original)
        self.assertNotEqual(self.synthetic.environments_dir, self.original.environments_dir)
        catalog = self.client.get('/play/games').get_json()
        self.assertEqual(catalog, {'groups': [
            {'source': 'synthetic', 'label': 'Synthetic', 'games': [
                {'source': 'synthetic', 'id': 'sc01-v1', 'title': 'Square Cross'}]},
            {'source': 'original', 'label': 'Original ARC-AGI-3', 'games': [
                {'source': 'original', 'id': 'tp01-v1', 'title': 'Input test fixture (not official)'}]}]})
        with self.client.get('/') as response:
            self.assertEqual(response.status_code, 200)
        initial = self.state()
        self.assertEqual(set(initial), {'frame', 'state', 'levels_completed', 'win_levels', 'game',
                                       'source', 'seed', 'palette', 'epoch', 'available_actions'})
        self.assertEqual(initial['seed'], 42)
        self.assertEqual(initial['frame'], self.session.env.observation_space.frame[-1].tolist())
        self.assertEqual(initial['palette'][9], '#1E93FFFF')
        moved = self.action('ACTION4').get_json()
        self.assertNotEqual(moved['frame'], initial['frame'])
        same = self.select('sc01-v1', 'synthetic').get_json()
        self.assertEqual(same, moved)
        alternate = self.select().get_json()
        self.assertEqual(alternate['available_actions'], [5, 6, 7])
        self.assertEqual(alternate['frame'][32][32], 9)
        self.assertEqual(alternate['frame'][0][0], 5)
        self.assertEqual(alternate['source'], 'original')
        self.assertNotEqual(alternate['epoch'], initial['epoch'])
        returned = self.select('sc01-v1', 'synthetic').get_json()
        self.assertEqual(returned['frame'], initial['frame'])
        self.assertEqual(returned['available_actions'], [1, 2, 3, 4])

    def test_action_validation_and_real_animation(self):
        initial = self.state()
        for invalid in ({}, {'action': []}, [], {'action': 'ACTION8', 'epoch': initial['epoch']}):
            self.assertEqual(self.client.post('/play/action', json=invalid).status_code, 400)
        self.assertEqual(self.client.post('/play/action', json={'action': 'RESET'}).status_code, 400)
        for action in ('ACTION5', 'ACTION7'):
            self.assertEqual(self.action(action).status_code, 400)
        self.assertEqual(self.action('ACTION6', data={'x': 0, 'y': 0}).status_code, 400)
        self.assertEqual(self.action('ACTION1', data={}).status_code, 400)
        self.assertEqual(self.state(), initial)
        self.select()
        alternate = self.state()
        for bad in ({}, {'x': 0}, {'y': 0}, {'x': 0, 'y': 0, 'extra': 1}, [], None):
            self.assertEqual(self.action('ACTION6', data=bad).status_code, 400)
        for key in ('x', 'y'):
            for invalid in (-1, 64, 1.0, True, '1', None):
                with self.subTest(key=key, invalid=invalid):
                    coordinates = {'x': 0, 'y': 0, key: invalid}
                    self.assertEqual(self.action('ACTION6', data=coordinates).status_code, 400)
        self.assertEqual(self.state(), alternate)
        for x, y in ((0, 0), (63, 63)):
            clicked = self.action('ACTION6', data={'x': x, 'y': y}).get_json()
            self.assertEqual(clicked['frame'][y][x], 9)
        self.assertEqual(clicked['frame'][2][2], 2)  # One real transition per click.
        animation = self.action('ACTION5').get_json()
        self.assertEqual(len(animation['frames']), 2)
        self.assertEqual(animation['frames'][0][63][63], 8)
        self.assertEqual(animation['frames'][1][63][63], 14)
        self.assertEqual(animation['frame'], animation['frames'][-1])
        self.assertNotIn('frames', self.state())  # Polling does not replay an old action.
        self.assertEqual(self.action('ACTION7').get_json()['frame'][32][32], 9)
        self.action('ACTION6', data={'x': 1, 'y': 1})
        self.assertEqual(self.state()['available_actions'], [6, 7])
        self.assertEqual(self.action('ACTION5').status_code, 400)

    def test_failed_switch_keeps_previous_session(self):
        before = self.action('ACTION4').get_json()
        for bad in ({}, {'game': '../ls20', 'source': 'original', 'epoch': before['epoch']},
                    {'game': 'sc01-v1', 'source': 'original', 'epoch': before['epoch']},
                    {'game': [], 'source': 'original', 'epoch': before['epoch']}):
            self.assertEqual(self.client.post('/play/select', json=bad).status_code, 400)
        for failure in (None, RuntimeError('private credential must not reach JSON'), ValueError('invalid seed')):
            kwargs = {'side_effect': failure} if isinstance(failure, Exception) else {'return_value': None}
            with patch.object(self.original, 'make', **kwargs), self.assertLogs(self.app.logger, level='ERROR'):
                response = self.select()
            self.assertEqual(response.status_code, 500)
            self.assertNotIn('private credential', response.get_data(as_text=True))
            self.assertEqual(self.state(), before)
        self.assertEqual(self.action('ACTION2').status_code, 200)

    def test_real_square_cross_selection_levels_through_public_actions(self):
        initial = self.state()
        self.assertEqual(initial['win_levels'], 8)
        for level in range(1, 9):
            state = self.state()
            self.assertEqual(state['levels_completed'], level - 1)
            self.assertEqual(state['available_actions'], [1, 2, 3, 4] + ([6] if level >= 7 else []))
            self.assertEqual(set(state), set(initial))  # No geometry or selection identity in JSON.
            scene = self.session.env._game.scene  # Test-only oracle; requests use public routes.
            for action, data in reference_actions(scene, level):
                if action == 6:
                    before = self.state()
                    for bad in ({'x': -1, 'y': 0}, {'x': 64, 'y': 0}, {'x': 0, 'y': True}, {}):
                        self.assertEqual(self.action('ACTION6', data=bad).status_code, 400)
                        self.assertEqual(self.state(), before)
                response = self.action(f'ACTION{action}', **({'data': data} if data is not None else {}))
                self.assertEqual(response.status_code, 200)
                if action == 6:
                    after = response.get_json()['frame']
                    changed = {(x, y) for y in range(64) for x in range(64)
                               if before['frame'][y][x] != after[y][x]}
                    x, y = data['x'], data['y']
                    self.assertEqual(changed, {(x-1, y-1), (x, y-1), (x-1, y), (x, y)})
                    self.assertTrue(all(after[y][x] == 0 for x, y in changed))
            if level in (6, 7):
                # An unselected arrow spends budget, RESET restores the seeded
                # level and marker-free frame with ACTION6 still advertised.
                before = self.state()
                self.action('ACTION1')
                reset = self.action('RESET').get_json()
                self.assertEqual(reset['frame'], before['frame'])
                self.assertEqual(reset['available_actions'], [1, 2, 3, 4, 6])
                self.assertNotEqual(reset['epoch'], before['epoch'])
        self.assertEqual(self.state()['state'], 'WIN')
        self.assertEqual(self.state()['levels_completed'], 8)
        reset = self.action('RESET').get_json()
        self.assertEqual(reset['frame'], initial['frame'])
        self.assertEqual(reset['available_actions'], [1, 2, 3, 4])

    def test_stale_epoch_and_concurrent_switch_have_no_side_effects(self):
        old = self.state()['epoch']
        self.select()
        before = self.state()
        for endpoint, body in (('/play/action', {'action': 'ACTION6', 'data': {'x': 0, 'y': 0}}),
                               ('/play/action', {'action': 'RESET'}),
                               ('/play/select', {'game': 'sc01-v1', 'source': 'synthetic'})):
            self.assertEqual(self.client.post(endpoint, json=dict(body, epoch=old)).status_code, 409)
            self.assertEqual(self.state(), before)
        reset = self.action('RESET').get_json()
        self.assertNotEqual(before['epoch'], reset['epoch'])
        self.assertEqual(self.client.post('/play/action', json={'action': 'RESET', 'epoch': before['epoch']}).status_code, 409)
        body = dict(game='sc01-v1', source='synthetic', epoch=reset['epoch'])
        def switch(_):
            with self.app.test_client() as client:
                return client.post('/play/select', json=body).status_code
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sorted(pool.map(switch, range(2))), [200, 409])

    def test_terminal_reset_and_action_failure(self):
        initial = self.state()
        env = self.session.env
        with patch.object(env, 'reset', return_value=None), self.assertLogs(self.app.logger, level='ERROR'):
            response = self.action('RESET')
        self.assertEqual(response.status_code, 500)
        self.assertIn('SDK action failed', response.get_json()['error'])
        self.assertEqual(self.action('RESET').status_code, 200)
        for _ in range(128):
            response = self.action('ACTION2')
            self.assertEqual(response.status_code, 200)
            if response.get_json()['state'] == 'GAME_OVER':
                break
        self.assertEqual(response.get_json()['state'], 'GAME_OVER')
        self.assertEqual(self.action('ACTION1').status_code, 409)
        self.assertEqual(self.action('RESET').get_json()['frame'], initial['frame'])

    def test_missing_originals_and_colliding_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = open_arcade('offline', str(Path(directory) / 'absent'))
            self.addCleanup(empty.close_scorecard)
            sources = dict(self.session.sources, original=(empty, None))
            session = PlaySession(sources, 'sc01-v1', 42)
            app = Flask('empty')
            web_routes(session)(self.synthetic, app)
            self.assertEqual(app.test_client().get('/play/games').get_json()['groups'][1]['games'], [])
        # Independent providers can expose the same ID; startup must not guess a source.
        duplicate = open_arcade('offline', str(ROOT / 'games'))
        self.addCleanup(duplicate.close_scorecard)
        with self.assertRaisesRegex(ValueError, 'unique discovered'):
            PlaySession(dict(self.session.sources, original=(duplicate, None)), 'sc01-v1', 42)


if __name__ == '__main__':
    unittest.main()
