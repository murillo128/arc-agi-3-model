"""Launcher/HTTP boundary tests with the real seeded SDK; no browser claims here."""

import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from arc_agi.server import create_app

from scripts.play_game import ROOT, web_routes
from arc3.envs.sdk import open_arcade


class LauncherTests(unittest.TestCase):
    def command(self, *args):
        with tempfile.TemporaryDirectory() as cwd:
            return subprocess.run([sys.executable, str(ROOT / "scripts/play_game.py"), *args],
                                  cwd=cwd, capture_output=True, text=True, timeout=20)

    def test_sdk_finite_and_cwd_independent(self):
        for limit in (0, 100):
            with self.subTest(limit=limit):
                result = self.command("--mode", "sdk", "--seed", "42", "--max-actions", str(limit))
                self.assertEqual(result.returncode, 0, result.stderr)
                summary = json.loads(result.stdout)
                self.assertLessEqual(summary["actions"], limit)
                if limit == 0:
                    self.assertEqual(summary["actions"], 0)
                    self.assertEqual(summary["state"], "NOT_FINISHED")
                    self.assertEqual(summary["levels_completed"], 0)
                else:
                    self.assertGreater(summary["actions"], 0)
                if summary['stop'] == 'max-actions':
                    self.assertEqual(summary['actions'], limit)
                    self.assertEqual(summary['state'], 'NOT_FINISHED')
                else:
                    self.assertEqual(summary['stop'], 'terminal')
                    self.assertIn(summary['state'], ('WIN', 'GAME_OVER'))

    def test_make_failure_is_visible(self):
        result = self.command("--mode", "sdk", "--game", "zz00-missing")
        self.assertEqual(result.returncode, 1)
        self.assertIn("SDK could not make", result.stderr)

    def test_real_http_observations_and_input_errors(self):
        arcade = open_arcade("offline", str(ROOT / "games"))
        self.addCleanup(arcade.close_scorecard)
        env = arcade.make("sc01-v1", seed=42)
        app, _ = create_app(arcade)
        app.testing = True
        web_routes(env, 42)(arcade, app)
        client = app.test_client()
        with client.get("/") as response:
            self.assertEqual(response.status_code, 200)
        initial = client.get("/play/state").get_json()
        self.assertEqual(set(initial), {'frame', 'state', 'levels_completed', 'win_levels',
                                        'game', 'seed', 'palette'})  # No privileged geometry/stride/budget.
        self.assertEqual(initial["seed"], 42)
        self.assertEqual(initial["frame"], env.observation_space.frame[-1].tolist())
        self.assertEqual(initial["palette"][9], "#1E93FFFF")
        for invalid in ({}, {"action": []}, {"action": "ACTION6"}, []):
            self.assertEqual(client.post("/play/action", json=invalid).status_code, 400)
        self.assertEqual(client.get("/play/state").get_json(), initial)
        moved = client.post("/play/action", json={"action": "ACTION4"}).get_json()
        self.assertNotEqual(moved["frame"], initial["frame"])
        reset = client.post("/play/action", json={"action": "RESET"}).get_json()
        self.assertEqual(reset["frame"], initial["frame"])
        # Failure injection is only for the transport error response, not gameplay proof.
        with patch.object(env, "reset", return_value=None), self.assertLogs(app.logger, level="ERROR"):
            response = client.post("/play/action", json={"action": "RESET"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("SDK action failed", response.get_json()["error"])
        self.assertEqual(client.post("/play/action", json={"action": "RESET"}).status_code, 200)
        for _ in range(128):
            response = client.post("/play/action", json={"action": "ACTION2"})
            self.assertEqual(response.status_code, 200)
            if response.get_json()['state'] == 'GAME_OVER':
                break
        self.assertEqual(response.get_json()["state"], "GAME_OVER")
        self.assertEqual(client.post("/play/action", json={"action": "ACTION1"}).status_code, 409)
        self.assertEqual(client.post("/play/action", json={"action": "RESET"}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
