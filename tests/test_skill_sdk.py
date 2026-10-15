"""Real offline SDK checks. Missing SDK is an explicit skip, never a mock pass."""
from importlib.util import find_spec
from pathlib import Path
import unittest

import numpy as np

from games.skill_core import SPECS
from skill_reference import plan

HAS_SDK = find_spec('arc_agi') is not None and find_spec('arcengine') is not None
if HAS_SDK:
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState
    from games.skill_curriculum import training_info

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_SDK, 'Requires project Python 3.12 and arc-agi==0.9.9; no SDK mock is used')
class OfflineSDKTests(unittest.TestCase):
    def setUp(self):
        self.arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                             environments_dir=str(ROOT/'games'))
        self.card = self.arcade.open_scorecard()
        self.addCleanup(self.arcade.close_scorecard,self.card)

    def test_discovery_all_levels_terminal_reset_and_teacher_alignment(self):
        ids = {info.game_id for info in self.arcade.get_environments()}
        self.assertTrue({s.game_id+'-v1' for s in SPECS}.issubset(ids))
        for spec in SPECS:
            with self.subTest(skill=spec.skill):
                env = self.arcade.make(spec.game_id+'-v1', seed=42, scorecard_id=self.card)
                self.assertIsNotNone(env)
                initial = env.observation_space
                self.assertEqual(initial.win_levels,6)
                self.assertEqual(initial.levels_completed,0)
                for level in range(1,7):
                    p = env._game.puzzle
                    self.assertEqual(p.level,level)
                    self.assertEqual(training_info(env)['level'],level)
                    np.testing.assert_array_equal(env.observation_space.frame[-1],p.render())
                    for command in plan(p):
                        x,y,w,h = p.button_rects()[command]
                        result = env.step(GameAction.ACTION6,data={'x':x+w//2,'y':y+h//2})
                        self.assertIsNotNone(result)
                        self.assertNotEqual(result.state,GameState.GAME_OVER)
                        self.assertTrue(result.frame)
                        for image in result.frame:
                            self.assertEqual(image.shape,(64,64))
                            self.assertTrue(np.all((image>=0)&(image<16)))
                    self.assertEqual(result.levels_completed,level)
                    transition = training_info(env)['last_transition']
                    self.assertEqual(transition['level'],level)
                    self.assertEqual(transition['outcome'],'level_complete')
                    self.assertTrue(all(g['achieved'] for g in transition['after']))
                self.assertEqual(result.state,GameState.WIN)
                reset = env.reset()
                self.assertEqual(reset.levels_completed,0)
                self.assertEqual(training_info(env)['level'],1)
                self.assertIsNone(training_info(env)['last_transition'])
                np.testing.assert_array_equal(reset.frame[-1],initial.frame[-1])

    def test_action_mapping_animation_undo_and_synthetic_guard(self):
        with self.assertRaises(ValueError): training_info(object())
        env = self.arcade.make('sy11-v1',seed=42,scorecard_id=self.card)
        p = env._game.puzzle
        command = plan(p)[0]
        result = env.step(GameAction.from_id(command+1))
        self.assertEqual(result.levels_completed,1)
        self.assertEqual(training_info(env)['last_transition']['level'],1)
        env = self.arcade.make('sy55-v1',seed=42,scorecard_id=self.card)
        p = env._game.puzzle
        for command in plan(p):
            x,y,w,h=p.button_rects()[command]
            result=env.step(GameAction.ACTION6,data={'x':x+w//2,'y':y+h//2})
        self.assertEqual(result.levels_completed,1)
        p=env._game.puzzle
        first=0; second=next(i for i,v in enumerate(p.deck) if v!=p.deck[first])
        for i in (first,second):
            x,y,w,h=p.button_rects()[i]
            result=env.step(GameAction.ACTION6,data={'x':x+w//2,'y':y+h//2})
        self.assertGreaterEqual(len(result.frame),2)
        self.assertFalse(np.array_equal(result.frame[0][4:61],result.frame[-1][4:61]))
        self.assertEqual(p.state,(0,-1,-1))
        result=env.step(GameAction.ACTION7)
        self.assertEqual(p.state[1],first)
        self.assertEqual(p.steps,3)
        reset=env.reset()
        self.assertEqual(training_info(env)['level'],2)
        self.assertEqual(env._game.puzzle.steps,0)
        self.assertEqual(env._game.puzzle.state,(0,-1,-1))


if __name__=='__main__':
    unittest.main()
