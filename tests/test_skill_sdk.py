"""Real offline SDK checks. Missing SDK is an explicit skip, never a mock pass."""
from importlib.util import find_spec
from pathlib import Path
import unittest

import numpy as np

from games.skill_core import SPECS, SKILL_SPECS
from skill_reference import plan

HAS_SDK = find_spec('arc_agi') is not None and find_spec('arcengine') is not None
if HAS_SDK:
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState
    from games.skill_curriculum import training_info

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_SDK, 'Requires project Python 3.12 and arc-agi==0.9.9; no SDK mock is used')
class OfflineSDKTests(unittest.TestCase):
    def click(self, env, command):
        x,y,w,h = env._game.puzzle.button_rects()[command]
        return env.step(GameAction.ACTION6, data={'x':x+w//2,'y':y+h//2})

    def setUp(self):
        self.arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                             environments_dir=str(ROOT/'games'))
        self.card = self.arcade.open_scorecard()
        self.addCleanup(self.arcade.close_scorecard,self.card)

    def test_discovery_all_levels_terminal_reset_and_teacher_alignment(self):
        ids = {info.game_id for info in self.arcade.get_environments()}
        expected = {s.game_id+'-v1' for s in SPECS} | {'sc01-v1','si01-v1'}
        self.assertEqual(ids, expected)
        self.assertEqual(len(self.arcade.get_environments()), len(ids))
        for spec in SKILL_SPECS:
            for seed in (0,1,42,123):
                with self.subTest(skill=spec.skill, seed=seed):
                    env = self.arcade.make(spec.game_id+'-v1', seed=seed, scorecard_id=self.card)
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

    def test_native_budget_final_action_loss_retry_and_untouched_resets(self):
        env = self.arcade.make('sy04-v1',seed=42,scorecard_id=self.card)
        initial = env.observation_space.frame[-1].copy()
        p = env._game.puzzle
        # The local SDK forwards unavailable enum actions. The synthetic
        # adapter treats them as misses; the browser separately rejects them.
        env.step(GameAction.ACTION1)
        self.assertEqual(p.steps,1)
        self.assertEqual(p.state,(0,))
        env.reset()
        p = env._game.puzzle
        for _ in range(p.budget-1):
            result = env.step(GameAction.ACTION6,data={'x':63,'y':63})
        self.assertEqual(p.steps,p.budget-1)
        self.assertEqual(p.state,(0,))
        result = self.click(env,0)
        self.assertEqual(result.levels_completed,1)
        self.assertEqual(training_info(env)['last_transition']['outcome'],'level_complete')
        level_two = result.frame[-1].copy()
        p = env._game.puzzle
        before = p.get_training_info()
        for _ in range(3):
            env.observation_space; p.render(); training_info(env)
        self.assertEqual(p.get_training_info(),before)
        for _ in range(p.budget):
            result = env.step(GameAction.ACTION7)  # Empty Undo still costs one.
        self.assertEqual(result.state,GameState.GAME_OVER)
        self.assertEqual(training_info(env)['last_transition']['outcome'],'lost')
        reset = env.reset()
        self.assertEqual(reset.levels_completed,1)
        np.testing.assert_array_equal(reset.frame[-1],level_two)
        self.assertIsNone(training_info(env)['last_transition'])
        reset = env.reset()
        self.assertEqual(reset.levels_completed,0)
        np.testing.assert_array_equal(reset.frame[-1],initial)
        np.testing.assert_array_equal(env.reset().frame[-1],initial)

    def test_reflection_axis_changes_after_real_sdk_rotation(self):
        env=self.arcade.make('sy11-v1',seed=0,scorecard_id=self.card)
        for _ in range(2):
            for command in plan(env._game.puzzle): self.click(env,command)
        p=env._game.puzzle
        self.assertEqual(p.level,3)
        self.assertEqual(p.cells(p.state),((1,1),(1,2),(2,1)))
        before=env.observation_space.frame[-1]
        self.assertEqual(int(before[54,44]),2)  # vertical reflection axis
        self.assertEqual(int(before[56,42]),p.palette[1])
        rotated=env.step(GameAction.ACTION6,data={'x':36,'y':56})
        self.assertEqual(p.cells(p.state),((0,1),(1,1),(1,2)))
        np.testing.assert_array_equal(rotated.frame[-1],p.render())
        self.assertEqual(int(rotated.frame[-1][56,42]),2)  # horizontal axis
        self.assertEqual(int(rotated.frame[-1][54,44]),p.palette[1])
        reflected=env.step(GameAction.ACTION6,data={'x':44,'y':56})
        self.assertEqual(p.cells(p.state),((0,1),(1,0),(1,1)))
        np.testing.assert_array_equal(reflected.frame[-1],p.render())
        np.testing.assert_array_equal(reflected.frame[-1][53:60,41:48],
                                      rotated.frame[-1][53:60,41:48])

    def test_nonmonotonic_labels_simultaneous_conditions_and_undo(self):
        env = self.arcade.make('sy26-v1',seed=42,scorecard_id=self.card)
        for command in plan(env._game.puzzle): self.click(env,command)
        self.assertEqual(training_info(env)['level'],2)
        info = training_info(env)
        self.assertEqual(info['subgoals'][0],dict(id='target-0',achieved=True,
                         current=1,target=1,monotonic=False,depends_on=[]))
        self.click(env,3)  # Parent not retracted: child extension is blocked.
        self.assertEqual(training_info(env)['subgoals'],info['subgoals'])
        self.click(env,0)
        self.assertFalse(training_info(env)['subgoals'][0]['achieved'])
        env.step(GameAction.ACTION7)
        self.assertEqual(training_info(env)['subgoals'],info['subgoals'])
        self.assertEqual(training_info(env)['steps'],3)
        info['subgoals'][0]['current'] = 'mutated'
        self.assertEqual(training_info(env)['subgoals'][0]['current'],1)
        env.reset()
        self.assertIsNone(training_info(env)['last_transition'])
        coupled = self.arcade.make('sy05-v1',seed=42,scorecard_id=self.card)
        self.click(coupled,0)  # Level one.
        self.assertEqual(coupled._game.puzzle.state,(0,0))
        self.assertEqual(coupled._game.puzzle.target,(1,1))
        self.click(coupled,0)
        transition = training_info(coupled)['last_transition']
        self.assertEqual(transition['level'],2)
        self.assertEqual([g['achieved'] for g in transition['before']],[False,False])
        self.assertEqual([g['achieved'] for g in transition['after']],[True,True])
        self.assertEqual(training_info(coupled)['level'],3)

    def test_native_program_prefixes_flow_stages_and_one_action_cost(self):
        for game in ('sy51-v1','sy43-v1'):
            env = self.arcade.make(game,seed=42,scorecard_id=self.card)
            # Sequential entry to a nontrivial level, never set_level.
            for _ in range(2):
                for command in plan(env._game.puzzle): self.click(env,command)
            p = env._game.puzzle
            steps = p.steps
            if game == 'sy51-v1':
                # Set three literal emit-0 instructions. Output prefixes must
                # contain 1, 2, 3 tokens, in that order, on the live output row.
                for command in (0,2,4): self.click(env,command)
                steps += 3
                result = env.step(GameAction.ACTION5)
                counts = [np.count_nonzero(f[18:22,2:54] == p.palette[0]) for f in result.frame[:3]]
                self.assertEqual(counts,[4,8,12])  # separated 2×2 emitted units in symbols-v2
            else:
                # Turn every branch on. Each source reaches the next lane in
                # sequence. Inspect native outlet pixels, not helper output.
                for i,bit in enumerate(p.state[0]):
                    if not bit: self.click(env,i); steps += 1
                result = env.step(GameAction.ACTION5)
                for stage, frame in enumerate(result.frame[:3],1):
                    for lane in range(1,4):
                        self.assertEqual(int(frame[11+5*lane,55]),p.palette[0] if lane<=stage else 5)
            self.assertEqual(p.steps,steps+1)
            self.assertGreaterEqual(len(result.frame),4)

    def test_sdk_frames_never_serialize_teacher_state(self):
        env = self.arcade.make('sy55-v1',seed=42,scorecard_id=self.card)
        data = vars(env.observation_space)
        self.assertFalse({'subgoals','deck','target','solution','training_info'} & data.keys())


if __name__=='__main__':
    unittest.main()
