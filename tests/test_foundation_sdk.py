"""Mandatory pinned real SDK boundary; absence is an error, never a skip."""
from pathlib import Path
import unittest

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from games.skill_core import FOUNDATION_SPECS
from games.skill_curriculum import training_info
from foundation_reference import plan, sdk_action

ROOT=Path(__file__).resolve().parents[1]


class FoundationSDK(unittest.TestCase):
    def setUp(self):
        self.arcade=Arcade(operation_mode=OperationMode.OFFLINE,environments_dir=str(ROOT/'games'))
        self.card=self.arcade.open_scorecard()
        self.addCleanup(self.arcade.close_scorecard,self.card)

    def make(self,game,seed=42):
        return self.arcade.make(game+'-v1',seed=seed,scorecard_id=self.card)

    def command(self,env,command):
        action,data=sdk_action(env._game.puzzle,command)
        self.assertIn(action,env.observation_space.available_actions)
        return env.step(GameAction.from_id(action),data=data)

    def test_all_six_levels_native_actions_transition_alignment_and_win_reset(self):
        ids={info.game_id for info in self.arcade.get_environments()}
        self.assertEqual(len(ids),70)
        for spec in FOUNDATION_SPECS:
            self.assertIn(spec.game_id+'-v1',ids)
            for seed in (0,1,42,123):
                env=self.make(spec.game_id,seed)
                initial=env.observation_space.frame[-1].copy()
                self.assertEqual(env.observation_space.win_levels,6)
                for level in range(1,7):
                    p=env._game.puzzle
                    self.assertEqual(p.level,level)
                    self.assertFalse(p.won)
                    commands=plan(p)
                    self.assertLessEqual(len(commands),3 if level==1 else 10)
                    for index,command in enumerate(commands):
                        before=p.get_subgoals(); steps=p.steps
                        result=self.command(env,command)
                        self.assertEqual(p.steps,steps+1)
                        info=training_info(env); transition=info['last_transition']
                        self.assertEqual(transition['level'],level)
                        self.assertEqual(transition['before'],before)
                        self.assertEqual(transition['after'],p.get_subgoals())
                        self.assertEqual(info['level'],min(level+1,6) if index==len(commands)-1 else level)
                        self.assertNotEqual(result.state,GameState.GAME_OVER)
                        self.assertFalse({'subgoals','deck','target','solution','training_info'} & vars(result).keys())
                        np.testing.assert_array_equal(result.frame[-1],env._game.puzzle.render())
                    self.assertEqual(result.levels_completed,level)
                    self.assertTrue(all(g['achieved'] for g in transition['after']))
                    self.assertEqual(transition['outcome'],'level_complete')
                self.assertEqual(result.state,GameState.WIN)
                reset=env.reset()
                self.assertEqual(reset.levels_completed,0)
                self.assertIsNone(training_info(env)['last_transition'])
                np.testing.assert_array_equal(reset.frame[-1],initial)

    def test_every_game_cost_undo_budget_loss_retry_and_full_reset(self):
        for spec in FOUNDATION_SPECS:
            env=self.make(spec.game_id)
            initial=env.observation_space.frame[-1].copy()
            for command in plan(env._game.puzzle): self.command(env,command)
            self.assertEqual(env.observation_space.levels_completed,1)
            level_two=env.observation_space.frame[-1].copy()
            p=env._game.puzzle
            # First action costs one even if it misses; Undo restores the
            # previous command state and never refunds budget.
            env.step(GameAction.ACTION6,data={'x':63,'y':0})
            self.assertEqual(p.steps,1); self.assertEqual(p.state,p.initial_state)
            self.command(env,plan(p)[0])
            if p.won:  # one-tap consolidation: retry it before budget proof
                env.reset(); env.reset()
                for command in plan(env._game.puzzle): self.command(env,command)
                p=env._game.puzzle
            else:
                env.step(GameAction.ACTION7)
                self.assertEqual(p.steps,3)
                self.assertEqual(p.state,p.initial_state)
            for _ in range(p.budget-p.steps): result=env.step(GameAction.ACTION7)
            self.assertEqual(result.state,GameState.GAME_OVER)
            self.assertEqual(training_info(env)['last_transition']['outcome'],'lost')
            retry=env.reset()
            self.assertEqual(retry.levels_completed,1)
            self.assertEqual(env._game.puzzle.steps,0)
            np.testing.assert_array_equal(retry.frame[-1],level_two)
            full=env.reset()
            self.assertEqual(full.levels_completed,0)
            np.testing.assert_array_equal(full.frame[-1],initial)

    def test_clicking_known_downstream_switches_cannot_bypass_links(self):
        env=self.make('bf02')
        for _ in range(5):
            for command in plan(env._game.puzzle): self.command(env,command)
        p=env._game.puzzle
        for command in (1,3,4):
            self.command(env,command)
            self.assertEqual(p.state,(False,)*5)
            self.assertEqual(training_info(env)['last_transition']['outcome'],'blocked')
        self.command(env,0); self.command(env,1); self.command(env,2)
        self.command(env,4)
        self.assertEqual(p.state,(True,True,True,False,False))
        self.command(env,3); self.command(env,4)
        self.assertEqual(env.observation_space.state,GameState.WIN)
        self.assertEqual(p.steps,9)

    def test_memory_animation_and_simultaneous_tank_labels(self):
        env=self.make('bf08')
        for _ in range(2):
            for command in plan(env._game.puzzle): self.command(env,command)
        p=env._game.puzzle
        self.command(env,0)
        other=next(i for i,v in enumerate(p.deck) if v!=p.deck[0])
        result=self.command(env,other)
        self.assertEqual(p.steps,2)
        self.assertGreaterEqual(len(result.frame),9)
        np.testing.assert_array_equal(result.frame[0],p.transient_frames[0])
        self.assertFalse(np.array_equal(result.frame[0][4:61],result.frame[-1][4:61]))
        self.assertEqual(p.state,(0,-1,-1))
        env.step(GameAction.ACTION7)
        self.assertEqual(p.state,(0,0,-1))
        env=self.make('bf07'); self.command(env,plan(env._game.puzzle)[0])
        t=training_info(env)['last_transition']
        self.assertEqual([g['achieved'] for g in t['before']],[False,False])
        self.assertEqual([g['achieved'] for g in t['after']],[True,True])
        self.assertEqual(t['level'],1)
        self.assertEqual(training_info(env)['level'],2)


if __name__=='__main__': unittest.main()
