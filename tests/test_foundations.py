"""Issue-62 rules: literal semantic checks, bounded instances, legacy preservation."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from games.skill_core import FOUNDATION_SPECS, SPECS, make_puzzle
from games.skill_common import Canvas
from foundation_reference import plan, sdk_action

ROOT=Path(__file__).resolve().parents[1]


class LiteralFoundationRules(unittest.TestCase):
    def test_two_colors_independent_and_all_targets_required(self):
        p=make_puzzle('bf01',0,5)
        self.assertEqual(p.transition((0,1,0),1),(0,0,0))
        self.assertEqual(p.transition((0,0,0),1),(0,1,0))
        p.target=(1,0,1)
        self.assertFalse(p.solved((1,0,0)))
        self.assertTrue(p.solved((1,0,1)))

    def test_switch_prerequisites_conjunction_and_permanence(self):
        p=make_puzzle('bf02',0,6)
        self.assertEqual(p.parents,((),(0,),(),(2,),(1,3)))
        self.assertEqual(p.transition((False,)*5,4),(False,)*5)
        self.assertEqual(p.transition((True,True,False,False,False),4),(True,True,False,False,False))
        self.assertEqual(p.transition((True,True,True,False,False),3),(True,True,True,True,False))
        self.assertEqual(p.transition((True,True,True,True,False),4),(True,)*5)
        self.assertFalse(p.solved((True,True,True,True,False)))
        self.assertTrue(p.solved((True,)*5))
        for command in range(5):
            self.assertEqual(p.transition((True,)*5,command),(True,)*5)
        p.act(0); p.act(1)
        self.assertEqual(p.state,(True,True,False,False,False))
        p.undo()
        self.assertEqual(p.state,(True,False,False,False,False))
        self.assertEqual(p.steps,3)
        self.assertEqual(p.get_subgoals()[-1]['depends_on'],['switch-1','switch-3'])
        self.assertTrue(all(g['monotonic'] for g in p.get_subgoals()))
        # Any known downstream index is still guarded; no public node shortcut.
        for level in range(1,7):
            p=make_puzzle('bf02',0,level)
            for i,parents in enumerate(p.parents):
                if parents: self.assertEqual(p.transition(p.state,i),p.state)

    def test_movement_and_fixed_mirror_literal_directions(self):
        p=make_puzzle('bf03'); q=make_puzzle('bf05')
        for obj in (p,q):
            self.assertEqual(obj.transition((1,2),0),(1,1))
            self.assertEqual(obj.transition((1,2),1),(1,3))
            self.assertEqual(obj.transition((1,2),2),(0,2))
            self.assertEqual(obj.transition((1,2),3),(2,2))
            self.assertEqual(obj.transition((0,0),0),(0,0))
            self.assertEqual(obj.transition((0,0),2),(0,0))
        self.assertEqual(q.project((0,2)),(2,2))
        self.assertEqual(q.project((1,2)),(1,2))
        q.target=(2,2)
        self.assertTrue(q.solved((0,2)))
        self.assertFalse(q.solved((2,2)))

    def test_reflection_has_a_visible_gap_between_fill_and_outline(self):
        p=make_puzzle('bf05',42,1)
        p.state=(1,2); p.destination=(0,0)
        for size in (3,5):
            p.size=size
            x,y=p.point(p.state); x=63-x
            frame=p.render()
            self.assertEqual(int(frame[y,x]),p.palette[0])
            self.assertEqual(int(frame[y,x+size//2+1]),5)
            self.assertEqual(int(frame[y,x+size//2+2]),p.palette[0])

    def test_rotation_fixed_center_clockwise_no_selection(self):
        p=make_puzzle('bf04',0,5)
        self.assertEqual(p.transition((3,1),0),(0,1))
        self.assertEqual(p.transition((3,1),1),(3,2))
        p.shape=0
        self.assertEqual(p.cells(0),((0,0),(1,0),(0,1)))
        self.assertEqual(p.cells(1),((0,0),(0,1),(-1,0)))
        self.assertEqual(p.cells(2),((0,0),(-1,0),(0,-1)))
        self.assertEqual(p.cells(3),((0,0),(0,-1),(1,0)))
        p.target=(0,2)
        self.assertFalse(p.solved((0,1)))
        self.assertTrue(p.solved((0,2)))

    def test_single_loop_moves_every_token_and_preserves_order(self):
        p=make_puzzle('bf06',0,5)
        self.assertEqual(p.transition((0,1,2,3),0),(3,0,1,2))
        self.assertEqual(p.transition((0,1,2,3),1),(1,2,3,0))
        self.assertEqual(p.transition(p.transition((0,1,2,3),0),1),(0,1,2,3))
        p.target=(3,0,1,2)
        self.assertFalse(p.solved((3,0,2,1)))

    def test_unit_transfer_capacity_empty_guard_and_simultaneous_goals(self):
        p=make_puzzle('bf07',0,6); p.capacity=4
        self.assertEqual(p.transition((2,0,1),0),(1,1,1))
        self.assertEqual(p.transition((2,0,1),1),(2,0,1))
        self.assertEqual(p.transition((2,4,0),0),(2,4,0))
        self.assertEqual(p.transition((2,0,1),3),(2,1,0))
        self.assertEqual(p.transition((2,1,0),2),(2,0,1))
        p.state=(1,1,0); p.target=(1,0,1)
        p.act(2)
        self.assertTrue(p.won)
        self.assertEqual([g['achieved'] for g in p.get_subgoals()],[True,True,True])

    def test_pair_reveal_matching_mismatch_feedback_and_undo(self):
        p=make_puzzle('bf08',0,3); p.deck=(0,1,0,1)
        self.assertEqual(p.transition((0,-1,-1),0),(0,0,-1))
        self.assertEqual(p.transition((0,0,-1),0),(0,0,-1))
        self.assertEqual(p.transition((0,0,-1),1),(0,-1,-1))
        self.assertEqual(p.transition((0,0,-1),2),(5,-1,-1))
        self.assertEqual(p.transition((5,-1,-1),0),(5,-1,-1))
        p.act(0); p.act(1)
        self.assertEqual(p.state,(0,-1,-1))
        self.assertEqual(len(p.transient_frames),8)
        self.assertFalse(np.array_equal(p.transient_frames[0][4:61],p.render()[4:61]))
        p.undo(); self.assertEqual(p.state,(0,0,-1))
        p.act(2); self.assertFalse(p.won)
        self.assertEqual([g['achieved'] for g in p.get_subgoals()],[True,False])
        p.act(1); p.act(3); self.assertTrue(p.won)


class UnclippedCanvas(Canvas):
    def rect(self,x,y,w,h,color,hollow=False):
        assert 0<=x and x+w<=64 and 4<=y and y+h<=62,(x,y,w,h)
        super().rect(x,y,w,h,color,hollow)


class FoundationInstances(unittest.TestCase):
    def test_all_bounded_instances_and_appearance_extremes(self):
        lengths=defaultdict(set); structures={}
        for spec in FOUNDATION_SPECS:
            for level in range(1,7):
                key=f'{spec.game_id}/L{level}'
                for seed in range(64):
                    original=make_puzzle(spec.game_id,seed,level)
                    commands=plan(original)
                    self.assertGreater(len(commands),0)
                    self.assertLessEqual(len(commands),3 if level==1 else 10)
                    lengths[key].add(len(commands))
                    structures[key]={name:getattr(original,name) for name in
                        ('n','distance','capacity','transfers','parents','shape') if hasattr(original,name)}
                    for appearance,size,offset in ((None,original.size,original.offset),(0,3,(0,0)),(63,5,(2,2))):
                        p=make_puzzle(spec.game_id,seed,level,appearance_seed=appearance)
                        p.size,p.offset=size,offset
                        self.assertEqual(p.state,original.state)
                        self.assertEqual(p.target,original.target)
                        self.assertEqual(p.get_subgoals(),original.get_subgoals())
                        for command in range(len(p.commands)):
                            self.assertEqual(p.transition(p.state,command),original.transition(original.state,command))
                        self.assertFalse(p.solved(p.state))
                        # Text glyph helpers are forbidden on these boards.
                        with patch.object(Canvas,'label',side_effect=AssertionError('letter')), \
                             patch.object(Canvas,'digit',side_effect=AssertionError('digit')):
                            image=p.render()
                        p.draw(UnclippedCanvas(p.palette,p.size))
                        self.assertEqual(image.shape,(64,64))
                        self.assertTrue(np.all((image>=0)&(image<16)))
                        rects=p.button_rects(); occupied=set()
                        for x,y,w,h in rects:
                            cells={(u,v) for u in range(x,x+w) for v in range(y,y+h)}
                            self.assertTrue(all(0<=u<64 and 4<=v<62 for u,v in cells),(key,rects))
                            self.assertFalse(occupied & cells,(key,rects))
                            occupied.update(cells)
                        if spec.game_id=='bf05':
                            for x in range(p.width):
                                for y in range(p.height):
                                    u,v=p.point((x,y))
                                    self.assertLess(u+5,31); self.assertGreater(63-u-5,32)
                        for command in commands:
                            before=p.steps
                            # Native actions move directly. Object clicks use
                            # inclusive corners to catch addressing clipping.
                            action,data=sdk_action(p,command,corner=int(size!=3))
                            p.input_action(action,**(data or {}))
                            self.assertEqual(p.steps,before+1)
                            self.assertFalse(p.lost)
                        self.assertTrue(p.won,(key,seed,appearance,commands))
                        self.assertTrue(all(g['achieved'] for g in p.get_subgoals()))
        report={key:{'actions':sorted(lengths[key]),'structure':structures[key]} for key in lengths}
        output=ROOT/'artifacts/synthetic-validation'; output.mkdir(parents=True,exist_ok=True)
        (output/'foundation-ranges.json').write_text(json.dumps(report,indent=2)+'\n')

    def test_exact_progression_and_consolidation(self):
        expected={
            'bf01':('n',[1,1,2,2,3,3]), 'bf02':('n',[1,2,2,3,4,5]),
            'bf03':('distance',[1,2,2,3,4,6]), 'bf04':('n',[1,1,1,1,2,2]),
            'bf05':('distance',[1,2,3,4,5,6]), 'bf06':('n',[2,2,3,3,4,4]),
            'bf07':('transfers',[1,2,3,4,4,6]), 'bf08':('n',[1,1,2,2,3,3])}
        for game,(field,values) in expected.items():
            self.assertEqual([getattr(make_puzzle(game,42,l),field) for l in range(1,7)],values)
        for seed in range(64):
            p=make_puzzle('bf03',seed,2)
            self.assertTrue(p.state[0]==p.destination[0] or p.state[1]==p.destination[1])
            p=make_puzzle('bf03',seed,3)
            self.assertTrue(p.state[0]!=p.destination[0] and p.state[1]!=p.destination[1])

    def test_consolidation_changes_layout_without_adding_rules(self):
        for game in ('bf01','bf08'):
            for seed in range(64):
                for odd in (1,3,5):
                    first=make_puzzle(game,seed,odd)
                    second=make_puzzle(game,seed,odd+1)
                    self.assertEqual(first.n,second.n)
                    self.assertNotEqual(first.button_rects(),second.button_rects(),(game,seed,odd))

    def test_detached_labels_and_reproducible_reset_instance(self):
        for spec in FOUNDATION_SPECS:
            p=make_puzzle(spec.game_id,42,6)
            initial=p.render().copy(); info=p.get_training_info(); saved=deepcopy(info)
            info['subgoals'][0]['current']='changed'; info['subgoals'][0]['depends_on'].append('bad')
            self.assertEqual(p.get_training_info(),saved)
            command=next(i for i in range(len(p.commands)) if not p.solved(p.transition(p.state,i)))
            p.act(command); p.undo()
            self.assertEqual(p.state,p.initial_state)
            self.assertEqual(p.steps,2)
            q=make_puzzle(spec.game_id,42,6)
            np.testing.assert_array_equal(q.render(),initial)
            for key in ('deck','solution','commands','parents'): self.assertNotIn(key,saved)
            self.assertTrue(all(set(g)=={'id','current','target','achieved','monotonic','depends_on'} for g in saved['subgoals']))

    def test_catalog_metadata_and_legacy_fingerprints(self):
        self.assertEqual(len(SPECS),68)
        self.assertEqual(len({s.game_id for s in SPECS}),68)
        for spec in FOUNDATION_SPECS:
            path=ROOT/'games/foundations'/spec.skill/'v1'
            meta=json.loads((path/'metadata.json').read_text())
            self.assertEqual(meta['game_id'],spec.game_id+'-v1')
            self.assertIn('foundation',meta['tags'])
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'current.json'
            subprocess.run([sys.executable,str(ROOT/'tests/legacy_fingerprint.py'),str(ROOT),str(output)],check=True)
            expected=json.loads((ROOT/'tests/fixtures/pre-direct-controls-fingerprints.json').read_text())['games']
            # Captured from accepted main, not this candidate. The older
            # issue-62 fixture predates palette randomization and click feedback;
            # retain that historical file without treating it as today's oracle.
            # Changed action contracts have separate rule/witness tests.
            unchanged=('sc01-v1','si01-v1','sy10-v1','sy47-v1','sy48-v1','sy55-v1','sy56-v1','sy57-v1')
            current=json.loads(output.read_text())
            self.assertEqual({g:current[g] for g in unchanged},{g:expected[g] for g in unchanged})


if __name__=='__main__': unittest.main()
