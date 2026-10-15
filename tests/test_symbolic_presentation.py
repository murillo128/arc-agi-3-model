"""Literal actor-frame contracts for symbols-v2; no OCR or renderer-derived oracle."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from games.skill_core import SPECS, Canvas, make_puzzle


class UnclippedCanvas(Canvas):
    def rect(self,x,y,w,h,color,hollow=False):
        assert 0<=x and x+w<=64 and 4<=y and y+h<=62,(x,y,w,h)
        super().rect(x,y,w,h,color,hollow)


class PresentationTests(unittest.TestCase):
    def puzzle(self, game, level=3):
        p=make_puzzle(game,42,level)
        p.palette=tuple(range(6,16))
        return p

    def test_all_authored_paths_avoid_text_and_preserve_read_purity(self):
        for spec in SPECS:
            for level in (1,3,6):
                for size in (3,5):
                    p=self.puzzle(spec.game_id,level); p.size=size
                    before=(p.state,p.target,deepcopy(p.get_training_info()),p.rng.getstate())
                    with patch.object(Canvas,'label',side_effect=AssertionError('alphabetic raster')), \
                         patch.object(Canvas,'digit',side_effect=AssertionError('decimal raster')):
                        p.render()
                        c=UnclippedCanvas(p.palette,p.size)
                        p.draw(c)
                        if not getattr(p,'board_controls',False): p.draw_controls(c)
                        for command in range(len(p.commands)):
                            after=p.transition(p.state,command)
                            p.transition_frames(p.state,command,after)
                    self.assertEqual((p.state,p.target,p.get_training_info(),p.rng.getstate()),before)

    def test_prerequisite_is_on_predecessor_link_separate_from_goal(self):
        for palette in (tuple(range(6,16)),tuple(range(15,5,-1))):
            p=self.puzzle('sy08'); p.palette=palette
            p.state=(2,0,0); p.target=(1,2,1)
            f=p.render()
            self.assertEqual(int(f[11,26]),palette[2])  # middle goal left outline
            self.assertEqual(int(f[39,20]),palette[3])  # predecessor enabling sample
            self.assertEqual(int(f[35,30]),2)          # closed incoming crossbar
            self.assertEqual(p.button_rects(),[(6,25,9,9),(26,25,9,9),(46,25,9,9)])
            p.click(30,29); self.assertEqual(p.state,(2,0,0))
            p.click(10,29); self.assertEqual(p.state,(3,0,0))
            self.assertEqual(int(p.render()[35,30]),0)
            p.click(30,29); self.assertEqual(p.state,(3,1,0))
            before=p.render()[35:42].copy()
            p.target=(3,3,3)
            np.testing.assert_array_equal(before,p.render()[35:42])
            p.click(10,29); self.assertEqual(p.state,(0,1,0))
            p.click(30,29); self.assertEqual(p.state,(0,1,0))

    def test_cycles_and_same_different_samples(self):
        p=self.puzzle('sy06',6)
        f=p.render()
        self.assertEqual([int(f[49,x]) for x in (5,12,19,26,33,40,47)],list(range(6,13)))
        self.assertTrue(np.all(f[55,5:48]==2))  # explicit cycle return
        p=self.puzzle('sy01'); p.equal=(True,False)
        f=p.render()
        self.assertEqual([int(f[20,x]) for x in (11,15,21,25)],[6,6,6,7])
        p=self.puzzle('sy05')
        f=p.render(); self.assertEqual(int(f[36,18]),0)
        p.click(8,31); self.assertEqual(p.state,(1,1,0))

    def test_small_arrows_and_same_coloured_objects_have_spatial_addressing(self):
        p=self.puzzle('sy06',6)
        np.testing.assert_array_equal(p.render()[48:51,7:10],[[5,2,5],[5,5,2],[5,2,5]])
        p=self.puzzle('sy07'); p.state=(0,0,0)
        f=p.render()
        self.assertTrue(np.all(f[39,10:59]==2))  # first object to top exchange sample
        self.assertTrue(np.all(f[35:56,50]==2))  # last object to lower sample
        self.assertEqual(int(f[36,30]),5)       # middle has no exchange connection
        p=self.puzzle('sy22'); p.state=(2,2,2)
        f=p.render()
        self.assertEqual(p.button_rects(),[(47,7,7,7),(55,7,7,7),(47,17,7,7),(55,17,7,7),(47,27,7,7),(55,27,7,7)])
        for y,color in ((10,7),(20,8),(30,9)):
            self.assertTrue(np.all(f[y,18:47]==color))
        p.click(50,10); self.assertEqual(p.state,(1,2,2))

    def test_linkage_prerequisite_and_reversible_goal(self):
        p=self.puzzle('sy26',2); p.target=(1,3)
        f=p.render()
        self.assertEqual(int(f[51,24]),2)
        self.assertEqual(int(f[46,16]),6)  # minimum parent sample on child link
        self.assertTrue(p.get_subgoals()[0]['achieved'])
        p.act(0)
        self.assertFalse(p.get_subgoals()[0]['achieved'])
        self.assertEqual(int(p.render()[50,24]),0)
        p.undo()
        np.testing.assert_array_equal(p.render()[4:61],f[4:61])

    def test_reflection_cue_matches_visible_axis_at_every_orientation(self):
        # Literal cells for the level-three L at anchor (2,2), before/after flip.
        cells = (
            (((2,2),(2,3),(3,2)), ((1,2),(2,2),(2,3))),
            (((1,2),(2,2),(2,3)), ((1,2),(2,1),(2,2))),
            (((1,2),(2,1),(2,2)), ((2,1),(2,2),(3,2))),
            (((2,1),(2,2),(3,2)), ((2,2),(2,3),(3,2))),
        )
        for palette in (tuple(range(6,16)),tuple(range(15,5,-1))):
            color=palette[1]
            vertical=[
                [2,2,2,2,2,2,2],
                [2,color,5,2,5,color,2],
                [2,color,5,2,5,color,2],
                [color,color,5,2,5,color,color],
                [2,5,5,2,5,5,2],
                [2,5,5,2,5,5,2],
                [2,2,2,2,2,2,2],
            ]
            horizontal=[
                [2,2,2,color,2,2,2],
                [2,color,color,color,5,5,2],
                [2,5,5,5,5,5,2],
                [2,2,2,2,2,2,2],
                [2,5,5,5,5,5,2],
                [2,color,color,color,5,5,2],
                [2,2,2,color,2,2,2],
            ]
            for size in (3,5):
                for rotation in range(4):
                    for flip in (0,1):
                        with self.subTest(palette=palette,size=size,rotation=rotation,flip=flip):
                            p=self.puzzle('sy11'); p.palette=palette; p.size=size
                            p.state=(2,2,rotation,flip,2); p.target=((4,4),)
                            cue=horizontal if rotation in (1,3) else vertical
                            self.assertEqual(p.cells(p.state),cells[rotation][flip])
                            np.testing.assert_array_equal(p.render()[53:60,41:48],cue)
                            p.click(44,56)
                            self.assertEqual(p.cells(p.state),cells[rotation][1-flip])
                            np.testing.assert_array_equal(p.render()[53:60,41:48],cue)

    def test_exact_quantity_trays_zero_missing_excess_and_maximum(self):
        p=self.puzzle('sy46',6)
        p.target=(0,1,2,0,0)
        p.state=((0,)*12,-1)
        f=p.render()
        # Independent cell positions: twelve live units, zero required units.
        for y in (14,17,20,23):
            for x in (3,6,9): np.testing.assert_array_equal(f[y:y+2,x:x+2],np.full((2,2),2))
        for y in (29,32,35,38):
            for x in (3,6,9): np.testing.assert_array_equal(f[y:y+2,x:x+2],np.full((2,2),6))
        np.testing.assert_array_equal(f[14:16,15:17],[[5,7],[7,7]])
        self.assertEqual(int(f[14,18]),2)  # second required tier-one slot empty
        self.assertEqual(int(f[29,15]),2)  # missing the one required item
        self.assertFalse(p.solved(p.state))
        p.state=((-1,1,1,2,2,-1,-1,-1,-1,-1,-1,-1),-1)
        f=p.render()
        self.assertEqual([int(f[30,x]) for x in (15,18,21)],[7,7,2])
        self.assertFalse(p.solved(p.state))  # excess tier-one item
        p.state=((-1,1,-1,2,2,-1,-1,-1,-1,-1,-1,-1),-1)
        self.assertTrue(p.solved(p.state))
        p.state=(p.state[0],3)
        self.assertEqual(int(p.render()[42,32]),0)  # selected block border

    def test_sigil_target_and_current_are_discrete_separate_quantities(self):
        p=self.puzzle('sy49'); p.target=(2,1,2); p.state=((0,1,2),0)
        f=p.render()
        np.testing.assert_array_equal(f[9:11,14:16],[[5,6],[6,6]])
        np.testing.assert_array_equal(f[17:19,14:16],np.full((2,2),2))
        self.assertEqual([int(f[18,x]) for x in (32,35,50,53)],[7,2,8,8])

    def test_hazard_track_matches_clock_after_not_before_action(self):
        p=self.puzzle('sy60',3); p.period=3; p.phases={1:1,3:2}; p.state=(0,0)
        f=p.render()
        self.assertEqual(int(f[10,10]),7)  # first hazard marked at middle phase
        self.assertEqual(int(f[13,18]),7)  # second at last phase
        self.assertEqual(int(f[7,57]),0)   # clock currently at top
        self.assertNotEqual(int(f[30,11]),7)
        self.assertIsNone(p.transition((0,0),0))
        p.state=(0,1); f=p.render()
        self.assertEqual(int(f[10,57]),0); self.assertEqual(int(f[30,11]),7)
        self.assertEqual(p.transition((0,1),0),(1,2))
        self.assertEqual(p.transition((0,2),1),(0,0))

    def test_program_switch_dictionary_noop_emit_call_and_full_output(self):
        p=self.puzzle('sy53',6); p.state=((0,)*6,(),False)
        f=p.render()
        # Switch configurations in the four examples, never decimal opcodes.
        self.assertEqual([int(f[y,x]) for y,x in ((43,2),(43,5),(43,34),(43,37),
                                                 (54,2),(54,5),(54,34),(54,37))],
                         [2,2,0,2,2,0,0,0])
        self.assertFalse(np.isin(f[40:49,11:29],[6,7]).any())  # no-op empty tray
        self.assertEqual(int(f[44,45]),6); self.assertEqual(int(f[55,13]),7)
        self.assertEqual([int(f[55,x]) for x in (45,48,51,54,57,60)],[6,6,6,7,7,7])
        self.assertEqual(int(f[51,43]),2)  # outer call/return bracket
        p.state=((3,)*6,(0,)*18+(1,)*18,True)
        f=p.render()
        self.assertEqual([int(f[19,x]) for x in range(4,56,3)],[6]*18)
        self.assertEqual([int(f[23,x]) for x in range(4,56,3)],[7]*18)
        self.assertEqual(p.button_rects()[0],(2,29,4,7))
        p=self.puzzle('sy54',6); f=p.render()
        self.assertEqual([int(f[44,x]) for x in (45,48)],[6,7])
        self.assertEqual([int(f[55,x]) for x in (13,16,19,22)],[6,6,7,7])

    def test_region_and_valve_controls_identify_their_actual_affected_set(self):
        p=self.puzzle('sy16',6); f=p.render()
        self.assertEqual(p.button_rects()[0],(2,44,10,7))
        self.assertEqual([int(f[45,x]) for x in (3,4,5,6)],[7]*4)
        self.assertEqual([int(f[46,x]) for x in (3,4,5,6)],[2]*4)
        p.click(7,47); self.assertEqual(p.state,(1,1,1,1)+(0,)*12)
        p=self.puzzle('sy43',3); p.state=((0,0,0),False)
        self.assertEqual(p.button_rects(),[(9,10,5,5),(16,15,5,5),(23,20,5,5),(51,45,10,14)])
        p.click(18,17); self.assertEqual(p.state,((0,1,0),False))


if __name__=='__main__': unittest.main()
