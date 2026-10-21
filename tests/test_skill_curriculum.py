"""Contracts, seed properties and solution witnesses for original micro-worlds.

Pure rules run without the SDK. Real offline-SDK checks live in a separate file;
passing these tests is not proof of the SDK or browser boundary.
"""
import ast
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import unittest

import numpy as np

from games.skill_core import (BY_ID, CATALOG, CONTEXTS, Canvas,
                             execute_program, make_puzzle)
from games.skill_core import SKILL_SPECS as SPECS
from skill_reference import expand, plan, sdk_action

ROOT = Path(__file__).resolve().parents[1]


class LiteralRulesTests(unittest.TestCase):
    def game(self, name, level=3):
        return make_puzzle(name, 42, level)

    def test_cycles_and_coupled_actions(self):
        p = self.game('efectos_acoplados')
        self.assertEqual(p.transition((0,0,0),0), (1,1,0))
        self.assertEqual(p.transition((2,2,0),0), (0,0,0))
        self.assertEqual(p.transition((0,0,0),2), (0,0,1))
        q = self.game('igualdad_desigualdad')
        q.target, q.equal = (1,1,2), (True,False)
        self.assertTrue(q.solved((1,1,0)))  # Inequality is not exact-color equality.
        self.assertFalse(q.solved((1,1,1)))
        self.assertFalse(q.solved((2,2,0)))  # Anchor is independently required.

    def test_key_preconditions_and_noncommuting_transforms(self):
        p = self.game('precondiciones')
        self.assertEqual(p.transition((0,0,0),1), (0,0,0))
        self.assertEqual(p.transition((3,0,0),1), (3,1,0))
        q = self.game('composicion_de_transformaciones')
        a = q.transition(q.transition((0,0,0),0),3)
        b = q.transition(q.transition((0,0,0),3),0)
        self.assertEqual(a, (0,0,1)); self.assertEqual(b, (1,0,0))

    def test_reflection_geometry_and_subset_coverage(self):
        p = self.game('simetria',1)
        self.assertEqual(p.cells((1,1,0,0,2)), ((1,1),(3,1)))
        self.assertEqual(p.cells((1,1,0,0,1)), ((1,1),))
        p.target = ((3,1),)
        self.assertTrue(p.solved((1,1,0,0,2)))
        self.assertFalse(p.solved((1,1,0,0,1)))
        self.assertEqual(p.transition((0,0,0,0,0),2), (0,0,0,0,0))

    def test_paint_overwrites_only_its_region(self):
        p = self.game('orden_no_conmutativo',2)
        start = (0,0,0,0)
        # Top row is colour 1, right column colour 2: order still matters.
        a = p.transition(p.transition(start,0),3)
        b = p.transition(p.transition(start,3),0)
        self.assertEqual(a, (1,2,0,2))
        self.assertEqual(b, (1,1,0,2))
        self.assertEqual(p.transition(start,0), (1,1,0,0))

    def test_wall_breaks_mirrored_invariants(self):
        p = self.game('ruptura_de_invariantes')
        self.assertEqual(p.transition((1,1,3,1),3), (2,1,2,1))
        self.assertEqual(p.transition((0,0,4,1),0), (0,0,4,0))
        p.walls = frozenset({(1,0)})
        self.assertEqual(p.transition((1,1,3,1),0), (1,1,3,0))

    def test_centroid_and_parent_child_positions(self):
        p = self.game('centroide',2)
        p.target = (6,)
        self.assertTrue(p.solved((2,4)))
        self.assertFalse(p.solved((2,3)))
        self.assertEqual(p.transition((2,4),1), (3,4))
        q = self.game('dependencias_jerarquicas',2)
        self.assertEqual(q.points((0,0)), [(9,9),(9,12)])
        self.assertEqual(q.points((1,0)), [(12,9),(12,12)])

    def test_nonmonotonic_staging_and_dependency(self):
        p = self.game('subobjetivos_no_monotonos',2)
        p.target = (1,3)
        self.assertEqual(p.transition((1,1),3), (1,1))
        self.assertTrue(p.get_subgoals()[0]['achieved'])
        p.act(0)
        self.assertFalse(p.get_subgoals()[0]['achieved'])
        p.act(3); p.act(3); p.act(1)
        self.assertTrue(p.won)

    def test_maze_key_door_and_safe_bridge(self):
        p = self.game('planificacion_en_estado_ampliado')
        p.n = 3; p.edges = {frozenset((0,1)),frozenset((1,2))}
        p.keys = {0:0}; p.doors = {frozenset((1,2)):0}; p.bridge = -1
        self.assertEqual(p.transition((1,0,0),3), (1,0,0))
        self.assertEqual(p.transition((1,1,0),3), (2,1,0))
        p.bridge = 1
        self.assertEqual(p.transition((0,1,0),3), (0,1,0))
        self.assertEqual(p.transition((0,1,1),3), (1,1,1))
        self.assertIsNone(p.transition((1,1,1),4))

    def test_loop_permutations_and_inverse(self):
        p = self.game('permutaciones',3)
        state = (0,1,2,3)
        self.assertEqual(p.transition(state,0), (3,0,1,2))
        self.assertEqual(p.transition(p.transition(state,0),1), state)
        self.assertEqual(Counter(p.transition(state,2)),Counter(state))
        self.assertNotEqual(p.transition(p.transition(state,0),2),
                            p.transition(p.transition(state,2),0))

    def test_peg_capture_and_permanent_support(self):
        p = self.game('reescritura_condicional',1)
        start = ((1,1,0,0,0,0,0,0,0),-1)
        result = p.transition(p.transition(start,0),2)
        self.assertEqual(result, ((0,0,1,0,0,0,0,0,0),-1))
        self.assertTrue(p.solved(result))
        start = ((1,2,0,0,0,0,0,0,0),-1)
        self.assertEqual(p.transition(p.transition(start,0),2),
                         ((0,2,1,0,0,0,0,0,0),-1))
        self.assertEqual(p.transition(p.transition(start,0),5), p.transition(start,0))

    def test_contact_impulse_and_irrelevant_objects(self):
        p = self.game('control_indirecto',1)
        self.assertEqual(p.transition((4,(5,6)),3), (5,(6,7)))
        self.assertEqual(p.transition((5,(6,7)),3), (5,(6,7)))
        q = self.game('dinamica_por_contacto',1)
        self.assertEqual(q.transition((4,(5,)),3), (4,(7,)))
        r = self.game('relevancia_del_objetivo',3)
        r.target = (10,)
        self.assertTrue(r.solved((0,(10,22))))
        self.assertFalse(r.solved((0,(11,10))))

    def test_other_agent_moves_while_waiting(self):
        p = self.game('inferencia_de_roles',2)
        p.role = -1
        self.assertEqual(p.transition((2,0),2), (2,4))
        self.assertEqual(p.transition((0,1),0), (0,0))

    def test_conservation_and_capacity(self):
        p = self.game('conservacion_y_transferencia',1)
        p.caps = (2,2)
        self.assertEqual(p.transition((2,0),0), (1,1))
        self.assertEqual(p.transition((0,2),0), (0,2))
        self.assertEqual(p.transition((1,2),0), (1,2))
        self.assertEqual(p.transition((1,1),1), (2,0))

    def test_signal_swap_versus_branch_and_execution_gate(self):
        p = self.game('propagacion',2)
        q = self.game('bifurcacion',2)
        self.assertEqual(p.output((1,1)), (2,4,1))
        self.assertEqual(q.output((1,1)), (1,3,7))
        p.state=((0,0),False); p.target=(2,4,1)
        p.act(0); p.act(1)
        self.assertFalse(p.won)
        self.assertTrue(all(g['achieved'] for g in p.get_subgoals()[:-1]))
        p.act(2)
        self.assertTrue(p.won); self.assertEqual(len(p.transient_frames),2)

    def test_gravity_resolves_the_complete_slide(self):
        p = self.game('dinamica_autonoma',2)
        p.n = 4; p.walls = frozenset({(1,2)})
        self.assertEqual(p.transition((0,0,1),1), (1,1,1))
        self.assertEqual(p.transition((1,1,1),0), (0,3,1))
        self.assertEqual(p.transition((0,3,1),2), (0,0,-1))

    def test_exact_merge_counts_and_type_guard(self):
        p = self.game('cardinalidad_exacta',2)
        start = ((0,0,1,-1),-1)
        self.assertEqual(p.transition(p.transition(start,0),1), ((-1,1,1,-1),-1))
        self.assertEqual(p.transition(p.transition(start,0),2), ((0,0,1,-1),2))
        p.target = (0,1,0,0,0)
        self.assertTrue(p.solved(((-1,1,-1,-1),-1)))
        self.assertFalse(p.solved(((-1,1,1,-1),-1)))

    def test_dictionary_composition(self):
        p = self.game('composicion_de_funciones',5)
        translated = p.phrase
        for table in p.maps:
            translated = tuple(table[symbol] for symbol in translated)
        self.assertEqual(translated,p.target)
        self.assertNotEqual(p.state,p.target)
        self.assertEqual(p.transition((0,1,2,3,0),2), (0,1,3,3,0))

    def test_real_call_return_and_recursive_arguments(self):
        output, peak = execute_program((1,3,2),call_depth=2)
        self.assertEqual(output,(0,0,0,1,1,1))
        self.assertGreaterEqual(peak,3)
        self.assertEqual(execute_program((2,1),recursive_arguments=True)[0], (0,0,1,1,0,1))
        self.assertEqual(execute_program((0,))[0], ())
        for depth in (1,2,3):
            self.assertEqual(execute_program((3,2,1,3),call_depth=depth)[0],
                             expand((3,2,1,3),depth))
        with self.assertRaisesRegex(ValueError,'bounded interpreter'):
            execute_program((3,)*200,call_depth=3)

    def test_sigil_requires_exact_pattern_then_clears(self):
        p = self.game('simbolos_y_significado',2)
        p.patterns = (3,12)
        self.assertEqual(p.transition(((0,0),0),0), ((0,0),1))
        self.assertEqual(p.transition(((0,0),1),1), ((1,0),0))
        self.assertEqual(p.transition(((0,0),17),1), ((0,0),19))

    def test_hidden_pair_has_transient_evidence(self):
        p = self.game('memoria_parcial',2)
        p.deck = (0,1,0,1)
        hidden = p.render().copy()
        p.act(0); first = p.render().copy(); p.act(1)
        self.assertEqual(p.state,(0,-1,-1))
        self.assertEqual(len(p.transient_frames),1)
        np.testing.assert_array_equal(p.render()[4:61],hidden[4:61])
        self.assertFalse(np.array_equal(first[4:61],p.transient_frames[0][4:61]))
        p.act(0); p.act(2)
        self.assertEqual(p.state,(5,-1,-1))
        self.assertTrue(p.get_subgoals()[0]['achieved'])
        p.undo()
        self.assertFalse(p.get_subgoals()[0]['achieved'])

    def test_history_changes_the_future_and_holds_plate(self):
        p = self.game('historial_como_estado',1)
        p.plates=((-1,0),)
        p.act(2); p.act(4)
        self.assertEqual(p.state,((0,0),(),((2,),),0))
        p.act(1)
        self.assertTrue(p.won)
        self.assertEqual(p.ghost_positions(((2,),),20),[(-1,0)])
        q = self.game('historial_como_estado',1)
        q.act(1)
        self.assertEqual(q.state[0],(0,0)); self.assertFalse(q.won)

    def test_new_phase_not_old_phase_determines_safety(self):
        p = self.game('orden_de_eventos',2)
        p.period=3; p.phases={1:1}
        self.assertIsNone(p.transition((0,0),0))
        self.assertEqual(p.transition((0,1),0),(1,2))
        self.assertEqual(p.transition((1,2),0),(2,0))
        self.assertIsNone(p.transition((1,0),1))


class LifecycleTests(unittest.TestCase):
    def test_win_has_priority_over_budget_and_terminal_needs_reset(self):
        p = make_puzzle('estados_ciclicos',0,1)
        p.budget = 1
        p.act(0)
        self.assertTrue(p.won); self.assertFalse(p.lost)
        state,steps = p.state,p.steps
        p.undo(); p.miss(); p.act(0)
        self.assertEqual((p.state,p.steps),(state,steps))
        q = make_puzzle('estados_ciclicos',0,1)
        q.budget = 1; q.miss()
        self.assertTrue(q.lost); self.assertFalse(q.won)

    def test_undo_restores_state_not_budget_and_render_is_pure(self):
        p = make_puzzle('estados_ciclicos',0,3)
        original = p.state
        p.act(0); p.undo()
        self.assertEqual(p.state,original); self.assertEqual(p.steps,2)
        state,steps,history = p.state,p.steps,list(p.history)
        for _ in range(3): p.render(); p.get_training_info()
        self.assertEqual((p.state,p.steps,p.history),(state,steps,history))
        p.budget = 3; p.undo()
        self.assertTrue(p.lost)

    def test_training_info_is_a_detached_snapshot(self):
        p = make_puzzle('historial_como_estado',3,4)
        info = p.get_training_info()
        self.assertTrue(info['training_only']); self.assertEqual(info['source'],'synthetic')
        self.assertEqual(info['game_id'],'sy58-v1')
        self.assertEqual(info['subgoals'][-1]['depends_on'],['plate-0','plate-1'])
        info['subgoals'][-1]['depends_on'].clear()
        info['subgoals'][0]['achieved'] = True
        self.assertEqual(p.get_subgoals()[-1]['depends_on'],['plate-0','plate-1'])
        self.assertFalse(p.get_subgoals()[0]['achieved'])
        self.assertNotIn('solution',json.dumps(info))

    def test_strict_inputs_and_offscreen_clicks(self):
        for args in ((123,),('missing',),('sy01',False),('sy01',0,0),('sy01',0,7)):
            with self.subTest(args=args), self.assertRaises(ValueError): make_puzzle(*args)
        with self.assertRaises(ValueError): make_puzzle('sy01',appearance_seed=True)
        p = make_puzzle('sy01')
        for x,y in ((-1,10),(64,10),(10,64),(True,10),(1.5,10)):
            p.click(x,y)
        self.assertEqual(p.steps,5); self.assertFalse(p.won)
        with self.assertRaises(ValueError): p.act(-1)
        x,y,w,h = p.button_rects()[0]
        p.click(x+w-1,y+h-1)
        self.assertTrue(p.won)

    def test_rectangle_clipping_never_wraps(self):
        c = Canvas(tuple(range(6,16)),3)
        c.rect(-10,0,3,3,0); c.rect(0,-10,3,3,0)
        self.assertTrue(np.all(c.pixels==5))
        c.rect(-1,-1,3,3,0)
        self.assertEqual(np.count_nonzero(c.pixels==0),4)


class GenerationTests(unittest.TestCase):
    def test_native_actions_replace_detached_panels(self):
        # The public contract is up to five general actions, plus clicks on
        # actual objects and Undo. Keep every semantic command reachable.
        from games.skill_core import SPECS as all_specs
        panel_contexts = {'transform','twins','centroid','linkage','maze','loops',
                          'boxes','agents','tanks','gravity','ghost','timing','paint',
                          'basic_move','basic_reflect','basic_loop','basic_transfer'}
        for spec in all_specs:
            for level in range(1,7):
                p=make_puzzle(spec.game_id,42,level)
                with self.subTest(game=spec.game_id,level=level):
                    native=p.action_commands()
                    clicks=p.click_commands()
                    self.assertLessEqual(set(native),{1,2,3,4,5})
                    self.assertEqual(set(native.values())|{i for i,_ in clicks},set(range(len(p.commands))))
                    self.assertEqual(6 in p.available_actions(),bool(clicks))
                    if spec.context in panel_contexts:
                        self.assertTrue(np.all(p.render()[53:61] == 5))
                        state=p.state
                        p.click(4,56)  # Former first button.
                        self.assertEqual(p.state,state)
                        self.assertEqual((p.event,p.steps),('miss',1))

    def test_direct_action_meanings_and_simplified_rule_limits(self):
        # Hand-calculated state changes; do not derive expectations from the
        # action map or the transition implementation used by witness helpers.
        cases = (
            ('sy22',6,(3,3),1,(3,2)), ('sy22',6,(3,3),4,(4,3)),
            ('sy25',6,(2,2),2,(2,3)), ('sy25',6,(2,2),3,(1,2)),
            ('sy40',6,(1,1,1),1,(1,2,0)), ('sy40',6,(1,1,1),4,(0,2,1)),
            ('sy16',2,(0,0,0,0),4,(0,2,0,2)),
            ('sy31',3,(0,1,2,3),4,(3,0,1,2)),
            ('sy31',3,(0,1,2,3),2,(2,0,1,3)),
            ('bf06',5,(0,1,2,3),3,(1,2,3,0)),
            ('bf07',6,(1,1,1),2,(1,0,2)),
        )
        for game,level,start,action,expected in cases:
            with self.subTest(game=game,action=action):
                p=make_puzzle(game,42,level)
                p.state=start
                p.input_action(action)
                self.assertEqual(p.state,expected)
                self.assertEqual(p.steps,1)
        for level in range(1,7):
            self.assertLessEqual(make_puzzle('sy22',42,level).n,2)
            self.assertLessEqual(make_puzzle('sy25',42,level).n,2)
            self.assertLessEqual(make_puzzle('sy40',42,level).n,3)
            self.assertLessEqual(len(make_puzzle('sy16',42,level).commands),5)
        self.assertEqual(make_puzzle('sy22',42,1).available_actions(),[3,4,7])
        self.assertEqual(make_puzzle('sy16',42,1).available_actions(),[1,2,7])

    def test_mirror_cycles_legal_positions_and_ghost_records_on_figure_click(self):
        p=make_puzzle('sy13',42,1)
        p.state=(1,1,0,0,1)
        p.input_action(5)
        self.assertEqual(p.state,(1,1,0,0,2))
        # Skip off-board reflected copies and wrap to the first legal axis.
        p=make_puzzle('sy13',42,1)
        p.state=(1,1,0,0,2)
        p.input_action(5)
        self.assertEqual(p.state,(1,1,0,0,1))
        p=make_puzzle('sy58',42,4)
        p.input_action(3)
        self.assertEqual(p.state,((-1,0),(2,),(),1))
        p.input_action(5)  # Space waits; it no longer records a ghost.
        self.assertEqual(p.state,((-1,0),(2,5),(),2))
        p.input_action(6,x=24,y=10)  # Actual mover after the left step.
        self.assertEqual(p.state,((0,0),(),((2,5),),0))
        p.input_action(7)
        self.assertEqual(p.state,((-1,0),(2,5),(),2))

    def test_late_levels_add_structure_in_saturated_contexts(self):
        # More attributes/values, rod range, sigil strokes or program slots;
        # this does not assert increasing shortest paths or learning difficulty.
        for game, fields in (('sy06', ('n','mod')), ('sy25', ('n','limit')),
                             ('sy49', ('n','weight')), ('sy54', ('n',))):
            signatures = []
            for level in range(1,7):
                p = make_puzzle(game,42,level)
                signatures.append(tuple(p.patterns[0].bit_count() if f=='weight'
                                        else getattr(p,f) for f in fields))
            with self.subTest(game=game):
                self.assertEqual(len(set(signatures)),6,(game,signatures))

    def test_large_spatial_boards_stay_above_controls(self):
        for game in ('sy11','sy19'):
            p = make_puzzle(game,42,6,appearance_seed=5)
            p.offset = (2,2); p.size = 5
            p.state = (p.m-1,p.m-1,0,0,0) if game == 'sy11' else (p.m,p.m,0,0)
            p.target = ((p.m,p.m),) if game == 'sy11' else p.target
            c = Canvas(p.palette,p.size)
            p.draw(c)
            with self.subTest(game=game):
                self.assertTrue(np.all(c.pixels[51:] == 5),game)

    def test_catalog_and_sdk_entrypoints(self):
        self.assertEqual(len(SPECS),60); self.assertEqual(len(CONTEXTS),30)
        self.assertEqual(len(CATALOG),len(BY_ID))
        for spec in SPECS:
            directory = ROOT/'games'/'skills'/spec.skill/'v1'
            meta = json.loads((directory/'metadata.json').read_text())
            self.assertEqual(meta['game_id'],spec.game_id+'-v1')
            self.assertIn('synthetic',meta['tags'])
            self.assertNotIn('baseline_actions',meta)
            tree = ast.parse((directory/(spec.game_id+'.py')).read_text())
            cls = next(n for n in tree.body if isinstance(n,ast.ClassDef))
            self.assertEqual(cls.name,spec.game_id.capitalize())

    def test_64_seeds_per_profile_and_level(self):
        for spec in SPECS:
            for level in range(1,7):
                for seed in range(64):
                    with self.subTest(skill=spec.skill,level=level,seed=seed):
                        p = make_puzzle(spec.skill,seed,level)
                        image = p.render()
                        self.assertEqual(image.shape,(64,64))
                        self.assertTrue(np.all((image>=0)&(image<16)))
                        self.assertFalse(p.solved(p.state))
                        goals = p.get_subgoals()
                        self.assertEqual(len({g['id'] for g in goals}),len(goals))
                        json.dumps(p.get_training_info())
                        if spec.context=='pegs': self.assertEqual(p.state[0].count(1),level+1)

    def test_appearance_randomization_preserves_rules_and_reachability(self):
        for spec in SPECS:
            sizes,palettes,images = set(),set(),set()
            for cosmetic in range(8):
                p = make_puzzle(spec.skill,42,4,appearance_seed=cosmetic)
                q = make_puzzle(spec.skill,42,4,appearance_seed=999)
                self.assertEqual(p.state,q.state); self.assertEqual(p.target,q.target)
                self.assertEqual(p.get_subgoals(),q.get_subgoals())
                for command in range(len(p.commands)):
                    self.assertEqual(p.transition(p.state,command),q.transition(q.state,command))
                sizes.add(p.size); palettes.add(p.palette); images.add(p.render().tobytes())
            self.assertGreater(len(sizes),1); self.assertGreater(len(palettes),1)
            self.assertGreater(len(images),1)

    def test_small_and_large_hitboxes_preserve_complete_witnesses(self):
        for spec in SPECS:
            for level in range(1,7):
                reference = make_puzzle(spec.skill,123,level)
                commands = plan(reference)
                for size in (3,5):
                    cosmetic = next(i for i in range(64)
                                    if make_puzzle(spec.skill,123,level,appearance_seed=i).size==size)
                    p = make_puzzle(spec.skill,123,level,appearance_seed=cosmetic)
                    q = make_puzzle(spec.skill,123,level)
                    rectangles = [rect for _,rect in p.click_commands()]
                    occupied = set()
                    for x,y,w,h in rectangles:
                        cells = {(u,v) for u in range(x,x+w) for v in range(y,y+h)}
                        self.assertTrue(all(0<=u<64 and 4<=v<62 for u,v in cells))
                        self.assertFalse(occupied & cells,(spec.game_id,level,size))
                        occupied.update(cells)
                    for command in commands:
                        # Opposite inclusive corners detect clipped or oversized
                        # addressing; midpoint-only witnesses miss such bugs.
                        action,data = sdk_action(p,command,corner=int(size!=3))
                        p.input_action(action,**(data or {}))
                        q.act(command)
                        self.assertEqual(p.state,q.state,(spec.game_id,level,size))
                        self.assertEqual(p.get_subgoals(),q.get_subgoals())
                    self.assertTrue(p.won,(spec.game_id,level,size))

    def test_seed_replays_and_global_rng_isolation(self):
        random.seed(123); expected=random.getstate()
        for spec in SPECS:
            p = make_puzzle(spec.game_id+'-v1',-7,3)
            q = make_puzzle(spec.skill,-7,3)
            np.testing.assert_array_equal(p.render(),q.render())
            for i in range(10):
                command=i%len(p.commands); p.act(command); q.act(command)
                self.assertEqual((p.state,p.steps,p.lost,p.won),(q.state,q.steps,q.lost,q.won))
                np.testing.assert_array_equal(p.render(),q.render())
        self.assertEqual(random.getstate(),expected)

    def test_solution_witnesses_across_profiles_levels_and_seeds(self):
        for spec in SPECS:
            for level in range(1,7):
                for seed in (0,1,7,42):
                    with self.subTest(skill=spec.skill,level=level,seed=seed):
                        p = make_puzzle(spec.skill,seed,level)
                        commands = plan(p)
                        self.assertGreater(len(commands),0)
                        self.assertLessEqual(len(commands),p.budget)
                        for command in commands:
                            # Exercise advertised actions and current figure
                            # hitboxes, not just the pure transition function.
                            action,data = sdk_action(p,command)
                            p.input_action(action,**(data or {}))
                            self.assertFalse(p.lost)
                        self.assertTrue(p.won)
                        self.assertTrue(all(g['achieved'] for g in p.get_subgoals()))


if __name__=='__main__':
    unittest.main()
