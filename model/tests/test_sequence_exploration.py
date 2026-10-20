"""Behavioral contracts for the auxiliary real-change prior and safe adoption."""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model import checkpoint
from model.config import SequenceConfig
from model.data import Action, Observation, random_action
from model.game_sequences import GameSession, Transition, SequenceLearner, collect_session
from model.sequence_exploration import (SequenceExplorationConfig, ChangeExplorer, ExplorationSession,
    features, interaction_counts, training_examples)
from model.sequence_training import load_engine, representation_for, restore_run, run
from model.tests.test_training_config import config
from model.training_config import load_plan


def board(x=10, y=12, colors=(0, 6, 11), shape=(64, 64), actions=(6,)):
    frame = np.full(shape, colors[0], dtype=np.uint8)
    frame[y:y+2, x:x+2] = colors[1]
    if min(shape) >= 48:
        frame[32:48, 32:48] = colors[2]  # Large visible distractor, not clickable.
    return Observation(frame[None], actions, 'NOT_FINISHED', 0, 2)


def transition(before, action, x, y):
    frame = before.final.copy()
    if action.id == 6 and x <= action.data['x'] < x+2 and y <= action.data['y'] < y+2:
        frame[y:y+2, x:x+2] = 2
    return Transition(before, action, replace(before, frames=frame[None]))


class SequenceExplorationTests(unittest.TestCase):
    def test_learned_prior_finds_small_targets_on_unseen_positions_and_preserves_palette_symmetry(self):
        torch.set_num_threads(1)
        c = SequenceExplorationConfig(policy='learned', replay_size=4096)
        model = ChangeExplorer(c)
        sessions = []
        rng = random.Random(50)
        # Observed interactions with small clickable targets and inert large decoys.
        # The explicit geometric hit rule above is independent of model features.
        for x, y in ((5, 8), (17, 24), (52, 14), (21, 54)):
            before = board(x, y)
            actions = [Action(6, dict(x=rng.randrange(64), y=rng.randrange(64))) for _ in range(300)]
            actions += [Action(6, dict(x=x, y=y)) for _ in range(50)]
            actions += [Action(6, dict(x=35, y=36)) for _ in range(50)]
            # Each probe starts from a reset copy of this board; no artificial carry.
            sessions.append(GameSession('fixture', 0, [replace(transition(before, a, x, y), reset_before=True) for a in actions]))
        model.fit(sessions); model.fit_replay(350)
        before = board(25, 6)
        candidates = [Action(6, dict(x=25, y=6)), Action(6, dict(x=35, y=36)), Action(6, dict(x=2, y=2))]
        probabilities = model.probabilities(before, candidates)
        self.assertGreater(probabilities[0], .7)
        self.assertLess(max(probabilities[1:]), .2)
        np.testing.assert_array_equal(features(before, candidates), features(board(25, 6, (5, 9, 3)), candidates))
        sampler = ExplorationSession(c, 9, model)
        ordinary = random.Random(0)
        chosen = [sampler.action(before, ordinary) for _ in range(500)]
        hits = sum(25 <= a.data['x'] < 27 and 6 <= a.data['y'] < 8 for a in chosen)
        self.assertGreater(hits / len(chosen), .35)
        uniform = random.Random(0)
        baseline = [random_action(before, uniform) for _ in range(500)]
        self.assertLess(sum(25 <= a.data['x'] < 27 and 6 <= a.data['y'] < 8 for a in baseline), 10)

    def test_uniform_is_legacy_exact_and_all_policies_preserve_action_id_stream(self):
        before = board(actions=(1, 3, 6, 7))
        for policy in ('uniform', 'mixed', 'learned'):
            c = SequenceExplorationConfig(policy=policy)
            sampler = ExplorationSession(c, 4, ChangeExplorer(c) if policy == 'learned' else None)
            a, b = random.Random(10), random.Random(10)
            for _ in range(128):
                expected, actual = random_action(before, a), sampler.action(before, b)
                self.assertEqual(expected.id, actual.id)
                if policy == 'uniform' or actual.id != 6:
                    self.assertEqual(expected, actual)
                else:
                    self.assertTrue(all(0 <= v <= 63 for v in actual.data.values()))
            self.assertEqual(a.getstate(), b.getstate())

    def test_missing_flat_and_small_frames_do_not_rescale_native_coordinates(self):
        c = SequenceExplorationConfig(policy='learned')
        model = ChangeExplorer(c)
        for before in (replace(board(), frames=np.zeros((1, 4, 8), dtype=np.uint8)),
                       replace(board(), frames=np.empty((0, 4, 8), dtype=np.uint8))):
            sampler = ExplorationSession(c, 0, model)
            a, b = random.Random(1), random.Random(1)
            self.assertEqual([sampler.action(before, a) for _ in range(32)],
                             [random_action(before, b) for _ in range(32)])
        small = board(2, 1, shape=(4, 8))
        sampler = ExplorationSession(c, 0, model); rng = random.Random(1)
        actions = [sampler.action(small, rng) for _ in range(256)]
        self.assertTrue(any(a.data['x'] >= 8 or a.data['y'] >= 4 for a in actions))
        inside = model.probabilities(small, [Action(6, dict(x=2, y=1))])
        self.assertTrue(np.isfinite(inside).all())

    def test_noop_penalty_expires_and_does_not_suppress_uniform_branch(self):
        c = SequenceExplorationConfig(policy='learned', noop_cooldown=2)
        model = ChangeExplorer(c); before = board()
        candidate = Action(6, dict(x=30, y=20)); other = Action(6, dict(x=40, y=20))
        sampler = ExplorationSession(c, 0, model)
        sampler.observe(before, candidate, before)
        def weights_once():
            with patch('model.sequence_exploration.click_pool', return_value=[candidate, other]), \
                 patch.object(sampler.rng, 'random', return_value=.9), \
                 patch.object(model, 'probabilities', return_value=np.ones(2)), \
                 patch.object(sampler.rng, 'choices', return_value=[other]) as choose:
                sampler.action(before, random.Random(4))
                return choose.call_args.kwargs['weights']
        penalized = weights_once()
        self.assertAlmostEqual(penalized[0]/penalized[1], .02)
        with patch.object(sampler.rng, 'random', return_value=0.):
            self.assertEqual(sampler.action(before, random.Random(4)), random_action(before, random.Random(4)))
        for _ in range(2): sampler.observe(before, Action(1), before)
        self.assertEqual(*weights_once())
        sampler.reset(); self.assertIsNone(sampler.previous)

    def test_targets_exclude_terminal_and_level_boundaries_but_retain_real_noops(self):
        before = board(); click = Action(6, dict(x=10, y=12))
        changed = transition(before, click, 10, 12)
        rows = [Transition(before, Action(1), before), changed,
                replace(changed, after=replace(changed.after, state='GAME_OVER')),
                replace(changed, after=replace(changed.after, levels_completed=1)),
                replace(changed, action=Action(0))]
        examples = list(training_examples([GameSession('x', 0, rows)]))
        self.assertEqual([r[1] for r in examples], [0., 1.])
        class Env:
            observation = before
            def step(self, action): return self.observation
        factory = SimpleNamespace(game_id='x', reserve_final_level=False, make=lambda *a, **kw: Env())
        c = SequenceExplorationConfig(policy='learned'); model = ChangeExplorer(c)
        session = collect_session(factory, [1], 0, SequenceConfig(session_steps=16), exploration=c, explorer=model)
        self.assertEqual(len(session.transitions), 16)
        self.assertEqual(interaction_counts(session)['unchanged_clicks'], 16)
        self.assertTrue(np.isfinite(model.fit([session])['loss']))
        session.transitions = [changed] * 16
        self.assertTrue(np.isfinite(model.fit([session])['loss']))

    def test_planner_prior_retains_ordinary_actions_and_click_count(self):
        c = SequenceExplorationConfig(policy='learned'); model = ChangeExplorer(c)
        actions = model.candidates(board(actions=(1, 2, 6, 7)), random.Random(3), 16)
        self.assertEqual([a.id for a in actions if a.id != 6], [1, 2, 7])
        clicks = [tuple(a.data.values()) for a in actions if a.id == 6]
        self.assertEqual(len(clicks), 16); self.assertEqual(len(set(clicks)), 16)
        self.assertEqual(model.candidates(board(actions=(2, 4)), random.Random(0)), [Action(2), Action(4)])

    def test_planner_success_still_outranks_a_change_prior(self):
        from contextlib import nullcontext
        from model.learning import State
        from model.path_planning import PathPlanner, PathConfig
        class Value(torch.nn.Module):
            def forward(self, z, h, goal, mask, rows, columns, remaining_budget):
                return torch.where(z[:, 0, 0] == 2, 8., -8.), torch.zeros(len(z))
        engine = SimpleNamespace(device=torch.device('cpu'), model=torch.nn.Identity(), autocast=nullcontext,
            actions=lambda actions:torch.tensor([[a.id] for a in actions]),
            predict=lambda z,h,a,height,width:(a[:, :1, None].float(), h, z))
        engine.model.config = SimpleNamespace(patch=1)
        prior = SimpleNamespace(candidates=lambda observation,rng,clicks:[Action(6, {'x':0, 'y':0}), Action(2)])
        planner = PathPlanner(engine, Value(), PathConfig(horizon=1), change_prior=prior)
        planner.goal = dict(z=torch.zeros(1, 1), mask=torch.ones(1, 1, 1), frame=np.zeros((1, 2)),
                            pixel_mask=np.ones((1, 2), dtype=bool))
        state = State(torch.zeros(1, 1), torch.zeros(1, 1), (1, 2))
        observation = replace(board(actions=(2, 6)), frames=np.zeros((1, 1, 2), dtype=np.uint8))
        self.assertEqual(planner.choose(state, observation, random.Random(0)), Action(2))

    def test_config_requires_uniform_support_and_bounded_candidates(self):
        for values in (dict(policy='unknown'), dict(uniform_probability=0.), dict(uniform_probability=float('nan')),
                       dict(candidate_clicks=257), dict(noop_cooldown=65), dict(updates=True)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                SequenceExplorationConfig(**values)

    def test_head_round_trip_uses_safe_checkpoint_and_preserves_next_update(self):
        c = SequenceExplorationConfig(policy='learned')
        model = ChangeExplorer(c)
        before = board(); rows = [transition(before, Action(6, dict(x=10, y=12)), 10, 12)]
        model.fit([GameSession('x', 0, rows)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'head.pt'; checkpoint.write(path, model.snapshot())
            restored = ChangeExplorer(c); restored.restore(checkpoint.load(path))
            self.assertEqual(model.fit_replay(1), restored.fit_replay(1))
            for a, b in zip(model.head.parameters(), restored.head.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                ChangeExplorer(replace(c, uniform_probability=.5)).restore(checkpoint.load(path))

    def test_explicit_adoption_preserves_world_state_then_ordinary_resume(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, value = config(root, [dict(source='synthetic', game='si01-v1', levels=[1])],
                sequence=dict(batch_games=1, session_steps=4, block_steps=2, level_steps=4,
                              context_steps=1, representation_updates=1))
            value['settings']['actions'] = 1000; path.write_text(json.dumps(value))
            args = SimpleNamespace(training_config=path, output=root/'run', source=None, resume=False,
                device='cpu', compile=False, max_batches=1)
            run(args)
            saved = checkpoint.load(root/'run/latest.pt')
            value['sequence_exploration'] = dict(policy='learned', updates=1)
            path.write_text(json.dumps(value)); plan = load_plan(path)
            engine = load_engine(plan, root/'run/latest.pt')
            learner = SequenceLearner(engine, SequenceConfig(**plan.payload['sequence']))
            rep = representation_for(engine, plan)
            explorer = ChangeExplorer(SequenceExplorationConfig(**plan.payload['sequence_exploration']))
            with self.assertRaisesRegex(ValueError, 'same sequence'):
                restore_run(saved, plan, learner, rep, explorer=explorer)
            counters = restore_run(saved, plan, learner, rep, adopt_exploration=True, explorer=explorer)
            self.assertEqual(counters, (saved['progress']['batches'], saved['progress']['actions']))
            for name, tensor in engine.model.state_dict().items():
                torch.testing.assert_close(tensor, saved['weights'][name], rtol=0, atol=0)
            for key, optimizer in (('predictor_optimizer', learner.optimizer), ('representation_optimizer', rep.optimizer)):
                for identity, state in optimizer.state_dict()['state'].items():
                    for name, tensor in state.items():
                        torch.testing.assert_close(tensor, saved[key]['state'][identity][name], rtol=0, atol=0)
            for section, key, replacement in (('settings','learning_rate',.002), ('sequence','session_steps',8)):
                bad = deepcopy(plan.payload); bad[section][key] = replacement
                with self.assertRaisesRegex(ValueError, 'same sequence'):
                    restore_run(saved, replace(plan, payload=bad), learner, rep, adopt_exploration=True, explorer=explorer)
            args.resume = True; args.adopt_exploration = True
            run(args)
            adopted = checkpoint.load(root/'run/latest.pt')
            self.assertEqual(adopted['exploration_adoptions'][0]['batch'], 1)
            args.adopt_exploration = False
            run(args)
            report = json.loads((root/'run/report.json').read_text())
            self.assertEqual([r['batch'] for r in report['history']], [1, 2, 3])
            self.assertEqual(report['history'][-1]['exploration']['policy'], 'learned')
            self.assertEqual(len(report['exploration_adoptions']), 1)
            self.assertEqual(checkpoint.load(root/'run/latest.pt')['exploration']['updates'], 2)
