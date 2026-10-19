"""Numerical reporting oracles, independent of any dashboard appearance."""
from dataclasses import replace
import random
import unittest

import numpy as np
import torch

from model.config import NetworkConfig, TrainConfig, SequenceConfig
from model.data import Action
from model.game_sequences import GameSession, Transition, evaluate_sessions, merge_evaluations, SequenceLearner
from model.learning import Engine
from model.networks import WorldModel
from model.sequence_diagnostics import SequenceDiagnostics
from model.sequence_training import fit_batch
from model.tests.test_agent import obs
from model.tests.test_sequence_training import session, scalar_engine


class SequenceDiagnosticsTests(unittest.TestCase):
    def test_pixel_categories_have_independent_counts_and_unchanged_is_not_success(self):
        before = np.full((2, 3), 5, dtype=np.uint8)
        after = np.array([[9, 9, 9], [5, 5, 5]], dtype=np.uint8)
        predicted = np.array([[9, 5, 14], [14, 5, 5]], dtype=np.uint8)
        row = GameSession('example', 0, [Transition(replace(obs(), frames=before[None]),
            Action(1), replace(obs(), frames=after[None]))])
        engine = scalar_engine()
        engine.predict = lambda z, h, action, height, width: (z, h, z)
        def decode(z, height, width):
            pixels = torch.tensor(predicted.astype(np.int64))
            return torch.nn.functional.one_hot(pixels, 16)[None].float()
        engine.decode = decode
        result = evaluate_sessions(engine, [row])
        expected = dict(wrong=3, union=4, transitions=1, delta_error=.75,
                        invented=1, missed=1, color=1, correct=1)
        self.assertEqual(result['all'], expected)
        self.assertEqual(result['example'], expected)
        merged = merge_evaluations([result, result])['all']
        self.assertEqual(merged['delta_error'], .75)
        self.assertEqual(merged['correct'], 2)
        self.assertEqual(merged['union'], 8)
        predicted[:] = before
        row.transitions[0] = replace(row.transitions[0], after=row.transitions[0].before)
        unchanged = evaluate_sessions(engine, [row])['all']
        self.assertIsNone(unchanged['delta_error'])
        self.assertEqual(unchanged['correct'], 0)
        self.assertEqual(unchanged['transitions'], 1)

    def test_diagnostics_use_real_context_and_preserve_parameters_gradients_and_rng(self):
        torch.manual_seed(10)
        engine = Engine(WorldModel(NetworkConfig.small()), TrainConfig())
        rows = [session((1, 2, 3, 4, 5, 6), game='first'), session((3, 4, 5, 6), game='second')]
        # A boundary is not a valid three-step rollout, although real memory carries on.
        rows[0].transitions[2] = replace(rows[0].transitions[2], reset_before=True)
        baseline = evaluate_sessions(engine, rows, reference_loss=True)
        before = {k: p.clone() for k, p in engine.model.state_dict().items()}
        for p in engine.model.parameters():
            p.grad = torch.ones_like(p)
        rng, python_rng = torch.get_rng_state().clone(), random.getstate()
        diagnostics = SequenceDiagnostics(engine, rows)
        diagnostics.start(1)
        measured = evaluate_sessions(engine, rows, reference_loss=True, diagnostics=diagnostics)
        summary, images = diagnostics.finish()
        self.assertEqual(measured, baseline)
        self.assertEqual(summary['sampled_transitions'], 8)
        self.assertEqual(summary['rollouts'], 2)  # first t=2 and second t=0 only
        self.assertIsNone(summary['latent_drift'])
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0]['previous'], rows[0].transitions[0].before.final.tolist())
        expected_reconstruction = engine.decode(engine.encode([rows[0].transitions[0].before])[0][None], 4, 4).argmax(-1)[0]
        self.assertEqual(images[0]['reconstruction'], expected_reconstruction.tolist())
        for k, p in engine.model.state_dict().items():
            torch.testing.assert_close(p, before[k], atol=0, rtol=0)
        self.assertTrue(all(torch.all(p.grad == 1) for p in engine.model.parameters()))
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        self.assertEqual(random.getstate(), python_rng)
        diagnostics.start(4)
        evaluate_sessions(engine, rows, diagnostics=diagnostics)
        repeated, _ = diagnostics.finish()
        self.assertEqual(repeated['latent_drift'], 0.)
        self.assertEqual(repeated['drift_reference_batch'], 1)

    def test_diagnostic_sample_budget_is_global_and_empty_evidence_is_missing(self):
        engine = Engine(WorldModel(NetworkConfig.small()), TrainConfig())
        rows = [session(tuple(range(8)), game=f'g{i}', seed=i) for i in range(12)]
        diagnostics = SequenceDiagnostics(engine, rows)
        self.assertEqual(len(diagnostics.selected), 32)
        diagnostics.start(1)
        evaluate_sessions(engine, rows, diagnostics=diagnostics)
        summary, images = diagnostics.finish()
        self.assertEqual(summary['sampled_transitions'], 32)
        self.assertLessEqual(summary['latent']['sample_count'], 64)
        self.assertLessEqual(len(images), 8)
        empty = SequenceDiagnostics(engine, [])
        empty.start(1)
        summary, images = empty.finish()
        self.assertIsNone(summary['reconstruction_error'])
        self.assertIsNone(summary['rollout_error'])
        self.assertIsNone(summary['latent']['std_mean'])
        self.assertEqual(images, [])

    def test_logged_loss_terms_sum_to_the_objective_without_changing_updates(self):
        from model.online_learning import OnlineRepresentation
        engine = Engine(WorldModel(NetworkConfig.small()), TrainConfig())
        learner = SequenceLearner(engine, SequenceConfig(session_steps=4, block_steps=2, representation_updates=2))
        representation = OnlineRepresentation(engine, 1e-4)
        result = fit_batch(learner, [session()], representation)
        self.assertAlmostEqual(sum(result['predictor_terms'].values()), result['loss'])
        self.assertAlmostEqual(sum(result['representation_terms'].values()), result['reconstruction_loss'], places=5)
        self.assertEqual(result['updates'], 2)
        self.assertEqual(result['examples'], 3)
