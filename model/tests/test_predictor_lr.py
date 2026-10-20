"""Predictor-only retuning preserves learned state; paired metrics use real counts."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from model import checkpoint
from model.compare_predictor_lr import compare, run as compare_run
from model.config import SequenceConfig
from model.game_sequences import SequenceLearner
from model.sequence_training import (load_engine, predictor_learning_rate, representation_for,
    restore_run, run)
from model.tests.test_training_config import config
from model.training_config import load_plan


torch.set_num_threads(2)


def arguments(path, output, **values):
    return SimpleNamespace(training_config=path, output=output, source=None,
        resume=False, device='cpu', compile=False, max_batches=1, **values)


class PredictorLRTests(unittest.TestCase):
    def test_retune_preserves_weights_moments_representation_and_survives_resume(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, _ = config(root, sequence=dict(batch_games=1, session_steps=4,
                block_steps=2, level_steps=2, representation_updates=1))
            args = arguments(path, root/'run')
            run(args)
            source = args.output/'latest.pt'
            saved = checkpoint.load(source)
            plan = load_plan(path)
            engine = load_engine(plan, source)
            learner = SequenceLearner(engine, SequenceConfig(**plan.payload['sequence']))
            representation = representation_for(engine, plan)
            restore_run(saved, plan, learner, representation)
            previous = deepcopy(learner.optimizer.state_dict())
            rates = representation.learning_rates.copy()
            predictor_learning_rate(learner, 1e-4)
            for name, weight in engine.model.state_dict().items():
                torch.testing.assert_close(weight, saved['weights'][name], rtol=0, atol=0)
            actual = learner.optimizer.state_dict()
            for key, state in previous['state'].items():
                for field, value in state.items():
                    torch.testing.assert_close(actual['state'][key][field], value, rtol=0, atol=0)
            for old, new in zip(previous['param_groups'], actual['param_groups'], strict=True):
                self.assertEqual(new, old | {'lr': 1e-4})
            self.assertEqual(representation.learning_rates, rates)
            self.assertEqual(learner.updates, saved['progress']['updates'])
            args.resume, args.unlimited = True, True
            args.predictor_learning_rate = 1e-4
            run(args)  # Override must happen after restoring the old optimizer LR.
            args.predictor_learning_rate = None
            run(args)  # Ordinary resume must retain the override, not the plan default.
            resumed = checkpoint.load(source)
            self.assertEqual(resumed['training']['learning_rate'], 1e-4)
            self.assertEqual(resumed['plan'], saved['plan'])
            self.assertEqual(resumed['progress']['batches'], saved['progress']['batches']+2)
            report = json.loads((args.output/'report.json').read_text())
            self.assertEqual(report['learning_rates'], {'predictor': 1e-4, **rates})
            self.assertEqual(len(report['history']), 3)
            self.assertTrue(all(g['lr'] == 1e-4 for g in resumed['predictor_optimizer']['param_groups']))
            for key, state in previous['state'].items():
                self.assertGreater(float(resumed['predictor_optimizer']['state'][key]['step']),
                                   float(state['step']))

    def test_invalid_override_fails_before_loading_or_mutating_run(self):
        for rate in (0., -1e-4, 1., float('nan'), float('inf')):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, 'learning rate'):
                run(SimpleNamespace(predictor_learning_rate=rate))

    def test_paired_comparison_requires_pixel_game_and_source_improvement(self):
        def row(wrong):
            return dict(wrong=wrong, union=100, transitions=10, invented=0,
                        missed=wrong, color=0, correct=100-wrong, delta_error=wrong/100)
        sources = dict(a='synthetic', b='sdk')
        baseline = dict(a=row(20), b=row(80))
        better = compare(baseline, dict(a=row(10), b=row(70)), sources)
        self.assertTrue(better['candidate_better'])
        self.assertAlmostEqual(better['pooled_difference'], -.1)
        self.assertAlmostEqual(better['macro_difference'], -.1)
        self.assertEqual(better['improved_games'], 2)
        self.assertFalse(compare(baseline, dict(a=row(0), b=row(85)), sources)['candidate_better'])
        tied = compare(baseline, baseline, sources)
        self.assertFalse(tied['candidate_better'])
        self.assertEqual(tied['tied_games'], 2)

    def test_cpu_experiment_keeps_checkpoint_and_uses_same_disjoint_corpus(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, _ = config(root, sequence=dict(batch_games=1, session_steps=2,
                block_steps=1, level_steps=1, representation_updates=1))
            run(arguments(path, root/'run'))
            compare_run(SimpleNamespace(training_config=path, checkpoint=root/'run/latest.pt',
                output=root/'experiment', device='cpu', probe_steps=2))
            result = json.loads((root/'experiment/comparison.json').read_text())
            self.assertTrue(result['source_checkpoint_unchanged'])
            self.assertEqual([t['learning_rate'] for t in result['trials']], [3e-5, 1e-4])
            self.assertTrue(set(result['training_seeds'].values()).isdisjoint(result['probe_seeds'].values()))
            a, b = result['trials']
            self.assertEqual(a['representation_rates'], b['representation_rates'])
            self.assertEqual(a['representation_rates'], a['final_representation_rates'])
            self.assertEqual(b['representation_rates'], b['final_representation_rates'])
            self.assertEqual(a['updates'], b['updates'])
            self.assertEqual([x['games'] for x in a['batches']], [x['games'] for x in b['batches']])


if __name__ == '__main__':
    unittest.main()
