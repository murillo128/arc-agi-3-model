"""Explicit catalog reduction preserves learning and starts a new metric segment.

The real CPU resume owns weight/Adam/phase continuity; literal subsets and
unrelated plan mutations own the safety boundary. Existing extension tests cover
adding games, which must retain its different representation-admission behavior.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from model import checkpoint
from model.config import SequenceConfig
from model.game_sequences import SequenceLearner
from model.sequence_control import SequenceControlConfig, SequencePhaseControl
from model.sequence_exploration import ChangeExplorer, SequenceExplorationConfig
from model.sequence_training import load_engine, representation_for, restricted_plan_matches, restore_run, run
from model.tests.test_training_config import config
from model.training_config import load_plan


class SequenceRestrictionTests(unittest.TestCase):
    def test_explicit_restriction_preserves_learning_and_resumes_only_retained_games(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, value = config(root, [
                dict(source='synthetic', game='sc01-v1', levels=[1]),
                dict(source='synthetic', game='bf01-v1', levels=[1]),
                dict(source='synthetic', game='sy01-v1', levels=[1])],
                sequence=dict(batch_games=3, session_steps=4, block_steps=2,
                              level_steps=4, representation_updates=1),
                sequence_control=dict(window_batches=2, diagnostic_samples=12),
                sequence_exploration=dict(policy='learned', updates=1, batch_size=4),
                thresholds=dict(representation_open=1., latent_drift=1., patience=1,
                                minimum_samples=1, minimum_changed=1))
            value['settings']['actions'] = 1000
            path.write_text(json.dumps(value))
            args = SimpleNamespace(training_config=path, output=root/'run', source=None,
                resume=False, device='cpu', compile=False, max_batches=2)
            run(args)
            saved = checkpoint.load(root/'run/latest.pt')
            self.assertEqual(saved['sequence_control']['phase'], 'predictor')
            for name in ('predictor_optimizer', 'representation_optimizer'):
                self.assertTrue(saved[name]['state'], 'exercise populated Adam moments')
            self.assertTrue(saved['exploration']['optimizer']['state'])
            self.assertGreater(len(saved['exploration']['replay_targets']), 0)
            previous_report = (root/'run/report.json').read_bytes()
            value['curriculum'].pop()
            path.write_text(json.dumps(value))
            args.resume, args.max_batches = True, 1
            with self.assertRaisesRegex(ValueError, 'same sequence'):
                run(args)
            self.assertEqual((root/'run/report.json').read_bytes(), previous_report)
            self.assertEqual(checkpoint.load(root/'run/latest.pt')['progress'], saved['progress'])

            plan = load_plan(path)
            engine = load_engine(plan, root/'run/latest.pt')
            learner = SequenceLearner(engine, SequenceConfig(**plan.payload['sequence']))
            rep = representation_for(engine, plan)
            control = SequencePhaseControl(plan.stages, plan.settings,
                SequenceControlConfig(**plan.payload['sequence_control']))
            explorer = ChangeExplorer(SequenceExplorationConfig(**plan.payload['sequence_exploration']))
            self.assertEqual(restore_run(saved, plan, learner, rep, control,
                restrict=True, explorer=explorer), (2, saved['progress']['actions']))
            for weights, previous in ((engine.model.state_dict(), saved['weights']),
                                      (explorer.head.state_dict(), saved['exploration']['weights'])):
                for name, tensor in weights.items():
                    torch.testing.assert_close(tensor, previous[name], rtol=0, atol=0)
            for optimizer, previous in ((learner.optimizer, saved['predictor_optimizer']),
                    (rep.optimizer, saved['representation_optimizer']),
                    (explorer.optimizer, saved['exploration']['optimizer'])):
                current = optimizer.state_dict()
                self.assertEqual(current['param_groups'], previous['param_groups'])
                self.assertEqual(current['state'].keys(), previous['state'].keys())
                for key, state in current['state'].items():
                    for name, tensor in state.items():
                        torch.testing.assert_close(tensor, previous['state'][key][name], rtol=0, atol=0)
            self.assertEqual(learner.updates, saved['progress']['updates'])
            self.assertEqual(rep.updates, saved['representation_state']['updates'])
            self.assertEqual(rep.sigreg_enabled, saved['representation_state']['sigreg_enabled'])
            self.assertEqual((control.phase, control.admitted_batch),
                             ('predictor', saved['sequence_control']['admitted_batch']))
            self.assertEqual(explorer.updates, saved['exploration']['updates'])
            self.assertEqual(explorer.examples, saved['exploration']['examples'])
            self.assertEqual(explorer.rng.bit_generator.state, saved['exploration']['rng'])
            self.assertEqual(len(explorer.replay), 0, 'discard unlabelled replay from removed games')
            torch.testing.assert_close(torch.get_rng_state(), saved['torch_rng'], rtol=0, atol=0)
            retained = {'sc01-v1/1', 'bf01-v1/1'}
            old_control = saved['sequence_control']
            self.assertEqual(control.streaks, {k: v for k, v in old_control['streaks'].items() if k in retained})
            self.assertEqual(control.observed_pairs, set(old_control['observed_pairs']) & retained)
            self.assertEqual(list(control.window), [dict(entry, pairs={
                k: v for k, v in entry['pairs'].items() if k in retained}) for entry in old_control['window']])

            args.restrict_curriculum = True
            run(args)
            report = json.loads((root/'run/report.json').read_text())
            self.assertEqual(report['games'], ['sc01-v1', 'bf01-v1'])
            self.assertEqual([r['batch'] for r in report['history']], [3])
            self.assertEqual({s['game'] for s in report['history'][0]['sessions']}, {'sc01-v1', 'bf01-v1'})
            self.assertEqual(report['phase_control']['phase'], 'predictor')
            self.assertEqual(report['curriculum_restrictions'][0]['batch'], 2)
            self.assertEqual(report['curriculum_restrictions'][0]['games_after'], ['sc01-v1', 'bf01-v1'])
            archive = root/'run/report-before-curriculum-restriction-2.json'
            self.assertEqual(json.loads(archive.read_text()), json.loads(previous_report))
            self.assertEqual(checkpoint.load(root/'run/latest.pt')['curriculum_restrictions'],
                             report['curriculum_restrictions'])
            run(args)  # Same plan plus flag must not clear metrics or replay again.
            args.restrict_curriculum = False
            run(args)
            report = json.loads((root/'run/report.json').read_text())
            self.assertEqual([r['batch'] for r in report['history']], [3, 4, 5])
            self.assertEqual(len(report['curriculum_restrictions']), 1)

    def test_restriction_rejects_unrelated_changes_and_other_adoptions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = config(root, [dict(source='synthetic', game='sc01-v1', levels=[1, 2, 3])],
                             sequence={}, sequence_control={})
            previous = load_plan(path)
            saved = dict(kind='game_sequences', version=1, plan=previous.payload,
                         sequence=previous.payload['sequence'])
            learner = SimpleNamespace(config=SequenceConfig(**saved['sequence']))
            restricted = deepcopy(previous.payload)
            restricted['stages'].pop(1)  # A subset need not be a prefix.
            self.assertTrue(restricted_plan_matches(previous.payload, restricted))
            mutations = [
                lambda p: p['settings'].update(learning_rate=1e-4),
                lambda p: p['stages'][0]['thresholds'].update(patience=99),
                lambda p: p['stages'].reverse(),
                lambda p: p['stages'].append(p['stages'][0]),
                lambda p: p['stages'][0].update(game='si01-v1'),
                lambda p: p['stages'].clear(),
                lambda p: p['sequence'].update(batch_games=8),
                lambda p: p['sequence_control'].update(window_batches=100),
                lambda p: p['sdk'].update(mode='online'),
                lambda p: p['initialization']['network'].update(latent=123),
            ]
            for mutation in mutations:
                candidate = deepcopy(restricted)
                mutation(candidate)
                with self.subTest(candidate=candidate), self.assertRaisesRegex(ValueError, 'same sequence'):
                    restore_run(saved, replace(previous, payload=candidate), learner, None, restrict=True)
            for other in ('adopt', 'extend', 'adopt_exploration'):
                with self.subTest(other=other), self.assertRaisesRegex(ValueError, 'combined'):
                    restore_run(saved, previous, learner, None, restrict=True, **{other: True})
            args = SimpleNamespace(training_config=path, output=root/'run', source=None,
                resume=False, restrict_curriculum=True, device='cpu', compile=False, max_batches=1)
            with self.assertRaisesRegex(ValueError, 'requires --resume'):
                run(args)
            self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
