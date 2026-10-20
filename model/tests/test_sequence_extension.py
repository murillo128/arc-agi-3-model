"""Append-only curriculum changes retain learned state and public data boundaries."""
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
from model.game_sequences import SequenceLearner, schedule_games
from model.sequence import public_split
from model.sequence_control import SequenceControlConfig, SequencePhaseControl
from model.sequence_training import load_engine, representation_for, restore_run, run
from model.tests.test_sequence_control import counts
from model.tests.test_training_config import config
from model.training_config import load_plan


class SequenceExtensionTests(unittest.TestCase):
    def test_explicit_extension_preserves_state_and_subsequent_resume(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, value = config(root,
                [dict(source='synthetic', game='sc01-v1', levels=[1, 2])],
                sequence=dict(batch_games=1, session_steps=4, block_steps=2,
                              level_steps=2, representation_updates=1),
                sequence_control=dict(window_batches=2, diagnostic_samples=8),
                thresholds=dict(representation_open=1., latent_drift=1., patience=1,
                                minimum_samples=1, minimum_changed=1))
            value['settings']['actions'] = 1000
            path.write_text(json.dumps(value))
            args = SimpleNamespace(training_config=path, output=root/'run', source=None,
                resume=False, device='cpu', compile=False, max_batches=2)
            run(args)
            saved = checkpoint.load(root/'run/latest.pt')
            self.assertEqual(saved['sequence_control']['phase'], 'predictor')
            for owner in ('predictor_optimizer', 'representation_optimizer'):
                self.assertTrue(saved[owner]['state'], 'exercise populated moments')
            old_report = (root/'run/report.json').read_bytes()
            value['curriculum'].append(dict(source='synthetic', game='si01-v1', levels=[1]))
            value['sequence_control']['window_batches'] = 4
            path.write_text(json.dumps(value))
            args.resume, args.max_batches = True, 1
            with self.assertRaisesRegex(ValueError, 'same sequence'):
                run(args)
            self.assertEqual((root/'run/report.json').read_bytes(), old_report)
            self.assertEqual(checkpoint.load(root/'run/latest.pt')['progress'], saved['progress'])

            plan = load_plan(path)
            engine = load_engine(plan, root/'run/latest.pt')
            learner = SequenceLearner(engine, SequenceConfig(**plan.payload['sequence']))
            rep = representation_for(engine, plan)
            control = SequencePhaseControl(plan.stages, plan.settings,
                SequenceControlConfig(**plan.payload['sequence_control']))
            progress = restore_run(saved, plan, learner, rep, control, extend=True)
            self.assertEqual(progress, (2, 8))
            for name, weight in engine.model.state_dict().items():
                torch.testing.assert_close(weight, saved['weights'][name], rtol=0, atol=0)
            for owner, optimizer in (('predictor_optimizer', learner.optimizer),
                                     ('representation_optimizer', rep.optimizer)):
                for key, state in optimizer.state_dict()['state'].items():
                    for name, tensor in state.items():
                        torch.testing.assert_close(tensor, saved[owner]['state'][key][name], rtol=0, atol=0)
            self.assertEqual(learner.updates, saved['progress']['updates'])
            self.assertEqual(rep.updates, saved['representation_state']['updates'])
            torch.testing.assert_close(torch.get_rng_state(), saved['torch_rng'], rtol=0, atol=0)
            self.assertEqual(control.phase, 'representation')
            self.assertEqual(rep.learning_rates, dict(encoder=3e-6, decoder=3e-6))
            self.assertEqual(control.streaks['si01-v1/1'], 0)
            for key, passes in saved['sequence_control']['streaks'].items():
                self.assertEqual(control.streaks[key], passes)
            self.assertEqual(list(control.window), saved['sequence_control']['window'])

            args.extend_curriculum = True
            run(args)
            report = json.loads((root/'run/report.json').read_text())
            self.assertEqual([r['batch'] for r in report['history']], [1, 2, 3])
            self.assertEqual(report['games'], ['sc01-v1', 'si01-v1'])
            self.assertEqual(report['curriculum_extensions'][0]['batch'], 2)
            self.assertEqual(report['curriculum_extensions'][0]['games_before'], ['sc01-v1'])
            self.assertEqual(checkpoint.load(root/'run/latest.pt')['curriculum_extensions'],
                             report['curriculum_extensions'])
            run(args)  # Identical plan plus the extension flag is idempotent.
            args.extend_curriculum = False
            run(args)  # Ordinary strict resume works after migration.
            report = json.loads((root/'run/report.json').read_text())
            self.assertEqual(report['progress']['batches'], 5)
            self.assertEqual(len(report['curriculum_extensions']), 1)

    def test_extension_rejects_unrelated_changes_and_accepts_first_sdk_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, value = config(root, [dict(source='synthetic', game='sc01-v1', levels=[1, 2])],
                                 sequence={}, sequence_control={})
            previous = load_plan(path)
            # These tests own plan validation; optimizers must not be touched on failure.
            saved = dict(kind='game_sequences', version=1, plan=previous.payload,
                         sequence=previous.payload['sequence'])
            learner = SimpleNamespace(config=SequenceConfig(**saved['sequence']))
            extended = deepcopy(previous.payload)
            extended['stages'].append(dict(extended['stages'][0], game='si01-v1'))
            mutations = [
                lambda p: p['settings'].update(learning_rate=1e-4),
                lambda p: p['stages'][0]['thresholds'].update(patience=99),
                lambda p: p['stages'].reverse(),
                lambda p: p['stages'].pop(0),
                lambda p: p['sequence'].update(batch_games=8),
                lambda p: p['sequence_control'].update(diagnostic_samples=1),
                lambda p: p['sequence_control'].update(window_batches=1),
                lambda p: p['initialization']['network'].update(latent=123),
            ]
            for mutation in mutations:
                candidate = deepcopy(extended)
                mutation(candidate)
                with self.subTest(candidate=candidate), self.assertRaisesRegex(ValueError, 'same sequence'):
                    restore_run(saved, replace(previous, payload=candidate), learner, None, extend=True)
            with self.assertRaisesRegex(ValueError, 'combined'):
                restore_run(saved, previous, learner, None, adopt=True, extend=True)
            from model.sequence_training import extended_plan_matches
            extended['sdk'] = dict(environments_dir='/public', mode='offline',
                                   split=dict(training_games=['ls20'], evaluation_games=['vc33']))
            self.assertTrue(extended_plan_matches(previous.payload, extended))
            prior_sdk = deepcopy(extended)
            extended['stages'].append(dict(extended['stages'][0], game='bf01-v1'))
            extended['sdk']['split']['evaluation_games'] = ['ft09']
            self.assertFalse(extended_plan_matches(prior_sdk, extended), 'existing split stays fixed')
            args = SimpleNamespace(training_config=path, output=root/'run', source=None,
                resume=False, extend_curriculum=True, device='cpu', compile=False, max_batches=1)
            with self.assertRaisesRegex(ValueError, 'requires --resume'):
                run(args)
            self.assertFalse(args.output.exists())

    def test_mixed_plan_matches_public_split_and_admits_only_reached_native_levels(self):
        root = Path(__file__).parents[2]
        mixed = load_plan(root/'model/sequence_mixed.example.json')
        synthetic = load_plan(root/'model/sequence_all_synthetic.example.json')
        self.assertEqual(mixed.stages[:424], synthetic.stages)
        sdk = mixed.payload['sdk']['split']
        public = sdk['training_games'] + sdk['evaluation_games']
        self.assertEqual(public_split(public), (sdk['training_games'], sdk['evaluation_games']))
        self.assertEqual({g.split('-')[0] for g in sdk['evaluation_games']},
                         {'ar25', 'cn04', 'lf52', 'tn36', 'vc33'})
        self.assertEqual(len(sdk['training_games']), 20)
        self.assertEqual(len(mixed.stages), 549)
        for game, cap in sdk['training_level_caps'].items():
            self.assertEqual([s['level'] for s in mixed.stages if s['game'] == game],
                             list(range(1, cap+1)))
        c = SequenceConfig(**mixed.payload['sequence'])
        games = list(dict.fromkeys(s['game'] for s in mixed.stages))
        control = SequencePhaseControl(mixed.stages, mixed.settings,
            SequenceControlConfig(**mixed.payload['sequence_control']))
        for index in range(36):  # Three complete shuffled rounds of 91 games.
            batch = set(schedule_games(games, index, c))
            evidence = {f"{s['game']}/{s['level']}": counts(samples=8, changed=4, drift_samples=8)
                for s in mixed.stages if s['game'] in batch and (s['source'] != 'sdk' or s['level'] == 1)}
            control.observe(index+1, evidence)
            if index < 35:
                self.assertEqual(control.phase, 'representation')
        self.assertEqual(control.phase, 'predictor')
        self.assertEqual(sum(r['required'] for r in control.rows()), 444)


if __name__ == '__main__':
    unittest.main()
