"""Training-data gates: independent numerical examples and real CPU lifecycle."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model import checkpoint
from model.config import NetworkConfig, TrainConfig, SequenceConfig
from model.curriculum import CurriculumConfig
from model.game_sequences import SequenceLearner, evaluate_sessions
from model.learning import Engine
from model.networks import WorldModel
from model.online_learning import OnlineRepresentation
from model.sequence_control import SequenceControlConfig, SequencePhaseControl, TrainingBatchDiagnostics
from model.sequence_training import fit_batch, run, restore_run, save_run, representation_for, collect_batch
from model.tests.test_sequence_training import session
from model.tests.test_training_config import config
from model.training_config import load_plan


def stages():
    gates = asdict(CurriculumConfig(patience=2, minimum_samples=2, minimum_changed=1))
    return [dict(game='game', level=i, thresholds=gates) for i in (1, 2)]


def counts(**changes):
    return dict(dict(samples=2, changed=1, reconstruction_wrong=1, reconstruction_union=20,
        drift_delta=.01, drift_scale=1., drift_samples=2, prediction_wrong=15, prediction_union=20), **changes)


class SequenceControlTests(unittest.TestCase):
    def controller(self, **options):
        return SequencePhaseControl(stages(), {'encoder_learning_rate_after_predictor': 3e-7},
                                    SequenceControlConfig(**options))

    def test_sparse_observed_levels_do_not_veto_qualified_peers(self):
        sparse_rows = (
            counts(changed=0, reconstruction_wrong=0, reconstruction_union=0),
            counts(samples=1, drift_samples=1),
            counts(drift_samples=1),
        )
        for sparse in sparse_rows:
            with self.subTest(sparse=sparse):
                c = self.controller(window_batches=1)
                for batch in (1, 2):
                    c.observe(batch, {'game/1': counts(), 'game/2': sparse})
                self.assertEqual((c.phase, c.admitted_batch), ('predictor', 2))
                row = c.rows()[1]
                self.assertTrue(row['observed'])
                self.assertTrue(row['required'], 'catalog membership is not a pass')
                self.assertFalse(row['eligible'])
                self.assertFalse(row['passing'])
                self.assertEqual(row['passes'], 0)
                self.assertEqual(row['samples'], sparse['samples'])

    def test_all_sparse_or_absent_evidence_cannot_admit_predictor(self):
        for sparse in (counts(changed=0), counts(samples=1, drift_samples=1)):
            with self.subTest(sparse=sparse):
                c = self.controller(window_batches=1)
                for batch in (1, 2, 3):
                    c.observe(batch, {'game/1': sparse, 'game/2': sparse})
                self.assertTrue(all(row['observed'] for row in c.rows()))
                self.assertFalse(any(row['eligible'] for row in c.rows()))
                self.assertEqual(c.phase, 'representation')
                c.observe(4, {})
                self.assertEqual(c.phase, 'representation', 'empty all() is not qualification')

    def test_sparse_level_participates_when_fresh_evidence_becomes_sufficient(self):
        c = self.controller(window_batches=1)
        c.observe(1, {'game/1': counts(), 'game/2': counts(changed=0)})
        c.observe(2, {'game/1': counts(), 'game/2': counts(reconstruction_wrong=10)})
        self.assertTrue(c.rows()[1]['eligible'])
        self.assertEqual(c.phase, 'representation', 'a newly eligible measured failure blocks')
        c.observe(3, {'game/1': counts(), 'game/2': counts()})
        self.assertEqual(c.phase, 'representation', 'the new level still needs two fresh passes')
        c.observe(4, {'game/1': counts(), 'game/2': counts()})
        self.assertEqual((c.phase, c.admitted_batch), ('predictor', 4))

    def test_pool_counts_fresh_confirmations_and_latched_admission(self):
        c = self.controller(window_batches=2)
        c.observe(1, {'game/1': counts(reconstruction_wrong=1, reconstruction_union=2), 'game/2': counts()})
        c.observe(2, {'game/1': counts(reconstruction_wrong=0, reconstruction_union=18), 'game/2': counts()})
        self.assertEqual(c.rows()[0]['reconstruction_error'], .05)  # 1/20, not mean(.5,0)
        self.assertEqual(c.rows()[0]['passes'], 1)
        self.assertEqual(c.phase, 'representation')
        c.observe(3, {'game/2': counts()})
        self.assertEqual(c.rows()[0]['passes'], 1, 'stale data cannot supply another passing batch')
        c.observe(4, {'game/1': counts(changed=0), 'game/2': counts()})
        self.assertFalse(c.rows()[0]['enough'], 'expired changed evidence must not qualify')
        self.assertEqual(c.rows()[0]['passes'], 0)
        self.assertEqual((c.phase, c.admitted_batch), ('predictor', 4),
                         'the sparse observed level no longer vetoes its qualified peer')
        c.observe(5, {'game/1': counts(), 'game/2': counts()})
        c.observe(6, {'game/1': counts(), 'game/2': counts()})
        self.assertEqual((c.phase, c.admitted_batch), ('predictor', 4))
        c.observe(7, {'game/1': counts(reconstruction_wrong=20), 'game/2': counts()})
        self.assertEqual(c.phase, 'predictor', 'admission latches; continuing quality stays visible')
        with self.assertRaisesRegex(ValueError, 'new batch'):
            c.observe(7, {})
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            c.observe(8, {'game/1': counts(drift_delta=float('nan'))})

    def test_sufficient_evidence_with_bad_or_undefined_metrics_still_blocks(self):
        for bad in (counts(reconstruction_wrong=10), counts(drift_delta=.04),
                    counts(reconstruction_union=0), counts(drift_scale=0)):
            c = self.controller()
            for batch in range(1, 6):
                c.observe(batch, {'game/1': counts(), 'game/2': bad})
            self.assertEqual(c.phase, 'representation')
            self.assertTrue(c.rows()[1]['eligible'])
            self.assertEqual(c.rows()[1]['passes'], 0)

    def test_native_unreached_levels_wait_but_first_levels_and_synthetics_are_required(self):
        catalog = [dict(s, source='sdk') for s in stages()]
        catalog += [dict(stages()[0], game='synthetic', source='synthetic')]
        c = SequencePhaseControl(catalog, {}, SequenceControlConfig(window_batches=2))
        c.observe(1, {'game/1': counts()})
        c.observe(2, {'game/1': counts()})
        self.assertEqual(c.phase, 'representation', 'synthetic evidence remains mandatory')
        self.assertFalse(c.rows()[2]['observed'])
        for batch in (3, 4):
            c.observe(batch, {'game/1': counts(), 'synthetic/1': counts()})
        self.assertEqual(c.phase, 'predictor')
        rows = {r['key']: r for r in c.rows()}
        self.assertTrue(rows['game/1']['required'])
        self.assertFalse(rows['game/2']['required'])
        self.assertFalse(rows['game/2']['passing'], 'unreached is not a pass')
        self.assertEqual(rows['game/2']['passes'], 0)

    def test_observed_native_level_remains_required_after_expiry_and_resume(self):
        catalog = [dict(s, source='sdk') for s in stages()]
        c = SequencePhaseControl(catalog, {}, SequenceControlConfig(window_batches=2))
        c.observe(1, {'game/1': counts(), 'game/2': counts(reconstruction_wrong=20)})
        c.observe(2, {})
        c.observe(3, {})
        c.observe(4, {'game/1': counts()})
        self.assertEqual(c.phase, 'representation')
        other = SequencePhaseControl(catalog, {}, c.config)
        other.restore(c.snapshot())
        self.assertTrue(other.rows()[1]['required'])
        self.assertTrue(other.rows()[1]['observed'])
        self.assertFalse(other.rows()[1]['eligible'], 'expired data is ignored, not a pass')
        self.assertEqual(other.rows()[1]['samples'], 0)
        other.observe(5, {'game/1': counts(), 'game/2': counts(reconstruction_wrong=20)})
        self.assertEqual(other.phase, 'representation', 'fresh sufficient data restores the veto')
        other.observe(6, {'game/1': counts(), 'game/2': counts()})
        other.observe(7, {'game/1': counts(), 'game/2': counts()})
        self.assertEqual(other.phase, 'representation', 'native levels still need patience')
        other.observe(8, {'game/1': counts(), 'game/2': counts()})
        self.assertEqual(other.phase, 'predictor')

    def test_observed_synthetic_coverage_survives_expiry_and_resume(self):
        c = self.controller(window_batches=1)
        c.observe(1, {'game/1': counts(), 'game/2': counts(changed=0)})
        c.observe(2, {})
        other = self.controller(window_batches=1)
        other.restore(c.snapshot())
        self.assertTrue(all(row['observed'] for row in other.rows()))
        self.assertFalse(any(row['eligible'] for row in other.rows()))
        for batch in (3, 4):
            other.observe(batch, {'game/1': counts()})
        self.assertEqual(other.phase, 'predictor', 'old observed sparse levels do not veto forever')
        self.assertFalse(other.rows()[1]['passing'])

    def test_legacy_snapshot_infers_coverage_without_altering_evidence(self):
        catalog = [dict(stages()[0], game=g, source='synthetic') for g in ('recent', 'confirmed')]
        catalog += [dict(stages()[0], game='native', source='sdk')]
        c = SequencePhaseControl(catalog, {}, SequenceControlConfig(window_batches=1))
        legacy = dict(version=1, config=asdict(c.config), phase='representation', admitted_batch=None,
                      window=[dict(batch=10, pairs={'recent/1': counts(changed=0)})],
                      streaks={'recent/1': 0, 'confirmed/1': 2, 'native/1': 0},
                      observed_native_levels=['native/1'])
        c.restore(legacy)
        self.assertTrue(all(row['observed'] for row in c.rows()))
        self.assertFalse(c.rows()[0]['passing'], 'coverage is not qualification')
        self.assertEqual(c.snapshot()['window'], legacy['window'])
        self.assertEqual(c.snapshot()['streaks'], legacy['streaks'])
        self.assertEqual(c.phase, 'representation')

    def test_exact_phase_rates_and_moments_survive_transition_and_restore(self):
        e = Engine(WorldModel(NetworkConfig.small()), TrainConfig())
        rep = OnlineRepresentation(e, 3e-5, encoder_learning_rate_after_predictor=3e-7)
        rep.update([session().transitions[0].before])
        moments = deepcopy(rep.optimizer.state_dict()['state'])
        c = self.controller()
        c.apply(rep)
        self.assertEqual(rep.learning_rates, dict(encoder=3e-5, decoder=3e-5))
        for batch in (1, 2):
            c.observe(batch, {'game/1': counts(), 'game/2': counts()})
        c.apply(rep)
        self.assertEqual(rep.learning_rates, dict(encoder=3e-7, decoder=3e-7))
        for key, row in rep.optimizer.state_dict()['state'].items():
            for name, value in row.items():
                torch.testing.assert_close(value, moments[key][name], rtol=0, atol=0)
        other = self.controller()
        other.restore(c.snapshot())
        self.assertEqual(other.report(), c.report())

    def test_changed_pixels_gate_and_same_observations_for_encoder_drift(self):
        e = Engine(WorldModel(NetworkConfig.small()), TrainConfig())
        # This test owns metric arithmetic, not encoder numerics. Exact features
        # avoid roundoff differences between batched and single-image forwards.
        e.encode = lambda observations: [
            torch.full((1, e.model.config.latent), float(o.final[0, 0]))
            for o in observations]
        rows = [session((1, 1))]
        actual = np.ones((1, 4, 4), np.uint8)
        actual[0, 0, 0] = 2
        rows[0].transitions[0] = replace(rows[0].transitions[0],
            after=replace(rows[0].transitions[0].after, frames=actual))
        e.decode = lambda z, h, w, **kw: torch.nn.functional.one_hot(
            torch.ones(len(z), h, w, dtype=torch.long), 16).float()
        probe = TrainingBatchDiagnostics(e, rows, 8)
        probe.start(1)
        evaluate_sessions(e, rows, diagnostics=probe)
        encode = e.encode
        e.encode = lambda observations: [z * 2 for z in encode(observations)]
        probe.after_representation()
        d, _ = probe.finish()
        self.assertEqual(d['reconstruction_error'], 0., 'old-frame global reconstruction is perfect')
        self.assertEqual(d['reconstruction_delta_error'], 1., 'missing the one real change is a total failure')
        self.assertAlmostEqual(d['latent_drift'], 1., places=6)
        self.assertEqual(probe.pairs['test/1']['changed'], 1)
        self.assertEqual(probe.pairs['test/1']['drift_samples'], 1)

    def test_representation_phase_does_not_update_dynamics(self):
        e = Engine(WorldModel(NetworkConfig.small()), TrainConfig())
        learner = SequenceLearner(e, SequenceConfig(session_steps=4, block_steps=2, representation_updates=2))
        rep = OnlineRepresentation(e, 3e-5)
        before = deepcopy(e.model.state_dict())
        result = fit_batch(learner, [session()], rep, train_predictor=False)
        self.assertEqual(result['updates'], 0)
        self.assertIsNone(result['loss'])
        self.assertEqual(learner.optimizer.state_dict()['state'], {})
        self.assertEqual(rep.updates, 2)
        for k, p in e.model.state_dict().items():
            if k.startswith('dynamics.'):
                torch.testing.assert_close(p, before[k], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(p, before[k]) for k, p in e.model.state_dict().items()
                            if k.startswith('encoder.')))

    def test_diagnostics_keep_rare_real_changes_and_cover_isolated_levels(self):
        e = Engine(WorldModel(NetworkConfig.small()), TrainConfig())
        a, b = session((1,) * 20), session((1,) * 20)
        a.transitions = [replace(s, level=2) for s in a.transitions]
        b.transitions = [replace(s, level=3) for s in b.transitions]
        a.transitions[7] = replace(a.transitions[7], after=session((1, 2)).transitions[0].after)
        probe = TrainingBatchDiagnostics(e, [a, b], 8)
        self.assertEqual(len(probe.selected), 8)
        self.assertIn((id(a), 7), probe.selected)
        self.assertEqual(sum(i == id(b) for i, _ in probe.selected), 4)
        with self.assertRaisesRegex(ValueError, 'cover every'):
            TrainingBatchDiagnostics(e, [a, b], 1)

    def test_all_catalog_window_allows_fresh_evidence_patience_without_hiding_bad_levels(self):
        from model.game_sequences import schedule_games
        plan = load_plan(Path(__file__).parents[1]/'sequence_all_synthetic.example.json')
        games = list(dict.fromkeys(s['game'] for s in plan.stages))
        c = SequenceConfig(**plan.payload['sequence'])
        self.assertEqual((len(games), len(plan.stages)), (71, 424))
        for window, bad, expected in ((18, False, 'predictor'), (4, False, 'representation'),
                                      (18, True, 'representation')):
            with self.subTest(window=window, bad=bad):
                controller = SequencePhaseControl(plan.stages, plan.settings,
                    SequenceControlConfig(window_batches=window, diagnostic_samples=384))
                for index in range(27):
                    batch = set(schedule_games(games, index, c, seed=plan.settings['seed']))
                    evidence = {f"{s['game']}/{s['level']}": counts(samples=8, changed=4, drift_samples=8,
                        reconstruction_wrong=20 if bad and s['game']=='sy60-v1' and s['level']==6 else 1)
                        for s in plan.stages if s['game'] in batch}
                    controller.observe(index + 1, evidence)
                    if index < 26:
                        self.assertEqual(controller.phase, 'representation', 'all pairs need three fresh visits')
                self.assertEqual(controller.phase, expected)

    def test_eight_representation_updates_include_every_full_or_tail_session(self):
        # The recorder observes which real frames fit_batch supplies; it does not select them.
        for size in (8, 7):
            with self.subTest(size=size):
                rows = [session((i, i + 1), game=f'g{i}', seed=i) for i in range(size)]
                seen = []
                def update(observations, **kwargs):
                    seen.append(int(observations[0].final[0, 0]))
                learner = SimpleNamespace(engine=SimpleNamespace(device=torch.device('cpu'), model=torch.nn.Identity()),
                    config=SequenceConfig(batch_games=8, representation_updates=8, game_schedule='shuffled'))
                fit_batch(learner, rows, SimpleNamespace(update=update), train_predictor=False)
                self.assertEqual(len(seen), 8)
                self.assertEqual(set(seen), set(range(size)))

    def test_configuration_rejects_invalid_control_and_decoder_rate(self):
        for options in ({'window_batches': 0}, {'diagnostic_samples': True},
                        {'decoder_learning_rate_after_predictor': float('nan')}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SequenceControlConfig(**options)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, values = config(root, sequence_control={})
            with self.assertRaisesRegex(ValueError, 'requires sequence'):
                load_plan(path)
            values['sequence'] = {}
            values['sequence_control']['decoder_learning_rate_after_predictor'] = .1
            path.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, 'decoder LR'):
                load_plan(path)

    def test_controlled_run_measures_current_batch_before_learning_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, values = config(root, [dict(source='synthetic', game='sc01-v1', levels=[1, 2])],
                sequence=dict(batch_games=4, session_steps=4, block_steps=2, level_steps=2, representation_updates=1),
                sequence_control=dict(diagnostic_samples=8),
                thresholds=dict(representation_open=1., latent_drift=1., patience=1,
                                minimum_samples=1, minimum_changed=1))
            values['settings']['actions'] = 32
            path.write_text(json.dumps(values))
            def args(name, resume=False, maximum=None):
                return SimpleNamespace(training_config=path, output=root/name, source=None, resume=resume,
                    device='cpu', compile=False, max_batches=maximum)
            events = []
            def collect(*a, **kw):
                self.assertFalse(kw.get('evaluation', False), 'no separate evaluation games')
                result = collect_batch(*a, **kw)
                events.append(('collect', id(result)))
                return result
            def evaluate(e, sessions, **kw):
                events.append(('measure', id(sessions)))
                return evaluate_sessions(e, sessions, **kw)
            def fit(l, sessions, *a, **kw):
                events.append(('fit', id(sessions)))
                return fit_batch(l, sessions, *a, **kw)
            with patch('model.sequence_training.collect_batch', side_effect=collect), \
                 patch('model.sequence_training.evaluate_sessions', side_effect=evaluate), \
                 patch('model.sequence_training.fit_batch', side_effect=fit):
                run(args('resumed', maximum=1))
                run(args('resumed', resume=True))
            self.assertEqual([name for name, _ in events], ['collect', 'measure', 'fit'] * 2)
            self.assertTrue(all(len({v for _, v in events[i:i+3]}) == 1 for i in (0, 3)))
            run(args('continuous'))
            a, b = [checkpoint.load(root/name/'latest.pt') for name in ('resumed', 'continuous')]
            self.assertEqual(a['sequence_control'], b['sequence_control'])
            self.assertEqual(a['progress'], b['progress'])
            for k in a['weights']:
                torch.testing.assert_close(a['weights'][k], b['weights'][k], rtol=0, atol=0)
            for key, state in a['representation_optimizer']['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, b['representation_optimizer']['state'][key][name], rtol=0, atol=0)
            report = json.loads((root/'resumed/report.json').read_text())
            self.assertEqual(report['measurement_source'], 'training_pre_update')
            self.assertEqual(report['phase_control']['phase'], 'predictor', report['phase_control'])
            self.assertEqual([r['phase'] for r in report['history']], ['representation', 'predictor'])
            self.assertEqual([r['updates'] for r in report['history']], [0, 2])
            self.assertEqual(report['history'][1]['learning_rates']['decoder'], 3e-7)

    def test_explicit_legacy_adoption_preserves_state_and_rejects_unrelated_plan_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, values = config(root, [dict(source='synthetic', game='sc01-v1', levels=[1])],
                                  sequence=dict(session_steps=4, block_steps=2))
            old = load_plan(path)
            e = Engine(WorldModel(NetworkConfig.small()), TrainConfig())
            learner, rep = SequenceLearner(e, SequenceConfig(**old.payload['sequence'])), representation_for(e, old)
            fit_batch(learner, [session()], rep)
            save_run(root/'old.pt', old, learner, rep, 5, 80)
            saved = checkpoint.load(root/'old.pt')
            values['sequence_control'] = {}
            values['settings']['encoder_learning_rate_after_predictor'] = 3e-8
            path.write_text(json.dumps(values))
            new = load_plan(path)
            c = SequencePhaseControl(new.stages, new.settings, SequenceControlConfig())
            # Without explicit adoption this remains a strict resume error.
            with self.assertRaisesRegex(ValueError, 'same sequence'):
                restore_run(saved, new, learner, rep, c)
            before = deepcopy(e.model.state_dict())
            self.assertEqual(restore_run(saved, new, learner, rep, c, adopt=True), (5, 80))
            self.assertEqual(c.phase, 'representation')
            self.assertEqual(learner.updates, saved['progress']['updates'])
            for key, value in e.model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            for owner, opt in (('predictor_optimizer', learner.optimizer), ('representation_optimizer', rep.optimizer)):
                for key, state in opt.state_dict()['state'].items():
                    for name, value in state.items():
                        torch.testing.assert_close(value, saved[owner]['state'][key][name], rtol=0, atol=0)
            values['settings']['learning_rate'] = 1e-4
            path.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, 'same sequence'):
                restore_run(saved, load_plan(path), learner, rep, c, adopt=True)
            save_run(root/'new.pt', new, learner, rep, 5, 80, c)
            with self.assertRaisesRegex(ValueError, 'legacy'):
                restore_run(checkpoint.load(root/'new.pt'), new, learner, rep, c, adopt=True)


if __name__ == '__main__':
    unittest.main()
