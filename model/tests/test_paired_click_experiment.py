"""Real-prefix fidelity and numerical contracts for the bounded A/B probe."""
from contextlib import redirect_stdout
import copy
from dataclasses import asdict, replace
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model import checkpoint
from model.config import NetworkConfig, SequenceConfig, TrainConfig
from model.data import Action, Observation
from model.game_sequences import GameSession, SequenceLearner, Transition
from model.learning import Engine
from model.networks import WorldModel
from model.paired_click_experiment import (replay_branch, seed_sets, counts, summarize,
    batches_for, encode_cache, cached_sessions, decision, run)


def observation(frame, terminal=False):
    return Observation(np.array([frame], dtype=np.uint8), (6,),
        'WIN' if terminal else 'NOT_FINISHED', int(terminal), 1)


class ClickEnvironment:
    def __init__(self, value=0):
        self.frame = np.full((4, 4), value, dtype=np.uint8)
        self.observation = observation(self.frame)
        self.calls = 0

    def step(self, action):
        self.calls += 1
        self.frame[action.data['y'], action.data['x']] = 1
        self.observation = observation(self.frame, self.calls == 2)
        return self.observation


class ClickFactory:
    def __init__(self, value=0):
        self.value = value
        self.seeds = []

    def make(self, seed, level, *, palette_seed):
        self.seeds.append((seed, level, palette_seed))
        return ClickEnvironment(self.value)


def fixture():
    factory = ClickFactory()
    env = factory.make(42, 1, palette_seed=42)
    steps = []
    for i, action in enumerate((Action(6, dict(x=0, y=0)), Action(6, dict(x=1, y=0)),
                                Action(6, dict(x=0, y=1)))):
        if env.observation.terminal:
            env = factory.make(1_000_042, 1, palette_seed=42)
        before = env.observation
        steps.append(Transition(before, action, env.step(action), i == 2, 1))
    return GameSession('test', 42, steps)


class PairedClickTests(unittest.TestCase):
    def test_fork_replays_resets_without_sibling_contamination(self):
        source, factory = fixture(), ClickFactory()
        a = replay_branch(factory, source, Action(6, dict(x=2, y=3)))
        b = replay_branch(factory, source, Action(6, dict(x=3, y=2)))
        self.assertEqual(factory.seeds, [(42,1,42),(1_000_042,1,42)]*2)
        self.assertIs(a.transitions[0], source.transitions[0])
        self.assertIs(a.transitions[-1].before, b.transitions[-1].before)
        for branch, location in ((source,(1,0)), (a,(3,2)), (b,(2,3))):
            expected = np.zeros((4,4), dtype=np.uint8); expected[location] = 1
            np.testing.assert_array_equal(branch.transitions[-1].after.final, expected)
        self.assertTrue(a.transitions[-1].reset_before)
        self.assertEqual(a.environment_actions, 3)

    def test_replay_rejects_mismatching_observation(self):
        with self.assertRaisesRegex(ValueError, 'Replay diverged before'):
            replay_branch(ClickFactory(3), fixture(), Action(6, dict(x=2,y=2)))

    def test_metrics_distinguish_omission_wrong_color_and_invention(self):
        row = summarize(counts(np.array([0,0,0,0,0]), np.array([1,1,1,0,0]),
                               np.array([1,0,2,1,0])))
        for key in ('correct','missed','color','invented'):
            self.assertEqual(row[key], 1)
        self.assertEqual(row['error'], .75)
        self.assertAlmostEqual(row['recall'], 1/3)
        self.assertEqual(row['invented_per_action'], 1)

    def test_seed_partitions_and_supervised_update_budgets(self):
        train, held = seed_sets(3, 2)
        a = {s for row in train.values() for s in row}
        b = {s for row in held.values() for s in row}
        self.assertFalse(a & b)
        self.assertTrue(all(s % 2 == 0 for s in a))
        self.assertTrue(all(s % 2 == 1 for s in b))
        data = dict(normal=list(range(16)), paired=[list(range(i,i+4)) for i in range(0,16,4)])
        for variant in ('normal', 'paired'):
            batches = list(batches_for(data, variant, 5, 8))
            self.assertEqual([len(batch) for batch in batches], [8]*5)
            self.assertEqual(batches, list(batches_for(data, variant, 5, 8)))

    def test_native_update_preserves_encoder_decoder_and_independent_memories(self):
        torch.set_num_threads(2); torch.manual_seed(7)
        engine = Engine(WorldModel(NetworkConfig.small()), TrainConfig())
        original = fixture()
        fork = replay_branch(ClickFactory(), original, Action(6,dict(x=2,y=2)))
        data = dict(normal=[original], paired=[[original,fork]], validation=[fork], ordinary=[], guards=[])
        cache = encode_cache(engine, data)
        rows = [original,fork]
        encoded = cached_sessions(engine, rows, cache)
        learner = SequenceLearner(engine, SequenceConfig(session_steps=3,block_steps=1,context_steps=2))
        memories = learner.prefix(encoded, 2)
        torch.testing.assert_close(memories[0], memories[1], rtol=0, atol=0)
        old = {k:v.clone() for k,v in engine.model.state_dict().items()}
        result = learner.update(encoded, 2, 3)
        self.assertEqual(result['examples'], 2)
        self.assertGreater(result['grad_norm'], 0)
        for key,value in engine.model.state_dict().items():
            if key.startswith(('encoder.','decoder.')):
                torch.testing.assert_close(value, old[key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(v,old[k]) for k,v in engine.model.state_dict().items()
                            if k.startswith('dynamics.')))

    def test_balancing_matches_effect_and_level_counts_but_preserves_pairs(self):
        data = dict(normal=[], paired=[])
        for level in range(1, 7):
            row = fixture(); step = row.transitions[-1]
            yes = replace(step, level=level)
            no = replace(step, level=level, after=step.before)
            pair = [GameSession('test',level,[yes]), GameSession('test',level,[no])]
            data['paired'].append(pair)
            data['normal'].extend(pair)
        batches = {variant:list(batches_for(data, variant, 7, 8))
                   for variant in ('normal_balanced','paired_balanced')}
        for normal, paired in zip(*batches.values()):
            self.assertEqual([s.transitions[-1].level_number for s in normal],
                             [s.transitions[-1].level_number for s in paired])
            for rows in (normal, paired):
                self.assertEqual(sum(np.count_nonzero(s.transitions[-1].after.final) > 0 for s in rows),4)
            for i in range(0,8,2):
                self.assertIs(paired[i].transitions[-1].before,paired[i+1].transitions[-1].before)

    def test_spatial_arm_preserves_adam_state_and_only_changes_all_adapter_rates(self):
        torch.set_num_threads(2); torch.manual_seed(7)
        config = replace(NetworkConfig.small(), patch=2, pixel_latent=True,
                         local_dynamics=True, rotation_adapter_width=4)
        engine = Engine(WorldModel(config), TrainConfig())
        learner = SequenceLearner(engine)
        # Distinct restored group rates catch accidental resets to constructor defaults.
        learner.optimizer.param_groups[1]['lr'] = 2e-5
        for parameter in engine.model.dynamics.parameters():
            parameter.grad = torch.full_like(parameter, .125)
        learner.optimizer.step()
        expected = {}
        names = {parameter: name for name, parameter in engine.model.dynamics.named_parameters()}
        for group in learner.optimizer.param_groups:
            options = {k: copy.deepcopy(v) for k, v in group.items() if k != 'params'}
            for parameter in group['params']:
                expected[names[parameter]] = (options, copy.deepcopy(learner.optimizer.state[parameter]))
        adapter = {name for name in expected if name.startswith('rotation_adapter.')}
        self.assertEqual(sum(name.endswith('.bias') for name in adapter), 4)
        self.assertTrue(any(not name.endswith('.bias') for name in adapter))
        self.assertTrue(all(state['step'].item() == 1 for _, state in expected.values()))
        self.assertTrue(all(torch.count_nonzero(state['exp_avg']) for _, state in expected.values()))

        source = fixture()
        branches = [source] + [replay_branch(ClickFactory(), source, Action(6, dict(x=x, y=2)))
                               for x in range(3)]
        data = dict(normal=[fixture() for _ in range(4)], paired=[branches], validation=branches,
                    ordinary=[source], guards=[source], counters={}, composition={})
        observed = []
        test = self

        class ObservedLearner(SequenceLearner):
            def update(self, *args, **kwargs):
                spatial_arm = len(observed) == 2
                current_names = {p: n for n, p in self.engine.model.dynamics.named_parameters()}
                seen = []
                for group in self.optimizer.param_groups:
                    for parameter in group['params']:
                        name = current_names[parameter]
                        seen.append(name)
                        options, state = expected[name]
                        target = dict(options, lr=1e-4) if spatial_arm and name in adapter else options
                        with test.subTest(arm=len(observed), parameter=name):
                            test.assertEqual({k: v for k, v in group.items() if k != 'params'}, target)
                            test.assertEqual(set(self.optimizer.state[parameter]), set(state))
                            for key, value in state.items():
                                torch.testing.assert_close(self.optimizer.state[parameter][key], value,
                                                           rtol=0, atol=0)
                test.assertCountEqual(seen, expected)  # No lost or duplicate parameters.
                result = super().update(*args, **kwargs)
                observed.append(result['examples'])
                return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root/'source.pt'
            checkpoint.write(source_path, dict(network=asdict(config), training=asdict(engine.config),
                weights=engine.model.state_dict(), predictor_optimizer=learner.optimizer.state_dict(),
                progress=dict(updates=1)))
            original_bytes = source_path.read_bytes()
            args = SimpleNamespace(source=source_path, output=root/'probe', updates=1, cpu_threads=2,
                train_scenes=1, validation_scenes=1, context=2, batch_size=4, device='cpu',
                balanced=False, spatial_lr=True)
            # Collection has separate real-prefix coverage; run the actual optimizer/update path.
            with patch('model.paired_click_experiment.collect_data', return_value=data), \
                 patch('model.paired_click_experiment.SequenceLearner', ObservedLearner), \
                 redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertEqual(source_path.read_bytes(), original_bytes)
        self.assertEqual(observed, [4, 4, 4])
        self.assertTrue(all(arm['frozen_representation'] for arm in report['trials'].values()))

    def test_adoption_gate_rejects_new_errors_outside_paired_examples(self):
        import copy
        baseline = dict(paired=dict(all=dict(recall=.2,error=.8,invented_per_action=2),
            levels={str(i):dict(error=.8) for i in range(1,7)}),
            ordinary=dict(all=dict(error=.7)), guards=dict(all=dict(error=.4)))
        candidate = copy.deepcopy(baseline)
        candidate['paired']['all'].update(recall=.3,error=.7)
        for row in candidate['paired']['levels'].values(): row['error']=.7
        self.assertTrue(decision(baseline,candidate)['passed'])
        stronger_source = copy.deepcopy(baseline)
        stronger_source['paired']['all']['recall']=.35
        self.assertFalse(decision(baseline,candidate,stronger_source)['passed'])
        candidate['ordinary']['all']['error']=.72
        self.assertFalse(decision(baseline,candidate)['passed'])
        candidate['ordinary']['all']['error']=.7
        candidate['guards']['all']['error']=.43
        self.assertFalse(decision(baseline,candidate)['passed'])


if __name__ == '__main__':
    unittest.main()
