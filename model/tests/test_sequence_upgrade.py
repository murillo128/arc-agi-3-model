"""Function-preserving spatial growth and explicit exploration migration."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import torch

from model import checkpoint
from model.config import NetworkConfig, SequenceConfig
from model.data import Action
from model.game_sequences import SequenceLearner
from model.learning import optimizer
from model.networks import WorldModel
from model.sequence_exploration import ChangeExplorer, ExplorationSession, SequenceExplorationConfig
from model.sequence_training import load_engine, representation_for, restore_run, run
from model.sequence_upgrade import load_upgraded_weights, upgraded_network, upgraded_optimizer, upgrade_plan_matches
from model.spatial_adapter import GaussianClickWidth, RotationAdapter, click_maps
from model.tests.test_sequence_exploration import board
from model.tests.test_training_config import config
from model.training_config import load_plan


torch.set_num_threads(2)


def local_config(**kwargs):
    return replace(NetworkConfig.small(), latent=64, pixel_latent=True,
                   local_dynamics=True, residual_scale=.1, **kwargs)


class SequenceUpgradeTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for scalar/tensor kernel regression')
    def test_initial_learned_width_matches_the_original_cuda_gaussian(self):
        action = torch.tensor([Action(6, dict(x=0, y=0)).features()], device='cuda')
        width = GaussianClickWidth().cuda()
        actual = click_maps(action, 8, 8, 8, 8, width())[0, 0]
        # Original fixed-width kernel at squared distances 0, 4**2+1, 5**2.
        # CUDA's tensor division can round differently from scalar division.
        expected = torch.exp(-torch.tensor([0., 17., 25.], device='cuda') / 4.5)
        torch.testing.assert_close(torch.stack((actual[0, 0], actual[1, 4], actual[0, 5])),
                                   expected, rtol=0, atol=0)

    def test_learned_gaussian_width_preserves_center_masks_and_has_finite_gradients(self):
        width = GaussianClickWidth()
        self.assertEqual(float(width()), 1.5)
        actions = torch.tensor([Action(6, dict(x=3, y=1)).features(),
                                Action(6, dict(x=5, y=1)).features(),
                                Action(6, dict(x=3)).features(), Action(4).features()])
        fixed = click_maps(actions, 3, 5, 4, 8)
        learned = click_maps(actions, 3, 5, 4, 8, width())
        torch.testing.assert_close(fixed, learned, atol=0, rtol=0)
        self.assertAlmostEqual(float(learned[0, 0, 1, 2]), math.exp(-1/4.5), places=7)
        self.assertAlmostEqual(float(learned[0, 0, 1, 1]), math.exp(-4/4.5), places=7)
        center_gradient = torch.autograd.grad(learned[0, 0, 1, 3], width.log_scale, retain_graph=True)[0]
        self.assertEqual(float(center_gradient), 0)
        learned[0, 0, 1, 2].backward()
        self.assertGreater(float(width.log_scale.grad), 0)
        self.assertTrue(torch.isfinite(width.log_scale.grad).all())
        with torch.no_grad():
            width.log_scale.fill_(math.atanh(.5))  # Doubles sigma to 3 pixels.
        wider = click_maps(actions, 3, 5, 4, 8, width())
        self.assertAlmostEqual(float(wider[0, 0, 1, 1]), math.exp(-4/18), places=7)
        self.assertEqual(float(wider[0, 0, 1, 3]), 1)
        self.assertEqual(int(wider[0, 0].argmax()), 11)
        torch.testing.assert_close(wider[:, 1:], fixed[:, 1:], atol=0, rtol=0)
        self.assertEqual(int(torch.count_nonzero(wider[1:])), 0)
        self.assertEqual(int(torch.count_nonzero(wider[:, :, 3:, :])), 0)
        self.assertEqual(int(torch.count_nonzero(wider[:, :, :, 5:])), 0)
        for raw, expected in ((-100., .375), (100., 6.)):
            with torch.no_grad():
                width.log_scale.fill_(raw)
            self.assertAlmostEqual(float(width()), expected, places=6)
            self.assertTrue(torch.isfinite(click_maps(actions, 3, 5, 4, 8, width())).all())
        with self.assertRaisesRegex(ValueError, 'requires a rotation adapter'):
            local_config(learnable_click_sigma=True)
        with self.assertRaisesRegex(ValueError, 'booleans'):
            local_config(rotation_adapter_width=2, learnable_click_sigma=1)

    def test_click_width_upgrade_preserves_trained_adapter_and_adam_then_learns(self):
        old = WorldModel(local_config(rotation_adapter_width=2))
        with torch.no_grad():
            old.dynamics.rotation_adapter.output.weight.normal_(0, .1)
        adam = optimizer(old.dynamics.parameters(), .001, False)
        sum(p.square().sum() for p in old.dynamics.parameters()).backward()
        adam.step()
        saved = dict(network=asdict(old.config), weights=deepcopy(old.state_dict()),
                     predictor_optimizer=adam.state_dict(), progress=dict(updates=1))
        saved['network'].pop('learnable_click_sigma')  # Legacy fixed-width model.
        target = replace(old.config, learnable_click_sigma=True)
        self.assertEqual(upgraded_network(saved, dict(network=asdict(target))), target)
        previous = dict(initialization=dict(network=saved['network']))
        current = dict(initialization=dict(network=asdict(target)))
        self.assertFalse(upgrade_plan_matches(previous, current))
        self.assertTrue(upgrade_plan_matches(previous, current, upgrade_predictor=True))
        self.assertFalse(upgrade_plan_matches(current, previous, upgrade_predictor=True))
        for invalid in (replace(target, memory=64), replace(target, rotation_adapter_width=3)):
            with self.assertRaisesRegex(ValueError, 'only enable'):
                upgraded_network(saved, dict(network=asdict(invalid)))
        new = WorldModel(target)
        load_upgraded_weights(new, saved)
        for name, value in saved['weights'].items():
            torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
        migrated = optimizer(new.dynamics.parameters(), .001, False)
        migrated.load_state_dict(upgraded_optimizer(saved, new.dynamics))
        parameters = dict(new.dynamics.named_parameters())
        for name, parameter in old.dynamics.named_parameters():
            for field, value in adam.state[parameter].items():
                torch.testing.assert_close(migrated.state[parameters[name]][field], value, rtol=0, atol=0)
        scalar = new.dynamics.rotation_adapter.click_width.log_scale
        self.assertNotIn(scalar, migrated.state)
        group = next(g for g in migrated.param_groups if any(p is scalar for p in g['params']))
        self.assertEqual(group['weight_decay'], 0.)
        self.assertEqual(group['lr'], .001)
        z = torch.randn(2, 4, 64); memory = torch.randn(2, 32)
        action = torch.tensor([Action(6, dict(x=3, y=2)).features(), Action(4).features()])
        for training in (False, True):
            old.train(training); new.train(training)
            with torch.no_grad():
                for left, right in zip(old.dynamics(z, memory, action, 5, 7),
                                       new.dynamics(z, memory, action, 5, 7)):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
        migrated.zero_grad(set_to_none=True)
        new.dynamics(z, memory, action, 5, 7)[0].square().mean().backward()
        self.assertGreater(float(scalar.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(scalar.grad).all())
        migrated.step()
        self.assertNotEqual(float(scalar), 0.)
        self.assertNotEqual(float(new.dynamics.rotation_adapter.click_sigma()), 1.5)
        broken = dict(saved, weights=dict(saved['weights']))
        broken['weights'].pop('dynamics.rotation_adapter.output.weight')
        with self.assertRaisesRegex(ValueError, 'every existing weight'):
            load_upgraded_weights(WorldModel(target), broken)

    def test_undo_is_excluded_before_every_policy_without_mutating_available_actions(self):
        before = board(actions=(1, 6, 7))
        for policy in ('uniform', 'mixed', 'learned'):
            c = SequenceExplorationConfig(policy=policy, allow_undo=False)
            collector = ExplorationSession(c, 5, ChangeExplorer(c) if policy == 'learned' else None)
            rng = random.Random(22)
            actions = [collector.action(before, rng) for _ in range(300)]
            self.assertEqual({a.id for a in actions}, {1, 6})
            self.assertEqual(before.available_actions, (1, 6, 7))
            with self.assertRaisesRegex(ValueError, 'excluding Undo'):
                collector.action(replace(before, available_actions=(7,)), rng)
        with self.assertRaises(ValueError):
            SequenceExplorationConfig(allow_undo=0)

    def test_native_click_maps_preserve_coordinates_and_mask_absence_and_padding(self):
        actions = torch.tensor([Action(6, dict(x=3, y=1)).features(),
                                Action(6, dict(x=63, y=1)).features(), Action(1).features()])
        maps = click_maps(actions, 3, 5, 4, 8)
        self.assertEqual(maps.shape, (3, 3, 4, 8))
        torch.testing.assert_close(maps[0, :, 1, 3], torch.tensor([1., 0., 0.]), atol=1e-7, rtol=0)
        self.assertEqual(int(maps[0, 0].argmax()), 11)  # Native (x=3,y=1), padded stride 8.
        self.assertAlmostEqual(float(maps[0, 1, 1, 4]), 1/63, places=7)
        self.assertEqual(int(torch.count_nonzero(maps[1:])), 0)
        self.assertEqual(int(torch.count_nonzero(maps[:, :, 3:, :])), 0)
        self.assertEqual(int(torch.count_nonzero(maps[:, :, :, 5:])), 0)

    def test_native_click_maps_reject_the_first_pixel_outside_each_frame_edge(self):
        # SDK clicks use integer 0..63 coordinates even on smaller native frames.
        for size in range(1, 64):
            for axis in ('x', 'y'):
                with self.subTest(size=size, axis=axis):
                    height, width = (3, size) if axis == 'x' else (size, 3)
                    outside = dict(x=0, y=0); outside[axis] = size
                    inside = dict(x=0, y=0); inside[axis] = size - 1
                    actions = torch.tensor([Action(6, outside).features(), Action(6, inside).features()])
                    maps = click_maps(actions, height, width, height + 1, width + 1)
                    self.assertEqual(int(torch.count_nonzero(maps[0])), 0)
                    torch.testing.assert_close(maps[1, :, inside['y'], inside['x']],
                                               torch.tensor([1., 0., 0.]), rtol=0, atol=0)

    def test_filter_bank_has_four_distinct_shared_orientations(self):
        adapter = RotationAdapter(local_config(rotation_adapter_width=1))
        with torch.no_grad():
            adapter.filters.weight.zero_()
            adapter.filters.weight[0, 0, 0, 2] = 1
        image = torch.zeros(1, 4, 9, 9); image[0, 0, 4, 4] = 1
        actual = adapter.oriented_features(image)
        expected = torch.zeros(1, 4, 9, 9)
        for k, (y, x) in enumerate(((6, 4), (4, 6), (2, 4), (4, 2))):
            expected[0, k, y, x] = 1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_zero_output_preserves_predictions_memory_and_decoder_then_learns(self):
        old = WorldModel(local_config())
        # Exercise a genuinely nonzero existing local predictor.
        with torch.no_grad():
            old.dynamics.local.output.weight.normal_(0, .1)
        saved = dict(network=asdict(old.config), weights=old.state_dict())
        new = WorldModel(replace(old.config, rotation_adapter_width=2))
        load_upgraded_weights(new, saved)
        z = torch.randn(2, 4, 64); memory = torch.randn(2, 32)
        action = torch.tensor([Action(6, dict(x=3, y=2)).features(), Action(4).features()])
        for training in (False, True):
            old.train(training); new.train(training)
            with torch.no_grad():
                a, b = old.dynamics(z, memory, action, 5, 7), new.dynamics(z, memory, action, 5, 7)
                for left, right in zip(a, b):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
                torch.testing.assert_close(old.decoder(a[0], 5, 7), new.decoder(b[0], 5, 7), atol=0, rtol=0)
        optimizer = torch.optim.AdamW(new.dynamics.parameters(), lr=.001)
        target = torch.randn_like(z)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = (new.dynamics(z, memory, action, 5, 7)[0] - target).square().mean()
            loss.backward()
            adapter = new.dynamics.rotation_adapter
            self.assertGreater(float(adapter.output.weight.grad.abs().sum()), 0)
            if step:
                for p in (adapter.filters.weight, adapter.click.weight, adapter.condition.weight):
                    self.assertGreater(float(p.grad.abs().sum()), 0)
                    self.assertTrue(torch.isfinite(p.grad).all())
            optimizer.step()

    def test_strict_weight_and_network_guards(self):
        old = WorldModel(local_config())
        saved = dict(network=asdict(old.config), weights=old.state_dict())
        target = replace(old.config, rotation_adapter_width=2)
        self.assertEqual(upgraded_network(saved, dict(network=asdict(target))), target)
        with self.assertRaisesRegex(ValueError, 'only enable'):
            upgraded_network(saved, dict(network=asdict(replace(target, memory=64))))
        broken = dict(saved, weights=dict(saved['weights']))
        broken['weights'].pop('decoder.color.bias')
        with self.assertRaisesRegex(ValueError, 'every existing weight'):
            load_upgraded_weights(WorldModel(target), broken)
        for width in (-1, True, 65):
            with self.assertRaises(ValueError):
                replace(old.config, rotation_adapter_width=width)
        with self.assertRaisesRegex(ValueError, 'local dynamics'):
            replace(NetworkConfig.small(), rotation_adapter_width=2)

    def test_optimizer_upgrade_rejects_incomplete_trained_adam_state(self):
        old = WorldModel(local_config())
        adam = optimizer(old.dynamics.parameters(), .001, False)
        # Populate every old parameter's real Adam state with one known update.
        sum(p.square().sum() for p in old.dynamics.parameters()).backward()
        adam.step()
        saved = dict(network=asdict(old.config), predictor_optimizer=adam.state_dict(),
                     progress=dict(updates=1))
        target = WorldModel(replace(old.config, rotation_adapter_width=2)).dynamics
        identity = saved['predictor_optimizer']['param_groups'][0]['params'][0]
        for missing in ('entry', 'all_entries', 'step', 'exp_avg', 'exp_avg_sq'):
            with self.subTest(missing=missing):
                broken = deepcopy(saved)
                states = broken['predictor_optimizer']['state']
                if missing == 'entry':
                    del states[identity]
                elif missing == 'all_entries':
                    states.clear()
                else:
                    del states[identity][missing]
                with self.assertRaisesRegex(ValueError, 'Missing predictor (optimizer state|AdamW fields)'):
                    upgraded_optimizer(broken, target)
        for field in ('step', 'exp_avg', 'exp_avg_sq'):
            with self.subTest(malformed=field):
                broken = deepcopy(saved)
                broken['predictor_optimizer']['state'][identity][field] = torch.zeros(2)
                with self.assertRaisesRegex(ValueError, 'Predictor optimizer (step|shape)'):
                    upgraded_optimizer(broken, target)
        broken = deepcopy(saved)
        broken['predictor_optimizer']['param_groups'][0]['amsgrad'] = True
        with self.assertRaisesRegex(ValueError, 'Missing predictor AdamW fields'):
            upgraded_optimizer(broken, target)

    def test_optimizer_upgrade_allows_empty_state_only_before_the_first_update(self):
        old = WorldModel(local_config())
        saved = dict(network=asdict(old.config),
                     predictor_optimizer=optimizer(old.dynamics.parameters(), .001, False).state_dict(),
                     progress=dict(updates=0))
        target = WorldModel(replace(old.config, rotation_adapter_width=2)).dynamics
        migrated = optimizer(target.parameters(), .001, False)
        migrated.load_state_dict(upgraded_optimizer(saved, target))
        self.assertFalse(migrated.state)
        self.assertEqual({id(p) for group in migrated.param_groups for p in group['params']},
                         {id(p) for p in target.parameters()})
        saved['progress']['updates'] = 1
        with self.assertRaisesRegex(ValueError, 'Missing predictor optimizer state'):
            upgraded_optimizer(saved, target)

    def test_real_checkpoint_upgrade_preserves_adam_then_saves_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, value = config(root, [dict(source='synthetic', game='bf04-v1', levels=[1, 2])],
                sequence=dict(batch_games=1, session_steps=6, block_steps=3, context_steps=1,
                              level_steps=3, representation_updates=1),
                sequence_exploration=dict(policy='learned', updates=1))
            value['settings']['actions'] = 1000; path.write_text(json.dumps(value))
            initial_path = root/'initial.json'
            initial = json.loads(initial_path.read_text()); initial['network'] = asdict(local_config())
            initial_path.write_text(json.dumps(initial))
            args = SimpleNamespace(training_config=path, output=root/'run', source=None, resume=False,
                                   device='cpu', compile=False, max_batches=1)
            run(args)
            saved = checkpoint.load(root/'run/latest.pt')
            self.assertTrue(saved['predictor_optimizer']['state'])
            self.assertTrue(saved['representation_optimizer']['state'])
            # An actual legacy checkpoint lacks both new default-valued fields.
            saved['network'].pop('rotation_adapter_width')
            saved['network'].pop('learnable_click_sigma')
            saved['plan']['initialization']['network'].pop('rotation_adapter_width')
            saved['plan']['initialization']['network'].pop('learnable_click_sigma')
            saved['plan']['sequence_exploration'].pop('allow_undo')
            saved['exploration']['config'].pop('allow_undo')
            checkpoint.write(root/'run/latest.pt', saved)
            legacy = load_plan(path)
            legacy_engine = load_engine(legacy, root/'run/latest.pt')
            legacy_learner = SequenceLearner(legacy_engine, SequenceConfig(**legacy.payload['sequence']))
            legacy_rep = representation_for(legacy_engine, legacy)
            old_explorer = ChangeExplorer(SequenceExplorationConfig(**legacy.payload['sequence_exploration']))
            restore_run(saved, legacy, legacy_learner, legacy_rep, explorer=old_explorer)

            initial['network']['rotation_adapter_width'] = 2; initial_path.write_text(json.dumps(initial))
            value['sequence_exploration']['allow_undo'] = False; path.write_text(json.dumps(value))
            plan = load_plan(path)
            engine = load_engine(plan, root/'run/latest.pt', upgrade_predictor=True)
            learner = SequenceLearner(engine, SequenceConfig(**plan.payload['sequence']))
            rep = representation_for(engine, plan)
            explorer = ChangeExplorer(SequenceExplorationConfig(**plan.payload['sequence_exploration']))
            with self.assertRaisesRegex(ValueError, 'same sequence'):
                restore_run(saved, plan, learner, rep, explorer=explorer)
            for section, key, changed in (('settings', 'learning_rate', .002), ('sequence', 'session_steps', 8)):
                bad = deepcopy(plan.payload); bad[section][key] = changed
                with self.assertRaisesRegex(ValueError, 'same sequence'):
                    restore_run(saved, replace(plan, payload=bad), learner, rep, explorer=explorer,
                                disable_undo=True, upgrade_predictor=True)
            counters = restore_run(saved, plan, learner, rep, explorer=explorer,
                                   disable_undo=True, upgrade_predictor=True)
            self.assertEqual(counters, (1, saved['progress']['actions']))
            torch.testing.assert_close(torch.get_rng_state(), saved['torch_rng'], rtol=0, atol=0)
            for name, tensor in saved['weights'].items():
                torch.testing.assert_close(engine.model.state_dict()[name], tensor, rtol=0, atol=0)
            for module, left, right in ((legacy_engine.model.dynamics, legacy_learner.optimizer, learner.optimizer),
                                       (legacy_engine.model, legacy_rep.optimizer, rep.optimizer)):
                current_parameters = dict(engine.model.dynamics.named_parameters() if module is legacy_engine.model.dynamics
                                          else engine.model.named_parameters())
                for name, old_parameter in module.named_parameters():
                    if old_parameter not in left.state:
                        continue
                    for key, tensor in left.state[old_parameter].items():
                        torch.testing.assert_close(tensor, right.state[current_parameters[name]][key], rtol=0, atol=0)
            for name, p in engine.model.dynamics.named_parameters():
                if name.startswith('rotation_adapter.'):
                    self.assertNotIn(p, learner.optimizer.state)
            for name, tensor in explorer.head.state_dict().items():
                torch.testing.assert_close(tensor, saved['exploration']['weights'][name], rtol=0, atol=0)
            self.assertEqual(explorer.updates, saved['exploration']['updates'])
            self.assertTrue(all(not row[0][40] for row in explorer.replay))
            broken = dict(saved, predictor_optimizer=deepcopy(saved['predictor_optimizer']))
            broken['predictor_optimizer']['param_groups'][0]['params'].pop()
            with self.assertRaisesRegex(ValueError, 'size mismatch|parameter IDs'):
                upgraded_optimizer(broken, engine.model.dynamics)

            args.resume = args.disable_undo = args.upgrade_predictor = True
            run(args)
            migrated = checkpoint.load(root/'run/latest.pt')
            self.assertEqual(migrated['predictor_adaptations'][0]['batch'], 1)
            self.assertGreater(float(migrated['weights']['dynamics.rotation_adapter.output.weight'].abs().sum()), 0)
            args.disable_undo = args.upgrade_predictor = False
            run(args)
            report = json.loads((root/'run/report.json').read_text())
            self.assertEqual([r['batch'] for r in report['history']], [2, 3])
            self.assertEqual(report['history'][-1]['exploration']['undo_actions'], 0)
            self.assertEqual(len(report['predictor_adaptations']), 1)
            self.assertEqual(report['network']['rotation_adapter_width'], 2)
            self.assertTrue((root/'run/report-before-predictor-adaptation-1.json').exists())

            initial['network']['learnable_click_sigma'] = True
            initial_path.write_text(json.dumps(initial))
            args.upgrade_predictor = True
            run(args)
            with_width = checkpoint.load(root/'run/latest.pt')
            name = 'dynamics.rotation_adapter.click_width.log_scale'
            self.assertNotEqual(float(with_width['weights'][name]), 0.)
            loaded = load_engine(load_plan(path), root/'run/latest.pt')
            torch.testing.assert_close(loaded.model.state_dict()[name], with_width['weights'][name], atol=0, rtol=0)
            state = with_width['predictor_optimizer']
            scalar_id = state['param_groups'][1]['params'][-1]
            previous_step = float(state['state'][scalar_id]['step'])
            args.upgrade_predictor = False
            run(args)
            resumed = checkpoint.load(root/'run/latest.pt')
            self.assertGreater(float(resumed['predictor_optimizer']['state'][scalar_id]['step']), previous_step)
            report = json.loads((root/'run/report.json').read_text())
            self.assertEqual([r['batch'] for r in report['history']], [4, 5])
            self.assertTrue(report['click_map']['learnable'])
            self.assertEqual(report['history'][-1]['click_sigma_pixels'], report['click_map']['sigma_pixels'])
            self.assertEqual(report['predictor_adaptations'][-1]['batch'], 3)
            self.assertTrue(report['predictor_adaptations'][-1]['learnable_click_sigma'])
            loaded = load_engine(load_plan(path), root/'run/latest.pt')
            self.assertEqual(float(loaded.model.dynamics.rotation_adapter.click_sigma()), report['click_map']['sigma_pixels'])


if __name__ == '__main__':
    unittest.main()
