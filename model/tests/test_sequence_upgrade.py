"""Function-preserving spatial growth and explicit exploration migration."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
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
from model.networks import WorldModel
from model.sequence_exploration import ChangeExplorer, ExplorationSession, SequenceExplorationConfig
from model.sequence_training import load_engine, representation_for, restore_run, run
from model.sequence_upgrade import load_upgraded_weights, upgraded_network, upgraded_optimizer
from model.spatial_adapter import RotationAdapter, click_maps
from model.tests.test_sequence_exploration import board
from model.tests.test_training_config import config
from model.training_config import load_plan


torch.set_num_threads(2)


def local_config(**kwargs):
    return replace(NetworkConfig.small(), latent=64, pixel_latent=True,
                   local_dynamics=True, residual_scale=.1, **kwargs)


class SequenceUpgradeTests(unittest.TestCase):
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
            saved['plan']['initialization']['network'].pop('rotation_adapter_width')
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


if __name__ == '__main__':
    unittest.main()
