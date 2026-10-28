"""Exact checkpoint export and publication lifetime, without GPU/model execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from model.lmex_sync import sync


class LMEXSyncTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.root = self.base / 'models'
        self.root.mkdir()
        self.architecture = self.base / 'architecture.json'
        self.architecture.write_bytes(b'{"schema_version": 1, "nodes": []}\n')
        self.checkpoint = self.base / 'latest.pt'
        self.public = self.root / 'arc3-delta-world-model'
        self.payload = {'network': {'width': 2, 'encoder_depth': 1, 'heads': 1, 'patch': 1},
                        'progress': {'actions': 64, 'updates': 4, 'episodes': 2, 'level_index': 0},
                        'weights': {'encoder.weight': torch.tensor([[1.25, -2.], [0., 8.]]),
                                    'decoder.bias': torch.tensor([.5, -.25], dtype=torch.bfloat16)},
                        'optimizers': [{'not_a_parameter': torch.tensor([999.])}]}
        self.save()

    def save(self):
        temporary = self.checkpoint.with_suffix('.tmp')
        torch.save(self.payload, temporary)
        temporary.replace(self.checkpoint)

    def publish(self):
        return sync(self.checkpoint, self.architecture, self.root)

    def test_exact_roundtrip_metadata_and_noop(self):
        report = self.publish()
        self.assertTrue(self.public.is_symlink())
        self.assertTrue(self.public.resolve().is_relative_to(self.root))
        self.assertEqual(report['actions'], 64)
        self.assertEqual(report['parameters'], 6)
        with safe_open(self.public / 'model.safetensors', framework='pt') as source:
            self.assertEqual(source.keys(), ['decoder.bias', 'encoder.weight'])
            self.assertEqual(source.get_tensor('encoder.weight').tolist(), [[1.25, -2.], [0., 8.]])
            self.assertEqual(source.get_tensor('decoder.bias').dtype, torch.bfloat16)
            self.assertEqual(source.get_tensor('decoder.bias').tolist(), [.5, -.25])
        config = json.loads((self.public / 'config.json').read_text())
        self.assertEqual(config['parameter_count'], 6)
        self.assertEqual(config['hidden_size'], 2)
        self.assertEqual(config['training_progress']['updates'], 4)
        self.assertEqual(config['name_or_path'], 'arc3-delta-world-model')
        self.assertEqual((self.public / 'architecture.json').read_bytes(), self.architecture.read_bytes())
        original = self.public.resolve()
        self.assertIsNone(self.publish())
        self.assertEqual(self.public.resolve(), original)

    def test_architecture_only_reuses_weights_and_replacement_refreshes(self):
        self.publish()
        original = self.public.resolve()
        weight_inode = (self.public / 'model.safetensors').stat().st_ino
        self.architecture.write_text('{"schema_version":1,"nodes":[],"name":"Edited"}')
        report = self.publish()
        self.assertFalse(report['weights_changed'])
        self.assertEqual((self.public / 'model.safetensors').stat().st_ino, weight_inode)
        self.assertFalse(original.exists())
        self.payload['progress']['actions'] = 96
        self.payload['weights']['encoder.weight'][0, 0] = 7
        self.save()  # same-shaped atomic checkpoint replacement
        report = self.publish()
        self.assertTrue(report['weights_changed'])
        self.assertEqual(report['actions'], 96)
        self.assertEqual(load_file(self.public / 'model.safetensors')['encoder.weight'][0, 0].item(), 7)
        packages = list(self.public.resolve().parent.glob('package-*'))
        self.assertEqual(packages, [self.public.resolve()])

    def test_portable_agent_exports_active_path_weights_and_keeps_world_names(self):
        self.payload = {'format': 'arc3-agent', 'version': 1, 'agent': {'mode': 'path_mpc'},
            'world': {'network': self.payload['network'], 'weights': self.payload['weights']},
            'components': {'path_evaluator': {'weights': {'readout.bias': torch.tensor([3.])}}},
            'outcome_value': {'weights': {'unused': torch.tensor([999.])}},
            'training_progress': {'actions': 2048, 'updates': 800, 'path_updates': 20}}
        self.save()
        report = self.publish()
        self.assertEqual(report['actions'], 2048)
        self.assertEqual(report['path_updates'], 20)
        tensors = load_file(self.public / 'model.safetensors')
        self.assertEqual(set(tensors), {'encoder.weight', 'decoder.bias', 'path_evaluator.readout.bias'})
        self.assertEqual(tensors['path_evaluator.readout.bias'].tolist(), [3.])
        self.assertEqual(tensors['encoder.weight'].tolist(), [[1.25, -2.], [0., 8.]])
        config = json.loads((self.public / 'config.json').read_text())
        self.assertEqual(config['architectures'], ['WorldModel', 'PathValue'])

    def test_sequence_export_includes_change_head_and_batch_progress(self):
        self.payload.update(kind='game_sequences', version=1,
            exploration={'version': 1, 'weights': {'layers.0.weight': torch.tensor([[.2, .8]])},
                         'optimizer': {'private': torch.tensor([999.])},
                         'replay_targets': torch.tensor([1.])})
        self.payload['progress']['batches'] = 12
        self.save()
        self.publish()
        tensors = load_file(self.public / 'model.safetensors')
        self.assertEqual(set(tensors), {'encoder.weight', 'decoder.bias', 'exploration.layers.0.weight'})
        torch.testing.assert_close(tensors['exploration.layers.0.weight'], torch.tensor([[.2, .8]]))
        config = json.loads((self.public / 'config.json').read_text())
        self.assertEqual(config['training_progress']['batches'], 12)
        self.assertEqual(config['architectures'], ['WorldModel', 'ChangeHead'])

    def planning_snapshot(self):
        from dataclasses import asdict
        from model.sequence_planning import SequencePlanningConfig, VisualGoal
        from model.path_planning import PathValue

        self.payload.update(kind='game_sequences', version=1)
        self.payload['network'].update(latent=2, memory=3)
        goal = VisualGoal(2, 3, 1, 4).state_dict()
        path = PathValue(2, 3, 4).state_dict()
        # Distinct, exactly representable values also detect accidental mixing.
        goal = {k: torch.full_like(v, .25, dtype=torch.bfloat16) for k, v in goal.items()}
        path = {k: torch.full_like(v, -.5) for k, v in path.items()}
        saved = dict(version=1, config=asdict(SequencePlanningConfig(width=4)), goal=goal, path=path,
                     goal_updates=12, path_updates=24, teacher_actions=128, learned_probability=.03125,
                     adopted_batch=10, rng=torch.tensor([99]),
                     goal_optimizer={'private': torch.tensor([999.])},
                     path_optimizer={'private': torch.tensor([999.])})
        self.payload['planning'] = saved
        self.save()
        return saved

    def test_sequence_planning_exports_exact_heads_and_public_progress(self):
        saved = self.planning_snapshot()
        self.publish()
        tensors = load_file(self.public / 'model.safetensors')
        expected = {**self.payload['weights'],
                    **{f'goal_planner.{k}': v for k, v in saved['goal'].items()},
                    **{f'path_evaluator.{k}': v for k, v in saved['path'].items()}}
        self.assertEqual(set(tensors), set(expected))
        for key, tensor in tensors.items():
            torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)
        config = json.loads((self.public / 'config.json').read_text())
        self.assertEqual(config['architectures'], ['WorldModel', 'VisualGoal', 'PathValue'])
        self.assertEqual(config['parameter_count'], sum(t.numel() for t in expected.values()))
        self.assertEqual(config['sequence_planning'], {k: saved[k] for k in (
            'version', 'config', 'goal_updates', 'path_updates', 'teacher_actions',
            'learned_probability', 'adopted_batch')})
        self.assertNotIn('torch_dtype', config)  # mixed dtypes are preserved

    def test_incomplete_or_unsupported_planning_keeps_previous_package(self):
        from copy import deepcopy
        valid = self.planning_snapshot()
        self.publish()
        original = self.public.resolve()
        for mutation in ('version', 'missing_path', 'missing_tensor', 'shape', 'empty', 'config'):
            with self.subTest(mutation=mutation):
                saved = deepcopy(valid)
                if mutation == 'version': saved['version'] = 99
                elif mutation == 'missing_path': del saved['path']
                elif mutation == 'missing_tensor': del saved['goal']['project.bias']
                elif mutation == 'shape': saved['path']['success.bias'] = torch.zeros(2)
                elif mutation == 'empty': saved['goal'] = {}
                else: del saved['config']
                self.payload['planning'] = saved
                self.save()
                with self.assertRaisesRegex(ValueError, 'planning'):
                    self.publish()
                self.assertEqual(self.public.resolve(), original)
                self.assertEqual(list(original.parent.glob('package-*')), [original])

    def test_producer_upgrade_reexports_unchanged_checkpoint_once(self):
        self.planning_snapshot()
        self.publish()
        # Simulate a package from the old producer, with the same source stamps.
        weights = self.public / 'model.safetensors'
        save_file(self.payload['weights'], str(weights))
        manifest_path = self.public / '.sync.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.pop('export_version', None)
        manifest_path.write_text(json.dumps(manifest))
        old_inode = weights.stat().st_ino
        report = self.publish()
        self.assertIsNotNone(report)
        self.assertTrue(report['weights_changed'])
        self.assertNotEqual(weights.stat().st_ino, old_inode)
        self.assertIn('goal_planner.project.weight', load_file(weights))
        self.assertIsNone(self.publish())

    def test_current_architecture_names_operations_and_binds_actual_modules(self):
        from model.config import NetworkConfig
        from model.networks import WorldModel
        from model.sequence_exploration import ChangeHead
        from model.sequence_planning import VisualGoal
        from model.path_planning import PathValue

        document = json.loads(Path('model/architecture.json').read_text())
        missing = [n['id'] for n in document['nodes']
                   if n['kind'] == 'operation' and not n.get('operation')]
        self.assertEqual(missing, [])
        for node in document['nodes']:
            self.assertEqual(len(node['ports']), len({p['id'] for p in node['ports']}), node['id'])
        network = json.loads(Path('model/spatial_adapter_config.example.json').read_text())['network']
        network['learnable_click_sigma'] = False
        with torch.device('meta'):
            expected = dict(WorldModel(NetworkConfig(**network)).state_dict())
            expected.update({f'exploration.{k}': v for k, v in ChangeHead().state_dict().items()})
            expected.update({f'goal_planner.{k}': v for k, v in VisualGoal(1024, 2048, 4, 64).state_dict().items()})
            expected.update({f'path_evaluator.{k}': v for k, v in PathValue(1024, 2048, 64).state_dict().items()})
        bindings = {p['name']: [d['value'] for d in p['shape']] for p in document['parameters']}
        self.assertEqual(bindings, {k: list(v.shape) for k, v in expected.items()})
        bound = {p for n in document['nodes'] for p in n['parameter_ids']}
        self.assertEqual(bound, {p['id'] for p in document['parameters']})
        click = next(n for n in document['nodes'] if n['id'] == 'predictor.latent.real.rotation_adapter.click_width')
        self.assertEqual(click['parameter_ids'], [])
        self.assertIn({'name': 'sigma_pixels', 'value': 1.5}, click['attributes'])

    def test_bad_input_preserves_last_complete_package(self):
        self.publish()
        original = self.public.resolve()
        self.payload['weights'] = {}
        self.save()
        with self.assertRaisesRegex(ValueError, 'state_dict'):
            self.publish()
        self.assertEqual(self.public.resolve(), original)
        self.assertEqual(load_file(self.public / 'model.safetensors')['encoder.weight'][0, 0].item(), 1.25)
        self.assertEqual(list(original.parent.glob('package-*')), [original])
        self.architecture.write_text('{"schema_version":')
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(self.public.resolve(), original)

    def test_changed_input_during_export_is_never_published(self):
        self.publish()
        original = self.public.resolve()
        self.payload['weights']['encoder.weight'][0, 0] = 5
        self.save()

        def concurrent_save(*args, **kwargs):
            save_file(*args, **kwargs)
            self.payload['weights']['encoder.weight'][0, 0] = 9
            self.save()

        with patch('model.lmex_sync.save_file', side_effect=concurrent_save):
            with self.assertRaisesRegex(RuntimeError, 'Inputs changed'):
                self.publish()
        self.assertEqual(self.public.resolve(), original)
        self.publish()
        self.assertEqual(load_file(self.public / 'model.safetensors')['encoder.weight'][0, 0].item(), 9)

    def test_unowned_destinations_are_not_overwritten(self):
        self.public.mkdir()
        important = self.public / 'keep.txt'
        important.write_text('untouched')
        with self.assertRaisesRegex(ValueError, 'Move the existing'):
            self.publish()
        self.assertEqual(important.read_text(), 'untouched')
        foreign = self.base / 'foreign'
        self.public.rename(foreign)
        self.public.symlink_to(foreign, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'unowned'):
            self.publish()
        self.assertEqual((foreign / 'keep.txt').read_text(), 'untouched')


if __name__ == '__main__':
    unittest.main()
