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
