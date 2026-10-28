"""Portable exports contain only inference state and strictly load native shapes."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from model.adaptation_checkpoint import export, load, from_payload, PURPOSE
from model.adaptation_heads import VisualGoal
from model.config import NetworkConfig, TrainConfig
from model.networks import WorldModel
from model.path_planning import PathValue


def sequence_snapshot():
    config = replace(NetworkConfig.small(), patch=2, latent=16, pixel_latent=True,
                     local_dynamics=True, rotation_adapter_width=4)
    world = WorldModel(config)
    goal = VisualGoal(config.latent, config.memory, config.patch, 8)
    path = PathValue(config.latent, config.memory, 8)
    return dict(kind='game_sequences', version=1, network=asdict(config), training=asdict(TrainConfig()),
                weights=world.state_dict(), planning=dict(version=1, config=dict(width=8),
                goal=goal.state_dict(), path=path.state_dict(), optimizer={'secret': 'discard me'},
                rng='discard me'), progress=dict(batches=8246), predictor_optimizer={'secret': 'discard me'},
                run='/private/training/path', torch_rng=torch.get_rng_state())


class AdaptationCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_export_roundtrip_excludes_training_data_and_preserves_source(self):
        torch.manual_seed(5)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.pt', Path(directory) / 'base.pt'
            snapshot = sequence_snapshot()
            torch.save(snapshot, source)
            source_before = source.read_bytes()
            report = export(source, output)
            self.assertEqual(source.read_bytes(), source_before)
            payload = torch.load(output, weights_only=True)
            self.assertEqual(set(payload), {'format', 'version', 'world', 'goal', 'path', 'metadata'})
            self.assertEqual(set(payload['world']), {'network', 'learning', 'weights'})
            self.assertNotIn('/private', str(payload['metadata']))
            self.assertEqual(report['source_batches'], 8246)
            self.assertEqual(report['purpose'], PURPOSE)
            rng = torch.get_rng_state().clone()
            bundle = load(output)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertIsNone(bundle.change)
            for key, value in bundle.engine.model.state_dict().items():
                self.assertTrue(torch.equal(snapshot['weights'][key], value), key)
            for key, value in bundle.goal.state_dict().items():
                self.assertTrue(torch.equal(snapshot['planning']['goal'][key], value), key)
            for key, value in bundle.path.state_dict().items():
                self.assertTrue(torch.equal(snapshot['planning']['path'][key], value), key)
            self.assertTrue(all(not p.requires_grad for p in bundle.engine.model.parameters()))
            self.assertEqual(bundle.engine.model.config.rotation_adapter_width, 4)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                export(source, source)

    def test_malformed_schema_shapes_and_nonfinite_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.pt', Path(directory) / 'base.pt'
            torch.save(sequence_snapshot(), source)
            export(source, output)
            original = torch.load(output, weights_only=True)
            mutations = [
                lambda p: p.update(rng=[1, 2]),
                lambda p: p['world']['network'].update(unrecognized=True),
                lambda p: p['world']['network'].update(residual_scale=True),
                lambda p: p['world']['learning'].update(false_change_weight=True),
                lambda p: p['world']['learning'].update(frame_chunk=1.5),
                lambda p: p['goal']['config'].update(latent=999),
                lambda p: p['path']['config'].update(objective_version=2),
                lambda p: p['metadata'].update(source_path='/private/source.pt'),
                lambda p: p['world']['weights'].update({'encoder.patch.weight': torch.ones(1)}),
                lambda p: p['goal']['weights'].update({'project.weight': p['goal']['weights']['project.weight'].double()}),
                lambda p: p['goal']['weights']['project.weight'].fill_(float('nan')),
                lambda p: p['world']['weights'].pop('encoder.patch.weight'),
                lambda p: p['path']['weights'].update(unexpected=torch.ones(1)),
                lambda p: p['path']['weights']['project.weight'].fill_(float('inf')),
                lambda p: p['goal']['weights']['project.weight'].fill_(float('-inf')),
            ]
            for mutation in mutations:
                payload = deepcopy(original)
                mutation(payload)
                torch.save(payload, output)
                with self.subTest(mutation=mutation), self.assertRaises((ValueError, TypeError)):
                    load(output)

    def test_read_only_asset_loads_without_filesystem_mapping_and_matches_mmap(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.pt', Path(directory) / 'base.pt'
            torch.save(sequence_snapshot(), source)
            export(source, output)
            before = output.read_bytes()
            reference = from_payload(torch.load(output, weights_only=True, mmap=True))
            output.chmod(0o444)
            with patch.object(torch.UntypedStorage, 'from_file',
                              side_effect=OSError('input filesystem cannot map storage')):
                actual = load(output)
            for expected_module, actual_module in ((reference.engine.model, actual.engine.model),
                                                   (reference.goal, actual.goal),
                                                   (reference.path, actual.path)):
                for key, expected in expected_module.state_dict().items():
                    value = actual_module.state_dict()[key]
                    self.assertTrue(torch.equal(value, expected), key)
                    self.assertNotEqual(value.data_ptr(), expected.data_ptr(), key)
                self.assertTrue(all(not p.requires_grad for p in actual_module.parameters()))
            self.assertEqual(actual.metadata, reference.metadata)
            self.assertEqual(output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
