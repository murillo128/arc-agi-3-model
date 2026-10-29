"""Portable exports contain only inference state and strictly load native shapes."""
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict, replace
import io
import os
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


def sequence_snapshot(*, object_slots=0):
    config = replace(NetworkConfig.small(), patch=2, latent=16, pixel_latent=True,
                     local_dynamics=True, rotation_adapter_width=4, object_slots=object_slots)
    world = WorldModel(config)
    goal = VisualGoal(config.latent, config.memory, config.patch, 8)
    path = PathValue(config.latent, config.memory, 8)
    return dict(kind='game_sequences', version=1, network=asdict(config), training=asdict(TrainConfig()),
                weights=world.state_dict(), planning=dict(version=1, config=dict(width=8),
                goal=goal.state_dict(), path=path.state_dict(), optimizer={'secret': 'discard me'},
                rng='discard me'), progress=dict(batches=8246), predictor_optimizer={'secret': 'discard me'},
                run='/private/training/path', torch_rng=torch.get_rng_state())


class SequentialReader:
    """A real read-only stream with a forward-only mounted-storage contract."""

    def __init__(self, stream, *, fail_after=None):
        self.stream = stream
        self.fail_after = fail_after
        self.read_sizes = []
        self.bytes_read = 0

    def read(self, size=-1):
        if not 0 < size <= 1024 * 1024:
            raise AssertionError(f'Source read must be bounded: {size}')
        self.read_sizes.append(size)
        if self.fail_after is not None and self.bytes_read >= self.fail_after:
            raise OSError('mounted source read failed')
        # Short reads also exercise the copy loop without a large checkpoint.
        block = self.stream.read(min(size, 4096))
        self.bytes_read += len(block)
        return block

    def seek(self, *args):
        raise AssertionError('mounted source cannot seek')

    def tell(self):
        raise AssertionError('mounted source cannot tell')

    def fileno(self):
        raise AssertionError('mounted source cannot expose a mapping descriptor')

    def __enter__(self):
        return self

    def close(self):
        self.stream.close()

    def __exit__(self, *args):
        self.close()


@contextmanager
def staging_boundary(source, local_directory, *, fail_after=None):
    """Instrument source IO and real temporary storage, leaving torch.load real."""
    real_open, real_temporary_file = io.open, tempfile.TemporaryFile
    readers, temporary_files = [], []

    def open_file(file, mode='r', *args, **kwargs):
        if not isinstance(file, int) and Path(file) == source:
            if mode != 'rb':
                raise AssertionError('mounted source must remain read-only')
            reader = SequentialReader(real_open(file, mode, *args, **kwargs), fail_after=fail_after)
            readers.append(reader)
            return reader
        return real_open(file, mode, *args, **kwargs)

    def temporary_file(*args, **kwargs):
        stream = real_temporary_file(*args, **kwargs)
        temporary_files.append((stream, stream.fileno()))
        return stream

    with patch('builtins.open', open_file), patch('io.open', open_file), \
            patch.object(tempfile, 'tempdir', str(local_directory)), \
            patch.object(tempfile, 'TemporaryFile', temporary_file), \
            patch.object(torch.UntypedStorage, 'from_file',
                         side_effect=AssertionError('input filesystem cannot map storage')):
        yield readers, temporary_files


class AdaptationCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_staging_cleaned(self, readers, temporary_files, local_directory):
        self.assertEqual(len(readers), 1)
        self.assertTrue(readers[0].stream.closed)
        self.assertEqual(len(temporary_files), 1)
        for stream, descriptor in temporary_files:
            self.assertTrue(stream.closed)
            with self.assertRaises(OSError):
                os.fstat(descriptor)
        self.assertEqual(list(local_directory.iterdir()), [])

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

    def test_legacy_v1_optional_object_defaults_preserve_weights_and_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.pt', Path(directory) / 'base.pt'
            snapshot = sequence_snapshot()
            torch.save(snapshot, source)
            export(source, output)
            original = torch.load(output, weights_only=True)
            # Pre-object v1 exports have neither field. Each new optional field
            # can also be omitted independently; other schema keys stay required.
            for missing in (('object_slots', 'object_width'), ('object_slots',), ('object_width',)):
                with self.subTest(missing=missing):
                    payload = deepcopy(original)
                    for key in missing:
                        del payload['world']['network'][key]
                    network_before = dict(payload['world']['network'])
                    bundle = from_payload(payload)
                    self.assertEqual(payload['world']['network'], network_before)
                    self.assertEqual(bundle.engine.model.config.object_slots, 0)
                    self.assertEqual(bundle.engine.model.config.object_width, 16)
                    self.assertFalse(hasattr(bundle.engine.model.dynamics, 'objects'))
                    for key, value in bundle.engine.model.state_dict().items():
                        self.assertTrue(torch.equal(value, snapshot['weights'][key]), key)
                    for invalid in ('unknown', 'missing-required'):
                        malformed = deepcopy(payload)
                        if invalid == 'unknown':
                            malformed['world']['network']['unknown'] = 1
                        else:
                            del malformed['world']['network']['latent']
                        with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'network config'):
                            from_payload(malformed)

    def test_enabled_object_export_preserves_typed_buffers_and_rejects_invalid_state(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.pt', Path(directory) / 'base.pt'
            snapshot = sequence_snapshot(object_slots=3)
            prefix = 'dynamics.objects.'
            snapshot['weights'][prefix+'updates'].fill_(17)
            snapshot['weights'][prefix+'gain'].fill_(1.)
            torch.save(snapshot, source)
            before = source.read_bytes()
            export(source, output)
            bundle = load(output)
            self.assertEqual(source.read_bytes(), before)
            head = bundle.engine.model.dynamics.objects
            self.assertEqual(head.updates.dtype, torch.int64)
            self.assertEqual(head.updates.shape, torch.Size([]))
            self.assertEqual(head.updates.item(), 17)
            self.assertEqual(head.gain.item(), 1.)
            for key, value in bundle.engine.model.state_dict().items():
                self.assertTrue(torch.equal(value, snapshot['weights'][key]), key)
            original = torch.load(output, weights_only=True)
            invalid_values = (
                ('updates', torch.tensor(-1, dtype=torch.int64)),
                ('updates', torch.tensor(17.)),
                ('updates', torch.tensor(17, dtype=torch.int32)),
                ('updates', torch.tensor(True)),
                ('updates', torch.tensor([17], dtype=torch.int64)),
                ('gain', torch.tensor(float('nan'))),
                ('gain', torch.tensor(1, dtype=torch.int64)),
            )
            for name, value in invalid_values:
                with self.subTest(name=name, value=value):
                    payload = deepcopy(original)
                    payload['world']['weights'][prefix+name] = value
                    with self.assertRaisesRegex(ValueError, 'Invalid world tensor'):
                        from_payload(payload)

    def test_malformed_schema_shapes_and_nonfinite_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.pt', Path(directory) / 'base.pt'
            local_directory = Path(directory) / 'local'
            local_directory.mkdir()
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
                before = output.read_bytes()
                with self.subTest(mutation=mutation):
                    with staging_boundary(output, local_directory) as (readers, temporary_files):
                        with self.assertRaises((ValueError, TypeError)):
                            load(output)
                    self.assert_staging_cleaned(readers, temporary_files, local_directory)
                    self.assertEqual(readers[0].bytes_read, len(before))
                    self.assertEqual(output.read_bytes(), before)

    def test_read_only_asset_loads_sequentially_and_matches_mmap(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.pt', Path(directory) / 'base.pt'
            local_directory = Path(directory) / 'local'
            local_directory.mkdir()
            torch.save(sequence_snapshot(), source)
            export(source, output)
            before = output.read_bytes()
            reference = from_payload(torch.load(output, weights_only=True, mmap=True))
            output.chmod(0o444)
            with staging_boundary(output, local_directory) as (readers, temporary_files):
                actual = load(output)
            self.assert_staging_cleaned(readers, temporary_files, local_directory)
            self.assertGreater(len(readers[0].read_sizes), 2)
            self.assertEqual(readers[0].bytes_read, len(before))
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
            self.assertEqual(output.stat().st_mode & 0o777, 0o444)

    def test_truncated_payload_and_source_read_failure_clean_up_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.pt', Path(directory) / 'base.pt'
            local_directory = Path(directory) / 'local'
            local_directory.mkdir()
            torch.save(sequence_snapshot(), source)
            export(source, output)
            complete = output.read_bytes()
            for failure in ('truncated', 'source-read'):
                with self.subTest(failure=failure):
                    output.chmod(0o644)
                    output.write_bytes(complete[:-64] if failure == 'truncated' else complete)
                    output.chmod(0o444)
                    before = output.read_bytes()
                    fail_after = 4096 if failure == 'source-read' else None
                    error = OSError if failure == 'source-read' else RuntimeError
                    message = 'mounted source read failed' if failure == 'source-read' else 'central directory'
                    with staging_boundary(output, local_directory, fail_after=fail_after) as boundary:
                        readers, temporary_files = boundary
                        with self.assertRaisesRegex(error, message):
                            load(output)
                    self.assert_staging_cleaned(readers, temporary_files, local_directory)
                    self.assertEqual(readers[0].bytes_read, fail_after or len(before))
                    self.assertEqual(output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
