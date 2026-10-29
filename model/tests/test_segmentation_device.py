"""Teacher residency and failure cleanup without an installed SAM or a CUDA GPU."""
from contextlib import ExitStack, contextmanager
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.segmentation import SAM2Segmenter


class TeacherDeviceTests(unittest.TestCase):
    @contextmanager
    def teacher(self, device='cuda:1', *, failure=None, bf16=True):
        events = []
        state = SimpleNamespace(selected=None, features=None, failure=failure, device=device)

        class PositionEmbeddingSine(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cache = {}

        class RoPEAttention(torch.nn.Module):
            pass

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.resident = 'cpu'
                self.position = PositionEmbeddingSine()
                self.rotary = RoPEAttention()

            def to(self, target):
                self.resident = str(target)
                events.append(('move', self.resident))
                if state.failure == 'transfer' and self.resident.startswith('cuda'):
                    raise RuntimeError('transfer failed')
                return self

        model = Model()

        def cache_tensors():
            # Plain tensors deliberately bypass Module.to(), as in pinned SAM.
            model.position.cache[(4, 4)] = torch.ones(4, 4)
            model.rotary.freqs_cis = torch.ones(2, dtype=torch.complex64)
            state.cpu_freqs = model.rotary.freqs_cis.clone()
            model.rotary.freqs_cis.cpu = lambda: state.cpu_freqs

        def build(config, checkpoint, **kwargs):
            events.append(('build', kwargs['device']))
            model.resident = str(kwargs['device'])
            if device.startswith('cuda'):
                self.assertEqual(state.selected, device)
                self.assertEqual(kwargs['hydra_overrides_extra'], [
                    '++model.image_encoder.neck.position_encoding.warmup_cache=false',
                    '++model.memory_encoder.position_encoding.warmup_cache=false'])
            cache_tensors()
            return model

        def reset():
            events.append(('reset', model.resident))
            state.features = None
            if state.failure == 'reset':
                raise RuntimeError('reset failed')

        def generate(rgb):
            events.append(('generate', model.resident))
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertFalse(model.training)
            self.assertFalse(model.weight.requires_grad)
            if device.startswith('cuda'):
                self.assertEqual(state.selected, device)
                self.assertTrue(torch.is_autocast_enabled('cuda'))
                self.assertEqual(torch.get_autocast_dtype('cuda'), torch.bfloat16)
            else:
                self.assertFalse(torch.is_autocast_enabled('cpu'))
            self.assertEqual(rgb.shape, (512, 512, 3))
            np.testing.assert_array_equal(rgb[128, 256], [255, 123, 204])
            state.features = object()
            cache_tensors()
            if state.failure == 'generate':
                raise RuntimeError('generate failed')
            if state.failure == 'labels':
                return [dict(segmentation=np.ones((8, 8)), predicted_iou='invalid')]
            mask = np.zeros((8, 8), bool)
            mask[2:4, 4:6] = True
            return [dict(segmentation=mask, predicted_iou=.85, stability_score=.9)]

        generator = SimpleNamespace(point_grids=[np.array([[.25, .25]], np.float32)],
                                    predictor=SimpleNamespace(reset_predictor=reset), generate=generate)

        def automatic(model_arg, **kwargs):
            self.assertIs(model_arg, model)
            self.assertEqual(kwargs, dict(points_per_side=8, points_per_batch=16,
                pred_iou_thresh=.7, stability_score_thresh=.8, crop_n_layers=0))
            return generator

        @contextmanager
        def cuda_device(target):
            prior = state.selected
            state.selected = str(target)
            events.append(('device', state.selected))
            try:
                yield
            finally:
                state.selected = prior

        def synchronize(target=None):
            self.assertEqual(str(target) if target is not None else state.selected, device)
            events.append(('sync', state.selected))

        def empty_cache():
            self.assertEqual(state.selected, device)
            events.append(('empty_cache', state.selected))

        def timestamp():
            events.append(('time',))
            return float(len(events))

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            checkpoint = Path(directory) / 'fixture.pt'
            checkpoint.write_bytes(b'offline teacher fixture')
            stack.enter_context(patch('model.segmentation.TINY_SHA256',
                hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
            stack.enter_context(patch.dict(sys.modules, {
                'sam2': SimpleNamespace(),
                'sam2.build_sam': SimpleNamespace(build_sam2=build),
                'sam2.automatic_mask_generator': SimpleNamespace(SAM2AutomaticMaskGenerator=automatic),
                'sam2.modeling': SimpleNamespace(),
                'sam2.modeling.position_encoding': SimpleNamespace(PositionEmbeddingSine=PositionEmbeddingSine),
                'sam2.modeling.sam': SimpleNamespace(),
                'sam2.modeling.sam.transformer': SimpleNamespace(RoPEAttention=RoPEAttention)}))
            stack.enter_context(patch('torch.cuda.device', cuda_device))
            stack.enter_context(patch('torch.cuda.is_available', return_value=True))
            support = stack.enter_context(patch('torch.cuda.is_bf16_supported', return_value=bf16))
            stack.enter_context(patch('torch.cuda.synchronize', synchronize))
            stack.enter_context(patch('torch.cuda.empty_cache', empty_cache))
            stack.enter_context(patch('torch.cuda.reset_peak_memory_stats',
                                      side_effect=AssertionError('trainer peaks must be retained')))
            stack.enter_context(patch('model.segmentation.time.perf_counter', timestamp))
            provider = SAM2Segmenter(checkpoint, device=device)
            yield provider, model, generator, events, state, support

    @staticmethod
    def frame():
        frame = np.zeros((4, 4), np.uint8)
        frame[1, 2] = 7
        return frame

    def assert_idle(self, model, state):
        self.assertEqual(model.resident, 'cpu')
        self.assertIsNone(state.features)
        self.assertIsNone(state.selected)
        if state.device.startswith('cuda'):
            self.assertEqual(model.position.cache, {})
            self.assertIs(model.rotary.freqs_cis, state.cpu_freqs)

    def test_cuda_is_cpu_resident_until_uncached_inference_then_releases_memory(self):
        with self.teacher() as (provider, model, generator, events, state, support):
            self.assertIn(('build', 'cpu'), events)
            self.assertEqual(events[-2:], [('sync', 'cuda:1'), ('empty_cache', 'cuda:1')])
            self.assert_idle(model, state)
            events.clear()
            result = provider.segment(self.frame())
            expected = np.zeros((1, 4, 4), bool)
            expected[0, 1, 2] = True
            np.testing.assert_array_equal(result.masks, expected)
            np.testing.assert_allclose(result.confidence, [.85])
            self.assertFalse(result.masks.flags.writeable)
            self.assertFalse(result.confidence.flags.writeable)
            self.assertFalse(result.complete)
            self.assert_idle(model, state)
            np.testing.assert_array_equal(generator.point_grids[0], [[.25, .25], [.625, .375]])
            support.assert_any_call(including_emulation=False)
            before_transfer = events.index(('move', 'cuda:1'))
            self.assertIn(('empty_cache', 'cuda:1'), events[:before_transfer])
            reset = events.index(('reset', 'cuda:1'))
            offload = events.index(('move', 'cpu'))
            self.assertLess(reset, offload)
            self.assertIn(('sync', 'cuda:1'), events[offload+1:])
            self.assertIn(('empty_cache', 'cuda:1'), events[offload+1:])
            self.assertEqual(events[-1], ('time',), 'Timing includes transfer and cleanup')
            report = provider.report()
            self.assertEqual((report['device'], report['precision'], report['resident_device']),
                             ('cuda:1', 'bfloat16', 'cpu'))
            self.assertEqual(report['calls'], 1)
            self.assertGreater(report['seconds'], 0)

    def test_cache_hits_and_exhausted_budget_do_not_touch_cuda(self):
        with self.teacher() as (provider, model, generator, events, state, support):
            provider.remaining = 1
            first = provider.segment(self.frame())
            self.assertEqual(provider.remaining, 0)
            events.clear()
            support.reset_mock()
            self.assertIs(provider.segment(self.frame().copy()), first)
            different = self.frame(); different[0, 0] = 2
            self.assertIsNone(provider.segment(different))
            self.assertIsNone(provider.segment(None))
            self.assertEqual(events, [])
            support.assert_not_called()
            self.assert_idle(model, state)
            report = provider.report()
            self.assertEqual((report['calls'], report['cache_hits'], report['deferred']), (1, 1, 1))

    def test_failures_release_predictor_and_offload_without_caching_or_fallback(self):
        for failure in ('transfer', 'generate', 'labels', 'reset'):
            with self.subTest(failure=failure), self.teacher(failure=failure) as values:
                provider, model, generator, events, state, support = values
                provider.remaining = 1
                with self.assertRaises((RuntimeError, ValueError)):
                    provider.segment(self.frame())
                self.assert_idle(model, state)
                self.assertEqual(provider.remaining, 0)
                self.assertEqual(provider.report()['calls'], 0)
                self.assertEqual(provider.report()['cached_frames'], 0)
                self.assertEqual(events.count(('generate', 'cuda:1')), int(failure != 'transfer'))
                offload = events.index(('move', 'cpu'))
                self.assertIn(('sync', 'cuda:1'), events[offload+1:])
                self.assertIn(('empty_cache', 'cuda:1'), events[offload+1:])

    def test_unsupported_bfloat16_fails_before_teacher_transfer(self):
        with self.teacher(bf16=False) as (provider, model, generator, events, state, support):
            with self.assertRaisesRegex(RuntimeError, 'cuda:1.*[Bb][Ff]16|cuda:1.*bfloat16'):
                provider.segment(self.frame())
            self.assertNotIn(('move', 'cuda:1'), events)
            self.assertFalse(any(event[0] == 'generate' for event in events))
            self.assert_idle(model, state)
            self.assertEqual(provider.report()['cached_frames'], 0)

    def test_cpu_keeps_float32_and_clears_predictor_on_success_or_failure(self):
        for failure in (None, 'generate'):
            with self.subTest(failure=failure), self.teacher('cpu', failure=failure) as values:
                provider, model, generator, events, state, support = values
                if failure:
                    with self.assertRaisesRegex(RuntimeError, 'generate failed'):
                        provider.segment(self.frame())
                else:
                    result = provider.segment(self.frame())
                    self.assertEqual(np.argwhere(result.masks).tolist(), [[0, 1, 2]])
                self.assert_idle(model, state)
                support.assert_not_called()
                self.assertFalse(any(event[0] in ('device', 'sync', 'empty_cache', 'move') for event in events))
                report = provider.report()
                self.assertEqual((report['device'], report['precision'], report['resident_device']),
                                 ('cpu', 'float32', 'cpu'))


if __name__ == '__main__':
    unittest.main()
