"""Instance telemetry preserves tiny overlapping masks at native resolution."""
import base64
import json
import unittest

import numpy as np

from model.object_preview import instance_preview, mask_instances
from model.segmentation import Segmentation


class ObjectPreviewTests(unittest.TestCase):
    def test_overlapping_masks_and_slot_ids_survive_serialization(self):
        masks = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                          [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                          [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                          [[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=bool)
        labels = Segmentation(masks[:2], np.array([1., .75]))
        preview = instance_preview(labels, labels, masks, np.array([1., .5, .49, 1.]))
        decoded = json.loads(json.dumps(preview, allow_nan=False))
        self.assertEqual(decoded['shape'], [3, 3])
        self.assertEqual(decoded['encoding'], 'packbits-little-base64')
        for group in ('current', 'next', 'predicted'):
            rows = decoded['sets'][group]
            self.assertEqual([r['id'] for r in rows], [0, 1])
            self.assertEqual([r['area'] for r in rows], [1, 9])
            # Row-major little-endian bits: centre = bit4; full grid = bits0..8.
            self.assertEqual(base64.b64decode(rows[0]['bits']), b'\x10\x00')
            self.assertEqual(base64.b64decode(rows[1]['bits']), b'\xff\x01')
        sparse = mask_instances(masks, np.array([.4, .75, 1., .6]))
        self.assertEqual([r['id'] for r in sparse], [1, 2])

    def test_bounded_native_payload_and_invalid_shapes(self):
        masks = np.ones((64, 64, 64), dtype=bool)
        encoded = mask_instances(masks, np.ones(64))
        self.assertEqual(len(encoded), 64)
        self.assertTrue(all(len(base64.b64decode(r['bits'])) == 512 for r in encoded))
        self.assertEqual(mask_instances(np.empty((0, 2, 3)), np.empty(0)), [])
        for shape in ((65, 4, 4), (1, 65, 4), (1, 4, 0)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                mask_instances(np.ones(shape), np.ones(shape[0]))
        with self.assertRaises(ValueError):
            mask_instances(np.ones((1, 3, 3)), np.ones(2))
        with self.assertRaises(ValueError):
            mask_instances(np.ones((1, 3, 3)), np.array([float('nan')]))


if __name__ == '__main__':
    unittest.main()
