"""Preview provenance, pixel semantics and bounded lifetime, without GPU/UI tests."""
import random
import unittest

import numpy as np

from model.prediction_samples import PredictionSamples


class PredictionSampleTests(unittest.TestCase):
    def test_both_deltas_share_reference_and_preserve_real_replacement_colors(self):
        previous = np.array([[9, 5, 5], [5, 15, 5]], dtype=np.uint8)
        actual = np.array([[5, 9, 5], [5, 15, 5]], dtype=np.uint8)
        predicted = np.array([[5, 5, 14], [5, 15, 5]], dtype=np.uint8)
        samples = PredictionSamples(seed=12)
        samples.add(previous, actual, predicted, game='example', level=1, training_action=17)
        previous[:] = 0  # The caller may reuse its buffers after publishing.
        sample = samples.draw()
        self.assertEqual(sample['previous'], [[9, 5, 5], [5, 15, 5]])
        self.assertEqual(sample['real_delta'], [[5, 9, 16], [16, 16, 16]])
        self.assertEqual(sample['predicted_delta'], [[5, 16, 14], [16, 16, 16]])
        self.assertEqual((sample['wrong'], sample['union']), (2, 3))
        self.assertEqual(sample['training_action'], 17)

    def test_bounded_sampling_is_independent_and_keeps_stationary_actions(self):
        training_rng = random.getstate()
        samples = PredictionSamples(limit=2, seed=4)
        self.assertIsNone(samples.draw())
        for value in (1, 2, 3):
            frame = np.full((2, 2), value, dtype=np.uint8)
            samples.add(frame, frame, frame, training_action=value)
        first, second = samples.draw(), samples.draw()
        self.assertEqual({first['training_action'], second['training_action']}, {2, 3})
        self.assertEqual(second['real_delta'], [[16, 16], [16, 16]])
        self.assertEqual((second['wrong'], second['union']), (0, 0))
        self.assertEqual(random.getstate(), training_rng)


if __name__ == '__main__':
    unittest.main()
