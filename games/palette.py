"""Appearance-only relabeling shared by bundled synthetic environments."""
import random

import numpy as np


def palette_for_seed(seed: int) -> np.ndarray:
    """A full bijection, independent of every puzzle/action RNG stream."""
    if type(seed) is not int or seed < 0:
        raise ValueError('Palette seed must be a nonnegative integer')
    return np.asarray(random.Random(f'synthetic-palette:{seed}').sample(range(16), 16),
                      dtype=np.int8)
