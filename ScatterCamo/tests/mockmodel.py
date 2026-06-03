"""A NumPy-only mock classifier so tests need neither torch nor a GPU.

The true class loses confidence as the mean perturbation grows past ``flip``,
so any non-trivial perturbation flips the prediction -- enough to exercise every
attack's full control flow (find adversarial, then minimize visibility).
"""

import numpy as np


class MockModel:
    def __init__(self, x_ref, k=10, true=0, flip=0.01):
        self.x_ref, self.k, self.true, self.flip = x_ref, k, true, flip

    def predict(self, x):
        img = np.asarray(x)
        if img.ndim == 4:
            img = img[0]
        pert = float(np.mean(np.abs(img - self.x_ref)))
        preds = np.full(self.k, -1.0)
        preds[self.true] = 1.0 - pert / self.flip
        preds[1] = 0.5
        return preds[None, :]


def small_dataset(n=3, size=24, seed=0):
    """Return a list of (image, true_label) pairs with distinct images."""
    rng = np.random.default_rng(seed)
    return [(rng.random((size, size, 3)), 0) for _ in range(n)]
