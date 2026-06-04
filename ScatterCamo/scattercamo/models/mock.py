"""A torch-free stand-in classifier for offline flag checks and demos.

``MockModel`` mimics the ``ImageNetModel`` interface (``predict`` + ``queries``
counter) without downloading weights or importing torch. The "true" class wins
on the clean image and loses as the perturbation grows, so attacks still
converge -- enough to exercise the full pipeline and verify that flags take
effect, without measuring anything about a real model.
"""

import numpy as np


class MockModel:
    """Deterministic mock: logits where the true class fades as the image is
    perturbed, and a runner-up class overtakes it past ``flip`` mean-abs change.
    """

    def __init__(self, x_ref, k=1000, true=0, flip=0.04):
        self.x_ref = np.asarray(x_ref)
        self.k = k
        self.true = true
        self.flip = flip
        self.queries = 0

    def predict(self, x):
        self.queries += 1
        img = np.asarray(x)
        if img.ndim == 4:
            img = img[0]
        pert = float(np.mean(np.abs(img - self.x_ref)))
        preds = np.full(self.k, -1.0)
        preds[self.true] = 1.0 - pert / self.flip     # decreases with perturbation
        preds[1] = 0.5                                 # the class that overtakes
        return preds[None, :]

    def __call__(self, x):
        return self.predict(x)
