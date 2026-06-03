"""Loss functions returning ``[is_adversarial, loss]`` for a candidate image.

Ported from CamoPatch / SA-MOO so the same interface drives every attack.
The attack *minimizes* the loss; ``is_adversarial`` short-circuits the search
into its L2-minimization phase via the prioritized domination relation.
"""

import math
import numpy as np

try:
    import torch
except ImportError:  # torch only needed for the to_pytorch path
    torch = None


def _to_pytorch(img):
    return torch.from_numpy(img).permute(2, 0, 1)


class UnTargeted:
    """Margin loss: f_true - f_other. Adversarial when argmax != true label."""

    def __init__(self, model, true, to_pytorch=False):
        self.model = model
        self.true = true
        self.to_pytorch = to_pytorch

    def _logits(self, img):
        if self.to_pytorch:
            x = _to_pytorch(img)[None].float()
            preds = self.model.predict(x).flatten()
            return int(torch.argmax(preds)), preds.tolist()
        preds = self.model.predict(np.expand_dims(img, 0)).flatten()
        return int(np.argmax(preds)), list(preds)

    def get_label(self, img):
        return self._logits(img)[0]

    def __call__(self, img):
        y, preds = self._logits(img)
        is_adversarial = y != self.true
        f_true = math.log(math.exp(preds[self.true]) + 1e-30)
        preds[self.true] = -math.inf
        f_other = math.log(math.exp(max(preds)) + 1e-30)
        return [is_adversarial, float(f_true - f_other)]


class Targeted:
    """Loss for forcing a specific target class. Adversarial when argmax == target."""

    def __init__(self, model, true, target, to_pytorch=False):
        self.model = model
        self.true = true
        self.target = target
        self.to_pytorch = to_pytorch

    def _logits(self, img):
        if self.to_pytorch:
            x = _to_pytorch(img)[None].float()
            preds = self.model.predict(x).flatten()
            return int(torch.argmax(preds)), preds.tolist()
        preds = self.model.predict(np.expand_dims(img, 0)).flatten()
        return int(np.argmax(preds)), list(preds)

    def get_label(self, img):
        return self._logits(img)[0]

    def __call__(self, img):
        y, preds = self._logits(img)
        is_adversarial = y == self.target
        f_target = preds[self.target]
        f_other = math.log(sum(math.exp(pi) for pi in preds))
        return [is_adversarial, float(f_other - f_target)]
