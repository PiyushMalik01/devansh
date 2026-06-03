"""ImageNet model wrappers with a common ``predict`` interface + query counting.

Black-box assumption: the attack only ever calls ``predict`` and reads the
returned logits/probabilities. ``queries`` tracks the budget consumed.
"""

import numpy as np
import torch
from torchvision import models as tvm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ImageNetModel:
    REGISTRY = {0: tvm.vgg16_bn, 1: tvm.resnet50}

    def __init__(self, idx):
        ctor = self.REGISTRY[idx]
        self.model = ctor(weights="DEFAULT").to(DEVICE).eval()
        self.mu = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
        self.sigma = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
        self.queries = 0

    def predict(self, x):
        self.queries += 1
        if not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        x = x.float().to(DEVICE)
        with torch.no_grad():
            normed = (x - self.mu) / self.sigma
            return self.model(normed).cpu()

    def __call__(self, x):
        return self.predict(x)
