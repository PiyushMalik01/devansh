"""Frechet Inception Distance (FID).

FID measures how far the *distribution* of generated (adversarial) images sits
from the real (clean) images, in the feature space of a pretrained InceptionV3.
It is a **set-level** metric: compute it over many images, never a single pair
(and it needs at least 2 images per set for a covariance).

Two pieces, deliberately split by dependency weight:

* ``fid_from_activations(real, fake)`` -- the Frechet distance between two sets of
  feature vectors. Pure NumPy + SciPy (both core deps), so it is fully testable
  offline without torch.
* ``inception_activations(images)`` -- 2048-d InceptionV3 pool3 features. Needs
  ``torch`` + ``torchvision`` (the ``real`` extra: ``uv sync --extra real``);
  raises a clear error otherwise.

Note: features come from torchvision's InceptionV3, not the exact TF-Inception
weights used in the original FID paper, so absolute values differ slightly from
some published numbers. For *relative* comparisons between methods on the same
data -- our use case -- it is self-consistent and fine.
"""

import numpy as np


def fid_from_activations(act_real, act_fake, eps=1e-6):
    """FID between two ``(N, D)`` arrays of feature activations."""
    act_real = np.asarray(act_real, dtype=np.float64)
    act_fake = np.asarray(act_fake, dtype=np.float64)
    if act_real.shape[0] < 2 or act_fake.shape[0] < 2:
        raise ValueError("FID needs at least 2 images per set (to form a covariance)")
    mu1, sigma1 = act_real.mean(axis=0), np.cov(act_real, rowvar=False)
    mu2, sigma2 = act_fake.mean(axis=0), np.cov(act_fake, rowvar=False)
    return _frechet_distance(mu1, sigma1, mu2, sigma2, eps)


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy import linalg

    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        # singular product -> jitter the diagonals and retry
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def inception_activations(images, batch_size=32, device=None):
    """2048-d InceptionV3 pool3 features for ``images`` ``(N, H, W, 3)`` in [0, 1].

    Requires torch + torchvision (the ``real`` extra). Downloads the InceptionV3
    weights on first use.
    """
    try:
        import torch
        from torchvision import models
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "FID feature extraction needs torch + torchvision. Install the extra:\n"
            "    uv sync --extra real"
        ) from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT,
                              aux_logits=True)
    net.fc = torch.nn.Identity()           # expose the 2048-d pool3 features
    net.eval().to(device)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    imgs = torch.from_numpy(np.asarray(images, dtype=np.float32)).permute(0, 3, 1, 2)
    feats = []
    with torch.no_grad():
        for i in range(0, imgs.shape[0], batch_size):
            batch = imgs[i:i + batch_size].to(device)
            batch = torch.nn.functional.interpolate(
                batch, size=(299, 299), mode="bilinear", align_corners=False)
            batch = (batch - mean) / std
            feats.append(net(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def fid(real_images, fake_images, batch_size=32, device=None):
    """End-to-end FID between two sets of images ``(N, H, W, 3)`` in [0, 1]."""
    act_real = inception_activations(real_images, batch_size, device)
    act_fake = inception_activations(fake_images, batch_size, device)
    return fid_from_activations(act_real, act_fake)
