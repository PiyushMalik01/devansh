"""Reporting metrics for adversarial examples.

Note the asymmetry with the search objective: the attack optimizes *squared*
L2 (``representation.l2_perturbation``) because it is monotonic and cheaper;
here we report the true L2 *norm* to match the conventions in both source papers.
"""

import numpy as np


def l0(adv, x, eps=1e-6):
    """Number of perturbed pixels (a pixel counts if any channel changed)."""
    per_pixel = np.abs(adv - x).sum(axis=-1)
    return int(np.count_nonzero(per_pixel > eps))


def l2(adv, x):
    """True L2 norm of the perturbation."""
    return float(np.sqrt(np.sum((adv - x) ** 2)))


def ssim(adv, x):
    """Structural similarity (1.0 = identical). Lazy-imports scikit-image."""
    from skimage.metrics import structural_similarity

    return float(structural_similarity(x, adv, channel_axis=-1, data_range=1.0))


def psnr(adv, x, data_range=1.0):
    """Peak signal-to-noise ratio in dB (higher = more similar).

    Returns ``inf`` when the images are identical (zero MSE).
    """
    mse = float(np.mean((adv - x) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def summarize(records):
    """Aggregate a list of per-image result dicts into headline statistics.

    Each record should have keys: ``success`` (bool), ``queries`` (int),
    and (for successful attacks) ``l0``, ``l2``, ``ssim``.
    """
    n = len(records)
    successes = [r for r in records if r.get("success")]
    asr = len(successes) / n if n else 0.0

    def avg(key):
        # PSNR can be inf (identical images); drop non-finite values from the mean.
        vals = [r[key] for r in successes if key in r and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else None

    return {
        "n": n,
        "asr": asr,
        "avg_l0": avg("l0"),
        "avg_l2": avg("l2"),
        "avg_psnr": avg("psnr"),
        "avg_ssim": avg("ssim"),
        "avg_queries": float(np.mean([r["queries"] for r in records])) if n else None,
    }
