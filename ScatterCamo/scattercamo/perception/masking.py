"""Perceptual hideability map + seeding + visibility-weighted L2.

Builds a per-pixel map ``W`` in ``[eps, 1]``, high where a perturbation is
imperceptible, from three human-vision *masking* signals computed cheaply from
the clean image (no model queries):

    dark    = 1 - luminance     (luminance / Weber masking: shadows hide change)
    edges   = |Sobel gradient|  (contrast masking: edges hide change)
    texture = local variance    (texture masking: busy regions hide change)

``W = floor(normalize(w_dark*dark + w_edges*edges + w_texture*texture))``.

Used two ways:
  * **seeding** -- sample initial shape centers with probability proportional to W
    (``seed_positions``);
  * **objective** -- reweight the L2 search objective by ``visibility = 1 - W`` so
    changes in visible pixels cost more (``weighted_l2``).

Reported metrics (``scattercamo.metrics``) are untouched and still measure the
true visible distortion. Pure NumPy, matching the renderer's no-OpenCV ethos.
"""

import numpy as np

# Rec. 601 luminance weights.
_LUMA = np.array([0.299, 0.587, 0.114])


def _normalize(a):
    """Min-max scale to [0, 1]. A constant array maps to all zeros."""
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.float64)
    return (a - lo) / (hi - lo)


def _luminance(x):
    """Grayscale luminance of an (h, w, 3) image in [0, 1]."""
    return x.astype(np.float64) @ _LUMA


def _convolve3x3(img, k):
    """Valid 3x3 convolution with edge replication. Pure NumPy."""
    p = np.pad(img, 1, mode="edge")
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            out += k[i, j] * p[i:i + h, j:j + w]
    return out


def _edge_magnitude(lum):
    """Sobel gradient magnitude of the luminance channel."""
    kx = np.array([[-1.0, 0.0, 1.0],
                   [-2.0, 0.0, 2.0],
                   [-1.0, 0.0, 1.0]])
    gx = _convolve3x3(lum, kx)
    gy = _convolve3x3(lum, kx.T)
    return np.hypot(gx, gy)


def _box_mean(img, window):
    """Mean over a ``window`` x ``window`` box per pixel, via an integral image."""
    r = window // 2
    pad = np.pad(img, r, mode="edge")
    integ = np.zeros((pad.shape[0] + 1, pad.shape[1] + 1), dtype=np.float64)
    integ[1:, 1:] = pad.cumsum(0).cumsum(1)
    h, w = img.shape
    win = 2 * r + 1
    total = (integ[win:win + h, win:win + w]
             - integ[0:h, win:win + w]
             - integ[win:win + h, 0:w]
             + integ[0:h, 0:w])
    return total / (win * win)


def _local_variance(lum, window):
    """Local variance E[x^2] - E[x]^2 over a box window."""
    mean = _box_mean(lum, window)
    mean_sq = _box_mean(lum * lum, window)
    return np.maximum(mean_sq - mean * mean, 0.0)


def hideability_map(x, w_dark=1.0, w_edges=1.0, w_texture=1.0, window=7, eps=0.05):
    """Per-pixel hideability map ``W`` in ``[eps, 1]`` for image ``x`` (h, w, 3).

    High where a perturbation is imperceptible. ``w_*`` weight the three masking
    signals (set any to 0 to ablate it). ``eps`` floors the map so it stays a
    soft prior rather than a hard ban. Degenerate inputs (all weights 0, or a
    constant image) collapse toward ``eps`` -> near plain-L2 / uniform behavior.
    """
    lum = _luminance(x)
    dark = _normalize(1.0 - lum)
    edges = _normalize(_edge_magnitude(lum))
    texture = _normalize(_local_variance(lum, window))

    total_w = w_dark + w_edges + w_texture
    if total_w <= 0:
        combined = np.zeros_like(lum)
    else:
        combined = _normalize(
            (w_dark * dark + w_edges * edges + w_texture * texture) / total_w
        )
    return eps + (1.0 - eps) * combined


def hideability_components(x, window=7):
    """Return the three normalized masking signals for visualization/analysis.

    Each is an ``(h, w)`` array in ``[0, 1]``: ``dark``, ``edges``, ``texture``.
    These are the un-weighted, un-floored ingredients that ``hideability_map``
    combines.
    """
    lum = _luminance(x)
    return {
        "dark": _normalize(1.0 - lum),
        "edges": _normalize(_edge_magnitude(lum)),
        "texture": _normalize(_local_variance(lum, window)),
    }


def seed_positions(W, m, rng):
    """Sample ``m`` shape centers with probability proportional to ``W``.

    Returns an ``(m, 2)`` array of ``(y, x)`` genes in ``[0, 1)`` ready to drop
    into the first two genome columns. Falls back to uniform if ``W`` sums to 0.
    """
    h, w = W.shape
    p = W.ravel().astype(np.float64)
    s = p.sum()
    if s <= 0:
        return np.stack([rng.random(m), rng.random(m)], axis=1)
    idx = rng.choice(h * w, size=m, p=p / s)
    yy, xx = np.divmod(idx, w)
    # sub-pixel jitter so co-located samples don't collapse to identical genes
    ys = (yy + rng.random(m)) / h
    xs = (xx + rng.random(m)) / w
    return np.stack([ys, xs], axis=1)


def weighted_l2(adv, x, visibility):
    """Visibility-weighted squared L2 distance.

    ``visibility`` is ``(h, w)`` or ``(h, w, 1)`` in ``[0, 1]`` (typically
    ``1 - W``): changes in visible (high-visibility) pixels are penalized more.
    """
    diff_sq = (adv - x) ** 2
    v = visibility[:, :, None] if visibility.ndim == 2 else visibility
    return float(np.sum(diff_sq * v))
