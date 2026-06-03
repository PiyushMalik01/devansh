"""Shape genome + renderer.

A solution is a genome of ``M`` shapes. Each shape is 7 numbers in [0, 1]:

    (y, x, radius, R, G, B, alpha)

generalizing CamoPatch's circle gene from a fixed square patch to the full
image canvas. The adversarial image is produced by sequentially alpha-blending
each shape onto the original image (later shapes paint over earlier ones).

Rendering is pure NumPy (no OpenCV dependency) so it runs anywhere.
"""

import numpy as np

GENES_PER_SHAPE = 7  # (y, x, radius, R, G, B, alpha)


def random_genome(m, rng):
    """Return a random genome of shape (m, 7) with all genes in [0, 1)."""
    return rng.random((m, GENES_PER_SHAPE))


def generate_image(genome, x, max_radius):
    """Alpha-blend ``genome``'s shapes onto image ``x`` (h, w, 3) in [0, 1].

    Args:
        genome: (M, 7) array, each row (y, x, radius, R, G, B, alpha) in [0, 1].
        x: original image, float array (h, w, 3) in [0, 1].
        max_radius: radius gene of 1.0 maps to this many pixels.

    Returns:
        Adversarial image, float64 (h, w, 3), clipped to [0, 1].
    """
    h, w = x.shape[:2]
    canvas = x.astype(np.float64).copy()
    yy, xx = np.ogrid[:h, :w]
    for row in genome:
        cy, cx = row[0] * h, row[1] * w
        r = row[2] * max_radius + 1.0
        color = row[3:6]
        alpha = float(row[6])
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        canvas[mask] = alpha * color + (1.0 - alpha) * canvas[mask]
    return np.clip(canvas, 0.0, 1.0)


def l2_perturbation(adv, x):
    """Squared L2 distance between adversarial and original image.

    Used as a search objective (monotonic in the true L2 norm, cheaper).
    Reporting uses the actual norm; see ``scattercamo.metrics.l2``.
    """
    return float(np.sum((adv - x) ** 2))
