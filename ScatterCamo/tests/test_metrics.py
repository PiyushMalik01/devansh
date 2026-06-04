"""Tests for PSNR and the FID Frechet-distance math (NumPy + SciPy, no torch).

Run from the repo root:
    python tests/test_metrics.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scattercamo import metrics
from scattercamo.metrics.fid import fid_from_activations


def test_psnr_identical_is_inf():
    rng = np.random.default_rng(0)
    x = rng.random((16, 16, 3))
    assert metrics.psnr(x, x) == float("inf")
    print("  [ok] PSNR of identical images is +inf")


def test_psnr_known_value():
    x = np.zeros((8, 8, 3))
    adv = x + 0.1                      # constant error -> MSE = 0.01 -> 20 dB
    assert abs(metrics.psnr(adv, x) - 20.0) < 1e-6
    print("  [ok] PSNR matches the closed-form value (0.1 error -> 20 dB)")


def test_fid_identical_sets_near_zero():
    rng = np.random.default_rng(1)
    act = rng.random((50, 32))
    d = fid_from_activations(act, act)
    assert d < 1e-6, d
    print(f"  [ok] FID of a set with itself is ~0 ({d:.2e})")


def test_fid_mean_shift_grows():
    rng = np.random.default_rng(2)
    a = rng.random((100, 16))
    b = a + 5.0                        # pure mean shift of 5 across 16 dims
    d = fid_from_activations(a, b)
    assert np.isfinite(d) and d > 16 * 5 ** 2 * 0.5   # dominated by ||mu1-mu2||^2
    print(f"  [ok] FID grows under a mean shift ({d:.1f})")


def test_fid_needs_two_images():
    try:
        fid_from_activations(np.zeros((1, 8)), np.zeros((1, 8)))
    except ValueError:
        print("  [ok] FID rejects sets smaller than 2")
        return
    raise AssertionError("expected ValueError for a single-image set")


if __name__ == "__main__":
    print("Running metrics tests...")
    test_psnr_identical_is_inf()
    test_psnr_known_value()
    test_fid_identical_sets_near_zero()
    test_fid_mean_shift_grows()
    test_fid_needs_two_images()
    print("ALL TESTS PASSED")
