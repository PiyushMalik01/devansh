"""Dependency-light tests for perceptual placement (NumPy only).

Run from the repo root:
    python tests/test_perception.py

Proves: the hideability map is in range and responds to each masking signal
(dark / edges / texture); seeding concentrates shapes in hideable regions;
the visibility-weighted L2 discounts changes in hideable regions; the plain-L2
path is byte-for-byte unchanged; and the full attack still converges with
perceptual mode on.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scattercamo.perception import hideability_map, seed_positions, weighted_l2
from scattercamo.perception.masking import _luminance
from scattercamo.representation import random_genome, l2_perturbation
from scattercamo.search import Solution
from scattercamo.losses import UnTargeted
from scattercamo.attack import ScatterCamoAttack

# Reuse the mock model from the smoke tests.
from test_smoke import MockModel


def test_map_range_and_shape():
    rng = np.random.default_rng(0)
    x = rng.random((20, 24, 3))
    W = hideability_map(x)
    assert W.shape == (20, 24)
    assert W.min() >= 0.0 and W.max() <= 1.0
    print("  [ok] hideability map: correct shape, range [eps, 1]")


def test_dark_signal():
    # Left half bright, right half dark. Dark-only weights -> right half hideable.
    x = np.ones((16, 16, 3))
    x[:, 8:, :] = 0.0
    W = hideability_map(x, w_dark=1.0, w_edges=0.0, w_texture=0.0)
    assert W[:, 12].mean() > W[:, 3].mean()
    print("  [ok] dark regions score higher under luminance masking")


def test_edge_signal():
    # A vertical edge at column 8. Edge-only weights -> W peaks near the edge.
    x = np.zeros((16, 16, 3))
    x[:, 8:, :] = 1.0
    W = hideability_map(x, w_dark=0.0, w_edges=1.0, w_texture=0.0)
    edge_band = W[:, 6:10].mean()
    flat_band = W[:, 0:3].mean()
    assert edge_band > flat_band
    print("  [ok] edges score higher under contrast masking")


def test_texture_signal():
    # Left half flat, right half noisy. Texture-only weights -> noisy is hideable.
    rng = np.random.default_rng(1)
    x = np.full((24, 24, 3), 0.5)
    x[:, 12:, :] = rng.random((24, 12, 3))
    W = hideability_map(x, w_dark=0.0, w_edges=0.0, w_texture=1.0)
    assert W[:, 18].mean() > W[:, 5].mean()
    print("  [ok] textured regions score higher under texture masking")


def test_seeding_biases_to_high_W():
    # W concentrated in the bottom-right quadrant; samples should follow.
    W = np.full((40, 40), 0.01)
    W[20:, 20:] = 1.0
    rng = np.random.default_rng(2)
    pos = seed_positions(W, 400, rng)            # (400, 2) in [0,1)
    in_quadrant = np.mean((pos[:, 0] >= 0.5) & (pos[:, 1] >= 0.5))
    assert in_quadrant > 0.8, in_quadrant
    print(f"  [ok] seeding concentrates in high-W region ({in_quadrant:.0%})")


def test_weighted_l2_discounts_hideable():
    x = np.zeros((10, 10, 3))
    adv = x.copy()
    adv[5, 5, :] += 0.3                          # one perturbed pixel
    visible = np.ones((10, 10))                  # everything visible
    hideable = np.ones((10, 10)) * 0.1           # mostly hideable
    assert weighted_l2(adv, x, hideable) < weighted_l2(adv, x, visible)
    # With all-ones visibility it equals the plain squared-L2.
    assert np.isclose(weighted_l2(adv, x, visible), l2_perturbation(adv, x))
    print("  [ok] weighted L2 discounts hideable regions, matches plain L2 at v=1")


def test_backward_compatible_plain_path():
    rng = np.random.default_rng(3)
    x = rng.random((12, 12, 3))
    g = random_genome(5, rng)
    loss = UnTargeted(MockModel(x), true=0, to_pytorch=False)
    s_plain = Solution(g.copy(), x, max_radius=4.0)              # visibility=None
    s_plain.evaluate(loss)
    adv = s_plain.generate_image()
    assert np.isclose(s_plain.fitnesses[1], l2_perturbation(adv, x))
    print("  [ok] visibility=None path identical to plain squared-L2")


def test_attack_converges_perceptual():
    rng = np.random.default_rng(0)
    x = rng.random((16, 16, 3))
    loss = UnTargeted(MockModel(x, flip=0.04), true=0, to_pytorch=False)
    attack = ScatterCamoAttack({
        "x": x, "M": 8, "queries": 300, "pop_size": 10, "seed": 0,
        "perceptual": True,
    })
    result = attack.optimise(loss)
    assert result["queries"] <= 300
    assert result["success"], "perceptual attack failed to find an adversarial solution"
    assert attack.W is not None and attack.visibility is not None
    print(f"  [ok] perceptual attack converged in {result['queries']} queries")


if __name__ == "__main__":
    print("Running ScatterCamo perception tests...")
    test_map_range_and_shape()
    test_dark_signal()
    test_edge_signal()
    test_texture_signal()
    test_seeding_biases_to_high_W()
    test_weighted_l2_discounts_hideable()
    test_backward_compatible_plain_path()
    test_attack_converges_perceptual()
    print("ALL TESTS PASSED")
