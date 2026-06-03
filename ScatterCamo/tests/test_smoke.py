"""Dependency-light smoke + unit tests (NumPy only -- no torch, no model download).

Run from the repo root:
    python tests/test_smoke.py

Proves: the renderer is deterministic and actually perturbs pixels; the
prioritized domination relation matches SA-MOO Definition 3.1; the genetic
operators preserve genome validity; and the full NSGA-II loop converges to an
adversarial solution against a mock model within budget.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scattercamo.representation import random_genome, generate_image, GENES_PER_SHAPE
from scattercamo.search import Solution
from scattercamo.operators import crossover, mutation
from scattercamo.losses import UnTargeted
from scattercamo.attack import ScatterCamoAttack


class MockModel:
    """Logits where the true class loses as the perturbation grows past `flip`."""

    def __init__(self, x_ref, k=10, true=0, flip=0.04):
        self.x_ref, self.k, self.true, self.flip = x_ref, k, true, flip

    def predict(self, x):
        img = np.asarray(x)
        if img.ndim == 4:
            img = img[0]
        pert = float(np.mean(np.abs(img - self.x_ref)))
        preds = np.full(self.k, -1.0)
        preds[self.true] = 1.0 - pert / self.flip   # decreases with perturbation
        preds[1] = 0.5                               # the class that overtakes
        return preds[None, :]


def _stub(is_adv, loss, l2):
    s = Solution(np.zeros((1, GENES_PER_SHAPE)), np.zeros((2, 2, 3)), 1.0)
    s.is_adversarial, s.loss = is_adv, loss
    s.fitnesses = np.array([loss, l2], dtype=float)
    return s


def test_domination():
    adv_lowL2 = _stub(True, 0.0, 1.0)
    adv_hiL2 = _stub(True, 0.0, 5.0)
    non_adv = _stub(False, 0.5, 0.0)
    assert adv_lowL2.dominates(non_adv)            # adversarial beats non-adversarial
    assert not non_adv.dominates(adv_lowL2)
    assert adv_lowL2.dominates(adv_hiL2)           # both adv -> lower L2 wins
    assert not adv_hiL2.dominates(adv_lowL2)
    a, b = _stub(False, 0.2, 9.0), _stub(False, 0.8, 0.0)
    assert a.dominates(b)                          # both non-adv -> lower loss wins
    print("  [ok] prioritized domination relation")


def test_renderer():
    rng = np.random.default_rng(7)
    x = rng.random((24, 24, 3))
    g = random_genome(6, rng)
    a1 = generate_image(g, x, max_radius=5.0)
    a2 = generate_image(g, x, max_radius=5.0)
    assert np.array_equal(a1, a2)                  # deterministic
    assert a1.shape == x.shape
    assert a1.min() >= 0.0 and a1.max() <= 1.0     # in range
    assert not np.array_equal(a1, x)               # actually perturbs
    print("  [ok] renderer: deterministic, in-range, perturbs")


def test_operators():
    rng = np.random.default_rng(1)
    g1, g2 = random_genome(8, rng), random_genome(8, rng)
    o1, o2 = crossover(g1, g2, pc=0.3, rng=rng)
    for o in (o1, o2):
        assert o.shape == g1.shape
    m = mutation(g1, pm=0.5, rng=rng)
    assert m.shape == g1.shape
    assert m.min() >= 0.0 and m.max() <= 1.0
    assert not np.array_equal(m, g1)               # mutation changed something
    print("  [ok] operators preserve genome shape and [0,1] range")


def test_attack_converges():
    rng = np.random.default_rng(0)
    x = rng.random((16, 16, 3))
    model = MockModel(x, flip=0.04)
    loss = UnTargeted(model, true=0, to_pytorch=False)
    attack = ScatterCamoAttack({
        "x": x, "M": 8, "queries": 300, "pop_size": 10,
        "pc": 0.3, "pm": 0.3, "seed": 0,
    })
    result = attack.optimise(loss)
    assert result["queries"] <= 300                # respected the budget
    assert result["success"], "attack failed to find an adversarial solution"
    assert result["best"].is_adversarial
    print(f"  [ok] attack converged: success in {result['queries']} queries, "
          f"best L2^2={result['best'].fitnesses[1]:.4f}")


if __name__ == "__main__":
    print("Running ScatterCamo smoke tests...")
    test_domination()
    test_renderer()
    test_operators()
    test_attack_converges()
    print("ALL TESTS PASSED")
