"""Each baseline runs end-to-end against the mock model, respects its budget,
and returns a unified AttackResult with an adversarial image."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from mockmodel import MockModel

from scattercamo.losses import UnTargeted
from scattercamo.runner.result import AttackResult
from scattercamo.baselines import SparseRSAttack, SAMOOAttack, CamoPatchAttack


def _loss(x):
    return UnTargeted(MockModel(x, true=0, flip=0.01), true=0, to_pytorch=False)


def test_sparse_rs():
    rng = np.random.default_rng(0)
    x = rng.random((24, 24, 3))
    res = SparseRSAttack({"x": x, "eps": 60, "queries": 200, "seed": 0}).optimise(_loss(x))
    assert isinstance(res, AttackResult)
    assert res.queries <= 200
    assert res.success and res.adv_image is not None
    print(f"  [ok] Sparse-RS: success in {res.queries} queries")


def test_samoo():
    rng = np.random.default_rng(1)
    x = rng.random((24, 24, 3))
    res = SAMOOAttack({"x": x, "eps": 60, "queries": 200, "pop_size": 10,
                       "seed": 0}).optimise(_loss(x))
    assert isinstance(res, AttackResult)
    assert res.queries <= 200
    assert res.success and res.adv_image is not None
    print(f"  [ok] SA-MOO: success in {res.queries} queries")


def test_camopatch():
    rng = np.random.default_rng(2)
    x = rng.random((24, 24, 3))
    res = CamoPatchAttack({"x": x, "N": 15, "s": 12, "queries": 200,
                           "seed": 0}).optimise(_loss(x))
    assert isinstance(res, AttackResult)
    assert res.queries <= 200
    assert res.success and res.adv_image is not None
    print(f"  [ok] CamoPatch: success in {res.queries} queries")


if __name__ == "__main__":
    print("Running baseline tests...")
    test_sparse_rs()
    test_samoo()
    test_camopatch()
    print("BASELINE TESTS PASSED")
