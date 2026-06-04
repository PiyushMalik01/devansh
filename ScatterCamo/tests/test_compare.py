"""Smoke test for the comparison harness against the mock model (no torch).

Run from the repo root:
    python tests/test_compare.py

Proves compare() runs every method end-to-end and returns well-formed summaries.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from mockmodel import MockModel

from scattercamo.losses import UnTargeted
from compare import compare, attack_factories


def test_compare_runs_all_methods():
    rng = np.random.default_rng(0)
    dataset = [(rng.random((28, 28, 3)), 0) for _ in range(2)]
    loss_factory = lambda x, y: UnTargeted(MockModel(x, true=0, flip=0.01),
                                           true=y, to_pytorch=False)

    args = argparse.Namespace(M=6, queries=200, seed=0, perceptual=False, eps=60)
    methods = ["scattercamo", "camopatch", "samoo", "sparse_rs"]
    results = compare(dataset, loss_factory, methods, attack_factories(args),
                      want_fid=False)

    assert set(results) == set(methods)
    for name, s in results.items():
        assert "asr" in s and "avg_psnr" in s and "avg_ssim" in s
        assert 0.0 <= s["asr"] <= 1.0
        assert s["avg_queries"] is not None and s["avg_queries"] <= 200
    print(f"  [ok] compare() ran {len(methods)} methods; "
          f"ScatterCamo ASR={results['scattercamo']['asr']:.2f}")


if __name__ == "__main__":
    print("Running compare tests...")
    test_compare_runs_all_methods()
    print("ALL TESTS PASSED")
