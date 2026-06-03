"""Batch runner: aggregates metrics, checkpoints, and resumes by skipping done work."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mockmodel import MockModel, small_dataset

from scattercamo.losses import UnTargeted
from scattercamo.attack import ScatterCamoAttack
from scattercamo.runner import BatchRunner


def _attack_factory(x):
    return ScatterCamoAttack({"x": x, "M": 6, "queries": 120, "pop_size": 8, "seed": 0})


def _loss_factory(x, label):
    return UnTargeted(MockModel(x, true=label, flip=0.01), true=label, to_pytorch=False)


def test_runner_aggregates():
    data = small_dataset(n=3, size=24)
    with tempfile.TemporaryDirectory() as d:
        runner = BatchRunner(_attack_factory, _loss_factory, out_dir=d, name="sc")
        summary = runner.run(data)
        assert summary["n"] == 3
        assert 0.0 <= summary["asr"] <= 1.0
        assert len(summary["records"]) == 3
        assert os.path.exists(os.path.join(d, "sc.json"))
        # successful records carry the visibility metrics
        for r in summary["records"]:
            if r["success"]:
                assert "l0" in r and "l2" in r and "ssim" in r
    print(f"  [ok] runner aggregated 3 images, ASR={summary['asr']:.2f}")


def test_runner_resumes():
    data = small_dataset(n=3, size=24)
    with tempfile.TemporaryDirectory() as d:
        BatchRunner(_attack_factory, _loss_factory, out_dir=d, name="sc").run(data)

        def exploding_factory(x):
            raise AssertionError("attack should not run for already-completed images")

        runner2 = BatchRunner(exploding_factory, _loss_factory, out_dir=d, name="sc")
        summary = runner2.run(data)        # everything is checkpointed -> no attack runs
        assert summary["n"] == 3
    print("  [ok] runner resumed from checkpoint without recomputing")


if __name__ == "__main__":
    print("Running runner tests...")
    test_runner_aggregates()
    test_runner_resumes()
    print("RUNNER TESTS PASSED")
