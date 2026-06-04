"""Analysis plots actually produce non-empty image files."""

import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scattercamo.analysis import (
    frontier_curve, pareto_plot, qualitative_grid, hideability_panel,
)


def _nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def test_frontier_curve():
    series = {
        "ScatterCamo": [(0.98, 0.85), (0.95, 0.90), (0.90, 0.93)],
        "Sparse-RS": [(0.92, 0.84), (0.88, 0.90)],
    }
    with tempfile.TemporaryDirectory() as d:
        out = frontier_curve(series, os.path.join(d, "frontier.png"))
        assert _nonempty(out)
    print("  [ok] frontier_curve produced a figure")


def test_pareto_plot():
    front = [SimpleNamespace(fitnesses=np.array([0.1 * i, 5.0 - i]),
                             is_adversarial=(i % 2 == 0)) for i in range(5)]
    with tempfile.TemporaryDirectory() as d:
        out = pareto_plot(front, os.path.join(d, "pareto.png"))
        assert _nonempty(out)
    print("  [ok] pareto_plot produced a figure")


def test_qualitative_grid():
    rng = np.random.default_rng(0)
    imgs = [rng.random((24, 24, 3)) for _ in range(3)]
    with tempfile.TemporaryDirectory() as d:
        out = qualitative_grid(imgs, ["orig", "ScatterCamo", "CamoPatch"],
                               os.path.join(d, "grid.png"))
        assert _nonempty(out)
    print("  [ok] qualitative_grid produced a figure")


def test_hideability_panel():
    rng = np.random.default_rng(0)
    x = rng.random((32, 32, 3))
    with tempfile.TemporaryDirectory() as d:
        out = hideability_panel(x, os.path.join(d, "hideability.png"))
        assert _nonempty(out)
    print("  [ok] hideability_panel produced a figure")


if __name__ == "__main__":
    print("Running analysis tests...")
    test_frontier_curve()
    test_pareto_plot()
    test_qualitative_grid()
    test_hideability_panel()
    print("ANALYSIS TESTS PASSED")
