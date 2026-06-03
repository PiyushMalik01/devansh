"""Sparse-RS (Croce et al. 2022) -- sparse-pixel black-box random search.

Perturbs a fixed number ``k`` of pixels with corner values in {-1, +1} (per
channel), added to the image and clipped. Each iteration resamples a fraction
of the perturbed pixels (locations + values) on a decreasing schedule and
accepts the change if the loss improves. Single-objective (loss only): it does
not minimize L2 -- which is exactly the visibility weakness ScatterCamo targets.
"""

import numpy as np

from scattercamo.runner.result import AttackResult


def _p_schedule(it, n_queries, p_init):
    """Sparse-RS fraction-of-pixels schedule (decreasing in query progress)."""
    t = int(it / max(n_queries, 1) * 10000)
    table = [(50, 2), (200, 4), (500, 5), (1000, 6), (2000, 8),
             (4000, 10), (6000, 12), (8000, 15)]
    for thresh, div in table:
        if t <= thresh:
            return p_init / div
    return p_init / 20


DEFAULTS = {"eps": 150, "queries": 10000, "p_init": 0.3, "seed": 0}


class SparseRSAttack:
    def __init__(self, params):
        self.p = {**DEFAULTS, **params}
        self.rng = np.random.default_rng(self.p["seed"])

    def _apply(self, x, idx, vals):
        h, w = x.shape[:2]
        adv = x.copy()
        rows, cols = idx // w, idx % w
        adv[rows, cols] = np.clip(adv[rows, cols] + vals, 0.0, 1.0)
        return adv

    def optimise(self, loss_function):
        x = self.p["x"]
        h, w = x.shape[:2]
        k, budget = self.p["eps"], self.p["queries"]
        n_pixels = h * w

        idx = self.rng.choice(n_pixels, size=k, replace=False)
        vals = self.rng.choice([-1.0, 1.0], size=(k, 3))
        adv = self._apply(x, idx, vals)
        is_adv, loss = loss_function(adv)
        used = 1

        best_adv = adv.copy() if is_adv else None
        best_loss = loss if is_adv else np.inf
        history = []

        while used < budget:
            p = _p_schedule(used, budget, self.p["p_init"])
            n_change = max(int(p * k), 1)

            new_idx, new_vals = idx.copy(), vals.copy()
            change = self.rng.choice(k, size=n_change, replace=False)
            free = np.setdiff1d(np.arange(n_pixels), idx, assume_unique=False)
            new_idx[change] = self.rng.choice(free, size=n_change, replace=False)
            new_vals[change] = self.rng.choice([-1.0, 1.0], size=(n_change, 3))

            cand = self._apply(x, new_idx, new_vals)
            is_adv, new_loss = loss_function(cand)
            used += 1

            if new_loss < loss:
                idx, vals, loss = new_idx, new_vals, new_loss
                adv = cand
            if is_adv and new_loss < best_loss:
                best_adv, best_loss = cand.copy(), new_loss
            history.append({"queries": used, "loss": float(loss)})

        return AttackResult(adv_image=best_adv, success=best_adv is not None,
                            queries=used, history=history)
