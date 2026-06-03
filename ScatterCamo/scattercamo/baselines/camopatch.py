"""CamoPatch (Williams & Li, NeurIPS 2023) -- single camouflaged square patch.

A genome of ``N`` semi-transparent circles is rendered into an s x s patch and
placed at a location on the image. Appearance is optimized by a (1+1)-ES; the
location by simulated annealing. Dual acceptance: while not adversarial, accept
on lower loss; once adversarial, accept on lower L2 to the covered region.

Pure-NumPy circle rendering (no OpenCV), consistent with ScatterCamo's renderer.
"""

import math
import numpy as np

from scattercamo.runner.result import AttackResult


def render_patch(genome, s, max_radius):
    """Render a genome of circles onto an s x s white canvas in [0, 1]."""
    canvas = np.ones((s, s, 3), dtype=np.float64)
    yy, xx = np.ogrid[:s, :s]
    for row in genome:
        cy, cx = row[0] * s, row[1] * s
        r = row[2] * max_radius + 1.0
        alpha = float(row[6])
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        canvas[mask] = alpha * row[3:6] + (1.0 - alpha) * canvas[mask]
    return np.clip(canvas, 0.0, 1.0)


def _mutate(genome, mut, rng):
    g = genome.copy()
    n, length = g.shape
    row = int(rng.integers(0, n))
    n_change = int(rng.integers(0, length + 1))
    sel = rng.choice(length, size=n_change, replace=False) if n_change else np.array([], int)
    if len(sel):
        if rng.random() < mut:
            g[row, sel] = rng.random(len(sel))
        else:
            g[row, sel] = np.clip(g[row, sel] + (rng.random(len(sel)) - 0.5) / 3.0, 0, 1)
    return g


DEFAULTS = {"N": 100, "s": 40, "queries": 10000, "mut": 0.3, "temp": 300.0,
            "update_loc_period": 4, "seed": 0}


class CamoPatchAttack:
    def __init__(self, params):
        self.p = {**DEFAULTS, **params}
        self.rng = np.random.default_rng(self.p["seed"])

    def _place(self, x, patch, loc, s):
        adv = x.copy()
        adv[loc[0]:loc[0] + s, loc[1]:loc[1] + s, :] = patch
        return np.clip(adv, 0.0, 1.0)

    @staticmethod
    def _l2_region(patch, x, loc, s):
        region = x[loc[0]:loc[0] + s, loc[1]:loc[1] + s, :]
        return float(np.sum((patch - region) ** 2))

    def optimise(self, loss_function):
        x = self.p["x"]
        h, w = x.shape[:2]
        s = min(self.p["s"], h, w)
        max_radius = s / 6.0
        budget = self.p["queries"]

        genome = self.rng.random((self.p["N"], 7))
        patch = render_patch(genome, s, max_radius)
        loc = self.rng.integers(0, h - s + 1, size=2)
        adv = self._place(x, patch, loc, s)
        is_adv, loss = loss_function(adv)
        l2 = self._l2_region(patch, x, loc, s)
        used = 1

        best_adv = adv.copy() if is_adv else None
        counter = 0
        history = []

        while used < budget:
            counter += 1
            if counter < self.p["update_loc_period"]:
                new_genome = _mutate(genome, self.p["mut"], self.rng)
                new_patch = render_patch(new_genome, s, max_radius)
                cand = self._place(x, new_patch, loc, s)
                is_adv_c, loss_c = loss_function(cand)
                used += 1
                l2_c = self._l2_region(new_patch, x, loc, s)
                accept = (l2_c < l2) if (is_adv and is_adv_c) else (loss_c < loss)
                if accept:
                    genome, patch, adv, loss, l2, is_adv = (
                        new_genome, new_patch, cand, loss_c, l2_c, is_adv_c)
            else:
                counter = 0
                shift = int(max((budget - used) / budget, 0) * 0.75 * h)
                new_loc = np.clip(loc + self.rng.integers(-shift - 1, shift + 2, size=2),
                                  0, h - s)
                cand = self._place(x, patch, new_loc, s)
                is_adv_c, loss_c = loss_function(cand)
                used += 1
                l2_c = self._l2_region(patch, x, new_loc, s)
                if is_adv and is_adv_c:
                    accept = l2_c < l2
                else:
                    temp = self.p["temp"] / (used + 1)
                    metropolis = math.exp(-min((loss_c - loss) / temp, 50))
                    accept = loss_c < loss or self.rng.random() < metropolis
                if accept:
                    loc, adv, loss, l2, is_adv = new_loc, cand, loss_c, l2_c, is_adv_c

            if is_adv:
                best_adv = adv.copy()
            history.append({"queries": used, "loss": float(loss), "l2": float(l2)})

        return AttackResult(adv_image=best_adv, success=best_adv is not None,
                            queries=used, history=history)
