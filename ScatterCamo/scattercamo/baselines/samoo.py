"""SA-MOO (Williams & Li, CVPR 2023) -- sparse-pixel multi-objective attack.

NSGA-II over a pixel genome: a fixed set of ``eps`` perturbed pixels, each with
a value in {-1, 0, +1} per channel (scaled by ``p_size``). Bi-objective
(loss, L2) with the prioritized domination relation. Reuses the shared NSGA-II
sort/crowding/tournament so it differs from ScatterCamo only in representation.
"""

import numpy as np
from copy import deepcopy
from operator import attrgetter

from scattercamo.search import (
    fast_nondominated_sort,
    calculate_crowding_distance,
    tournament_selection,
)
from scattercamo.runner.result import AttackResult


class PixelSolution:
    def __init__(self, pixels, values, x, p_size):
        self.pixels = pixels          # (eps,) flat indices
        self.values = values          # (eps, 3) in {-1, 0, 1}
        self.x = x
        self.p_size = p_size
        self.w = x.shape[1]
        self.fitnesses = None
        self.is_adversarial = None
        self.loss = None
        self.domination_count = None
        self.dominated_solutions = None
        self.rank = None
        self.crowding_distance = None

    def copy(self):
        return deepcopy(self)

    def generate_image(self):
        adv = self.x.copy()
        rows, cols = self.pixels // self.w, self.pixels % self.w
        adv[rows, cols] = np.clip(adv[rows, cols] + self.values * self.p_size, 0.0, 1.0)
        return adv

    def evaluate(self, loss_function):
        adv = self.generate_image()
        is_adv, loss = loss_function(adv)
        self.is_adversarial = bool(is_adv)
        self.loss = float(loss)
        self.fitnesses = np.array([self.loss, float(np.sum((adv - self.x) ** 2))])

    def dominates(self, other):
        if self.is_adversarial and not other.is_adversarial:
            return True
        if not self.is_adversarial and other.is_adversarial:
            return False
        if self.is_adversarial and other.is_adversarial:
            return self.fitnesses[1] < other.fitnesses[1]
        return self.fitnesses[0] < other.fitnesses[0]


DEFAULTS = {"eps": 150, "queries": 10000, "pop_size": 20, "pc": 0.1, "pm": 0.4,
            "p_size": 1.0, "zero_prob": 0.3, "tournament_size": 2, "seed": 0}


class SAMOOAttack:
    def __init__(self, params):
        self.p = {**DEFAULTS, **params}
        self.rng = np.random.default_rng(self.p["seed"])
        np.random.seed(self.p["seed"])

    def _sample_values(self, n):
        ones = (1 - self.p["zero_prob"]) / 2
        return self.rng.choice([-1, 1, 0], size=(n, 3),
                               p=(ones, ones, self.p["zero_prob"]))

    def _new_solution(self, n_pixels):
        pixels = self.rng.choice(n_pixels, size=self.p["eps"], replace=False)
        return PixelSolution(pixels, self._sample_values(self.p["eps"]),
                             self.p["x"], self.p["p_size"])

    def _mutate(self, soln, n_pixels):
        child = soln.copy()
        eps = len(soln.pixels)
        n_change = max(int(eps * self.p["pm"]), 1)
        keep = self.rng.choice(eps, size=eps - n_change, replace=False)
        free = np.setdiff1d(np.arange(n_pixels), soln.pixels)
        new_pix = self.rng.choice(free, size=n_change, replace=False)
        child.pixels = np.concatenate([soln.pixels[keep], new_pix])
        child.values = np.concatenate([soln.values[keep], self._sample_values(n_change)])
        return child

    def _crossover(self, a, b):
        eps = len(a.pixels)
        l = max(int(eps * self.p["pc"]), 1)
        diff = np.array([i for i in range(eps) if b.pixels[i] not in a.pixels])
        off = a.copy()
        if len(diff) > 0:
            l = min(l, len(diff))
            sw = self.rng.choice(diff, size=l, replace=False)
            off.pixels[sw] = b.pixels[sw].copy()
            off.values[sw] = b.values[sw].copy()
        return off

    def _survival(self, combined, pop_size):
        fronts = fast_nondominated_sort(combined)
        survivors, i = [], 0
        while i < len(fronts) and len(survivors) + len(fronts[i]) <= pop_size:
            calculate_crowding_distance(fronts[i])
            survivors.extend(fronts[i])
            i += 1
        if len(survivors) < pop_size and i < len(fronts) and fronts[i]:
            calculate_crowding_distance(fronts[i])
            fronts[i].sort(key=attrgetter("crowding_distance"), reverse=True)
            survivors.extend(fronts[i][: pop_size - len(survivors)])
        return survivors

    @staticmethod
    def _best_adv(pop):
        adv = [s for s in pop if s.is_adversarial]
        return min(adv, key=lambda s: s.fitnesses[1]) if adv else None

    def optimise(self, loss_function):
        x = self.p["x"]
        n_pixels = x.shape[0] * x.shape[1]
        pop_size, budget = self.p["pop_size"], self.p["queries"]

        pop = [self._new_solution(n_pixels) for _ in range(pop_size)]
        for s in pop:
            s.evaluate(loss_function)
        used = pop_size
        best = self._best_adv(pop)
        history = []

        while used < budget:
            for f in fast_nondominated_sort(pop):
                calculate_crowding_distance(f)
            parents = tournament_selection(pop, self.p["tournament_size"])
            children = []
            for p1, p2 in parents:
                child = self._mutate(self._crossover(p1, p2), n_pixels)
                children.append(child)
            if not children:
                break
            for c in children:
                if used >= budget:
                    break
                c.evaluate(loss_function)
                used += 1
            pop = self._survival(pop + children, pop_size)
            cur = self._best_adv(pop)
            if cur is not None and (best is None or cur.fitnesses[1] < best.fitnesses[1]):
                best = cur
            history.append({"queries": used,
                            "best_l2": float(best.fitnesses[1]) if best else None})

        return AttackResult(
            adv_image=best.generate_image() if best else None,
            success=best is not None, queries=used, history=history)
