"""The ScatterCamo attack: NSGA-II over scattered camouflaged shapes.

Loop (per generation):
    1. non-dominated sort + crowding distance on the current population
    2. binary-tournament selection of parents
    3. crossover + mutation -> children (each child costs one model query)
    4. (mu + lambda) survival: keep the best ``pop_size`` by rank then crowding

Termination: the query budget is exhausted. The result carries the best
adversarial solution found (lowest L2 among adversarial), the final Pareto
front, the query count, and a per-generation history for plotting.
"""

import logging
import numpy as np
from operator import attrgetter

from scattercamo.representation import random_genome
from scattercamo.perception import hideability_map, seed_positions
from scattercamo.search import (
    Solution,
    fast_nondominated_sort,
    calculate_crowding_distance,
    tournament_selection,
)
from scattercamo.operators import generate_offspring

log = logging.getLogger(__name__)


DEFAULTS = {
    "M": 10,                # shapes per solution (the sparsity knob, swept externally)
    "max_radius_frac": 0.10,  # max shape radius as a fraction of min(h, w)
    "queries": 10000,       # model-query budget
    "pop_size": 20,
    "pc": 0.3,              # crossover rate (fraction of shapes swapped)
    "pm": 0.3,              # mutation rate (re-roll vs jitter)
    "tournament_size": 2,
    "seed": 0,
    # Perceptual placement (off by default -> identical to plain ScatterCamo).
    "perceptual": False,    # master switch for hideability-aware placement
    "mask_dark": 1.0,       # luminance-masking weight  (used only if perceptual)
    "mask_edges": 1.0,      # edge-masking weight        (used only if perceptual)
    "mask_texture": 1.0,    # texture-masking weight     (used only if perceptual)
    "mask_window": 7,       # local-variance window for the texture signal
}


class ScatterCamoAttack:
    def __init__(self, params):
        self.p = {**DEFAULTS, **params}
        if self.p["pop_size"] < 2:
            raise ValueError("pop_size must be >= 2 for tournament selection")
        self.rng = np.random.default_rng(self.p["seed"])
        np.random.seed(self.p["seed"])  # tournament uses the global RNG
        h, w = self.p["x"].shape[:2]
        self.max_radius = self.p["max_radius_frac"] * min(h, w)
        self.history = []

        # Perceptual placement: build the hideability map once from the clean
        # image, then seed positions from it and reweight the L2 objective.
        if self.p["perceptual"]:
            self.W = hideability_map(
                self.p["x"], self.p["mask_dark"], self.p["mask_edges"],
                self.p["mask_texture"], self.p["mask_window"],
            )
            self.visibility = 1.0 - self.W
        else:
            self.W = None
            self.visibility = None

    def _new_solution(self):
        genome = random_genome(self.p["M"], self.rng)
        if self.W is not None:
            genome[:, :2] = seed_positions(self.W, self.p["M"], self.rng)
        return Solution(genome, self.p["x"], self.max_radius, self.visibility)

    @staticmethod
    def _best_adversarial(population):
        adv = [s for s in population if s.is_adversarial]
        return min(adv, key=lambda s: s.fitnesses[1]) if adv else None

    def _survival(self, combined, pop_size):
        fronts = fast_nondominated_sort(combined)
        survivors, i = [], 0
        while i < len(fronts) and survivors.__len__() + len(fronts[i]) <= pop_size:
            calculate_crowding_distance(fronts[i])
            survivors.extend(fronts[i])
            i += 1
        if len(survivors) < pop_size and i < len(fronts) and fronts[i]:
            calculate_crowding_distance(fronts[i])
            fronts[i].sort(key=attrgetter("crowding_distance"), reverse=True)
            survivors.extend(fronts[i][: pop_size - len(survivors)])
        return survivors

    def optimise(self, loss_function):
        pop_size, budget = self.p["pop_size"], self.p["queries"]
        log.info("start: M=%d pop=%d budget=%d perceptual=%s max_radius=%.2fpx",
                 self.p["M"], pop_size, budget, self.W is not None, self.max_radius)

        population = [self._new_solution() for _ in range(pop_size)]
        for s in population:
            s.evaluate(loss_function)
        used = pop_size
        best = self._best_adversarial(population)
        log.info("step init: evaluated %d solutions, adversarial_found=%s",
                 used, best is not None)

        gen = 0
        while used < budget:
            gen += 1
            for front in fast_nondominated_sort(population):
                calculate_crowding_distance(front)

            parents = tournament_selection(population, self.p["tournament_size"])
            children = generate_offspring(parents, self.p["pc"], self.p["pm"], self.rng)
            if not children:
                log.warning("step gen %d: no children produced, stopping early", gen)
                break

            evaluated = []
            for c in children:
                if used >= budget:
                    break
                c.evaluate(loss_function)
                used += 1
                evaluated.append(c)        # only evaluated children enter survival

            population = self._survival(population + evaluated, pop_size)
            current_best = self._best_adversarial(population)
            if current_best is not None and (
                best is None or current_best.fitnesses[1] < best.fitnesses[1]
            ):
                best = current_best

            min_loss = float(min(s.loss for s in population))
            best_l2 = float(best.fitnesses[1]) if best else None
            self.history.append({"queries": used, "best_l2": best_l2, "min_loss": min_loss})
            log.debug("step gen %d: queries=%d/%d best_l2=%s min_loss=%.4f",
                      gen, used, budget,
                      f"{best_l2:.4f}" if best_l2 is not None else "n/a", min_loss)

        log.info("done: generations=%d queries=%d success=%s best_l2=%s",
                 gen, used, best is not None,
                 f"{best.fitnesses[1]:.4f}" if best is not None else "n/a")

        return {
            "best": best,
            "adv_image": best.generate_image() if best is not None else None,
            "front": fast_nondominated_sort(population)[0],
            "queries": used,
            "success": best is not None,
            "history": self.history,
        }
