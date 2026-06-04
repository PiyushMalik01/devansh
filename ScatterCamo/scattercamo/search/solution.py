"""NSGA-II machinery with SA-MOO's prioritized domination relation.

The non-dominated sort, crowding distance, and tournament selection are the
standard NSGA-II operators (Deb et al. 2002), adapted from SA-MOO's MOAA.
The novelty we inherit is the *prioritized* domination in ``Solution.dominates``:
an adversarial solution always beats a non-adversarial one; ties broken by L2
(if both adversarial) or by loss (if neither is).
"""

import numpy as np
from operator import attrgetter

from scattercamo.representation import generate_image, l2_perturbation
from scattercamo.perception import weighted_l2


class Solution:
    """A candidate attack: a genome of shapes plus its evaluation state."""

    def __init__(self, genome, x, max_radius, visibility=None):
        self.genome = genome          # (M, 7)
        self.x = x                    # original image (h, w, 3)
        self.max_radius = max_radius
        self.visibility = visibility  # (h, w) in [0, 1] = 1 - W; None -> plain L2

        self.fitnesses = None         # np.array([loss, l2_squared])
        self.is_adversarial = None
        self.loss = None

        # NSGA-II bookkeeping
        self.domination_count = None
        self.dominated_solutions = None
        self.rank = None
        self.crowding_distance = None

    def copy(self):
        """Lightweight clone: copy the (tiny) genome, share the immutable image
        and visibility map by reference.

        ``x`` and ``visibility`` are never mutated in place (the renderer copies
        ``x`` before drawing), so sharing them avoids deep-copying the full image
        for every child each generation. The clone starts with fresh evaluation
        state — every child is re-evaluated before it is used.
        """
        return Solution(self.genome.copy(), self.x, self.max_radius, self.visibility)

    def generate_image(self):
        return generate_image(self.genome, self.x, self.max_radius)

    def evaluate(self, loss_function):
        adv = self.generate_image()
        is_adv, loss = loss_function(adv)
        self.is_adversarial = bool(is_adv)
        self.loss = float(loss)
        if self.visibility is None:
            l2 = l2_perturbation(adv, self.x)
        else:
            l2 = weighted_l2(adv, self.x, self.visibility)
        self.fitnesses = np.array([self.loss, l2], dtype=np.float64)

    def dominates(self, other):
        """Prioritized domination (SA-MOO Definition 3.1)."""
        if self.is_adversarial and not other.is_adversarial:
            return True
        if not self.is_adversarial and other.is_adversarial:
            return False
        if self.is_adversarial and other.is_adversarial:
            return self.fitnesses[1] < other.fitnesses[1]   # lower L2 wins
        return self.fitnesses[0] < other.fitnesses[0]        # lower loss wins


def fast_nondominated_sort(population):
    """Partition ``population`` into Pareto fronts (front 0 = best)."""
    fronts = [[]]
    for individual in population:
        individual.domination_count = 0
        individual.dominated_solutions = []
        for other in population:
            if individual.dominates(other):
                individual.dominated_solutions.append(other)
            elif other.dominates(individual):
                individual.domination_count += 1
        if individual.domination_count == 0:
            individual.rank = 0
            fronts[0].append(individual)
    i = 0
    while len(fronts[i]) > 0:
        nxt = []
        for individual in fronts[i]:
            for other in individual.dominated_solutions:
                other.domination_count -= 1
                if other.domination_count == 0:
                    other.rank = i + 1
                    nxt.append(other)
        i += 1
        fronts.append(nxt)
    return fronts


def calculate_crowding_distance(front):
    if len(front) == 0:
        return
    n = len(front)
    for ind in front:
        ind.crowding_distance = 0
    for m in range(len(front[0].fitnesses)):
        front.sort(key=lambda ind: ind.fitnesses[m])
        front[0].crowding_distance = 1e9
        front[n - 1].crowding_distance = 1e9
        values = [ind.fitnesses[m] for ind in front]
        scale = max(values) - min(values)
        if scale == 0:
            scale = 1
        for i in range(1, n - 1):
            front[i].crowding_distance += (
                front[i + 1].fitnesses[m] - front[i - 1].fitnesses[m]
            ) / scale


def crowding_operator(a, b):
    """Return 1 if ``a`` is preferred to ``b`` (lower rank, or more crowded gap)."""
    if (a.rank < b.rank) or (a.rank == b.rank and a.crowding_distance > b.crowding_distance):
        return 1
    return -1


def _tournament(population, k):
    idx = np.random.choice(len(population), size=k, replace=False)
    best = None
    for i in idx:
        cand = population[i]
        if best is None or crowding_operator(cand, best) == 1:
            best = cand
    return best


def tournament_selection(population, k):
    """Return len(population)//2 parent pairs by binary-tournament selection."""
    parents = []
    while len(parents) < len(population) // 2:
        parents.append((_tournament(population, k), _tournament(population, k)))
    return parents
