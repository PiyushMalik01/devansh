"""Genetic operators for the shape genome.

- crossover: swap a subset of *whole shapes* between two parents (SA-MOO style,
  but the unit of exchange is a shape rather than a pixel).
- mutation: pick one shape and either re-roll or jitter a subset of its genes
  (CamoPatch style). Jittering position relocates the shape on the canvas.
"""

import numpy as np

from scattercamo.representation import GENES_PER_SHAPE


def crossover(g1, g2, pc, rng):
    """Swap ``ceil(M*pc)`` whole shapes between two genomes. Returns two offspring."""
    m = g1.shape[0]
    l = max(int(m * pc), 1)
    idx = rng.choice(m, size=l, replace=False)
    o1, o2 = g1.copy(), g2.copy()
    o1[idx], o2[idx] = g2[idx].copy(), g1[idx].copy()
    return o1, o2


def mutation(genome, pm, rng):
    """Mutate a single shape's genes in place on a copy. Returns the new genome."""
    g = genome.copy()
    m = g.shape[0]
    s = int(rng.integers(0, m))
    n_genes = int(rng.integers(1, GENES_PER_SHAPE + 1))
    sel = rng.choice(GENES_PER_SHAPE, size=n_genes, replace=False)
    if rng.random() < pm:
        g[s, sel] = rng.random(len(sel))                       # re-roll
    else:
        g[s, sel] += (rng.random(len(sel)) - 0.5) / 3.0        # small jitter
        g[s, sel] = np.clip(g[s, sel], 0.0, 1.0)
    return g


def generate_offspring(parents, pc, pm, rng):
    """Produce two children per parent pair (crossover then mutation)."""
    children = []
    for p1, p2 in parents:
        o1, o2 = crossover(p1.genome, p2.genome, pc, rng)
        c1 = p1.copy()
        c1.genome = mutation(o1, pm, rng)
        c2 = p2.copy()
        c2.genome = mutation(o2, pm, rng)
        children.extend([c1, c2])
    return children
