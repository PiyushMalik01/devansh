# ScatterCamo — Design Decisions

This captures the brainstorm that produced the repo, so the rationale is not lost.

## Goal

A **paper-grade novel hybrid** of CamoPatch and SA-MOO — not a reproduction, a
benchmark toolkit, or a defense study.

## Fusion direction (chosen)

**Representation fusion** — "sparse camouflaged shapes": K small camouflaged
shapes scattered anywhere, instead of one square patch (CamoPatch) or scattered
single pixels (SA-MOO). It subsumes both papers as special cases.

## Realization: Approach B over Approach A

Three realizations were considered:

| | Genome | Objectives | Risk | Verdict |
|---|---|---|---|---|
| **A** | variable-M, presence-gated | `(loss, L2, L0)` tri-objective | high | future work |
| **B** | fixed-M | `(loss, L2)` bi-objective | low | **chosen** |
| C | two-stage SA-MOO→CamoPatch | mixed | low | weak novelty, rejected |

**Why B, not A.** A is the most novel-sounding and exactly where these projects
die under a black-box query budget:

1. **3-objective search loses selection pressure** — far more solutions become
   mutually non-dominated, so NSGA-II drifts instead of converging. SA-MOO
   itself deliberately collapsed 3 objectives to 2; A re-introduces what they
   engineered away.
2. **Variable-length genomes are fragile** — add/remove-shape operators easily
   degenerate, and the L0 objective gets gamed (drop all shapes).
3. **Query budget** — 3-objective convergence needs more evaluations than the
   ~10k ceiling allows.

B is the intersection of the parts *proven to work* in both papers (SA-MOO's
bi-objective NSGA-II + CamoPatch's camouflaged representation) recombined into
something unpublished. It converges, it ships, and `M`-sweeps trace the
frontier the conventional way. B is a strict code subset of A, so A remains a
clean v2 / future-work extension.

## Constraints

- **Compute**: RTX 4060 for fast dev; rentable cloud for full runs. Repo is
  config-driven (`configs/dev.yaml` vs `configs/full.yaml`), seeded, and
  designed for checkpoint/resume (to survive cloud preemption — TODO).
- **Eval scope (lean & deep)**: black-box, untargeted, ImageNet only; ~3 models;
  baselines CamoPatch / SA-MOO / Sparse-RS; ~10k queries. All energy on showing
  ScatterCamo wins the L0/L2/SSIM-vs-ASR frontier.

## Module boundaries (why this structure)

`representation/`, `search/`, `operators/` are the three loci of novelty and are
each independently testable (deterministic render; pure-predicate domination;
validity-preserving operators). `baselines/` adapting to the *same* runner API
is what makes the comparison fair. `models/` + `losses/` are lifted near-verbatim
from the source repos since they already share an interface.

## Verification

`tests/test_smoke.py` runs NumPy-only (no torch / no model download) and proves:
domination correctness, renderer determinism + actual perturbation, operator
genome validity, and full NSGA-II convergence to an adversarial solution within
budget against a mock model. All passing as of scaffold.
