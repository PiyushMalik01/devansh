# ScatterCamo — Method

## Threat model

- **Black-box**: only the model's output logits/probabilities are observed.
- **Untargeted**: succeed when the predicted label differs from the true label.
- **Query-limited**: a hard budget (default 10,000 queries) caps evaluations.
- **Dataset**: ImageNet (224×224), classifiers ResNet50 / VGG16-bn / one
  adversarially-trained model.

## Representation

A solution is a genome of `M` shapes, each 7 genes in `[0, 1]`:

```
(y, x, radius, R, G, B, alpha)
```

The adversarial image is produced by sequentially alpha-blending each shape
onto the original image (later shapes paint over earlier ones), then clipping
to `[0, 1]`. This generalizes CamoPatch's circle gene from a fixed square
patch to the full canvas. Rendering is pure NumPy (`representation/shapes.py`).

## Objectives

```
F(δ) = ( L(x + δ),  ||δ||₂² )
```

- `L` is the untargeted margin loss `f_true − f_other` (adversarial when < 0).
- The search uses *squared* L2 (monotonic, cheap); reporting uses the true norm.

Sparsity (L0) is **not** a search objective here. It is controlled by `M` and
swept externally over `{1, 5, 10, 20, 40}` to trace the frontier — the standard
convention in the sparse-attack literature, and the reason this (Approach B) is
far more likely to converge under a tight query budget than a 3-objective search.

## Search — NSGA-II with prioritized domination

Standard NSGA-II (fast non-dominated sort + crowding distance + binary
tournament + (μ+λ) survival), with SA-MOO's **prioritized domination relation**:

> `A` dominates `B` if:
> 1. `A` is adversarial and `B` is not; or
> 2. both adversarial and `‖δ_A‖₂ < ‖δ_B‖₂`; or
> 3. both non-adversarial and `L(A) < L(B)`.

This drives the population to *first* find adversarial solutions, *then*
minimize their visibility — without ever trading away adversariality.

### Operators

- **Crossover**: swap a subset of whole shapes between two parents.
- **Mutation**: pick one shape; re-roll or jitter a random subset of its genes.
  Jittering `(y, x)` relocates the shape on the canvas.

## Evaluation plan (lean & deep)

- Models: ResNet50, VGG16-bn, one adversarially-trained (RobustBench).
- Baselines: CamoPatch, SA-MOO, Sparse-RS — all wrapped to the same runner.
- Metrics: ASR, L0, L2, SSIM, queries. Headline plot: SSIM (or L2) vs ASR as
  `M` varies — show ScatterCamo dominates the invisibility/strength frontier.

## Future work — the variable-M, tri-objective extension (Approach A)

Add a per-shape presence gene and a third objective (active-shape count / area),
turning `M` from a fixed hyperparameter into something the search discovers.
Higher novelty, but 3-objective search loses selection pressure and needs more
queries — deferred until Approach B produces a solid baseline result.
