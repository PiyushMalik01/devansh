# Perceptually-weighted ScatterCamo — design spec

**Date:** 2026-06-04
**Status:** approved, implementing
**Scope:** C+A (perceptual objective + seeding) with full per-signal ablation toggles.

## Problem

Today shape centers are drawn uniformly over the image (`shapes.random_genome`),
and the visibility objective is a plain squared-L2. Perturbations therefore land
in visually exposed regions (smooth sky, skin) as readily as in hideable ones.
Human-vision *masking* says perturbations are far less perceptible in three kinds
of region — dark, edge, and textured — so we should bias both placement and the
search objective toward them.

## Approach

Build a per-pixel **hideability map** `W(y,x) ∈ [0,1]` from the clean image (zero
model queries), high where a change is imperceptible:

```
dark    = 1 − luminance          (luminance / Weber masking)
edges   = |Sobel gradient|       (contrast / edge masking)
texture = local variance         (texture masking)
W = floor( normalize( w_dark·dark + w_edges·edges + w_texture·texture ) )
```

Each signal is min–max normalized; the weighted sum is normalized again and given
a small floor `eps` so `W ∈ [eps, 1]` (a soft prior, never a hard ban). Degenerate
configs (all weights 0, or a constant image) collapse `W` toward 0 → plain-L2
behavior, i.e. a safe no-op.

Use `W` two ways:

- **A — seeding:** initial shape centers sampled with probability ∝ `W`.
- **C — objective:** replace the second fitness with **visibility-weighted**
  squared-L2, `Σ (adv−x)² · (1−W)`, so visible-pixel changes cost more and the
  search continuously evolves patches toward hideable regions.

The genome, renderer, genetic operators, and NSGA-II machinery are **unchanged**.
Reported metrics (`metrics.l0/l2/ssim`) are **unchanged** — they still measure
true visible distortion, keeping all comparisons fair.

## Components

| Unit | File | Responsibility |
|---|---|---|
| `hideability_map(x, w_dark, w_edges, w_texture, window, eps)` | NEW `scattercamo/perception/masking.py` | Build `W` (pure NumPy). |
| `seed_positions(W, m, rng)` | same | Sample `m` centers ∝ `W`, return as `(m,2)` genes in `[0,1)`. |
| `weighted_l2(adv, x, visibility)` | same | Visibility-weighted squared-L2. |
| `Solution(..., visibility=None)` | `search/solution.py` | When `visibility` set, second fitness uses `weighted_l2`; else identical to today. |
| Attack wiring | `attack/scattercamo.py` | Build `W`/`visibility` once when `perceptual`; seed new solutions; thread `visibility` in. |
| CLI | `run_attack.py` | `--perceptual`, `--mask_dark/edges/texture`, `--mask_window`. |

## Config (defaults preserve current behavior)

```
--perceptual        off       master switch
--mask_dark    1.0            luminance weight   (only used when --perceptual)
--mask_edges   1.0            edge weight        (only used when --perceptual)
--mask_texture 1.0            texture weight     (only used when --perceptual)
--mask_window  7              local-variance window (odd)
```

Ablation runs: uniform = perceptual off; dark-only = `--perceptual --mask_edges 0
--mask_texture 0`; etc.

## Data flow

`x → (perceptual?) build W once → visibility = 1−W`. Seed initial genomes'
positions ∝ `W`. Each `Solution.evaluate` computes `weighted_l2` from `visibility`.
NSGA-II unchanged. Children inherit `visibility` through `Solution.copy()`. Final
report uses untouched true metrics.

## Error handling / edge cases

- Constant signal → that signal normalizes to 0 (no contribution).
- All weights 0 → `W ≈ eps` everywhere → near-plain-L2, uniform seeding.
- `seed_positions` with zero-sum `W` → uniform fallback.
- `visibility` broadcast `(h,w)→(h,w,1)` over RGB.

## Testing (`tests/test_perception.py`, NumPy-only)

1. `W` shape `(h,w)`, range `[eps,1]`.
2. Dark half of an image → higher `W` there (dark-only weights).
3. Vertical edge → `W` peaks at the edge column (edge-only weights).
4. Noisy vs flat region → higher `W` in noisy region (texture-only weights).
5. `seed_positions` concentrates samples in a high-`W` quadrant.
6. `weighted_l2`: same-magnitude change costs less where `W` high.
7. Backward-compat: `Solution(visibility=None)` second fitness == `l2_perturbation`.
8. Integration: `ScatterCamoAttack(perceptual=True)` converges on the mock model.

## Out of scope (future)

- Biased mutation (option B) — the objective already supplies continuous pressure.
- Hard masking (option D).
- Variable-M / tri-objective (already tracked in `docs/method.md`).

## Docs deliverable

Update `USAGE.md` §3 with the new flags and the `--mask_* depends on --perceptual`
dependency; add a short "perceptual placement" section and the hideability-map
visualization.
