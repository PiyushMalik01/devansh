# ScatterCamo

**Multi-objective scattered camouflaged adversarial patches** — a black-box,
query-limited, untargeted attack on image classifiers that hybridizes two
NeurIPS/CVPR 2023 works by Phoenix Williams & Ke Li:

- **CamoPatch** (NeurIPS 2023) — camouflaged, semi-transparent shape representation.
- **SA-MOO** (CVPR 2023) — NSGA-II with a *prioritized* domination relation.

## The idea

Instead of one contiguous square patch (CamoPatch) or scattered single pixels
(SA-MOO), a perturbation is a set of **M small camouflaged shapes placed
anywhere on the image**. They are optimized by NSGA-II under a bi-objective
formulation `(loss, L2)` with SA-MOO's prioritized domination: *first* become
adversarial, *then* minimize how visible the perturbation is.

`M=1` with a large shape recovers something close to CamoPatch; many tiny
shapes approach SA-MOO's sparse regime. Sweeping `M` traces the
sparsity / invisibility frontier.

> Approach B of the design (fixed-M, bi-objective). The variable-M, tri-objective
> extension is documented as future work in `docs/method.md`.

## Layout

```
scattercamo/
  representation/  shape genome + pure-NumPy renderer
  search/          NSGA-II: Solution, non-dominated sort, crowding, domination
  operators/       crossover + mutation for the shape genome
  attack/          the ScatterCamo attack loop
  baselines/       Sparse-RS, SA-MOO, CamoPatch (shared loss/result interface)
  runner/          AttackResult + batch runner with checkpoint/resume
  analysis/        frontier curves, Pareto plots, qualitative grids
  losses/          untargeted margin loss (shared interface)
  models/          ImageNet model wrappers + query counting
  metrics/         ASR, L0, L2, SSIM, aggregation
run_attack.py      single-image entry point
configs/           dev.yaml (laptop) · full.yaml (rented GPU)
tests/             dependency-light smoke + unit tests (13, all passing)
docs/              method notes + design decisions
```

## Quick start

```bash
uv sync                                  # core deps (no torch); see USAGE.md §1

# Verify the engine (NumPy only, no GPU / model download):
uv run python tests/test_smoke.py

# Try the flags with no model download (torch-free mock):
uv run python run_attack.py --mock --image docs/sample.jpg --M 10 --queries 500

# Real attack on an ImageNet image (needs torch):  uv sync --extra real
uv run python run_attack.py --model 1 --image path/to/img.JPEG \
    --M 10 --queries 10000 --save out --out_image adv.png
```

## Status

- [x] Shape representation + renderer
- [x] NSGA-II search with prioritized domination
- [x] Genetic operators (crossover + mutation)
- [x] Attack loop + single-image runner
- [x] Smoke + unit tests passing
- [x] Baseline adapters (CamoPatch, SA-MOO, Sparse-RS) on a shared runner
- [x] Batch experiment runner + checkpoint/resume
- [x] Analysis: frontier curves, Pareto plots, qualitative grids

All 13 tests pass (NumPy/scikit-image/matplotlib only — no torch or GPU needed):
`python tests/test_smoke.py && python tests/test_baselines.py && python tests/test_runner.py && python tests/test_analysis.py`

### Running a comparison

```python
from scattercamo.runner import BatchRunner
from scattercamo.attack import ScatterCamoAttack
from scattercamo.baselines import SparseRSAttack, SAMOOAttack, CamoPatchAttack
from scattercamo.losses import UnTargeted
from scattercamo.models import ImageNetModel

model = ImageNetModel(1)                       # resnet50
loss_factory = lambda x, y: UnTargeted(model, y, to_pytorch=True)

for name, factory in {
    "scattercamo": lambda x: ScatterCamoAttack({"x": x, "M": 10, "queries": 10000}),
    "sparse_rs":   lambda x: SparseRSAttack({"x": x, "eps": 150, "queries": 10000}),
    "samoo":       lambda x: SAMOOAttack({"x": x, "eps": 150, "queries": 10000}),
    "camopatch":   lambda x: CamoPatchAttack({"x": x, "queries": 10000}),
}.items():
    summary = BatchRunner(factory, loss_factory, out_dir="results", name=name).run(dataset)
    print(name, summary["asr"], summary["avg_l2"], summary["avg_ssim"])
```

See `docs/method.md` for the algorithm and `docs/design.md` for the decisions
behind this approach.

## License

ScatterCamo is released under the [MIT License](LICENSE). The license covers
only this folder; the sibling reference repos remain their authors' IP — see
[`../CREDITS.md`](../CREDITS.md).
