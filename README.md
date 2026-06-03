# devansh — Camouflaged & Sparse Black-Box Adversarial Attacks

Research workspace exploring stealthy, query-limited, black-box adversarial
attacks, building toward a novel hybrid method.

## Contents

| Folder | What it is |
|---|---|
| [`CamoPatch/`](CamoPatch/) | Reference code for **CamoPatch** (Williams & Li, NeurIPS 2023) — camouflaged adversarial patches via an evolutionary strategy. Includes the paper PDF. |
| [`Black-Box-Sparse-Adversarial-Attack/`](Black-Box-Sparse-Adversarial-Attack/) | Reference code for **SA-MOO** (Williams & Li, CVPR 2023) — sparse adversarial attacks via multi-objective optimisation. Includes the paper PDF. |
| [`ScatterCamo/`](ScatterCamo/) | **New hybrid** developed here: *multi-objective scattered camouflaged patches*. Fuses CamoPatch's camouflaged-shape representation with SA-MOO's NSGA-II + prioritized domination. Fully implemented and tested. |

The two reference folders are the original authors' public code (histories
removed so this is a single repo); see their respective READMEs for citations.
The contribution of this workspace lives in [`ScatterCamo/`](ScatterCamo/) —
see [`ScatterCamo/docs/design.md`](ScatterCamo/docs/design.md) for the rationale
and [`ScatterCamo/docs/method.md`](ScatterCamo/docs/method.md) for the algorithm.

## Quick start (ScatterCamo)

```bash
cd ScatterCamo
pip install -r requirements.txt
python tests/test_smoke.py        # verify the engine (NumPy only)
```
