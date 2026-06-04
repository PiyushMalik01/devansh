# USAGE — how to run ScatterCamo

A practical, copy-paste guide. If you don't yet know *what* this does, read
**`UNDERSTAND_ME.md`** first, then come back here.

> All commands below are written for **Windows PowerShell** (your setup). The
> backtick `` ` `` at the end of a line continues it onto the next line — the
> PowerShell equivalent of the `\` you see in Linux examples.

---

## 1. One-time setup

```powershell
# from inside the ScatterCamo folder
pip install -r requirements.txt
```

This installs NumPy, PyTorch, torchvision, Pillow, scikit-image, matplotlib, PyYAML.
The first real attack will also **download the model weights** (resnet50 /
vgg16_bn) automatically — that needs internet and a few hundred MB, once.

> **torch is only needed for *real* attacks.** Everything else — the tests, the
> hideability visualization, and the `--mock` flag-check mode (§2b) — runs on just
> NumPy + Pillow + scikit-image + matplotlib. If you only want to try the flags,
> you can skip installing torch/torchvision.

### Sanity check (no GPU, no download needed)

```powershell
python tests/test_smoke.py
```

If that prints passing tests, the search engine works on your machine. You can run
the *whole* test suite the same way:

```powershell
python tests/test_smoke.py; python tests/test_perception.py; python tests/test_baselines.py; python tests/test_runner.py; python tests/test_analysis.py
```

---

## 2. Attack a single image

This is the main thing you'll do. Entry point: `run_attack.py`.

```powershell
python run_attack.py --model 1 --image path\to\img.JPEG --true_label 8 `
    --M 10 --queries 10000 --save out
```

What it prints:

```
success=True  queries=10000  model_queries=10000
  L0=1234  L2=2.91  SSIM=0.987  pred=435 (true=8)
```

- `success=True` → it found a smudge that fools the model.
- `pred=435 (true=8)` → the model now says class 435 instead of the correct 8.
- `L0 / L2 / SSIM` → how big/visible the smudge is (small L2 + high SSIM = good).
- `--save out` writes `out.npy` containing the original image, the adversarial
  image, the winning genome, and the per-generation history (for plotting).

---

## 2b. Offline flag-check mode (`--mock`, no model download)

Want to confirm the flags work — or experiment on a laptop — **without** the
torch install or the weights download? Pass `--mock`. It swaps in a tiny
torch-free fake classifier, so the *entire pipeline* runs (preprocessing,
seeding, NSGA-II, rendering, metrics, saving), exercising every flag exactly as a
real run would.

```powershell
# runs in a second or two, no torch, no download:
python run_attack.py --mock --image docs/sample.jpg --M 10 --queries 500
```

> **What `--mock` is and isn't.** It verifies that flags are wired up and the code
> path runs end-to-end. The `L2` / `SSIM` numbers it prints are **not meaningful**
> — they come from a fake model, not resnet50. Use it for plumbing checks and
> demos, never for reporting attack quality. Drop `--mock` for a real attack.

You can also dump the resulting picture with `--out_image` to eyeball it:

```powershell
python run_attack.py --mock --image docs/sample.jpg --M 10 --queries 500 --out_image out/adv.png
```

---

## 3. Every flag explained

| Flag | Required? | Default | What it does |
|---|---|---|---|
| `--image` | **yes** | — | Path to the image file (`.JPEG`/`.jpg`/`.png`). The only truly required flag. |
| `--model` | no | `1` | Which classifier to attack. `0` = vgg16_bn, `1` = resnet50. |
| `--true_label` | no | *auto* | The **correct** ImageNet class number (0–999). If omitted, it's auto-detected from the model's clean prediction (§4). |
| `--M` | no | `10` | Number of blobs in the smudge. The main dial. Try `1, 5, 10, 20, 40`. |
| `--queries` | no | `10000` | Budget: how many times we may poke the model. More = better result, slower. |
| `--pop_size` | no | `20` | Crowd size per generation. **Must be ≥ 2.** |
| `--pc` | no | `0.3` | Crossover rate, in `[0, 1]` (how much parents mix when breeding). |
| `--pm` | no | `0.3` | Mutation rate, in `[0, 1]` (how often a child gets randomly tweaked). |
| `--seed` | no | `0` | Random seed. Same seed → same run, for reproducibility. |
| `--perceptual` | no | *off* | Hide patches in dark / edge / textured regions (§5b). Off → plain ScatterCamo. |
| `--mask_dark` | no | `1.0` | Luminance-masking weight. **Only used with `--perceptual`.** |
| `--mask_edges` | no | `1.0` | Edge-masking weight. **Only used with `--perceptual`.** |
| `--mask_texture` | no | `1.0` | Texture-masking weight. **Only used with `--perceptual`.** |
| `--mask_window` | no | `7` | Window size for the texture (local-variance) signal. |
| `--mock` | no | *off* | Use a torch-free fake classifier — no weights download, no GPU. For checking that flags/plumbing work offline (§2b). Not a real attack. |
| `--out_image` | no | *(none)* | Path **and filename** to save the adversarial image as a viewable picture, e.g. `out/adv.png`. Format is taken from the extension; parent folders are created automatically. |
| `--save` | no | *(none)* | Filename prefix to save the full `.npy` result (orig + adv + genome + history). Omit to just print. |

For your first runs, **only touch `--image` and `--M`.** Leave the rest at defaults.

### Dependencies & constraints (which flag affects which)

Most flags are independent — but a few interact. Read this before combining them:

- **`--save` gates the output file.** The `.npy` (original + adversarial image +
  genome + history) is written **only if you pass `--save`**. Without it, the run
  still happens and prints results, but nothing is saved to disk. Everything under
  "saving" depends on this one flag being present.
- **`--true_label` changes the startup behavior** (see §4 for the full story):
  - *Omitted* → the model's own clean prediction is used; **no** correctness check.
  - *Provided* → it's cross-checked against the clean prediction; if they disagree
    the image is already misclassified and the run **exits early** without attacking.
  So whether the "already-misclassified" skip happens depends entirely on this flag.
- **`--pop_size` must be ≥ 2** — tournament selection picks 2 candidates, so a
  population of 1 is rejected with an error. It also sets the *ceiling* of how many
  children are bred per generation, so very small `--pop_size` with a large
  `--queries` just means more generations of a tiny crowd.
- **`--pc` and `--pm` must be in `[0, 1]`** — they're rates/probabilities, not counts.
- **`--queries` is the only stop condition.** The run ends when the budget is spent,
  regardless of `--M`, `--pop_size`, etc. Bigger `--M` or `--pop_size` means each
  generation costs more queries, so you get *fewer* generations for the same budget.

- **`--mask_dark` / `--mask_edges` / `--mask_texture` / `--mask_window` depend on
  `--perceptual`.** They are read **only when `--perceptual` is passed**; on their
  own they do nothing. See §5b for the full perceptual-placement guide.
- **`--mock` changes which model is used** — the torch-free fake classifier — so
  `--model` (vgg16 vs resnet50) is ignored under `--mock`, and no weights download
  happens. See §2b.
- **`--out_image` and `--save` are independent** and can be used together or
  separately. `--out_image` writes a viewable picture (PNG/JPG); `--save` writes
  the full `.npy` data bundle. Both are skipped if no adversarial image was found.

---

## 4. The `--true_label` flag (your question — now automatic)

You noticed correctly: you *used* to have to look up and type the correct class
number by hand, e.g. `--true_label 8`. **That is now optional** — if you omit it,
the program asks the model what the clean image is and uses that as the reference
label. So this is enough:

```powershell
python run_attack.py --model 1 --image path\to\img.JPEG --M 10 --queries 10000 --save out
```

It will print `auto true_label = <N> (model's clean-image prediction)`. You can
still pass `--true_label` explicitly if you know the ground-truth class and want
to force it.

**Why does the program want it at all?** This is an *untargeted* attack — its goal
is "make the model's answer **wrong**." To know whether the answer became wrong,
it needs to know what the **right** answer was in the first place. That reference
is `--true_label`. Internally it's fed to the loss (`UnTargeted(model, true_label)`
in `run_attack.py`), which compares the correct class's score against the
runner-up to decide if the model has been fooled (see `losses/margin.py`).

### How the automatic version works

Before attacking, the code **asks the model what it thinks the clean image is and
uses *that* as the true label.** Reasoning:

- If the model already classifies the clean image correctly, its own prediction
  *is* the true label — so nothing is lost.
- If the model is already wrong on the clean image, there's nothing to attack
  (it's already "fooled" with zero effort). In that case the program prints a
  `WARNING ... Nothing to attack.` and exits, so you don't waste a run.

This is exactly how the original CamoPatch / SA-MOO experiments pick labels, so
it's faithful to the method, not a hack. The logic lives near the top of
`main()` in `run_attack.py`.

---

## 5. Sweeping `M` (the headline experiment)

The whole point of ScatterCamo is the trade-off between *how many changes* and
*how invisible*. You explore it by running the same image at several `M` values:

```powershell
foreach ($m in 1,5,10,20,40) {
    python run_attack.py --model 1 --image path\to\img.JPEG --true_label 8 `
        --M $m --queries 10000 --save "out_M$m"
}
```

Each run saves `out_M1.npy ... out_M40.npy`. Low `M` = a few bigger blobs; high
`M` = many tiny specks. Comparing their L2 / SSIM shows the sparsity-vs-invisibility
frontier described in the README.

---

## 5b. Perceptual placement (hiding patches in hard-to-see regions)

By default, patch positions are chosen uniformly across the image. Pass
`--perceptual` to instead bias them toward regions where a change is hard to
notice — **dark** areas, **edges**, and **textured** areas — using a per-pixel
*hideability map* built once from the clean image (no extra model queries). It
does two things at once: it **seeds** the initial patches into hideable regions,
and it **reweights the invisibility objective** so the search keeps them there.

```powershell
# all three masking signals on (equal weight):
python run_attack.py --image path\to\img.JPEG --M 10 --queries 10000 --perceptual --save out
```

> The reported `L2` / `SSIM` numbers still measure *true* visible distortion, so
> perceptual vs non-perceptual runs remain directly comparable.

### Ablation: isolate each signal

Each signal has its own weight, so you can turn the others off to study which one
helps most (great for a writeup):

```powershell
# dark only
python run_attack.py --image img.JPEG --M 10 --perceptual --mask_edges 0 --mask_texture 0 --save out_dark
# edges only
python run_attack.py --image img.JPEG --M 10 --perceptual --mask_dark 0 --mask_texture 0 --save out_edges
# texture only
python run_attack.py --image img.JPEG --M 10 --perceptual --mask_dark 0 --mask_edges 0 --save out_texture
# combined (baseline for the ablation)
python run_attack.py --image img.JPEG --M 10 --perceptual --save out_combined
# uniform (control — perceptual off)
python run_attack.py --image img.JPEG --M 10 --save out_uniform
```

Compare the `L2` / `SSIM` each prints (lower L2 / higher SSIM at the same success
= better hiding).

### Seeing the map

To *visualize* where it decides to hide patches (like `docs/hideability_demo.png`):

```python
from scattercamo.analysis import hideability_panel
from run_attack import load_image
hideability_panel(load_image("img.JPEG"), "my_map.png")   # original + 3 signals + W
```

### Reminder on dependencies

`--mask_dark`, `--mask_edges`, `--mask_texture`, and `--mask_window` are read
**only when `--perceptual` is set**. Without `--perceptual` they are ignored and
the run is identical to plain ScatterCamo.

---

## 6. Using a config file instead of long flags

`configs/dev.yaml` (laptop-friendly) and `configs/full.yaml` (rented GPU) hold
preset values so you don't retype flags. For example `dev.yaml`:

```yaml
model: 1
M: 8
queries: 500     # tiny budget, runs in seconds
pop_size: 10
...
```

> Note: `run_attack.py` itself currently reads **command-line flags, not the YAML**
> — the configs are consumed by the batch experiment runner (see §7). If you'd like
> `run_attack.py` to accept `--config configs/dev.yaml` too, that's another small
> change I can make for you.

---

## 7. Comparing against the baselines (programmatic)

To benchmark ScatterCamo against the original methods (CamoPatch, SA-MOO,
Sparse-RS) over a whole dataset, use the `BatchRunner` from Python (full example
in `README.md`, "Running a comparison"):

```python
from scattercamo.runner import BatchRunner
from scattercamo.attack import ScatterCamoAttack
from scattercamo.losses import UnTargeted
from scattercamo.models import ImageNetModel

model = ImageNetModel(1)                                  # resnet50
loss_factory = lambda x, y: UnTargeted(model, y, to_pytorch=True)
factory = lambda x: ScatterCamoAttack({"x": x, "M": 10, "queries": 10000})

summary = BatchRunner(factory, loss_factory, out_dir="results", name="scattercamo").run(dataset)
print(summary["asr"], summary["avg_l2"], summary["avg_ssim"])
```

`asr` = Attack Success Rate (fraction of images fooled). The runner checkpoints,
so you can stop and resume long sweeps.

---

## 8. Reading a saved `.npy` result

```python
import numpy as np
d = np.load("out.npy", allow_pickle=True).item()
d["orig"]     # original image  (H, W, 3)
d["adv"]      # adversarial image (H, W, 3) — eyeball them side by side
d["genome"]   # the winning blobs (M, 7)
d["history"]  # loss/L2 per generation, for plotting convergence
```

You can `matplotlib.pyplot.imshow` `orig` and `adv` next to each other to *see*
how invisible the smudge turned out.

---

## 9. Quick troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Download stalls on first run | Model weights are fetching — needs internet, happens once. |
| Very slow per image | No GPU → PyTorch is on CPU. Lower `--queries` (e.g. 500) for testing, or use a GPU machine for real runs. |
| `success=False` | Budget too small, or `M` too small for this image. Raise `--queries` or `--M`. |
| `pop_size must be >= 2` | You set `--pop_size 1`. Tournament selection needs at least 2. |
| Unsure what `--true_label` to use | Just omit it — it's auto-detected now (§4). |

---

That's everything you need to run it. Start with §1 (setup) → §2 (one image) →
§5 (sweep `M`). When the `--true_label` typing gets old, come back to §4 and I'll
wire up auto-detection.
