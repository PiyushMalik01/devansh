"""Compare ScatterCamo against the original Phoenix Williams methods.

Runs ScatterCamo and the baselines -- CamoPatch (Williams & Li, NeurIPS 2023),
SA-MOO (Williams & Li, CVPR 2023), and Sparse-RS -- over the *same* images and
the *same* model, then reports ASR, avg L2, PSNR, SSIM, queries, and (optionally)
FID for each method in one table. This is the apples-to-apples comparison.

Real comparison needs a real classifier + a folder of images:
    uv sync --extra real
    uv run python compare.py --images path/to/imgs --queries 10000 --fid

Plumbing check with no torch / no download (numbers are not meaningful):
    uv run python compare.py --mock --n 4 --queries 400
"""

import argparse
import csv
import logging
import os

import numpy as np

from scattercamo import metrics
from scattercamo.losses import UnTargeted
from scattercamo.runner.result import normalize_result
from scattercamo.attack import ScatterCamoAttack
from scattercamo.baselines import SparseRSAttack, SAMOOAttack, CamoPatchAttack

log = logging.getLogger("compare")

METHOD_ORDER = ["scattercamo", "camopatch", "samoo", "sparse_rs"]


def attack_factories(args):
    """Map method name -> callable(x) -> attack instance, sharing the budget."""
    return {
        "scattercamo": lambda x: ScatterCamoAttack({
            "x": x, "M": args.M, "queries": args.queries, "seed": args.seed,
            "perceptual": args.perceptual}),
        "camopatch": lambda x: CamoPatchAttack({
            "x": x, "queries": args.queries, "seed": args.seed}),
        "samoo": lambda x: SAMOOAttack({
            "x": x, "eps": args.eps, "queries": args.queries, "seed": args.seed}),
        "sparse_rs": lambda x: SparseRSAttack({
            "x": x, "eps": args.eps, "queries": args.queries, "seed": args.seed}),
    }


def evaluate_method(name, make_attack, dataset, loss_factory, want_fid):
    """Run one method over the dataset; return its aggregate summary dict."""
    records, cleans, advs = [], [], []
    for i, (x, y) in enumerate(dataset):
        loss = loss_factory(x, y)
        result = normalize_result(make_attack(x).optimise(loss))
        rec = {"success": bool(result.success), "queries": int(result.queries)}
        produced = x                                  # what this method "shows"
        if result.success and result.adv_image is not None:
            produced = result.adv_image
            rec["l0"] = metrics.l0(produced, x)
            rec["l2"] = metrics.l2(produced, x)
            rec["psnr"] = metrics.psnr(produced, x)
            rec["ssim"] = metrics.ssim(produced, x)
        records.append(rec)
        cleans.append(x)
        advs.append(produced)
        log.info("%s: image %d/%d success=%s queries=%d",
                 name, i + 1, len(dataset), rec["success"], rec["queries"])

    summary = metrics.summarize(records)
    summary["fid"] = None
    if want_fid and len(advs) >= 2:
        try:
            summary["fid"] = metrics.fid(np.stack(cleans), np.stack(advs))
        except ImportError as exc:
            log.warning("FID skipped: %s", str(exc).splitlines()[0])
    return summary


def compare(dataset, loss_factory, methods, factories, want_fid):
    """Return ``{method_name: summary}`` for the requested methods."""
    results = {}
    for name in methods:
        log.info("=== method: %s ===", name)
        results[name] = evaluate_method(
            name, factories[name], dataset, loss_factory, want_fid)
    return results


def print_table(results):
    cols = ["method", "ASR", "avgL2", "PSNR", "SSIM", "queries", "FID"]
    widths = [12, 6, 8, 7, 7, 9, 8]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for name, s in results.items():
        def fmt(v, spec):
            return ("n/a" if v is None else format(v, spec))
        row = [
            name.ljust(widths[0]),
            fmt(s["asr"], ".2f").ljust(widths[1]),
            fmt(s["avg_l2"], ".3f").ljust(widths[2]),
            fmt(s["avg_psnr"], ".2f").ljust(widths[3]),
            fmt(s["avg_ssim"], ".4f").ljust(widths[4]),
            fmt(s["avg_queries"], ".0f").ljust(widths[5]),
            fmt(s["fid"], ".2f").ljust(widths[6]),
        ]
        print("  ".join(row))


def save_csv(results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["method", "asr", "avg_l0", "avg_l2", "avg_psnr", "avg_ssim",
              "avg_queries", "fid"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for name, s in results.items():
            w.writerow({"method": name, **{k: s.get(k) for k in fields[1:]}})


def load_dataset(args, model, to_pytorch):
    """Build a list of (image, true_label). Real: from a folder, label = the
    model's clean prediction. Mock: synthetic images, label 0."""
    if args.mock:
        rng = np.random.default_rng(args.seed)
        return [(rng.random((64, 64, 3)), 0) for _ in range(args.n)]

    # Real path: load images from a directory and label by clean prediction.
    from run_attack import load_image
    paths = sorted(
        os.path.join(args.images, f) for f in os.listdir(args.images)
        if f.lower().endswith((".jpg", ".jpeg", ".png")))[: args.n]
    if not paths:
        raise SystemExit(f"no images found in {args.images}")
    dataset = []
    for p in paths:
        x = load_image(p)
        label = UnTargeted(model, 0, to_pytorch=to_pytorch).get_label(x)
        dataset.append((x, label))
    return dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", type=str, help="folder of images to attack")
    src.add_argument("--mock", action="store_true",
                     help="synthetic images + torch-free mock model (plumbing only; "
                          "numbers are not meaningful)")
    ap.add_argument("--model", type=int, default=1, help="0=vgg16_bn, 1=resnet50")
    ap.add_argument("--n", type=int, default=10, help="number of images to use")
    ap.add_argument("--queries", type=int, default=10000)
    ap.add_argument("--M", type=int, default=10, help="ScatterCamo shapes")
    ap.add_argument("--eps", type=int, default=150, help="L0 budget for SA-MOO / Sparse-RS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perceptual", action="store_true",
                    help="run ScatterCamo with perceptual placement")
    ap.add_argument("--methods", type=str, default=",".join(METHOD_ORDER),
                    help="comma-separated subset of: " + ", ".join(METHOD_ORDER))
    ap.add_argument("--fid", action="store_true",
                    help="also compute FID (needs torch; uv sync --extra real)")
    ap.add_argument("--out", type=str, default="results/comparison.csv",
                    help="CSV path for the results")
    ap.add_argument("--log", type=str, default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.mock:
        from scattercamo.models import MockModel
        dataset = load_dataset(args, None, False)
        # per-image mock so each attack's reference is the image it attacks
        loss_factory = lambda x, y: UnTargeted(MockModel(x), y, to_pytorch=False)
    else:
        from scattercamo.models import ImageNetModel
        model = ImageNetModel(args.model)
        dataset = load_dataset(args, model, True)
        loss_factory = lambda x, y: UnTargeted(model, y, to_pytorch=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    factories = attack_factories(args)
    results = compare(dataset, loss_factory, methods, factories, args.fid)

    print()
    print_table(results)
    save_csv(results, args.out)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
