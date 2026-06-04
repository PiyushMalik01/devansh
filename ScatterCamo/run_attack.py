"""Run ScatterCamo against an ImageNet classifier on a single image.

Example:
    python run_attack.py --model 1 --image path/to/img.JPEG --true_label 8 \
        --M 10 --queries 10000 --save out

Sweeping --M over {1,5,10,20,40} traces the sparsity / invisibility frontier.
"""

import argparse
import numpy as np

from scattercamo.losses import UnTargeted
from scattercamo.attack import ScatterCamoAttack
from scattercamo import metrics


def load_image(path):
    """Load an image as (224, 224, 3) float in [0, 1] using PIL + NumPy only.

    Replicates torchvision's ``Resize(256) -> CenterCrop(224) -> ToTensor`` (the
    standard ImageNet preprocessing) without depending on torch/torchvision, so
    the pipeline runs on a minimal environment.
    """
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)                            # resize shorter side to 256
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2         # center-crop 224x224
    img = img.crop((left, top, left + 224, top + 224))
    return np.asarray(img, dtype=np.float32) / 255.0   # (224, 224, 3) in [0, 1]


def save_image(arr, path):
    """Save a float image ``(h, w, 3)`` in ``[0, 1]`` as a viewable picture.

    The file format is taken from ``path``'s extension (``.png``, ``.jpg``, ...).
    Parent directories are created if needed.
    """
    import os
    from PIL import Image

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    pixels = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
    Image.fromarray(pixels).save(path)


# Built-in defaults. A --config YAML overrides these; explicit CLI flags override
# both. ``image`` has no default -- it must come from the CLI or the config file.
DEFAULTS = {
    "model": 1, "true_label": None, "M": 10, "queries": 10000,
    "pop_size": 20, "pc": 0.3, "pm": 0.3, "seed": 0,
    "max_radius_frac": 0.10, "perceptual": False, "mask_dark": 1.0,
    "mask_edges": 1.0, "mask_texture": 1.0, "mask_window": 7, "mock": False,
    "out_image": None, "save": None,
}


def resolve_config(ap, cli):
    """Merge built-in defaults < ``--config`` YAML < explicit CLI flags.

    ``cli`` is ``vars(parsed_args)`` where unset options are absent (the parser
    uses ``SUPPRESS`` defaults), so only flags the user actually typed appear and
    correctly win over the config file. Returns an ``argparse.Namespace``.
    """
    cfg = dict(DEFAULTS)
    config_path = cli.pop("config", None)
    if config_path:
        import yaml
        with open(config_path) as f:
            loaded = yaml.safe_load(f) or {}
        ignored = []
        for key, value in loaded.items():
            if key in DEFAULTS or key == "image":
                cfg[key] = value
            else:
                ignored.append(key)
        if ignored:
            print(f"note: ignoring config keys not used by run_attack: "
                  f"{', '.join(ignored)}")
    cfg.update(cli)                       # explicit CLI flags win over the config
    if not cfg.get("image"):
        ap.error("--image is required (pass --image or set 'image' in --config)")
    return argparse.Namespace(**cfg)


def main():
    ap = argparse.ArgumentParser(
        description="Run ScatterCamo on one image. Precedence: CLI flags override "
                    "a --config YAML, which overrides the built-in defaults.")
    S = argparse.SUPPRESS                 # unset flags stay absent so config can fill them
    ap.add_argument("--config", type=str, default=None,
                    help="YAML config file; any flag below may be set there "
                         "(explicit CLI flags override it)")
    ap.add_argument("--image", type=str, default=S)
    ap.add_argument("--model", type=int, default=S, help="0=vgg16_bn, 1=resnet50")
    ap.add_argument("--true_label", type=int, default=S,
                    help="correct ImageNet class (0-999); if omitted, the model's "
                         "own prediction on the clean image is used")
    ap.add_argument("--M", type=int, default=S)
    ap.add_argument("--queries", type=int, default=S)
    ap.add_argument("--pop_size", type=int, default=S)
    ap.add_argument("--pc", type=float, default=S)
    ap.add_argument("--pm", type=float, default=S)
    ap.add_argument("--seed", type=int, default=S)
    ap.add_argument("--max_radius_frac", type=float, default=S,
                    help="max shape radius as a fraction of min(H, W); lower = "
                         "smaller circles (default 0.10)")
    # Perceptual placement (hide patches in dark / edge / textured regions).
    ap.add_argument("--perceptual", action="store_true", default=S,
                    help="bias placement + visibility objective toward hideable regions")
    ap.add_argument("--mask_dark", type=float, default=S,
                    help="luminance-masking weight (only with --perceptual)")
    ap.add_argument("--mask_edges", type=float, default=S,
                    help="edge-masking weight (only with --perceptual)")
    ap.add_argument("--mask_texture", type=float, default=S,
                    help="texture-masking weight (only with --perceptual)")
    ap.add_argument("--mask_window", type=int, default=S,
                    help="local-variance window for the texture signal")
    ap.add_argument("--mock", action="store_true", default=S,
                    help="use a torch-free mock classifier (no weights download) to "
                         "verify the pipeline / flags without running a real model")
    ap.add_argument("--out_image", type=str, default=S,
                    help="path + filename to save the adversarial image as a viewable "
                         "picture, e.g. out/adv.png (format from the extension)")
    ap.add_argument("--save", type=str, default=S, help="path prefix for .npy result")

    args = resolve_config(ap, vars(ap.parse_args()))

    x = load_image(args.image)
    if args.mock:
        from scattercamo.models import MockModel
        model = MockModel(x)              # torch-free, no weights download
        to_pytorch = False
    else:
        from scattercamo.models import ImageNetModel
        model = ImageNetModel(args.model)
        to_pytorch = True

    # The clean prediction is needed either way: to auto-fill --true_label, and
    # to warn when the model is already wrong (nothing left to attack).
    clean_pred = UnTargeted(model, 0, to_pytorch=to_pytorch).get_label(x)
    if args.true_label is None:
        true_label = clean_pred
        print(f"auto true_label = {true_label} (model's clean-image prediction)")
    else:
        true_label = args.true_label
        if clean_pred != true_label:
            print(f"WARNING: model already misclassifies the clean image as "
                  f"{clean_pred} (true={true_label}); it is adversarial with no "
                  f"perturbation. Nothing to attack.")
            return

    loss = UnTargeted(model, true_label, to_pytorch=to_pytorch)

    attack = ScatterCamoAttack({
        "x": x, "M": args.M, "queries": args.queries, "pop_size": args.pop_size,
        "pc": args.pc, "pm": args.pm, "seed": args.seed,
        "max_radius_frac": args.max_radius_frac,
        "perceptual": args.perceptual, "mask_dark": args.mask_dark,
        "mask_edges": args.mask_edges, "mask_texture": args.mask_texture,
        "mask_window": args.mask_window,
    })
    result = attack.optimise(loss)
    adv = result["best"].generate_image() if result["best"] is not None else None

    print(f"success={result['success']}  queries={result['queries']}  "
          f"model_queries={model.queries}")
    if adv is not None:
        print(f"  L0={metrics.l0(adv, x)}  L2={metrics.l2(adv, x):.4f}  "
              f"SSIM={metrics.ssim(adv, x):.4f}  "
              f"pred={loss.get_label(adv)} (true={true_label})")

    if args.save and adv is not None:
        import os
        parent = os.path.dirname(args.save)
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.save(args.save + ".npy", {
            "orig": x, "adv": adv, "genome": result["best"].genome,
            "queries": result["queries"], "history": result["history"],
        }, allow_pickle=True)
        print(f"  saved -> {args.save}.npy")

    if args.out_image and adv is not None:
        save_image(adv, args.out_image)
        print(f"  saved image -> {args.out_image}")


if __name__ == "__main__":
    main()
