"""Run ScatterCamo against an ImageNet classifier on a single image.

Example:
    python run_attack.py --model 1 --image path/to/img.JPEG --true_label 8 \
        --M 10 --queries 10000 --save out

Sweeping --M over {1,5,10,20,40} traces the sparsity / invisibility frontier.
"""

import argparse
import numpy as np

from scattercamo.models import ImageNetModel
from scattercamo.losses import UnTargeted
from scattercamo.attack import ScatterCamoAttack
from scattercamo import metrics


def load_image(path):
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    tensor = tf(Image.open(path).convert("RGB"))      # (3, H, W)
    return tensor.permute(1, 2, 0).numpy()            # (H, W, 3) in [0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, default=1, help="0=vgg16_bn, 1=resnet50")
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--true_label", type=int, required=True)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--queries", type=int, default=10000)
    ap.add_argument("--pop_size", type=int, default=20)
    ap.add_argument("--pc", type=float, default=0.3)
    ap.add_argument("--pm", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default=None, help="path prefix for .npy result")
    args = ap.parse_args()

    x = load_image(args.image)
    model = ImageNetModel(args.model)
    loss = UnTargeted(model, args.true_label, to_pytorch=True)

    attack = ScatterCamoAttack({
        "x": x, "M": args.M, "queries": args.queries, "pop_size": args.pop_size,
        "pc": args.pc, "pm": args.pm, "seed": args.seed,
    })
    result = attack.optimise(loss)

    print(f"success={result['success']}  queries={result['queries']}  "
          f"model_queries={model.queries}")
    if result["best"] is not None:
        adv = result["best"].generate_image()
        print(f"  L0={metrics.l0(adv, x)}  L2={metrics.l2(adv, x):.4f}  "
              f"SSIM={metrics.ssim(adv, x):.4f}  "
              f"pred={loss.get_label(adv)} (true={args.true_label})")
        if args.save:
            np.save(args.save + ".npy", {
                "orig": x, "adv": adv, "genome": result["best"].genome,
                "queries": result["queries"], "history": result["history"],
            }, allow_pickle=True)
            print(f"  saved -> {args.save}.npy")


if __name__ == "__main__":
    main()
