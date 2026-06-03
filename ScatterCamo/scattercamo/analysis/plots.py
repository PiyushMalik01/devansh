"""Plotting for the paper figures. Uses a non-interactive backend (saves to disk).

matplotlib is imported lazily so the rest of the package (and the NumPy-only
tests) never depend on it.
"""


def _plt():
    import matplotlib
    matplotlib.use("Agg")          # headless / no display
    import matplotlib.pyplot as plt
    return plt


def frontier_curve(series, out_path, xlabel="avg SSIM", ylabel="ASR"):
    """Plot the invisibility/strength frontier for several methods.

    Args:
        series: dict ``{method_name: [(x, y), ...]}`` where points are e.g.
                (avg SSIM, ASR) at successive sparsity settings (M / k).
        out_path: file to save the figure to.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, points in series.items():
        points = sorted(points)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Invisibility vs. attack success")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def pareto_plot(front, out_path):
    """Scatter the (loss, L2) objective values of a Pareto front (one image)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 4))
    losses = [s.fitnesses[0] for s in front]
    l2s = [s.fitnesses[1] for s in front]
    colors = ["tab:green" if s.is_adversarial else "tab:red" for s in front]
    ax.scatter(losses, l2s, c=colors, s=30)
    ax.set_xlabel("loss")
    ax.set_ylabel(r"$\|\delta\|_2^2$")
    ax.set_title("Pareto front (green = adversarial)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def qualitative_grid(images, titles, out_path, ncols=None):
    """Save a row/grid of images (e.g. original + adversarial per method)."""
    plt = _plt()
    n = len(images)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
