"""Visual explanation of *how* HOG works, and where it breaks (task 5.5).

Complements the report figure of task 5.4, which shows what the descriptor looks like. This
one shows the machinery and, more importantly, the quantity that would expose a problem:
per-cell gradient energy under noise, the condition HOG is predicted to handle worst.

Generates three figures into figures/demo/hog/:

  hog_block_normalisation.png -- what L2-Hys does, and what it cannot do
  hog_noise_response.png      -- per-cell gradient magnitude as sigma rises
  hog_cell_size.png           -- why 6 px cells beat 8 px, at the level of one sign

Run:
    poetry run python scripts/demo/hog_mechanics.py

Everything is computed with `gtsrb.representations.hog` and `gtsrb.degradations` themselves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gtsrb import cache, config, data, degradations
from gtsrb.representations.hog import HOGRepresentation, as_images

DEMO_CLASS = 5  # Speed limit 80 -- the class HOG confuses most (task 5.3)


def sample(frame, images, class_id: int = DEMO_CLASS):
    rows = frame[frame["class_id"] == class_id].nlargest(1, "roi_h")
    index = list(frame.index).index(rows.index[0])
    return images[index], str(frame.iloc[index]["path"])


def cell_energy(phi: HOGRepresentation, image: np.ndarray) -> np.ndarray:
    """Mean gradient magnitude per cell -- the raw material HOG histograms."""
    grey = as_images(image[None])[0]
    if grey.ndim == 3:
        grey = grey[..., -1]
    gy, gx = np.gradient(grey.astype(np.float32))
    magnitude = np.hypot(gx, gy)
    rows, cols = phi.cell_grid()
    ph, pw = phi.pixels_per_cell
    return magnitude[: rows * ph, : cols * pw].reshape(rows, ph, cols, pw).mean(axis=(1, 3))


def figure_block_normalisation(out_dir: Path, image: np.ndarray) -> Path:
    """What L2-Hys removes (a linear contrast change) and what it does not (gamma)."""
    phi = HOGRepresentation(pixels_per_cell=(6, 6)).fit(image[None])
    base = as_images(image[None])[0]
    # All kept as float in [0, 1] so display and descriptor paths are identical for each.
    variants = {
        "original": base,
        "contrast x0.5\n(linear)": np.clip(0.5 * base + 0.25, 0, 1),
        "gamma 2.5\n(non-linear)": as_images(
            degradations.apply(image[None], "gamma", 2.5, ["k"])
        )[0],
    }
    reference = phi.transform(image[None])[0]

    fig, axes = plt.subplots(2, len(variants), figsize=(3.3 * len(variants), 6.2))
    for col, (label, variant) in enumerate(variants.items()):
        axes[0, col].imshow(variant, cmap="gray", vmin=0.0, vmax=1.0,
                            interpolation="nearest")
        axes[0, col].set_title(label, fontsize=10)
        descriptor = phi.transform(variant[None])[0]
        delta = float(np.abs(descriptor - reference).mean())
        axes[1, col].plot(reference[:120], lw=1.2, color="#2b6cb0", label="original")
        axes[1, col].plot(descriptor[:120], lw=1.0, color="#c05621", ls="--", label=label.split(chr(10))[0])
        axes[1, col].set_ylim(0, max(0.5, float(reference[:120].max()) * 1.15))
        axes[1, col].set_title(f"mean |Δdescriptor| = {delta:.4f}", fontsize=9,
                               color="#2f855a" if delta < 1e-6 else "#c05621")
        axes[1, col].legend(fontsize=7)
        axes[1, col].grid(alpha=0.3)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

    axes[1, 0].set_ylabel("descriptor value\n(first 120 dims)", fontsize=9)
    fig.suptitle(
        "Block normalisation (L2-Hys) removes a linear contrast change exactly — and gamma only partly\n"
        "A linear change scales every gradient in a block by the same factor, which L2 divides "
        "back out.\nA gamma curve does not, so it survives — which is why the 9.5 gamma panel "
        "is a real measurement.",
        fontsize=10.5, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = out_dir / "hog_block_normalisation.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_noise_response(out_dir: Path, image: np.ndarray, key: str) -> Path:
    """The demo's failure test: per-cell gradient energy as noise rises."""
    phi = HOGRepresentation(pixels_per_cell=(6, 6)).fit(image[None])
    levels = [level for level in degradations.levels_for("noise")]
    clean_energy = cell_energy(phi, image)
    reference = phi.transform(image[None])[0]

    fig, axes = plt.subplots(3, len(levels), figsize=(2.5 * len(levels), 7.6))
    shifts = []
    for col, sigma in enumerate(levels):
        noisy = degradations.apply(image[None], "noise", sigma, [key])[0]
        energy = cell_energy(phi, noisy)
        descriptor = phi.transform(noisy[None])[0]
        shifts.append(float(np.abs(descriptor - reference).mean()))

        axes[0, col].imshow(noisy, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[0, col].set_title(f"σ = {sigma:g}", fontsize=10, fontweight="bold")
        axes[1, col].imshow(energy, cmap="magma", vmin=0,
                            vmax=float(clean_energy.max()) * 3.0)
        axes[2, col].imshow(energy / np.maximum(clean_energy, 1e-6), cmap="coolwarm",
                            vmin=0, vmax=6)
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_ylabel("input", fontsize=9, fontweight="bold")
    axes[1, 0].set_ylabel("gradient energy\nper cell\n(shared scale)", fontsize=9,
                          fontweight="bold")
    axes[2, 0].set_ylabel("ratio to clean\n(blue 0 … red 6x)", fontsize=9, fontweight="bold")
    fig.suptitle(
        "Why HOG is predicted to suffer most under noise — the mechanism, per cell\n"
        "HOG differentiates, and differentiation is a high-pass operation: it *amplifies* "
        "noise rather than averaging it.\n"
        "Bottom row is energy relative to clean — by σ = 40 the low-contrast background cells "
        "carry several times their true energy,\nso orientation votes there are close to random. "
        f"Mean |Δdescriptor| at σ=40: {shifts[-1]:.4f}.",
        fontsize=10, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path = out_dir / "hog_noise_response.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_cell_size(out_dir: Path, image: np.ndarray) -> Path:
    """Why 6 px cells beat 8 px, at the level of one sign (task 5.2 measured +3.5 pp)."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0].set_title("input 48×48", fontsize=10)

    for ax, ppc in zip(axes[1:], (8, 6), strict=True):
        phi = HOGRepresentation(pixels_per_cell=(ppc, ppc)).fit(image[None])
        rows, cols = phi.cell_grid()
        _, rendered = phi.visualize(image)
        from skimage import exposure

        ax.imshow(exposure.rescale_intensity(rendered, in_range=(0, rendered.max() * 0.35)),
                  cmap="gray", interpolation="nearest")
        for r in range(1, rows):
            ax.axhline(r * ppc - 0.5, color="#c05621", lw=0.6, alpha=0.6)
        for c in range(1, cols):
            ax.axvline(c * ppc - 0.5, color="#c05621", lw=0.6, alpha=0.6)
        ax.set_title(f"{ppc} px cells → {rows}×{cols} grid, {phi.n_features} dims",
                     fontsize=10)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Cell size is the binding constraint on a 48×48 crop, not orientation count\n"
        "Task 5.2 measured 8 px → 6 px worth +3.5 pp macro-F1, while 9 → 12 orientations was "
        "worth nothing.\nAt 8 px a whole sign gets only a 6×6 grid, and the digits fall inside "
        "single cells.",
        fontsize=10.5, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path = out_dir / "hog_cell_size.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="HOG mechanics demo (task 5.5).")
    parser.add_argument("--preproc", default="raw_gray")
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR / "demo" / "hog")
    args = parser.parse_args()

    config.set_seeds()
    args.out.mkdir(parents=True, exist_ok=True)
    train, _ = data.train_val_split()
    images = cache.load_images(train, args.preproc)
    image, key = sample(train, images)

    for path in (
        figure_block_normalisation(args.out, image),
        figure_noise_response(args.out, image, key),
        figure_cell_size(args.out, image),
    ):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
