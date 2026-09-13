"""Visual explanation of *how* the PCA representation works (task 4.1).

Complements the eigensigns figure of task 4.4, which is the polished report artifact. This
one shows the machinery: what the subspace keeps, what it throws away, how much of it is
illumination, and what happens when a degraded image is pushed through a basis that was
learned on clean data.

Generates four figures into figures/demo/pca/:

  pca_reconstruction.png  -- the same signs rebuilt from k = 2 ... 256 components
  pca_spectrum.png        -- scree + cumulative variance for the three preproc configs
  pca_pc1_brightness.png  -- PC1 is illumination: scatter (r = 0.997) + signs sorted by it
  pca_degraded.png        -- clean basis vs degraded input: reconstruction and residual

Run:
    poetry run python scripts/demo/pca_mechanics.py

Everything is computed with `gtsrb.representations.pca` and `gtsrb.degradations` themselves,
so the figures describe the code that runs during evaluation rather than a re-derivation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gtsrb import cache, config, data, degradations
from gtsrb.representations.pca import PCARepresentation, as_matrix

#: Component counts for the reconstruction ladder -- the task 4.2 sweep plus k = 2, which
#: is far too few and is there to make the progression legible.
LADDER = (2, 8, 32, 128, 256)

#: One sign from each of four visually distinct super-categories, so the ladder is not four
#: speed-limit discs. (round-prohibitory, triangular-warning, octagonal, round-mandatory)
DEMO_CLASSES = (14, 25, 13, 38)


def pick_samples(frame, n_per_class: int = 1) -> np.ndarray:
    """Indices of large, well-resolved examples of `DEMO_CLASSES`, deterministically.

    Large crops are chosen because the figures are about what PCA does to *structure*; a
    20 px sign upsampled to 48x48 is already blurred by the resize and would confound the
    reconstruction ladder with interpolation.
    """
    chosen: list[int] = []
    positions = {label: i for i, label in enumerate(frame.index)}
    for class_id in DEMO_CLASSES:
        rows = frame[frame["class_id"] == class_id].nlargest(n_per_class, "roi_h")
        chosen.extend(positions[label] for label in rows.index)
    return np.asarray(chosen)


def _show(ax, image: np.ndarray) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def figure_reconstruction(out_dir: Path, train, train_images) -> Path:
    """Rebuild the same signs from progressively more components."""
    picks = pick_samples(train)
    originals = train_images[picks]

    models = {k: PCARepresentation(k).fit(train_images) for k in LADDER}
    errors = {k: models[k].reconstruction_error(train_images) for k in LADDER}

    ncols = len(LADDER) + 1
    fig, axes = plt.subplots(len(picks), ncols, figsize=(1.55 * ncols, 1.62 * len(picks)))

    for r in range(len(picks)):
        _show(axes[r, 0], originals[r])
        axes[r, 0].set_ylabel(config.CLASS_NAMES[train.iloc[picks[r]]["class_id"]][:18],
                              fontsize=7)
        for c, k in enumerate(LADDER, start=1):
            _show(axes[r, c], models[k].reconstruct(originals[r : r + 1])[0])
        if r == 0:
            axes[r, 0].set_title("original\n(2304 dims)", fontsize=9, fontweight="bold")
            for c, k in enumerate(LADDER, start=1):
                axes[r, c].set_title(f"k = {k}\n{errors[k]:.1f} levels err", fontsize=9)

    fig.suptitle(
        "PCA reconstruction: what the subspace keeps\n"
        "Error is mean absolute reconstruction error over the whole training split, "
        "in gray levels — directly comparable with the σ of task 3.1",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "pca_reconstruction.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_spectrum(out_dir: Path, train) -> Path:
    """Scree and cumulative variance for all three preprocessing configs."""
    configs = ("raw_gray", "clahe_gray", "clahe_hsv")
    colours = plt.cm.viridis(np.linspace(0.15, 0.8, len(configs)))

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6))

    for colour, preproc in zip(colours, configs, strict=True):
        images = cache.load_images(train, preproc)
        phi = PCARepresentation(256, preproc=preproc).fit(images)
        ratios = phi.explained_variance_ratio_
        cumulative = phi.cumulative_variance()
        ks = np.arange(1, len(ratios) + 1)

        left.semilogy(ks, ratios, color=colour, label=preproc, lw=1.6)
        right.plot(ks, cumulative, color=colour, label=preproc, lw=1.8)

        k80 = phi.components_for_variance(0.80)
        if k80 is not None:
            right.plot([k80], [cumulative[k80 - 1]], "o", color=colour, ms=7)
            right.annotate(f"{k80}", (k80, cumulative[k80 - 1]), textcoords="offset points",
                           xytext=(6, -12), fontsize=9, color=colour, fontweight="bold")

    left.set_xlabel("component")
    left.set_ylabel("explained variance ratio (log)")
    left.set_title("Scree — variance per component")
    left.legend(fontsize=9)
    left.grid(alpha=0.3)

    right.axhline(0.80, color="0.5", ls="--", lw=1)
    right.annotate("80 % of variance", (256, 0.80), textcoords="offset points",
                   xytext=(-95, 5), fontsize=9, color="0.35")
    right.set_xlabel("components retained")
    right.set_ylabel("cumulative explained variance")
    right.set_title("Components needed: 8 → 25 → 106")
    right.legend(fontsize=9, loc="lower right")
    right.grid(alpha=0.3)

    fig.suptitle(
        "CLAHE makes the data harder to compress — and that is the point of it\n"
        "The dominant direction it normalises away was illumination, "
        "which had been absorbing half the variance",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = out_dir / "pca_spectrum.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_pc1_brightness(out_dir: Path, train) -> Path:
    """The leading component is illumination, not sign identity."""
    images = cache.load_images(train, "raw_gray")
    phi = PCARepresentation(8, preproc="raw_gray").fit(images)
    scores = phi.transform(images)[:, 0]
    brightness = as_matrix(images).mean(axis=1) * 255.0
    correlation = np.corrcoef(scores, brightness)[0, 1]

    fig = plt.figure(figsize=(12, 5.0))
    grid = fig.add_gridspec(2, 12, height_ratios=[2.4, 1.0], hspace=0.35)

    scatter = fig.add_subplot(grid[0, :6])
    sample = np.linspace(0, len(scores) - 1, 4000).astype(int)
    scatter.scatter(brightness[sample], scores[sample], s=3, alpha=0.25, color="#2b6cb0")
    scatter.set_xlabel("mean image intensity (gray levels)")
    scatter.set_ylabel("PC1 score")
    scatter.set_title(f"PC1 score vs. brightness — r = {correlation:+.4f}", fontsize=10)
    scatter.grid(alpha=0.3)

    bars = fig.add_subplot(grid[0, 6:])
    ratios = phi.explained_variance_ratio_
    bars.bar(np.arange(1, len(ratios) + 1), ratios * 100, color="#2b6cb0")
    bars.set_xlabel("component")
    bars.set_ylabel("% of total variance")
    bars.set_title(f"PC1 alone holds {ratios[0] * 100:.1f} % of the variance", fontsize=10)
    bars.grid(alpha=0.3, axis="y")

    order = np.argsort(scores)
    strip = np.linspace(0, len(order) - 1, 12).astype(int)
    for column, position in enumerate(strip):
        ax = fig.add_subplot(grid[1, column])
        _show(ax, images[order[position]])
        if column == 0:
            ax.set_xlabel("low PC1", fontsize=8)
        elif column == len(strip) - 1:
            ax.set_xlabel("high PC1", fontsize=8)

    fig.suptitle(
        "Over half of what PCA models on raw grayscale is illumination, not sign identity\n"
        "Sorting the training set by PC1 sorts it by exposure — a nuisance factor with "
        "no class information",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path = out_dir / "pca_pc1_brightness.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_degraded(out_dir: Path, train, train_images) -> Path:
    """A degraded image pushed through a basis fitted on clean data.

    Two numbers per row, and they answer different questions:

    `|res|` is the part of the degraded image the subspace cannot express. **A small
    residual is not good news.** It means the damage was representable -- the projection
    reproduced it faithfully and passed it on.

    `shift` is what actually predicts classification damage: how far the 128-d feature
    vector moved, in units of the median distance between two random clean training images.
    A shift near 1.0 means the degraded sign is as far from its clean self as two unrelated
    signs are from each other, which is where a linear classifier starts losing it.

    Neither is an accuracy claim -- task 9.5 measures that.
    """
    phi = PCARepresentation(128).fit(train_images)

    # Median distance between two random clean training images, in feature space: the ruler
    # the per-condition shift is expressed in.
    rng = config.rng_for("pca_demo", "scale")
    scores = phi.transform(train_images)
    pairs = rng.choice(len(scores), size=(4000, 2))
    scale = float(np.median(np.linalg.norm(scores[pairs[:, 0]] - scores[pairs[:, 1]], axis=1)))

    picks = pick_samples(train)[:2]
    conditions = [("noise", 40.0), ("blur", 15.0), ("gamma", 2.5)]

    rows = len(picks) * len(conditions)
    fig, axes = plt.subplots(rows, 4, figsize=(7.2, 1.75 * rows))

    for r, (index, (name, level)) in enumerate(
        [(i, c) for i in picks for c in conditions]
    ):
        clean = train_images[index : index + 1]
        key = [str(train.iloc[index]["path"])]
        degraded = degradations.apply(clean, name, level, key)
        rebuilt = phi.reconstruct(degraded)
        residual = degraded[0].astype(np.int16) - rebuilt[0].astype(np.int16)

        _show(axes[r, 0], clean[0])
        _show(axes[r, 1], degraded[0])
        _show(axes[r, 2], rebuilt[0])
        axes[r, 3].imshow(residual, cmap="coolwarm", vmin=-60, vmax=60,
                          interpolation="nearest")
        axes[r, 3].set_xticks([])
        axes[r, 3].set_yticks([])
        shift = float(
            np.linalg.norm(phi.transform(degraded)[0] - phi.transform(clean)[0]) / scale
        )
        axes[r, 0].set_ylabel(f"{name} {level:g}", fontsize=9, fontweight="bold")
        axes[r, 2].set_xlabel(f"shift = {shift:.2f}", fontsize=8, fontweight="bold")
        axes[r, 3].set_xlabel(f"|res| = {np.abs(residual).mean():.1f}", fontsize=8)

        if r == 0:
            for c, title in enumerate(
                ["clean", "degraded", "rebuilt from\n128 components", "residual\n(not representable)"]
            ):
                axes[r, c].set_title(title, fontsize=9, fontweight="bold")

    fig.suptitle(
        "Degraded input, clean basis — the projection is never re-fitted to the stressor\n"
        "|res| = what the subspace cannot express.  shift = how far the feature vector moved,\n"
        "in units of the median distance between two random clean signs.  A small residual "
        "is not good news:\nit means the damage was representable, so the projection passed "
        "it through intact.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "pca_degraded.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="PCA mechanics figures (task 4.1).")
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR / "demo" / "pca")
    args = parser.parse_args()

    config.set_seeds()
    args.out.mkdir(parents=True, exist_ok=True)

    train, _ = data.train_val_split()
    train_images = cache.load_images(train, "clahe_gray")

    for path in (
        figure_reconstruction(args.out, train, train_images),
        figure_spectrum(args.out, train),
        figure_pc1_brightness(args.out, train),
        figure_degraded(args.out, train, train_images),
    ):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
