"""Task 5.4: the HOG visualisation figure, one sample per super-category.

    poetry run python scripts/figure_hog.py

Renders the descriptor at the configuration task 5.2 selected, read from the sweep CSV rather
than hard-coded, so the figure shows what the reported results were produced with.

What the rendering is, and is not
---------------------------------
skimage draws, in each cell, a star of line segments whose brightness is that orientation's
histogram weight. It is a picture of **where gradient energy sits and which way it points** --
built from the *un-normalised* cell histograms.

The classifier never sees that. It sees the **block-normalised** vector, in which a contrast
change has been divided out (note 12 section 2.2). So the visualisation shows the structure
HOG captures, not the values it hands over. The figure states this, because a reader who takes
the rendering for the descriptor will draw the wrong conclusion about the gamma panel at 9.5.

Super-categories, not classes
-----------------------------
Four shapes are shown rather than four arbitrary classes, because the point is what HOG does
to *shape*: round-prohibitory, triangular-warning, octagonal, round-mandatory. Task 5.3 found
HOG failing on speed-limit digits and succeeding on the end-of-restriction strikethrough, and
both are shape-vs-detail distinctions the grid makes visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

from gtsrb import cache, config, data, tuning
from gtsrb.figures import save_demo_and_report
from gtsrb.representations.hog import HOGRepresentation

#: One class per shape family, with the reason each is here.
SAMPLES: tuple[tuple[int, str], ...] = (
    (14, "osmougaoni"),               # Stop
    (25, "trougaoni, upozorenje"),    # Road work
    (38, "okrugli, obaveza"),         # Keep right
    (5, "okrugli, zabrana"),          # Speed limit 80 -- HOG's worst confusion (task 5.3)
)


def pick(frame, class_id: int) -> int:
    """Index of a large, well-resolved example -- small crops are blurred by the upscale."""
    rows = frame[frame["class_id"] == class_id].nlargest(1, "roi_h")
    return list(frame.index).index(rows.index[0])


def figure(phi: HOGRepresentation, frame, images: np.ndarray, out_dir: Path,
           preproc: str) -> Path:
    fig, axes = plt.subplots(2, len(SAMPLES), figsize=(3.1 * len(SAMPLES), 6.4))

    for col, (class_id, shape) in enumerate(SAMPLES):
        index = pick(frame, class_id)
        image = images[index]
        _, rendered = phi.visualize(image)

        axes[0, col].imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[0, col].set_title(f"{config.CLASS_NAMES[class_id][:26]}\n({shape})", fontsize=9)
        # Rescale for display only. The raw rendering is dominated by a few strong cells and
        # the rest is near-black, so the structure that matters is invisible without it.
        # Purely cosmetic -- the descriptor is untouched.
        shown = exposure.rescale_intensity(rendered, in_range=(0, rendered.max() * 0.35))
        axes[1, col].imshow(shown, cmap="gray", interpolation="nearest")

        rows, cols = phi.cell_grid()
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
        # the cell grid, drawn on the HOG panel so the pooling unit is visible
        for r in range(1, rows):
            axes[1, col].axhline(r * phi.pixels_per_cell[0] - 0.5, color="#c05621",
                                 lw=0.5, alpha=0.55)
        for c in range(1, cols):
            axes[1, col].axvline(c * phi.pixels_per_cell[1] - 0.5, color="#c05621",
                                 lw=0.5, alpha=0.55)

    axes[0, 0].set_ylabel("ulaz\n48×48", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel(f"HOG\n{phi.cell_grid()[0]}×{phi.cell_grid()[1]} ćelija",
                          fontsize=10, fontweight="bold")

    title = fig.suptitle(
        f"HOG on four sign shapes — {preproc}, {phi.pixels_per_cell[0]} px cells, "
        f"{phi.orientations} orientations, {phi.n_features} dimensions\n"
        f"Each cell shows a star of segments weighted by orientation energy; the orange grid "
        f"is the cell boundary — the unit HOG pools within.\n"
        f"Note this is drawn from the UN-normalised histograms: the classifier sees the "
        f"block-normalised vector, in which contrast has been divided out.",
        fontsize=10.5,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hog_visualization.png"
    save_demo_and_report(fig, path, title, dpi=170, bare_rect=None)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="HOG visualisation figure (task 5.4).")
    parser.add_argument("--sweep", type=Path,
                        default=config.RESULTS_DIR / "sweeps" / "hog_configs.csv")
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR / "hog")
    args = parser.parse_args()

    config.set_seeds()
    selected = tuning.best_from_sweep(args.sweep)
    preproc = str(selected["preproc"])
    ppc = int(selected["pixels_per_cell"])

    train, _ = data.train_val_split()
    images = cache.load_images(train, preproc)
    phi = HOGRepresentation(orientations=int(selected["orientations"]),
                            pixels_per_cell=(ppc, ppc), preproc=preproc).fit(images[:1])
    print(f"{phi!r}")
    print(f"wrote {figure(phi, train, images, args.out, preproc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
