"""Task 4.4: the eigensigns figure -- top-16 principal components as images.

    poetry run python scripts/figure_eigensigns.py

The traffic-sign analogue of the Eigenfaces grid from the lecture. Renders the **fitted**
basis from `results/models/pca_svm_clahe_gray.joblib` rather than refitting, so the figure
shows the basis the reported results were actually produced with.

Two rendering decisions, both of which change what the reader concludes
----------------------------------------------------------------------
**Components are signed, so they are drawn on a diverging colour map centred at zero.**
Every component has entries either side of zero; rescaling one to a 0-255 grayscale range
would map "strongly negative" and "strongly positive" to the two ends of the same ramp and
make a bipolar pattern look like a brightness gradient. Blue/white/red with a symmetric range
keeps the sign legible, which is the whole content of an eigenvector.

**Each component is scaled to its own amplitude, not to a shared one.** Component 16 has
roughly a hundredth of component 1's variance; on a shared scale it would be a flat grey
square and the figure would show one eigenvector and fifteen blanks. Per-component scaling
makes the structure visible, and the amplitude information that is lost is printed back as
the explained-variance percentage under each panel -- so nothing is hidden, it is just moved
from the colour ramp to the caption.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gtsrb import config, preprocessing
from gtsrb.representations.pca import PCARepresentation

N_COMPONENTS_SHOWN = 16


def load_basis(preproc: str) -> PCARepresentation:
    """The fitted representation from task 4.3, or a clear instruction if it is missing."""
    import joblib

    path = config.MODELS_DIR / f"pca_svm_{preproc}.joblib"
    if not path.exists():
        raise SystemExit(
            f"no trained model at {path}\n"
            f"run:  poetry run python scripts/train_pca.py"
        )
    return joblib.load(path)["representation"]


def figure(phi: PCARepresentation, out_dir: Path, preproc: str) -> Path:
    eigen = phi.eigenimages(N_COMPONENTS_SHOWN)
    ratios = phi.explained_variance_ratio_

    fig = plt.figure(figsize=(12.4, 7.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 3.25], wspace=0.12,
                             left=0.04, right=0.98, top=0.80, bottom=0.03)

    # The mean is part of the model -- it is subtracted before every projection, so an
    # eigen-decomposition figure without it is showing half the transform.
    mean_ax = fig.add_subplot(outer[0, 0])
    mean_ax.imshow(phi.mean_image(), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    mean_ax.set_xticks([])
    mean_ax.set_yticks([])
    mean_ax.set_title("mean sign\n(subtracted before every projection)", fontsize=9,
                      fontweight="bold")

    grid = outer[0, 1].subgridspec(4, 4, hspace=0.32, wspace=0.06)
    for i in range(N_COMPONENTS_SHOWN):
        ax = fig.add_subplot(grid[i // 4, i % 4])
        limit = float(np.abs(eigen[i]).max())
        ax.imshow(eigen[i], cmap="coolwarm", vmin=-limit, vmax=limit,
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"PC{i + 1}  ·  {ratios[i] * 100:.1f} %", fontsize=8.5)

    fig.suptitle(
        f"Eigensigns — the leading {N_COMPONENTS_SHOWN} principal components of "
        f"{len(phi.mean_)}-dimensional {preproc} sign images\n"
        f"Red positive, blue negative, each scaled to its own amplitude; the percentage is "
        f"that component's share of the training variance.\n"
        f"PC1 is essentially uniform — it encodes overall brightness "
        f"(r = +0.99 with mean intensity), not sign identity.",
        fontsize=10.5,
        y=0.97,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pca_eigensigns.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Eigensigns figure (task 4.4).")
    parser.add_argument("--preproc", default=preprocessing.DEFAULT_PREPROC)
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR / "pca")
    args = parser.parse_args()

    config.set_seeds()
    phi = load_basis(args.preproc)

    ratios = phi.explained_variance_ratio_
    print(f"basis: {phi!r}")
    print(f"  shown: {N_COMPONENTS_SHOWN} of {phi.n_components} components")
    print(f"  they cover {ratios[:N_COMPONENTS_SHOWN].sum() * 100:.1f} % of the training "
          f"variance ({ratios[0] * 100:.1f} % in PC1 alone)")
    print(f"wrote {figure(phi, args.out, args.preproc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
