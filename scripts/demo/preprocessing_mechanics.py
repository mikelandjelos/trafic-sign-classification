"""Visual scrutiny of the two preprocessing decisions that were argued but never shown.

`preprocessing_pipeline.py` shows what the pipeline produces. This one interrogates *why*
two parameters were chosen, because both were justified by reasoning alone in note 08:

  1. interpolation chosen per image (INTER_AREA shrinking, INTER_LINEAR enlarging)
  2. CLAHE tile grid (4, 4) rather than OpenCV's default (8, 8)

The second is the weaker claim -- "6x6 px tiles amplify sensor noise into structure" was
asserted, never measured. If it is wrong, this figure should say so.

Run:
    poetry run python scripts/demo/preprocessing_mechanics.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np

from gtsrb import config, data
from gtsrb import preprocessing as pp

TARGET = config.IMAGE_SIZE[0]


def _grad_energy(image: np.ndarray) -> float:
    """Mean absolute gradient -- how much local structure an image contains."""
    a = image.astype(float)
    return float((np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()) / 2)


def figure_interpolation(frame, out_dir: Path) -> Path:
    """INTER_AREA vs INTER_LINEAR on a shrinking case and an enlarging case."""
    small = frame[frame["height"] <= 28].iloc[0]
    large = frame.nlargest(1, "height").iloc[0]

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    for r, row in enumerate((large, small)):
        grey = pp.to_gray(pp.load_image(row["path"]))
        h, w = grey.shape
        shrinking = TARGET * TARGET < h * w
        chosen = "INTER_AREA" if shrinking else "INTER_LINEAR"

        area = cv2.resize(grey, (TARGET, TARGET), interpolation=cv2.INTER_AREA)
        linear = cv2.resize(grey, (TARGET, TARGET), interpolation=cv2.INTER_LINEAR)
        diff = np.abs(area.astype(int) - linear.astype(int))

        direction = "downsampled" if shrinking else "upsampled"
        # Report the factor in the direction it actually moves, so both rows read as ">1x".
        factor = max(h, w) / TARGET if shrinking else TARGET / max(h, w)
        panels = [
            (grey, f"source {w}×{h}", f"{direction} {factor:.1f}×"),
            (area, "INTER_AREA", f"structure {_grad_energy(area):.1f}"),
            (linear, "INTER_LINEAR", f"structure {_grad_energy(linear):.1f}"),
        ]
        for c, (image, title, sub) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(title, fontsize=10,
                         fontweight="bold" if title.endswith(chosen) else "normal",
                         color="#1a7f37" if title == chosen else "black")
            ax.set_xlabel(sub + ("   ← adaptive picks this" if title == chosen else ""),
                          fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        ax = axes[r, 3]
        im = ax.imshow(diff, cmap="magma", interpolation="nearest")
        ax.set_title("|AREA − LINEAR|", fontsize=10)
        ax.set_xlabel(f"mean {diff.mean():.2f}, max {diff.max()} gray levels", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "Interpolation is chosen per image — 39.2 % of crops shrink, 60.8 % enlarge\n"
        "INTER_AREA averages over the source footprint (right when shrinking); "
        "INTER_LINEAR degenerates toward nearest-neighbour when enlarging",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=2.2)
    path = out_dir / "preprocessing_interpolation.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_clahe_tiles(frame, out_dir: Path, n_stats: int = 300) -> Path:
    """Does a (8,8) grid really amplify noise into structure at 48x48?"""
    grids = [(4, 4), (8, 8)]

    # A controlled case: flat grey plus mild noise contains NO real structure, so whatever
    # structure appears afterwards was invented by the operator.
    rng = np.random.default_rng(config.SEED)
    flat = np.clip(128 + rng.normal(0, 4, (TARGET, TARGET)), 0, 255).astype(np.uint8)

    row = frame[(frame["height"] >= 60) & (frame["height"] <= 120)].iloc[0]
    real = pp.resize(pp.to_gray(pp.load_image(row["path"])))

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))

    for r, (image, label) in enumerate(((real, "real sign"), (flat, "flat grey + σ=4 noise"))):
        ax = axes[r, 0]
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"input — {label}", fontsize=10)
        ax.set_xlabel(f"structure {_grad_energy(image):.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        for c, grid in enumerate(grids, start=1):
            out = pp.clahe(image, tile_grid=grid)
            ax = axes[r, c]
            ax.imshow(out, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            for k in range(1, grid[0]):
                position = k * TARGET / grid[0]
                ax.axhline(position, color="#20c0ff", linewidth=0.6, alpha=0.75)
                ax.axvline(position, color="#20c0ff", linewidth=0.6, alpha=0.75)
            is_default = grid == pp.CLAHE_TILE_GRID
            ax.set_title(
                f"CLAHE tiles {grid[0]}×{grid[1]}  ({TARGET // grid[0]} px each)"
                + ("  ← used" if is_default else ""),
                fontsize=10, fontweight="bold" if is_default else "normal",
                color="#1a7f37" if is_default else "black",
            )
            amplification = _grad_energy(out) / max(_grad_energy(image), 1e-9)
            ax.set_xlabel(f"structure {_grad_energy(out):.2f}  "
                          f"(×{amplification:.2f})", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[r, 3].axis("off")

    # Statistic over many real images: how much structure does each grid add?
    sample = frame.sample(min(n_stats, len(frame)), random_state=config.SEED)
    ratios = {grid: [] for grid in grids}
    for _, item in sample.iterrows():
        base = pp.resize(pp.to_gray(pp.load_image(item["path"])))
        before = _grad_energy(base)
        if before < 1e-6:
            continue
        for grid in grids:
            ratios[grid].append(_grad_energy(pp.clahe(base, tile_grid=grid)) / before)

    ax = fig.add_axes((0.76, 0.12, 0.2, 0.72))
    positions = np.arange(len(grids))
    means = [float(np.mean(ratios[g])) for g in grids]
    ax.bar(positions, means, color=["#1a7f37", "#888888"], width=0.6)
    for p, m in zip(positions, means, strict=True):
        ax.text(p, m + 0.01, f"×{m:.2f}", ha="center", fontsize=9)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{g[0]}×{g[1]}" for g in grids])
    ax.set_ylabel("structure after ÷ before")
    ax.set_title(f"Amplification\nover {len(ratios[grids[0]])} real images", fontsize=10)

    fig.suptitle(
        "Does an 8×8 tile grid invent structure at 48×48?  "
        "Tile boundaries drawn in blue.\n"
        "The flat-grey row is the control: it contains no real structure, so anything "
        "CLAHE produces there was manufactured",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 0.74, 0.88), h_pad=2.2)
    path = out_dir / "preprocessing_clahe_tiles.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_resize_direction(frame, out_dir: Path) -> Path:
    """The shrink/enlarge split as a picture rather than two percentages."""
    heights = frame["height"].to_numpy()
    widths = frame["width"].to_numpy()
    shrinking = (TARGET * TARGET) < (heights * widths)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    bins = np.arange(20, 140, 2)
    ax.hist(heights[shrinking], bins=bins, color="#1f77b4", alpha=0.8,
            label=f"downsampled → INTER_AREA   ({shrinking.mean() * 100:.1f} %)")
    ax.hist(heights[~shrinking], bins=bins, color="#d95f02", alpha=0.8,
            label=f"upsampled → INTER_LINEAR   ({(~shrinking).mean() * 100:.1f} %)")
    ax.axvline(TARGET, color="black", linestyle="--", linewidth=1.6)
    ax.annotate("48 px target", (TARGET, ax.get_ylim()[1] * 0.92), xytext=(8, 0),
                textcoords="offset points", fontsize=9)
    ax.set_xlabel("source image height (px)   ·   tail beyond 140 px not shown")
    ax.set_ylabel("images")
    ax.set_title(
        "Neither branch is a corner case — a single fixed interpolation would be wrong "
        "for a large share of the dataset", fontsize=11
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / "preprocessing_resize_direction.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=config.FIGURES_DIR / "demo" / "preprocessing")
    parser.add_argument("--samples", type=int, default=300)
    args = parser.parse_args()

    config.set_seeds()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = data.load_annotations("train")

    for path in (
        figure_interpolation(frame, args.out),
        figure_clahe_tiles(frame, args.out, args.samples),
        figure_resize_direction(frame, args.out),
    ):
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
