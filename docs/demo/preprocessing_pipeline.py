"""Visual walkthrough of the preprocessing pipeline (tasks 2.1-2.2).

Generates two figures, intended both as a sanity check and as report material:

  preprocessing_pipeline.png  -- one image through every stage, with the intensity
                                 histogram before and after CLAHE
  preprocessing_configs.png   -- the three ablation configs across several signs

Run:
    poetry run python docs/demo/preprocessing_pipeline.py
    poetry run python docs/demo/preprocessing_pipeline.py --class-id 14 --out /tmp/figs

Nothing here reimplements the pipeline -- it calls `gtsrb.preprocessing` directly, so the
figures show exactly what the models are fed. If a figure looks wrong, the pipeline is
wrong.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to files, never shown

import cv2
import matplotlib.pyplot as plt
import numpy as np

from gtsrb import config, data
from gtsrb import preprocessing as pp


# cv2.imread gives BGR; matplotlib expects RGB. Getting this wrong renders every red
# prohibition sign blue -- and it is the kind of error that survives to print.
def _rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _pick_illustrative(frame, min_h: int = 60, max_h: int = 130):
    """The sign where CLAHE does the most visible work, while staying legible.

    Picking the *darkest* image (minimum std) is the obvious choice and the wrong one: it
    lands on a near-black frame where the sign is invisible before and barely visible
    after, so the figure demonstrates nothing. What reads well is a genuinely
    under-exposed but still discernible sign, i.e. the largest contrast *gain* among
    images whose mean intensity is not pinned to either end of the range.
    """
    candidates = frame[(frame["height"] >= min_h) & (frame["height"] <= max_h)]
    if candidates.empty:
        candidates = frame
    sample = candidates.sample(min(300, len(candidates)), random_state=config.SEED)

    best, best_gain = None, -np.inf
    for _, row in sample.iterrows():
        resized = pp.resize(pp.to_gray(pp.load_image(row["path"])))
        if not (25 <= resized.mean() <= 210):  # skip frames pinned to black or white
            continue
        gain = pp.clahe(resized).std() - resized.std()
        if gain > best_gain:
            best, best_gain = row, gain
    return best if best is not None else sample.iloc[0]


def _show(ax, image, title, subtitle=""):
    ax.imshow(image, cmap=None if image.ndim == 3 else "gray",
              vmin=None if image.ndim == 3 else 0, vmax=None if image.ndim == 3 else 255,
              interpolation="nearest")
    ax.set_title(title, fontsize=10, pad=6)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=8, labelpad=4)
    ax.set_xticks([])
    ax.set_yticks([])


def figure_pipeline(row, out_dir: Path) -> Path:
    """One image through every stage of `clahe_gray`, plus the CLAHE histogram effect."""
    bgr = pp.load_image(row["path"])
    gray = pp.to_gray(bgr)
    resized = pp.resize(gray)
    equalised = pp.clahe(resized)

    h, w = bgr.shape[:2]
    direction = "downsampled" if 48 * 48 < h * w else "upsampled"
    interp = "INTER_AREA" if 48 * 48 < h * w else "INTER_LINEAR"

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
    _show(axes[0], _rgb(bgr), "1. source PPM", f"{w}x{h} BGR, as the dataset frames it")
    _show(axes[1], gray, "2. to_gray()", f"{w}x{h}, 1 channel")
    _show(axes[2], resized, "3. resize(48,48)", f"48x48, {direction} ({interp})")
    _show(axes[3], equalised, "4. clahe()", f"48x48, clip={pp.CLAHE_CLIP_LIMIT} "
                                            f"tiles={pp.CLAHE_TILE_GRID}")

    axes[4].hist(resized.ravel(), bins=48, range=(0, 255), alpha=0.6,
                 label=f"before (std {resized.std():.0f})", color="#888888")
    axes[4].hist(equalised.ravel(), bins=48, range=(0, 255), alpha=0.6,
                 label=f"after  (std {equalised.std():.0f})", color="#1f77b4")
    axes[4].set_title("intensity histogram", fontsize=10, pad=6)
    axes[4].set_xlabel("CLAHE spreads the used range", fontsize=8)
    axes[4].set_yticks([])
    axes[4].legend(fontsize=7, loc="upper left")

    name = config.CLASS_NAMES[int(row["class_id"])]
    fig.suptitle(
        f"Preprocessing pipeline — clahe_gray   |   class {int(row['class_id'])}: {name}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    path = out_dir / "preprocessing_pipeline.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_configs(frame, out_dir: Path, n_signs: int = 5) -> Path:
    """The three ablation configs, side by side, on a spread of classes."""
    rng = np.random.default_rng(config.SEED)
    classes = rng.choice(config.N_CLASSES, size=n_signs, replace=False)
    rows = [
        _pick_illustrative(frame[frame["class_id"] == int(c)], min_h=40, max_h=200)
        for c in classes
    ]

    names = list(pp.PREPROC_CONFIGS)
    fig, axes = plt.subplots(len(names) + 1, n_signs, figsize=(2.0 * n_signs, 8.2))

    for col, row in enumerate(rows):
        bgr = pp.load_image(row["path"])
        label = config.CLASS_NAMES[int(row["class_id"])]
        _show(axes[0, col], _rgb(pp.resize(bgr)), label[:24] if col else f"source\n{label[:20]}")
        if col == 0:
            axes[0, col].set_ylabel("source\n(resized for display)", fontsize=9)

        for r, cfg_name in enumerate(names, start=1):
            out = pp.preprocess(bgr, cfg_name)
            shown = _rgb(cv2.cvtColor(out, cv2.COLOR_HSV2BGR)) if out.ndim == 3 else out
            _show(axes[r, col], shown, "")
            if col == 0:
                cfg = pp.get_config(cfg_name)
                axes[r, col].set_ylabel(f"{cfg_name}\n{cfg.flat_dim} dims", fontsize=9)

    fig.suptitle(
        "The three preprocessing configs (task 2.2) — each adds one factor to the previous",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    path = out_dir / "preprocessing_configs.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-id", type=int, default=None,
                        help="class for the pipeline figure (default: auto-pick)")
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR / "demo")
    parser.add_argument("--signs", type=int, default=5, help="columns in the configs figure")
    args = parser.parse_args()

    config.set_seeds()
    args.out.mkdir(parents=True, exist_ok=True)

    frame = data.load_annotations("train")
    subset = frame if args.class_id is None else frame[frame["class_id"] == args.class_id]
    if subset.empty:
        raise SystemExit(f"no images for class {args.class_id}")

    row = _pick_illustrative(subset)
    print(f"pipeline sample: {row['path']}  "
          f"({row['width']}x{row['height']}, class {int(row['class_id'])})")

    for path in (figure_pipeline(row, args.out), figure_configs(frame, args.out, args.signs)):
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
