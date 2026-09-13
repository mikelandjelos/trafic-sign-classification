"""Task 3.5: contact sheet -- every degradation at every level, on one sample.

A deliverable figure (PROJECT_TASKS.md section 8), not just a diagnostic: it is what lets a
reader of the report see what "sigma = 40" or "k = 15" actually does to a sign, instead of
inferring it from an accuracy drop.

Run:
    poetry run python docs/demo/degradation_contact_sheet.py
    poetry run python docs/demo/degradation_contact_sheet.py --preproc raw_gray

What is shown is the **48x48 preprocessed model input**, degraded exactly as
`gtsrb.degradations` degrades it during evaluation -- same function, same path-keyed seed.
The figure therefore shows the pixels the classifiers actually receive, not an illustration
of them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from gtsrb import cache, config, data, degradations
from gtsrb import preprocessing as pp

#: Order the rows so the reader meets them as the report discusses them.
ROW_ORDER = ("noise", "blur", "gamma")

UNITS = {"noise": "σ = {}", "blur": "k = {} px", "gamma": "γ = {}"}


def pick_sharp_sign(frame, preproc: str, min_h: int = 45, max_h: int = 120):
    """A well-exposed, high-contrast sign that reads clearly at 48x48.

    Ranked by global contrast (std), deliberately *not* by edge energy: maximising edge
    energy selects for busy backgrounds -- fences, poles, foliage -- which dominate the
    picture and make the degradation harder to see rather than easier. Contrast favours a
    clean sign against a plain background, which is what a report figure needs.
    """
    candidates = frame[(frame["height"] >= min_h) & (frame["height"] <= max_h)]
    sample = candidates.sample(min(250, len(candidates)), random_state=config.SEED)
    images = cache.load_images(sample, preproc)
    grey = images if images.ndim == 3 else images[..., 2]
    flat = grey.reshape(len(grey), -1).astype(float)
    brightness, contrast = flat.mean(axis=1), flat.std(axis=1)
    # High-frequency clutter competes with the sign; penalise it rather than reward it.
    clutter = np.abs(np.diff(grey.astype(float), axis=2)).mean(axis=(1, 2))
    score = contrast - 1.5 * clutter
    score[(brightness < 70) | (brightness > 185)] = -np.inf
    return sample.iloc[int(np.argmax(score))]


def contact_sheet(row, preproc: str, out_dir: Path) -> Path:
    key = [row["path"]]
    clean = cache.load_images(row.to_frame().T, preproc)

    n_levels = max(len(degradations.levels_for(d)) for d in ROW_ORDER)
    fig, axes = plt.subplots(
        len(ROW_ORDER), n_levels, figsize=(2.0 * n_levels, 2.25 * len(ROW_ORDER))
    )

    for r, name in enumerate(ROW_ORDER):
        spec = degradations.get_degradation(name)
        for c, level in enumerate(spec.levels):
            image = degradations.apply(clean, name, level, key)[0]
            shown = image if image.ndim == 2 else image[..., 2]
            ax = axes[r, c]
            ax.imshow(shown, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            is_identity = level == spec.identity
            label = UNITS[name].format(level)
            ax.set_title(
                f"{label}   (clean)" if is_identity else label,
                fontsize=9,
                fontweight="bold" if is_identity else "normal",
                color="#1a7f37" if is_identity else "black",
            )
            if is_identity:
                # Gamma's identity sits mid-row, which is worth making visually obvious.
                ax.add_patch(
                    Rectangle((0, 0), shown.shape[1] - 1, shown.shape[0] - 1,
                              fill=False, edgecolor="#1a7f37", linewidth=2.5)
                )
            if c == 0:
                ax.set_ylabel(f"{name}\n", fontsize=11, fontweight="bold")

        for c in range(len(spec.levels), n_levels):
            axes[r, c].axis("off")

    class_name = config.CLASS_NAMES[int(row["class_id"])]
    fig.suptitle(
        f"Controlled degradations — 48×48 model input ({preproc})\n"
        f"class {int(row['class_id'])}: {class_name}   ·   green = identity level "
        f"(note gamma's sits mid-row)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.6)
    path = out_dir / "degradation_contact_sheet.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preproc", default=pp.DEFAULT_PREPROC,
                        choices=sorted(pp.PREPROC_CONFIGS))
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR / "demo")
    parser.add_argument("--class-id", type=int, default=None)
    args = parser.parse_args()

    config.set_seeds()
    args.out.mkdir(parents=True, exist_ok=True)

    frame = data.load_annotations("test")
    if args.class_id is not None:
        frame = frame[frame["class_id"] == args.class_id]
        if frame.empty:
            raise SystemExit(f"no test images for class {args.class_id}")

    row = pick_sharp_sign(frame, args.preproc)
    print(f"sample: {row['path']} (class {int(row['class_id'])}, {row['width']}x{row['height']})")
    print(f"  wrote {contact_sheet(row, args.preproc, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
