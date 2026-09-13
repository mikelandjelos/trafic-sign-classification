"""Visual explanation of *how* each degradation works (tasks 3.1-3.3).

Complements `degradation_contact_sheet.py`, which shows what the degradations *do* to a
sign. This one shows the machinery: the convolution kernels, the transfer curves, and the
perturbation distributions.

Generates three figures into figures/demo/:

  degradation_blur_kernels.png  -- the line kernels themselves, at several angles
  degradation_gamma_curves.png  -- the five transfer curves on one axis
  degradation_noise_stats.png   -- perturbation histograms per sigma, with clipping visible

Run:
    poetry run python scripts/demo/degradation_mechanics.py

Everything is computed with `gtsrb.degradations` itself, so the figures describe the code
that runs during evaluation rather than a re-derivation of it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gtsrb import cache, config, data, degradations
from gtsrb import preprocessing as pp

ANGLES = (0, 30, 45, 90, 135)


def figure_blur_kernels(out_dir: Path) -> Path:
    """The line kernels, plus the angle/pixel-count property that looks like a bug."""
    sizes = [s for s in degradations.levels_for("blur") if s != 0]
    fig, axes = plt.subplots(len(sizes), len(ANGLES), figsize=(2.0 * len(ANGLES), 2.3 * len(sizes)))

    for r, size in enumerate(sizes):
        for c, angle in enumerate(ANGLES):
            kernel = degradations.motion_blur_kernel(int(size), angle)
            ax = axes[r, c]
            ax.imshow(kernel, cmap="viridis", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            # RMS distance of kernel mass from the centre -- the effective blur extent,
            # and the quantity that must not depend on the angle.
            centre = (kernel.shape[0] - 1) / 2.0
            ys, xs = np.mgrid[0 : kernel.shape[0], 0 : kernel.shape[0]]
            radius = np.sqrt((((xs - centre) ** 2 + (ys - centre) ** 2) * kernel).sum())
            ax.set_title(f"{angle}°  ·  r = {radius:.2f}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"k = {int(size)}", fontsize=10, fontweight="bold")

    fig.suptitle(
        "Motion-blur kernels — normalised (Σ = 1, so brightness is unchanged)\n"
        "Built by sub-pixel sampling, so the effective extent r is near-constant across "
        "angles; rasterising a line instead made it vary by up to 29 %",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=1.8)
    path = out_dir / "degradation_blur_kernels.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_gamma_curves(out_dir: Path, sample: np.ndarray) -> Path:
    """The transfer curves, and what they do to a real intensity histogram."""
    levels = degradations.levels_for("gamma")
    identity = degradations.identity_for("gamma")
    colours = plt.cm.coolwarm(np.linspace(0.05, 0.95, len(levels)))

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(256)

    for gamma, colour in zip(levels, colours, strict=True):
        lut = degradations._gamma_lut(float(gamma))
        is_identity = gamma == identity
        left.plot(
            x, lut,
            color="black" if is_identity else colour,
            linewidth=2.6 if is_identity else 1.8,
            linestyle="--" if is_identity else "-",
            label=f"γ = {gamma}" + ("  (identity)" if is_identity else ""),
        )
    left.set_xlabel("input intensity")
    left.set_ylabel("output intensity")
    left.set_title("Gamma transfer curves  ·  out = 255·(in/255)^γ", fontsize=11)
    left.set_xlim(0, 255)
    left.set_ylim(0, 255)
    left.legend(fontsize=8, loc="upper left")
    left.grid(alpha=0.25)
    left.set_aspect("equal")

    for gamma, colour in zip(levels, colours, strict=True):
        out = degradations.apply(sample, "gamma", gamma, ["k"] * len(sample))
        right.hist(
            out.ravel(), bins=64, range=(0, 255), histtype="step",
            color="black" if gamma == identity else colour,
            linewidth=2.4 if gamma == identity else 1.5,
            linestyle="--" if gamma == identity else "-",
            label=f"γ = {gamma}  (mean {out.mean():.0f})",
        )
    right.set_xlabel("intensity")
    right.set_yticks([])
    right.set_title(
        "Effect on a real intensity histogram\n"
        "γ = 2.5 piles mass against 0 — that clipping is irreversible",
        fontsize=11,
    )
    right.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "degradation_gamma_curves.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_noise_stats(out_dir: Path, sample: np.ndarray, keys: list[str]) -> Path:
    """Why the realised std falls below sigma at the top level: clipping."""
    levels = [s for s in degradations.levels_for("noise") if s != 0]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6))
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(levels)))

    realised, requested = [], []
    for sigma, colour in zip(levels, colours, strict=True):
        out = degradations.apply(sample, "noise", sigma, keys)
        delta = out.astype(int) - sample.astype(int)
        realised.append(delta.std())
        requested.append(sigma)
        left.hist(
            delta.ravel(), bins=120, range=(-120, 120), histtype="step",
            color=colour, linewidth=1.6,
            label=f"σ = {sigma}  →  realised {delta.std():.1f}",
        )
    left.set_xlabel("per-pixel change (output − input)")
    left.set_yticks([])
    left.set_title(
        "Realised perturbation per σ\n"
        "the spike at 0 is already-saturated pixels clipped back to where they started",
        fontsize=11,
    )
    left.legend(fontsize=8)

    right.plot(requested, requested, "k--", linewidth=1.4, label="ideal (no clipping)")
    right.plot(requested, realised, "o-", color="#c2452d", linewidth=2, label="realised")
    for rq, rl in zip(requested, realised, strict=True):
        right.annotate(f"−{rq - rl:.1f}", (rq, rl), textcoords="offset points",
                       xytext=(6, -12), fontsize=8, color="#c2452d")
    right.set_xlabel("requested σ")
    right.set_ylabel("realised std")
    right.set_title(
        "Clipping costs more at higher σ\n"
        "the gap is saturation, not a defect", fontsize=11
    )
    right.legend(fontsize=9)
    right.grid(alpha=0.25)

    fig.tight_layout()
    path = out_dir / "degradation_noise_stats.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR / "demo" / "degradation")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--preproc", default=pp.DEFAULT_PREPROC,
                        choices=sorted(pp.PREPROC_CONFIGS))
    args = parser.parse_args()

    config.set_seeds()
    args.out.mkdir(parents=True, exist_ok=True)

    frame = data.load_annotations("test").head(args.samples)
    sample = cache.load_images(frame, args.preproc)
    keys = frame["path"].tolist()
    print(f"statistics over {len(sample)} test images ({args.preproc})")

    for path in (
        figure_blur_kernels(args.out),
        figure_gamma_curves(args.out, sample),
        figure_noise_stats(args.out, sample, keys),
    ):
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
