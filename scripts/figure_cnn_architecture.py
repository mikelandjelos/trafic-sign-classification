"""Task 7.1: architecture diagram for the small CNN.

    poetry run python scripts/figure_cnn_architecture.py

Shapes and parameter counts are **read off the real model** with forward hooks, not typed
into the script, so the figure cannot drift from `gtsrb.representations.cnn`.

The point of the figure is the split at the end: one trunk, two outputs. `cnn_e2e` continues
into the softmax head; `cnn_feat_svm` branches off the 128-unit penultimate layer into the
same `LinearSVC` every other representation uses. That branch is the only reason the CNN can
be compared like for like with PCA, HOG and BoVW.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from gtsrb import config
from gtsrb.representations.cnn import SmallCNN

CONV_COLOUR = "#2b6cb0"
FC_COLOUR = "#4a5568"
E2E_COLOUR = "#c05621"
SVM_COLOUR = "#2f855a"


def trace(model: SmallCNN) -> list[dict]:
    """Output shape and parameter count of every stage, measured from the model itself."""
    stages: list[dict] = []
    handles = []

    def hook(label: str):
        def record(module, _inputs, output):
            stages.append({
                "label": label,
                "shape": tuple(output.shape[1:]),
                "params": sum(p.numel() for p in module.parameters()),
            })
        return record

    for i, block in enumerate(model.trunk):
        handles.append(block.register_forward_hook(hook(f"block {i + 1}")))
    handles.append(model.embedding.register_forward_hook(hook("embedding")))
    handles.append(model.head.register_forward_hook(hook("head")))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, model.in_channels, *config.IMAGE_SIZE))
    for handle in handles:
        handle.remove()
    return stages


def box(ax, x, y, w, h, text, colour, fontsize=8.5, alpha=0.15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=1.6, edgecolor=colour,
                                facecolor=colour, alpha=alpha, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="black", zorder=3, linespacing=1.45)


def arrow(ax, x0, y0, x1, y1, colour="#4a5568", style="-|>", lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, linestyle=ls,
                                 color=colour, lw=lw, mutation_scale=13, zorder=1,
                                 shrinkA=0, shrinkB=0))


def figure(model: SmallCNN, out_dir: Path) -> Path:
    stages = trace(model)
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(1, 37)
    ax.axis("off")

    channels = model.in_channels
    height, width = config.IMAGE_SIZE
    main_y, box_h = 24.0, 8.0

    # --- input ---------------------------------------------------------------------------
    box(ax, 1, main_y, 9, box_h, f"input\n{channels}×{height}×{width}", "#718096", alpha=0.10)
    cursor = 10.0

    # --- three conv blocks ----------------------------------------------------------------
    for stage in stages[:3]:
        c, h, w = stage["shape"]
        arrow(ax, cursor, main_y + box_h / 2, cursor + 3, main_y + box_h / 2)
        cursor += 3
        box(ax, cursor, main_y, 14, box_h,
            "(conv3×3 → BN → ReLU) × 2\nmaxpool 2×2", CONV_COLOUR)
        ax.text(cursor + 7, main_y - 2.4, f"{c}×{h}×{w}", ha="center", va="top",
                fontsize=9, fontweight="bold", color=CONV_COLOUR)
        ax.text(cursor + 7, main_y + box_h + 1.0, f"{stage['params']:,} params",
                ha="center", va="bottom", fontsize=7.5, color="#718096")
        cursor += 14

    # --- embedding ------------------------------------------------------------------------
    arrow(ax, cursor, main_y + box_h / 2, cursor + 3, main_y + box_h / 2)
    cursor += 3
    flat = stages[2]["shape"][0] * stages[2]["shape"][1] * stages[2]["shape"][2]
    box(ax, cursor, main_y, 15, box_h,
        f"flatten {flat:,}\ndropout → FC {model.embedding_dim} → ReLU", FC_COLOUR)
    ax.text(cursor + 7.5, main_y + box_h + 1.0, f"{stages[3]['params']:,} params",
            ha="center", va="bottom", fontsize=7.5, color="#718096")
    embed_x = cursor + 15
    cursor = embed_x

    # --- the two heads ----------------------------------------------------------------------
    arrow(ax, cursor, main_y + box_h / 2, cursor + 4, main_y + box_h / 2, colour=E2E_COLOUR)
    box(ax, cursor + 4, main_y, 12, box_h, f"FC {config.N_CLASSES}\nsoftmax", E2E_COLOUR)
    ax.text(cursor + 10, main_y + box_h + 1.0, f"{stages[4]['params']:,} params",
            ha="center", va="bottom", fontsize=7.5, color="#718096")
    ax.text(cursor + 10, main_y - 3.0, "method  cnn_e2e", ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=E2E_COLOUR)

    # the branch: penultimate features out to the shared classifier
    arrow(ax, embed_x - 7.5, main_y, embed_x - 7.5, main_y - 9, colour=SVM_COLOUR, ls=(0, (3, 2)))
    ax.text(embed_x - 6.6, main_y - 4.5, f"{model.embedding_dim}-d\nfeatures", ha="left",
            va="center", fontsize=8.5, fontweight="bold", color=SVM_COLOUR)
    box(ax, embed_x - 17, main_y - 17.5, 19, box_h,
        "LinearSVC\n(the same one PCA, HOG,\nBoVW are classified with)", SVM_COLOUR,
        fontsize=8)
    ax.text(embed_x - 7.5, main_y - 19.5, "method  cnn_feat_svm", ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=SVM_COLOUR)

    ax.text(2, 3.5,
            f"total {model.n_parameters():,} parameters  ·  "
            f"conv trunk {sum(s['params'] for s in stages[:3]):,}  ·  "
            f"first FC layer {stages[3]['params']:,}",
            fontsize=9, color="#4a5568")

    fig.suptitle(
        "Small CNN — one trunk, two methods\n"
        "The penultimate layer branches into the shared LinearSVC, which is the only way to "
        "compare a learned\nrepresentation like-for-like with PCA, HOG and BoVW; the softmax "
        "head is the end-to-end ceiling.",
        fontsize=11, y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cnn_architecture.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="CNN architecture diagram (task 7.1).")
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR / "cnn")
    args = parser.parse_args()

    config.set_seeds()
    model = SmallCNN(in_channels=1)
    print(f"{model.n_parameters():,} parameters")
    print(f"wrote {figure(model, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
