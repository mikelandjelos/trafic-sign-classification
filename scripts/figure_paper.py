"""Figures for the IEEESTEC paper (docs/paper/), in English and sized for the two-column template.

    poetry run python scripts/figure_paper.py

The report figures in figures/report/ carry Serbian labels and are laid out for a full A4 page,
so at the paper's column width their text would shrink to ~4 pt. These are redrawn at the
template's column width (3.5 in) or text width (7.16 in), with 8 pt serif text as the template
asks for figure labels.

**No title or caption is drawn in any image.** Every caption is a "figure caption" paragraph in
the .doc, so an image carries only axes, labels, a legend and the panel letters its caption
refers to.

Nothing is re-derived: the data come from the functions the report figures already use
(`figure_robustness.grid`, `degradation_contact_sheet.pick_sharp_sign`, `gtsrb.degradations`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from gtsrb import cache, config, data, degradations, evaluation
from gtsrb.representations import cnn as cnn_module

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent / "demo")]
import analyse_clean
import cnn_mechanics
import degradation_contact_sheet as contact
import figure_robustness as robustness

OUT = config.FIGURES_DIR / "paper"

COLUMN_IN = 3.5
TEXT_IN = 7.16
DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
})

#: Names used in the paper's text and tables, in the report's method order and colours.
PAPER_NAMES = {
    "pca_svm": "PCA",
    "hog_svm": "HOG",
    "bovw_svm": "BoVW",
    "cnn_feat_svm": "CNN-feat",
    "cnn_e2e": "CNN-e2e",
}


def system_modules() -> Path:
    """The five-module system with the recognition module marked as this work's scope."""
    modules = (
        ("M1", "Detection"),
        ("M2", "Optical-flow\ntracking"),
        ("M3", "Recognition"),
        ("M4", "Temporal\naggregation"),
        ("M5", "Semantic\ninterpretation"),
    )
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 0.95))
    ax.set_xlim(0, COLUMN_IN)
    ax.set_ylim(0, 0.95)
    ax.axis("off")

    width, gap, y, height = 0.66, 0.05, 0.28, 0.52
    for i, (code, name) in enumerate(modules):
        x = i * (width + gap)
        scope = code == "M3"
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0,rounding_size=0.04",
            facecolor="#dbe8f5" if scope else "white",
            edgecolor="black" if scope else "#8a8a8a",
            linewidth=1.1 if scope else 0.6, linestyle="-" if scope else (0, (3, 2)),
        ))
        ax.text(x + width / 2, y + height - 0.1, code, ha="center", va="center",
                fontsize=8, fontweight="bold", color="black" if scope else "#6b6b6b")
        ax.text(x + width / 2, y + height / 2 - 0.06, name, ha="center", va="center",
                fontsize=6.3, linespacing=0.95, color="black" if scope else "#6b6b6b")
        if i:
            ax.add_patch(FancyArrowPatch(
                (x - gap + 0.01, y + height / 2), (x - 0.01, y + height / 2),
                arrowstyle="-|>", mutation_scale=6, lw=0.6, color="#555555",
            ))
        if scope:
            ax.text(x + width / 2, 0.13, "this work", ha="center", va="center",
                    fontsize=7.5, fontstyle="italic")

    path = OUT / "system_modules.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return path


def degradation_sheet() -> Path:
    """Every degradation at every level on one sign; the identity level is outlined."""
    frame = data.load_annotations("test")
    row = contact.pick_sharp_sign(frame, "raw_gray")
    clean = cache.load_images(row.to_frame().T, "raw_gray")
    key = [row["path"]]

    names = {"noise": "Noise", "blur": "Blur", "gamma": "Gamma"}
    units = {"noise": "σ = {:g}", "blur": "k = {:g}", "gamma": "γ = {:g}"}
    n_levels = max(len(degradations.levels_for(d)) for d in contact.ROW_ORDER)

    fig, axes = plt.subplots(len(contact.ROW_ORDER), n_levels,
                             figsize=(COLUMN_IN, 2.1))
    for r, name in enumerate(contact.ROW_ORDER):
        spec = degradations.get_degradation(name)
        for c, level in enumerate(spec.levels):
            image = degradations.apply(clean, name, level, key)[0]
            ax = axes[r, c]
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            identity = level == spec.identity
            ax.set_title(units[name].format(level), fontsize=7.5, pad=2,
                         fontweight="bold" if identity else "normal")
            if identity:
                ax.add_patch(Rectangle((-0.5, -0.5), image.shape[1], image.shape[0],
                                       fill=False, edgecolor="#1a7f37", linewidth=1.6))
            if c == 0:
                ax.set_ylabel(names[name], fontsize=8)
    fig.subplots_adjust(left=0.06, right=1.0, top=0.93, bottom=0.0, wspace=0.06, hspace=0.22)

    path = OUT / "degradation_contact_sheet.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return path


def robustness_curves() -> Path:
    """Macro-F1 retained against each method's own clean score, one panel per degradation."""
    grid = robustness.grid("macro_f1")
    panels = (
        ("noise", "(a)", "Noise standard deviation σ (gray levels)", False),
        ("blur", "(b)", "Motion blur kernel length k (pixels)", False),
        ("gamma", "(c)", "Gamma γ", True),
    )
    markers = {"pca_svm": "o", "hog_svm": "s", "bovw_svm": "^",
               "cnn_feat_svm": "D", "cnn_e2e": "v"}

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_IN, 2.25), sharey=True)
    for ax, (degradation, letter, xlabel, log_x) in zip(axes, panels, strict=True):
        for method, _, colour in robustness.METHODS:
            levels, retained = grid[method]["curves"][degradation]
            ax.plot(levels, retained, marker=markers[method], ms=3.2, color=colour,
                    label=PAPER_NAMES[method])
        ax.axhline(100, color="#a0a0a0", lw=0.6, ls=":")
        ax.axvline(degradations.identity_for(degradation), color="#a0a0a0", lw=0.6, ls="--")
        if log_x:
            ax.set_xscale("log")
            ax.set_xticks(list(degradations.levels_for(degradation)))
            ax.set_xticks([], minor=True)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        else:
            ax.set_xticks(list(degradations.levels_for(degradation)))
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.25, lw=0.5)
        ax.text(0.03, 0.05, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold")
    axes[0].set_ylabel("Retained macro-F1 (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.0)

    path = OUT / "robustness_curves.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return path


def representations() -> Path:
    """What each representation keeps of the same two signs, read from the trained artifacts.

    PCA: the image rebuilt from its 256 coordinates. HOG: the cell orientation histograms.
    BoVW: the visual word assigned at each dense keypoint (colours are nominal word ids).
    CNN: the mean activation of the last convolutional block, the grid the head reads.
    """
    models = {name: joblib.load(config.MODELS_DIR / f"{name}_raw_gray.joblib")["representation"]
              for name in ("pca_svm", "hog_svm", "bovw_svm")}
    network = cnn_mechanics.load_trained("raw_gray")
    train, _ = data.train_val_split()
    images = cache.load_images(train, "raw_gray")
    samples = (25, 5)  # Road work (triangle) and Speed limit 80 (HOG's hardest class)

    columns = ("Input", "PCA", "HOG", "BoVW", "CNN")
    fig, axes = plt.subplots(len(samples), len(columns), figsize=(COLUMN_IN, 1.55))
    for row, class_id in enumerate(samples):
        image = images[cnn_mechanics.pick(train, class_id)]

        reconstruction = models["pca_svm"].reconstruct(image[None])[0]
        _, hog_render = models["hog_svm"].visualize(image)
        bovw = models["bovw_svm"]
        grid = bovw.extractor.grid_shape
        words = bovw._fitted().predict(bovw.extractor.describe(image)).reshape(grid)
        with torch.no_grad():
            x = cnn_module.as_batch(image[None])
            for block in network.trunk:
                x = block(x)
        activation = x[0].numpy().mean(axis=0)

        panels = (
            (image, {"cmap": "gray", "vmin": 0, "vmax": 255}),
            (reconstruction, {"cmap": "gray", "vmin": 0, "vmax": 255}),
            # HOG brightness rescaled for display only, as in the report's HOG figure.
            (hog_render, {"cmap": "gray", "vmin": 0, "vmax": hog_render.max() * 0.35}),
            (words % 20, {"cmap": "tab20", "vmin": 0, "vmax": 19}),
            (activation, {"cmap": "magma"}),
        )
        for col, (panel, style) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel, interpolation="nearest", **style)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            if row == 0:
                ax.set_title(columns[col], fontsize=8, pad=2)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.0, wspace=0.05, hspace=0.06)

    path = OUT / "representations.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return path


def accuracy_by_size() -> Path:
    """Clean-test accuracy per sign-size bucket, from the same predictions as Table II."""
    test = data.load_annotations("test")
    y_true = test["class_id"].to_numpy()
    scored = analyse_clean.predict_all(test, y_true)
    buckets = evaluation.size_buckets(test["roi_h"])
    order = ["[0,32)", "[32,48)", "[48,72)", "[72,inf)"]
    labels = ["<32", "32–48", "48–72", "≥72"]
    markers = {"pca_svm": "o", "hog_svm": "s", "bovw_svm": "^",
               "cnn_feat_svm": "D", "cnn_e2e": "v"}

    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.05))
    counts = None
    for method, _, colour in robustness.METHODS:
        by_group = evaluation.accuracy_by_group(y_true, scored[method][1], buckets)
        counts = [by_group[b][1] for b in order]
        ax.plot(range(len(order)), [by_group[b][0] for b in order], marker=markers[method],
                ms=3.2, color=colour, label=PAPER_NAMES[method])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{lab}\n(n = {n:,})" for lab, n in zip(labels, counts, strict=True)])
    ax.set_xlabel("Sign height (pixels)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.70, 1.0)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(ncol=3, frameon=False, loc="lower center", handlelength=1.8,
              columnspacing=1.0, fontsize=7)

    path = OUT / "accuracy_by_size.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return path


def main() -> int:
    config.set_seeds()
    OUT.mkdir(parents=True, exist_ok=True)
    builds = (system_modules, degradation_sheet, robustness_curves, representations,
              accuracy_by_size)
    for build in builds:
        print(f"wrote {build()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
