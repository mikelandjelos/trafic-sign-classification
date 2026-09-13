"""Task 5.2: choose the HOG cell geometry and orientation count on validation.

    poetry run python scripts/sweep_hog.py

Sweeps `pixels_per_cell` x `orientations` x `preproc` x `C` x `class_weight` and selects by
validation macro-F1 -- the same protocol `scripts/sweep_pca.py` uses, via `gtsrb.tuning`.
96 grid points, roughly two hours; `--from-csv` redraws the figure without refitting.

Why the same shape of sweep as PCA
----------------------------------
Two findings from task 4.2 carry over directly and are the reason this is not a smaller grid:

- **`C` interacts with feature dimensionality.** PCA's best `C` fell from 10 to 0.01 as *k*
  grew 32 -> 256. HOG's four configurations span 900 to 2352 dimensions, so fixing `C` and
  sweeping geometry alone would pick the geometry that happens to suit one `C`.
- **Each preprocessing config needs its own hyperparameters.** Reusing `clahe_gray`'s cost
  ~1 pp on the other two there, about 39 % of the preprocessing effect being measured, and
  biased toward the config the hyperparameters came from.

Holding the protocol identical across methods is what "fixed classifier" means here (Q4), so
the grid is the same shape even where a hypothesis says part of it is unnecessary. HOG's
features are already block-normalised, so it may well need less regularisation headroom than
PCA -- but that is a hypothesis, and narrowing the grid on it would be assuming the answer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gtsrb import cache, config, data, preprocessing, tuning
from gtsrb.representations.hog import HOGRepresentation

#: (pixels_per_cell, orientations), ordered by the feature dimension they produce.
GEOMETRY_GRID: tuple[tuple[tuple[int, int], int], ...] = (
    ((8, 8), 9),    # 6x6 cells ->  900
    ((8, 8), 12),   # 6x6 cells -> 1200
    ((6, 6), 9),    # 8x8 cells -> 1764
    ((6, 6), 12),   # 8x8 cells -> 2352
)


def config_label(pixels_per_cell, orientations) -> str:
    return f"{pixels_per_cell[0]}px / {orientations}o"


def run_sweep(preprocs: list[str], verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    train, val = data.train_val_split()
    y_train = train["class_id"].to_numpy()
    y_val = val["class_id"].to_numpy()

    rows: list[dict] = []
    for preproc in preprocs:
        if verbose:
            print(f"\n=== {preproc} ===", flush=True)
        train_images = cache.load_images(train, preproc)
        val_images = cache.load_images(val, preproc)

        for pixels_per_cell, orientations in GEOMETRY_GRID:
            phi = HOGRepresentation(
                orientations=orientations, pixels_per_cell=pixels_per_cell, preproc=preproc
            ).fit(train_images[:1])
            if verbose:
                print(f"  {config_label(pixels_per_cell, orientations)}  "
                      f"dim={phi.n_features}", flush=True)
            result = tuning.tune_linear_svc(
                phi.transform(train_images), y_train,
                phi.transform(val_images), y_val,
                extra_params={
                    "preproc": preproc,
                    "pixels_per_cell": pixels_per_cell[0],
                    "orientations": orientations,
                    "feature_dim": phi.n_features,
                },
                verbose=verbose,
            )
            rows.extend(result.table())

    frame = pd.DataFrame(rows)
    return frame, frame.loc[frame["macro_f1"].idxmax()].to_dict()


def figure(frame: pd.DataFrame, best: dict, out_dir: Path) -> Path:
    """Left: the sweep for the winning preproc. Right: the three preprocs compared."""
    headline = best["preproc"]
    frame = frame.copy()
    frame["label"] = [config_label((p, p), o)
                      for p, o in zip(frame["pixels_per_cell"], frame["orientations"],
                                      strict=True)]
    order = [config_label(p, o) for p, o in GEOMETRY_GRID]
    positions = {label: i for i, label in enumerate(order)}

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.9))

    subset = frame[frame["preproc"] == headline]
    for class_weight, style in ((None, "-"), ("balanced", "--")):
        rows = subset[subset["class_weight"].isna() if class_weight is None
                      else subset["class_weight"] == class_weight]
        colours = plt.cm.viridis(np.linspace(0.15, 0.8, rows["C"].nunique()))
        for colour, (C, group) in zip(colours, rows.groupby("C"), strict=True):
            group = group.assign(x=[positions[v] for v in group["label"]]).sort_values("x")
            left.plot(group["x"], group["macro_f1"], style, color=colour, marker="o", ms=4,
                      lw=1.6, label=f"C={C:g}" + (", balanced" if class_weight else ""))

    best_x = positions[config_label((int(best["pixels_per_cell"]),) * 2,
                                    int(best["orientations"]))]
    left.plot([best_x], [best["macro_f1"]], "o", ms=13, mfc="none", mec="#c05621", mew=2.2,
              zorder=6)
    left.annotate(
        f"selected\n{config_label((int(best['pixels_per_cell']),) * 2, int(best['orientations']))}"
        f", C = {best['C']:g}\nclass_weight = {best['class_weight']}\n"
        f"macro-F1 = {best['macro_f1']:.3f}",
        xy=(best_x, best["macro_f1"]), xytext=(0.04, 0.30), textcoords="axes fraction",
        fontsize=8.5, color="#c05621", fontweight="bold", ha="left", va="top",
        arrowprops={"arrowstyle": "-|>", "color": "#c05621", "lw": 1.1,
                    "linestyle": "dotted", "shrinkA": 4, "shrinkB": 10,
                    "mutation_scale": 11},
        zorder=7,
    )
    left.set_ylabel("validation macro-F1")
    left.set_title(f"{headline}: cell geometry x C x class_weight", fontsize=10)
    left.legend(fontsize=7, ncol=2, loc="lower right")

    configs = ("raw_gray", "clahe_gray", "clahe_hsv")
    colours = dict(zip(configs, plt.cm.plasma(np.linspace(0.1, 0.7, len(configs))),
                       strict=True))
    for preproc in configs:
        rows = frame[frame["preproc"] == preproc]
        if rows.empty:
            continue
        per_config = rows.groupby("label")["macro_f1"].max().reindex(order)
        right.plot(range(len(order)), per_config.to_numpy(), "o-", color=colours[preproc],
                   lw=2.0, ms=5, label=preproc)

    right.set_ylabel("best validation macro-F1")
    right.set_title("Each preprocessing config swept independently", fontsize=10)
    right.legend(fontsize=8.5)

    for axis in (left, right):
        axis.set_xticks(range(len(order)))
        axis.set_xticklabels(
            [f"{label}\n{d} dims" for label, d in
             zip(order, [900, 1200, 1764, 2352], strict=True)], fontsize=8)
        axis.set_xlabel("cell size / orientations")
        axis.grid(alpha=0.3)

    fig.suptitle(
        f"HOG configuration sweep — selected {headline}, "
        f"{config_label((int(best['pixels_per_cell']),) * 2, int(best['orientations']))}, "
        f"C = {best['C']:g}, class_weight = {best['class_weight']}\n"
        f"Geometry, C and class_weight swept jointly, and each preprocessing config "
        f"independently — the same protocol as the PCA sweep",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hog_config_sweep.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="HOG configuration sweep (task 5.2).")
    parser.add_argument("--preproc", nargs="+", default=list(preprocessing.PREPROC_CONFIGS))
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "sweeps")
    parser.add_argument("--figures", type=Path, default=config.FIGURES_DIR / "hog")
    parser.add_argument("--from-csv", action="store_true",
                        help="redraw the figure from the saved grid, without re-fitting")
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "hog_configs.csv"

    if args.from_csv:
        frame = pd.read_csv(csv_path)
        best = frame.loc[frame["macro_f1"].idxmax()].to_dict()
    else:
        frame, best = run_sweep(args.preproc)
        frame.to_csv(csv_path, index=False)

    print("\n--- best per (preproc, geometry) ---")
    per_config = frame.loc[
        frame.groupby(["preproc", "pixels_per_cell", "orientations"])["macro_f1"].idxmax()
    ]
    print(per_config[["preproc", "pixels_per_cell", "orientations", "feature_dim", "C",
                      "class_weight", "macro_f1", "accuracy"]].to_string(index=False))

    print("\n--- best per preproc ---")
    per_preproc = frame.loc[frame.groupby("preproc")["macro_f1"].idxmax()]
    print(per_preproc[["preproc", "pixels_per_cell", "orientations", "feature_dim", "C",
                       "class_weight", "macro_f1", "accuracy"]].to_string(index=False))

    unconverged = frame[~frame["converged"]]
    if len(unconverged):
        print(f"\n{len(unconverged)} of {len(frame)} grid points did not converge")

    print(f"\nselected: preproc={best['preproc']} "
          f"pixels_per_cell={int(best['pixels_per_cell'])} "
          f"orientations={int(best['orientations'])} C={best['C']:g} "
          f"class_weight={best['class_weight']} macro_f1={best['macro_f1']:.4f} "
          f"accuracy={best['accuracy']:.4f}")
    if not args.from_csv:
        print(f"wrote {csv_path}")
    print(f"wrote {figure(frame, best, args.figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
