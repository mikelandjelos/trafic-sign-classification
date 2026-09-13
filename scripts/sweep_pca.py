"""Task 4.2: choose `n_components` for the PCA representation on validation.

    poetry run python scripts/sweep_pca.py

Sweeps k x C x class_weight jointly and selects by macro-F1 on the validation split.

Why the sweep is joint
----------------------
`k` and `C` interact: more components means a higher-dimensional, easier-to-separate space,
which shifts the regularisation that suits it. Picking `k` at a fixed `C` would select the
`k` that happens to suit that one `C`, so the two are swept together. `class_weight` is on
the grid for the same reason -- it is flagged as "consider" in PROJECT_TASKS §10, it
interacts with macro-F1 exactly where the 10.7x imbalance bites, and guessing it would be
the kind of unexamined default this project exists to avoid.

Why not select on explained variance
------------------------------------
Because variance and accuracy disagree here, which is the point of the figure. Retaining
95 % of the variance needs 155 components under `clahe_gray`; that number describes
*reconstruction*, and nothing says the discarded 5 % is the unimportant 5 % for
*discrimination*. Task 4.1 already measured the same disagreement across preprocessing
configs: CLAHE retains less variance at fixed k yet classifies better.

Writes:
    results/sweeps/pca_components.csv    every grid point, tidy
    figures/pca/pca_component_sweep.png  the selection figure
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
from gtsrb.representations.pca import PCARepresentation

#: The component counts under consideration (PROJECT_TASKS task 4.2).
COMPONENT_GRID: tuple[int, ...] = (32, 64, 128, 256)


def run_sweep(preproc: str, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Fit PCA at each k, tune the classifier on each, and return the tidy grid."""
    train, val = data.train_val_split()
    train_images = cache.load_images(train, preproc)
    val_images = cache.load_images(val, preproc)
    y_train = train["class_id"].to_numpy()
    y_val = val["class_id"].to_numpy()

    rows: list[dict] = []
    for k in COMPONENT_GRID:
        if verbose:
            print(f"  k = {k}")
        phi = PCARepresentation(k, preproc=preproc).fit(train_images)
        result = tuning.tune_linear_svc(
            phi.transform(train_images), y_train,
            phi.transform(val_images), y_val,
            extra_params={
                "n_components": k,
                "preproc": preproc,
                "cumulative_variance": float(phi.cumulative_variance()[-1]),
            },
            verbose=verbose,
        )
        rows.extend(result.table())

    frame = pd.DataFrame(rows)
    best = frame.loc[frame["macro_f1"].idxmax()].to_dict()
    return frame, best


def figure(frame: pd.DataFrame, best: dict, out_dir: Path) -> Path:
    """Accuracy and macro-F1 against k, plus the variance curve they disagree with."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for class_weight, style in ((None, "-"), ("balanced", "--")):
        subset = frame[frame["class_weight"].isna() if class_weight is None
                       else frame["class_weight"] == class_weight]
        colours = plt.cm.viridis(np.linspace(0.15, 0.8, subset["C"].nunique()))
        for colour, (C, group) in zip(colours, subset.groupby("C"), strict=True):
            group = group.sort_values("n_components")
            left.plot(group["n_components"], group["macro_f1"], style, color=colour,
                      marker="o", ms=4, lw=1.6,
                      label=f"C={C:g}" + (", balanced" if class_weight else ""))

    # Mark the selection with an open ring rather than a filled marker: at k = 256 the
    # eight curves converge into a few pixels, and anything solid covers the very lines the
    # reader is trying to compare. The ring sits around the point and the label is placed
    # clear of the cluster, with an arrow doing the pointing.
    left.plot([best["n_components"]], [best["macro_f1"]], "o", ms=13, mfc="none",
              mec="#c05621", mew=2.2, zorder=6)
    left.annotate(
        f"selected\nk = {int(best['n_components'])}, C = {best['C']:g}\n"
        f"class_weight = {best['class_weight']}\nmacro-F1 = {best['macro_f1']:.3f}",
        xy=(best["n_components"], best["macro_f1"]),
        xytext=(0.58, 0.30), textcoords="axes fraction",
        fontsize=8.5, color="#c05621", fontweight="bold", ha="left", va="top",
        arrowprops={"arrowstyle": "-|>", "color": "#c05621", "lw": 1.1,
                    "linestyle": "dotted", "shrinkA": 4, "shrinkB": 10,
                    "mutation_scale": 11},
        zorder=7,
    )
    left.set_xscale("log", base=2)
    left.set_xticks(COMPONENT_GRID)
    left.set_xticklabels([str(k) for k in COMPONENT_GRID])
    left.set_xlabel("components retained (k)")
    left.set_ylabel("validation macro-F1")
    left.set_title("Selection criterion: macro-F1 on validation", fontsize=10)
    left.legend(fontsize=7, ncol=2)
    left.grid(alpha=0.3)

    # The disagreement: variance saturates long before accuracy does.
    per_k = frame.groupby("n_components").agg(
        macro_f1=("macro_f1", "max"), variance=("cumulative_variance", "first")
    ).reset_index()
    right.plot(per_k["n_components"], per_k["variance"], "o-", color="#718096",
               label="cumulative explained variance")
    right.plot(per_k["n_components"], per_k["macro_f1"], "o-", color="#c05621",
               label="best validation macro-F1")
    right.set_xscale("log", base=2)
    right.set_xticks(COMPONENT_GRID)
    right.set_xticklabels([str(k) for k in COMPONENT_GRID])
    right.set_xlabel("components retained (k)")
    right.set_ylabel("fraction")
    right.set_title("Variance is not the objective — the two curves disagree", fontsize=10)
    right.legend(fontsize=9, loc="lower right")
    right.grid(alpha=0.3)

    fig.suptitle(
        f"PCA component sweep — selected k = {int(best['n_components'])}, "
        f"C = {best['C']:g}, class_weight = {best['class_weight']}\n"
        f"k and C are swept jointly because they interact: more components means an easier "
        f"space to separate,\nwhich shifts the regularisation that suits it",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pca_component_sweep.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="PCA n_components sweep (task 4.2).")
    parser.add_argument("--preproc", default=preprocessing.DEFAULT_PREPROC)
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "sweeps")
    parser.add_argument("--figures", type=Path, default=config.FIGURES_DIR / "pca")
    parser.add_argument("--from-csv", action="store_true",
                        help="redraw the figure from the saved grid, without re-fitting")
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "pca_components.csv"

    if args.from_csv:
        # The grid costs ~10 minutes of SVM fits; a figure tweak should not.
        frame = pd.read_csv(csv_path)
        best = frame.loc[frame["macro_f1"].idxmax()].to_dict()
    else:
        frame, best = run_sweep(args.preproc)
        frame.to_csv(csv_path, index=False)

    print("\n--- best per k (by macro-F1) ---")
    per_k = frame.loc[frame.groupby("n_components")["macro_f1"].idxmax()]
    print(per_k[["n_components", "C", "class_weight", "macro_f1", "accuracy",
                 "cumulative_variance", "fit_seconds"]].to_string(index=False))

    unconverged = frame[~frame["converged"]]
    if len(unconverged):
        print(f"\n{len(unconverged)} of {len(frame)} grid points did not converge:")
        print(unconverged[["n_components", "C", "class_weight", "n_iter"]].to_string(index=False))

    print(f"\nselected: k={int(best['n_components'])}, C={best['C']:g}, "
          f"class_weight={best['class_weight']}, macro_f1={best['macro_f1']:.4f}, "
          f"accuracy={best['accuracy']:.4f}")
    if not args.from_csv:
        print(f"wrote {csv_path}")
    print(f"wrote {figure(frame, best, args.figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
