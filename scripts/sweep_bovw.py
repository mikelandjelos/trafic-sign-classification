"""Task 6.5: choose the BoVW configuration on validation.

    poetry run python scripts/sweep_bovw.py --preproc clahe_gray --vocab 500   # stage 1
    poetry run python scripts/sweep_bovw.py                                    # full grid

Sweeps `(keypoint_size, step)` x `k` x `C` x `class_weight` and selects by validation
macro-F1 -- the same protocol as `sweep_pca.py` and `sweep_hog.py`, through `gtsrb.tuning`.

Why geometry is on the grid at all
----------------------------------
The plan (task 6.1) specified `step=6, keypoint_size=12`. Those values cost a great deal:
measured on the full split at k=500, they give macro-F1 **0.3402**, where `size=6, step=3`
gives **0.5248**. Sampling density moved the result by 18.5 pp, far more than `k` or `C` moved
anything, so it belongs in the sweep rather than being fixed by the plan.

`size` and `step` are also **decoupled** here, which no earlier probe did -- they were always
varied together, so it was not known whether the gain came from finer *spacing* (more
descriptors) or finer *scale* (each descriptor covering less of the sign).

**Measured answer: it is scale.** `step` fixes the keypoint count independently of `size`, so
the two separate cleanly -- at a fixed 256 kp, scale 6 -> 4 is worth **+12.3 pp**, while at a
fixed scale, 2.25x the density is worth **+2.1 pp**. Roughly 6x apart. An earlier version of
this docstring called the lever "sampling density"; that was imprecise, and the two had been
confounded in every probe that produced the claim. See `14-bovw.md` section 7.2.

Why `k` matters more than the literature check suggested
--------------------------------------------------------
Published BoVW work on traffic signs succeeds with a **300-word** codebook, inside the range
already tested, which was read as "vocabulary is not the missing ingredient". **That was
wrong:** 500 -> 1000 words is worth a consistent **+6 pp** at two separate geometries. The
published 300 words feed a *pLSA topic model*, not a linear SVM -- a codebook size was
transferred across a difference in what sits on top of it. See `14-bovw.md` section 7.3.

Cost
----
Descriptor extraction dominates and must be redone per (preproc, geometry) -- roughly 80-120 s
for the training split. The vocabulary and the SVM fits are cheap by comparison (200-1000
dimensions, versus HOG's 900-2352). Descriptors are therefore extracted **once per geometry**
and reused across every `k`, which is the only reason the full grid is affordable.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from gtsrb import cache, config, data, preprocessing, tuning
from gtsrb.representations.bovw import (
    DESCRIPTOR_DIM,
    DenseSIFT,
    normalise_histograms,
)

#: (keypoint_size, step). Decoupled deliberately -- see the module docstring.
#:
#: The (6,3)/(6,2)/(4,3)/(4,2) block is a 2x2 factorial in scale x spacing and is what
#: separated the two effects; it is kept even though those cells are no longer competitive,
#: because that separation is the reportable finding, not the winning cell.
GEOMETRY_GRID: tuple[tuple[int, int], ...] = (
    (12, 6),   # the plan's values, kept as the baseline the change is measured against
    (6, 3),    # ---- factorial: coarse scale, coarse spacing
    (6, 2),    # ---- factorial: coarse scale, fine spacing   -> isolates spacing
    (4, 3),    # ---- factorial: fine scale,   coarse spacing -> isolates scale
    (4, 2),    # ---- factorial: fine scale,   fine spacing
    (3, 2),
    (2, 2),    # best on clahe_gray; the curve had still not flattened
)

#: 500 -> 1000 was worth a consistent +6 pp and had not plateaued, so 2000 is on the grid.
VOCAB_GRID: tuple[int, ...] = (500, 1000, 2000)
VOCAB_SAMPLES = 200_000

#: BoVW wants a LARGER `C` than the other methods -- its features are the lowest-dimensional
#: in the study, and C=10 won at the top of `tuning.C_GRID`. Extended so the selected value is
#: not again pinned to the grid edge (the caveat PCA's k=256 carries at 4.2).
C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)


def label(size: int, step: int) -> str:
    return f"size {size} / step {step}"


def encode(extractor: DenseSIFT, kmeans, images: np.ndarray,
           budget_bytes: int = 256 * 1024**2) -> np.ndarray:
    """Encode a split, with the chunk sized by MEMORY rather than by image count.

    A fixed image count is a trap here: the descriptor block is
    `chunk x n_keypoints x 128 x 4` bytes, so a 4,000-image chunk costs 0.12 GB at 64
    keypoints and **1.10 GB** at 576. The first version of this script used 4,000 flat and
    was killed by the OOM reaper on the finest geometry.
    """
    chunk = max(1, budget_bytes // (extractor.n_keypoints * DESCRIPTOR_DIM * 4))
    out = np.zeros((len(images), kmeans.n_clusters), dtype=np.float32)
    for start in range(0, len(images), chunk):
        block = images[start:start + chunk]
        words = kmeans.predict(
            extractor.describe_batch(block).reshape(-1, DESCRIPTOR_DIM)
        ).reshape(len(block), extractor.n_keypoints)
        for i, row in enumerate(words):
            out[start + i] = np.bincount(row, minlength=kmeans.n_clusters)
    return normalise_histograms(out, "power_l2")


def run_sweep(preprocs: list[str], geometries, vocabs, c_grid=C_GRID,
              verbose: bool = True):
    train, val = data.train_val_split()
    y_train = train["class_id"].to_numpy()
    y_val = val["class_id"].to_numpy()

    rows: list[dict] = []
    for preproc in preprocs:
        if verbose:
            print(f"\n=== {preproc} ===", flush=True)
        train_images = cache.load_images(train, preproc)
        val_images = cache.load_images(val, preproc)

        for size, step in geometries:
            extractor = DenseSIFT(step=step, keypoint_size=size)
            started = time.perf_counter()
            # Extracted once per geometry and reused for every k -- the descriptors do not
            # depend on the vocabulary, and this is what makes the grid affordable.
            sample = extractor.sample_descriptors(
                train_images, VOCAB_SAMPLES, seed_parts=("bovw", preproc, size, step)
            )
            if verbose:
                print(f"  {label(size, step)}: {extractor.n_keypoints} kp/img, "
                      f"sampled {len(sample)} in {time.perf_counter() - started:.0f}s",
                      flush=True)

            for k in vocabs:
                started = time.perf_counter()
                kmeans = MiniBatchKMeans(k, random_state=config.SEED, n_init=3,
                                         batch_size=4096, max_iter=100).fit(sample)
                features_train = encode(extractor, kmeans, train_images)
                features_val = encode(extractor, kmeans, val_images)
                if verbose:
                    print(f"    k={k}: vocabulary + encoding "
                          f"{time.perf_counter() - started:.0f}s", flush=True)
                result = tuning.tune_linear_svc(
                    features_train, y_train, features_val, y_val,
                    c_grid=c_grid,
                    extra_params={"preproc": preproc, "keypoint_size": size, "step": step,
                                  "n_keypoints": extractor.n_keypoints, "n_words": k},
                    verbose=verbose,
                )
                rows.extend(result.table())

    frame = pd.DataFrame(rows)
    return frame, frame.loc[frame["macro_f1"].idxmax()].to_dict()


def figure(frame: pd.DataFrame, best: dict, out_dir: Path) -> Path:
    """Macro-F1 against descriptor *scale*, which is the lever that actually moves it.

    The left panel plots against keypoint size rather than keypoint count deliberately: at
    step 2 the count is 576 regardless of size, so a count axis would stack four very
    different results on one x value and hide the effect entirely.
    """
    frame = frame.copy()
    frame["label"] = [label(int(s), int(t))
                      for s, t in zip(frame["keypoint_size"], frame["step"], strict=True)]
    per_config = frame.loc[frame.groupby(["label", "n_keypoints", "n_words"])["macro_f1"].idxmax()]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for k, group in per_config.groupby("n_words"):
        group = group.sort_values("keypoint_size")
        left.plot(group["keypoint_size"], group["macro_f1"], "o-", lw=1.8, ms=6,
                  label=f"k = {int(k)} words")
    left.plot([best["keypoint_size"]], [best["macro_f1"]], "o", ms=13, mfc="none",
              mec="#c05621", mew=2.2, zorder=6)
    left.invert_xaxis()  # finer scale to the right, so the curve reads left-to-right as "better"
    left.set_xlabel("descriptor scale — keypoint size in px (finer →)")
    left.set_ylabel("validation macro-F1")
    left.set_title("Scale is the lever; vocabulary size is worth a further ~6 pp",
                   fontsize=10)
    left.legend(fontsize=9)
    left.grid(alpha=0.3)

    order = [label(s, t) for s, t in GEOMETRY_GRID]
    present = [x for x in order if x in set(per_config["label"])]
    best_by_geom = per_config.groupby("label")["macro_f1"].max().reindex(present)
    right.bar(range(len(present)), best_by_geom.to_numpy(), color="#2b6cb0")
    right.set_xticks(range(len(present)))
    right.set_xticklabels(present, fontsize=8, rotation=20, ha="right")
    right.set_ylabel("best validation macro-F1")
    right.set_title("size and step decoupled", fontsize=10)
    right.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"BoVW configuration sweep — selected {label(int(best['keypoint_size']), int(best['step']))}, "
        f"k = {int(best['n_words'])}, C = {best['C']:g}, class_weight = {best['class_weight']}\n"
        f"macro-F1 = {best['macro_f1']:.4f}, accuracy = {best['accuracy']:.4f}. "
        f"The plan's size 12 / step 6 is included as the baseline.",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bovw_config_sweep.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="BoVW configuration sweep (task 6.5).")
    # `raw_gray` only by default: preprocessing is held FIXED across all six configurations
    # (plan change 2026-09-16), so sweeping it here would reintroduce the per-method
    # preprocessing the comparison no longer uses. See PROJECT_TASKS.md section 1.
    parser.add_argument("--preproc", nargs="+", default=["raw_gray"],
                        choices=list(preprocessing.PREPROC_CONFIGS))
    parser.add_argument("--vocab", nargs="+", type=int, default=list(VOCAB_GRID))
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "sweeps")
    parser.add_argument("--figures", type=Path, default=config.FIGURES_DIR / "bovw")
    parser.add_argument("--csv-name", default="bovw_configs.csv")
    parser.add_argument("--from-csv", action="store_true")
    parser.add_argument("--c-grid", nargs="+", type=float, default=list(C_GRID),
                        help="stage 1 found C=10 winning at the TOP of tuning.C_GRID, "
                             "so this method uses an extended grid (see C_GRID)")
    parser.add_argument("--geometry", nargs="+", type=int, default=None,
                        metavar="SIZE:STEP-as-pairs",
                        help="flat list of size step size step ...; default: GEOMETRY_GRID")
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / args.csv_name

    if args.from_csv:
        frame = pd.read_csv(csv_path)
        best = frame.loc[frame["macro_f1"].idxmax()].to_dict()
    else:
        geometries = GEOMETRY_GRID
        if args.geometry:
            flat = args.geometry
            geometries = tuple(zip(flat[::2], flat[1::2], strict=True))
        frame, best = run_sweep(args.preproc, geometries, tuple(args.vocab),
                                c_grid=tuple(args.c_grid))
        frame.to_csv(csv_path, index=False)

    print("\n--- best per (preproc, geometry, k) ---")
    grouped = frame.loc[
        frame.groupby(["preproc", "keypoint_size", "step", "n_words"])["macro_f1"].idxmax()
    ]
    print(grouped[["preproc", "keypoint_size", "step", "n_keypoints", "n_words", "C",
                   "class_weight", "macro_f1", "accuracy"]].to_string(index=False))

    print(f"\nselected: preproc={best['preproc']} "
          f"size={int(best['keypoint_size'])} step={int(best['step'])} "
          f"k={int(best['n_words'])} C={best['C']:g} "
          f"class_weight={best['class_weight']} "
          f"macro_f1={best['macro_f1']:.4f} accuracy={best['accuracy']:.4f}")
    if not args.from_csv:
        print(f"wrote {csv_path}")
    print(f"wrote {figure(frame, best, args.figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
