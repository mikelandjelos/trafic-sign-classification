"""How much is spatial layout worth to BoVW?

    poetry run python scripts/experiment_spatial_pyramid.py

Runs the same descriptors, the same vocabulary and the same classifier with and without
spatial binning, so the **only** difference between the rows is whether a descriptor's
position is recorded. That makes it the study's central structural claim -- discarding spatial
arrangement is expensive on rigid, aligned objects -- reduced to a single controlled
manipulation, and a far cleaner experiment than the BoVW/HOG gap, which confounds six
differences at once.

Measured at size 2 / step 2, k=500: L0 0.8143 -> L1 0.8973 -> L2 **0.9184** macro-F1.
Layout is worth **+10.4 pp**, and L2 lands within 0.1 pp of HOG (0.9175).

STATUS: **this is a diagnostic, and it stays one.** It was briefly promoted to a sixth
configuration in the comparison (2026-09-16) and reverted a day later (2026-09-17): at the
k=1000 the 6.5 sweep selected, L=2 is 21,000 dimensions and ~6 GB resident, which does not fit
this machine. The science was fine; the engineering was not. See note 14 section 9.4.

So the numbers above are reported in the **discussion**, not in Table 1, and this script --
together with `gtsrb.representations.bovw.BoVWSpatialPyramid` and its tests -- is what keeps
them reproducible. The comparison itself is the proposal's five configurations.

A note on `class_weight`
------------------------
This sweeps only `C` and passes `class_weight=None`, which **predates the Q8 policy** (fixed
`class_weight="balanced"` for every method, 2026-09-17). The recorded numbers were produced
that way and are not re-run, because the claim they support is a **delta between levels** and
all three levels used the identical setting -- so the +10.4 pp is unaffected. Only the
absolute values would shift, and they are not quoted as method results anywhere.

How SPM works
-------------
Instead of one histogram over all keypoints, the image is partitioned into increasingly fine
grids -- level 0 is the whole image (= plain BoVW), level 1 is 2x2, level 2 is 4x4 -- a
histogram is built per cell, and they are concatenated. Weights follow Lazebnik: level `l`
gets 1/2^(L-l), and level 0 shares level 1's weight, so finer levels count for more.

The keypoints lie on a regular grid, so which spatial cell a descriptor belongs to is known
from its index alone -- no extra bookkeeping is needed, which is exactly why the position
information was *available* all along and simply thrown away.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans

from gtsrb import cache, config, data, tuning
from gtsrb.representations.bovw import (
    DESCRIPTOR_DIM,
    DenseSIFT,
    chunk_for_budget,
    normalise_histograms,
)


def pyramid_encode(extractor: DenseSIFT, kmeans, images: np.ndarray, levels: int,
                   budget_bytes: int = 128 * 1024**2) -> np.ndarray:
    """Spatial pyramid histogram. `levels=0` reduces exactly to plain BoVW.

    Memory-bounded chunking, for the same reason `sweep_bovw.encode` is: the descriptor
    block is chunk x n_keypoints x 128 x 4 bytes and a fixed image count silently scales
    with the geometry.
    """
    k = kmeans.n_clusters
    rows, cols = extractor.grid_shape
    cells = [(2 ** level, 2 ** level) for level in range(levels + 1)]
    width = k * sum(r * c for r, c in cells)

    # Lazebnik's weights: level l gets 1/2^(L-l); level 0 shares level 1's weight.
    weights = []
    for level, (r, c) in enumerate(cells):
        w = 1.0 / (2 ** levels) if level == 0 else 1.0 / (2 ** (levels - level + 1))
        weights.extend([w] * (r * c))

    # Sizing delegates to `bovw.chunk_for_budget` -- this had its own copy of the formula,
    # which is how it kept the pre-fix version after sweep_bovw.py was corrected.
    chunk = chunk_for_budget(extractor.n_keypoints, kmeans.n_clusters, budget_bytes)
    out = np.zeros((len(images), width), dtype=np.float32)

    for start in range(0, len(images), chunk):
        block = images[start:start + chunk]
        words = kmeans.predict(
            extractor.describe_batch(block).reshape(-1, DESCRIPTOR_DIM)
        ).reshape(len(block), rows, cols)

        for i, grid in enumerate(words):
            pieces = []
            for r, c in cells:
                # Which spatial cell each keypoint falls into, from its grid index alone.
                row_bin = np.minimum((np.arange(rows) * r) // rows, r - 1)
                col_bin = np.minimum((np.arange(cols) * c) // cols, c - 1)
                for rb in range(r):
                    for cb in range(c):
                        sub = grid[np.ix_(row_bin == rb, col_bin == cb)]
                        pieces.append(np.bincount(sub.ravel(), minlength=k))
            out[start + i] = np.concatenate(pieces)

    out *= np.repeat(np.array(weights, dtype=np.float32), k)
    return normalise_histograms(out, "power_l2")


def main() -> int:
    parser = argparse.ArgumentParser(description="How much is spatial layout worth to BoVW?")
    parser.add_argument("--preproc", default="raw_gray")
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--words", type=int, default=500)
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    config.set_seeds()
    # Bound sklearn's own distance chunking; the 1024 MB default is the largest single
    # allocation in this script.
    sklearn.set_config(working_memory=128)
    train, val = data.train_val_split()
    train_images = cache.load_images(train, args.preproc)
    val_images = cache.load_images(val, args.preproc)
    y_train = train["class_id"].to_numpy()
    y_val = val["class_id"].to_numpy()

    extractor = DenseSIFT(step=args.step, keypoint_size=args.size)
    print(f"{args.preproc}, size {args.size} / step {args.step} "
          f"({extractor.n_keypoints} kp on a {extractor.grid_shape[0]}x"
          f"{extractor.grid_shape[1]} grid), k = {args.words}\n", flush=True)

    started = time.perf_counter()
    kmeans = MiniBatchKMeans(args.words, random_state=config.SEED, n_init=3,
                             batch_size=4096, max_iter=100).fit(
        extractor.sample_descriptors(train_images, 200_000,
                                     seed_parts=("spm", args.size, args.step)))
    print(f"vocabulary: {time.perf_counter() - started:.0f}s "
          f"(shared by every level -- only the pooling differs)\n", flush=True)

    print(f"{'levels':>6} {'cells':>7} {'dims':>7} {'macro_f1':>9} {'acc':>8} {'C':>6}",
          flush=True)
    baseline = None
    for levels in args.levels:
        features_train = pyramid_encode(extractor, kmeans, train_images, levels)
        features_val = pyramid_encode(extractor, kmeans, val_images, levels)
        best = max(
            (tuning.fit_and_score(features_train, y_train, features_val, y_val,
                                  C=C, class_weight=None)
             for C in (1.0, 10.0)),
            key=lambda r: r.macro_f1)
        n_cells = sum(4 ** level for level in range(levels + 1))
        # Only level 0 is a baseline. Taking the first row run as the baseline made a solo
        # `--levels 2` invocation print "(+0.0 pp)" against itself, which reads as "the
        # pyramid bought nothing" -- the exact opposite of the result.
        if levels == 0:
            baseline = best.macro_f1
            delta = "  (baseline = plain BoVW)"
        elif baseline is None:
            delta = "  (no L=0 in this run -- not comparable)"
        else:
            delta = f"  ({(best.macro_f1 - baseline) * 100:+.1f} pp vs plain)"
        print(f"{levels:>6} {n_cells:>7} {features_train.shape[1]:>7} "
              f"{best.macro_f1:>9.4f} {best.accuracy:>8.4f} {best.params['C']:>6g}{delta}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
