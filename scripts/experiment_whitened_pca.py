"""Ablation: does whitening change PCA's noise robustness? (note 11 §2.1)

    poetry run python scripts/experiment_whitened_pca.py

Why this is worth running
-------------------------
PCA's noise robustness is the study's headline result: it is last of five on clean test data
and **first** at sigma=40, ahead of both CNNs (note 15 section 3). Note 11 section 2.1 made a
specific mechanistic claim about *why*, recorded as a fact about the operator rather than as
a justification for the default:

> Whitening divides each component by its own standard deviation, which amplifies the
> low-variance retained directions -- the ones where an isotropic perturbation has the largest
> share relative to signal. Whitened PCA is therefore plausibly a *differently robust
> representation*, not merely a rescaled one.

That is directly testable, and it is the right kind of test: **the prediction is that whitening
should HURT noise robustness**, so the run can falsify a claim this project already leans on.
Note 11 also records that the first draft of that section used the noise mechanism as a
*reason* for `whiten=False`, which was wrong -- the reason is fidelity to the Eigenfaces
construction. So the honest version of that correction requires actually measuring the thing
that was improperly used as an argument.

Protocol
--------
Identical to the reported PCA row in every respect except `whiten`: same k=256, same
`raw_gray` cache, same train split, same fixed `class_weight="balanced"` (Q8), `C` re-tuned on
validation for this feature space (Q4 -- whitening changes the feature scale by construction,
so reusing the unwhitened `C` would confound the comparison with a mis-set dial).

Evaluated on **test**, clean plus the full noise ladder, so the retention numbers are directly
comparable with note 15 section 2. Nothing is written to `results.csv`: this is an ablation,
not a sixth row, and mixing it into the reported grid would be exactly the Q5-class mistake
`evaluate_grid.py` exists to prevent.
"""

from __future__ import annotations

import argparse

from gtsrb import cache, config, data, degradations, evaluation, preprocessing, tuning
from gtsrb.representations.pca import PCARepresentation

NOISE_LEVELS = (0, 5, 10, 20, 40)


def build(whiten: bool, train_images, y_train, val_images, y_val, k: int, c_grid):
    """Fit PCA, tune `C` on validation for THIS feature space, return the fitted pair."""
    phi = PCARepresentation(k, preproc=preprocessing.DEFAULT_PREPROC, whiten=whiten)
    phi.fit(train_images)
    features_train = phi.transform(train_images)
    features_val = phi.transform(val_images)

    spread = features_train.std(axis=0)
    print(f"  whiten={whiten}: feature sigma spans "
          f"{spread.max() / spread.min():.1f}x "
          f"({spread.min():.3f} .. {spread.max():.3f})", flush=True)

    tuned = tuning.tune_linear_svc(features_train, y_train, features_val, y_val,
                                   c_grid=c_grid, verbose=False)
    print(f"  whiten={whiten}: selected C={tuned.best.params['C']:g}, "
          f"val macro-F1 {tuned.best.macro_f1:.4f}", flush=True)

    from sklearn.svm import LinearSVC

    classifier = LinearSVC(C=tuned.best.params["C"],
                           class_weight=tuned.best.params["class_weight"],
                           max_iter=5000, random_state=config.SEED)
    classifier.fit(features_train, y_train)
    return phi, classifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Whitened-PCA noise ablation.")
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--c-grid", nargs="+", type=float,
                        default=[0.001, 0.01, 0.1, 1.0, 10.0],
                        help="wider than tuning.C_GRID at BOTH ends: whitening rescales every "
                             "feature, so the right C for this space need not be near the "
                             "unwhitened one, and a grid-edge selection would be uninformative")
    args = parser.parse_args()

    config.set_seeds()
    train, val = data.train_val_split()
    test = data.load_annotations("test")
    preproc = preprocessing.DEFAULT_PREPROC

    train_images = cache.load_images(train, preproc)
    val_images = cache.load_images(val, preproc)
    test_images = cache.load_images(test, preproc)
    y_train = train["class_id"].to_numpy()
    y_val = val["class_id"].to_numpy()
    y_test = test["class_id"].to_numpy()
    keys = [str(path) for path in test["path"]]

    print(f"k={args.k}, preproc={preproc}, class_weight fixed to "
          f"{tuning.CLASS_WEIGHT!r} (Q8)\n")

    models = {}
    for whiten in (False, True):
        models[whiten] = build(whiten, train_images, y_train, val_images, y_val,
                               args.k, tuple(args.c_grid))

    print(f"\n{'sigma':>6} | {'unwhitened':>21} | {'whitened':>21}")
    print(f"{'':>6} | {'macro-F1':>9} {'retained':>11} | {'macro-F1':>9} {'retained':>11}")
    print("-" * 58)

    baselines: dict[bool, float] = {}
    rows = []
    for sigma in NOISE_LEVELS:
        degraded = degradations.apply(test_images, "noise", sigma, keys)
        line = {}
        for whiten, (phi, classifier) in models.items():
            result = evaluation.evaluate(y_test, classifier.predict(phi.transform(degraded)))
            if sigma == 0:
                baselines[whiten] = result.macro_f1
            line[whiten] = (result.macro_f1, result.macro_f1 / baselines[whiten] * 100)
        rows.append((sigma, line))
        print(f"{sigma:>6} | {line[False][0]:>9.4f} {line[False][1]:>10.1f}% | "
              f"{line[True][0]:>9.4f} {line[True][1]:>10.1f}%", flush=True)

    plain40, white40 = rows[-1][1][False], rows[-1][1][True]
    print()
    print(f"clean macro-F1 : unwhitened {baselines[False]:.4f}  whitened {baselines[True]:.4f}"
          f"  ({(baselines[True] - baselines[False]) * 100:+.2f} pp)")
    print(f"retained at 40 : unwhitened {plain40[1]:.1f}%  whitened {white40[1]:.1f}%"
          f"  ({white40[1] - plain40[1]:+.1f} pp)")
    verdict = ("CONFIRMED - whitening hurts noise robustness"
               if white40[1] < plain40[1] else
               "FALSIFIED - whitening does NOT hurt noise robustness")
    print(f"\nnote 11 section 2.1 mechanism: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
