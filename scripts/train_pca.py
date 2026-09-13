"""Task 4.3: train the final PCA + `LinearSVC` model and record its cost.

    poetry run python scripts/train_pca.py

Trains at the configuration task 4.2 selected, records validation performance and the cost
metrics for Table 1 (task 9.1) into `results/results.csv`, and saves the fitted model for
the evaluation grid at 8.1 to reuse without retraining.

What this script does NOT do: touch the test set. Test evaluation happens once, at task 8.1,
for all five methods at once. Rows written here are therefore `val_*` metrics; 8.1 writes the
plain `accuracy` / `macro_f1` rows from test.

Training data
-------------
**The train split only (31,379 images) -- not train+val.** Refitting on train+val after
selection is the usual practice and would give a slightly better model, but the CNN cannot
do it: it needs val for early stopping. Giving PCA, HOG and BoVW 25 % more training data than
the CNN can use would confound the comparison the whole project rests on, so every method is
trained on the same 31,379 rows. Recorded as a deliberate, small sacrifice of absolute
accuracy for comparability.

Hyperparameters
---------------
Read from `results/sweeps/pca_components.csv`, the sweep that chose them, rather than written
as literals here -- a literal would drift silently the first time the sweep is re-run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.svm import LinearSVC

from gtsrb import cache, config, data, evaluation, results, timing, tuning
from gtsrb.representations.pca import PCARepresentation

METHOD = "pca_svm"


def model_path(preproc: str) -> Path:
    return config.MODELS_DIR / f"{METHOD}_{preproc}.joblib"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train final PCA + LinearSVC (task 4.3).")
    parser.add_argument("--sweep", type=Path,
                        default=config.RESULTS_DIR / "sweeps" / "pca_components.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="train and print, but write nothing to results.csv")
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()

    selected = tuning.best_from_sweep(args.sweep)
    preproc = str(selected["preproc"])
    k = int(selected["n_components"])
    C = float(selected["C"])
    class_weight = selected["class_weight"]
    print(f"selected at 4.2: preproc={preproc} k={k} C={C:g} class_weight={class_weight}")

    train, val = data.train_val_split()
    train_images = cache.load_images(train, preproc)
    val_images = cache.load_images(val, preproc)
    y_train = train["class_id"].to_numpy()
    y_val = val["class_id"].to_numpy()

    # --- training cost: the WHOLE method, basis included ---------------------------------
    # PCA fit is part of training this representation. Timing only the SVM would understate
    # PCA's cost and flatter it against HOG, which has no fitted stage at all.
    phi = PCARepresentation(k, preproc=preproc)
    classifier = LinearSVC(C=C, class_weight=class_weight, max_iter=5000,
                           random_state=config.SEED)

    def fit_everything() -> None:
        phi.fit(train_images)
        classifier.fit(phi.transform(train_images), y_train)

    train_timing = timing.time_training(fit_everything, label="train")
    print(f"train: {train_timing.median_s:.1f} s (PCA fit + LinearSVC, single run)")

    # --- validation performance ----------------------------------------------------------
    val_features = phi.transform(val_images)
    val_result = evaluation.evaluate(y_val, classifier.predict(val_features))
    print(val_result.summary())

    # --- inference cost ------------------------------------------------------------------
    # Measured end to end (project + classify), because that is what deploying this method
    # would cost. Timed on val: the test set stays untouched until 8.1.
    infer_timing = timing.time_inference(
        lambda: classifier.predict(phi.transform(val_images)),
        n_items=len(val_images), label="inference",
    )
    print(f"inference (batched): {infer_timing.ms_per_item:.4f} ms/img "
          f"(median of {infer_timing.repeats}, first discarded)")

    # Batched ms/img amortises a 7830-image matrix multiply and is the right number for
    # "what does the grid cost". It is the WRONG number for a latency claim: a camera
    # presents one frame at a time, with no batch to amortise over. Both are recorded, so
    # nobody can quote the throughput figure as a per-frame latency.
    single = val_images[:1]
    single_timing = timing.time_inference(
        lambda: classifier.predict(phi.transform(single)),
        n_items=1, label="inference_single",
    )
    print(f"inference (single image): {single_timing.ms_per_item:.4f} ms/img "
          f"-- {single_timing.ms_per_item / infer_timing.ms_per_item:.0f}x the batched figure")

    # --- model size ----------------------------------------------------------------------
    path = model_path(preproc)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"representation": phi, "classifier": classifier, "preproc": preproc}, path)
    size_mb = path.stat().st_size / 1024**2
    print(f"model: {path} ({size_mb:.2f} MB)")

    if args.dry_run:
        print("\n--dry-run: nothing written to results.csv")
        return 0

    # --- record --------------------------------------------------------------------------
    run_id = results.start_run()
    rows = [
        # Validation metrics are prefixed so they can never be confused with the test rows
        # task 8.1 writes under the same (method, preproc, clean, 0) key.
        *[{**row, "metric": f"val_{row['metric']}"}
          for row in results.rows_from_evaluation(
              run_id, METHOD, preproc, "clean", results.NO_LEVEL, val_result, per_class=True)],
        *results.rows_from_timing(run_id, METHOD, preproc, train_timing),
        *results.rows_from_timing(run_id, METHOD, preproc, infer_timing),
        *results.rows_from_timing(run_id, METHOD, preproc, single_timing),
    ]
    # The selected hyperparameters travel with the results, as Q4 requires: a reader must be
    # able to see which dial each method was given without opening a sweep file.
    for metric, value in (
        ("selected_n_components", float(k)),
        ("selected_C", C),
        ("selected_class_weight_balanced", 1.0 if class_weight == "balanced" else 0.0),
        ("feature_dim", float(phi.n_features)),
        ("model_size_mb", size_mb),
        ("n_train_images", float(len(train_images))),
    ):
        rows.append({"run_id": run_id, "method": METHOD, "preproc": preproc,
                     "degradation": "clean", "level": results.NO_LEVEL,
                     "metric": metric, "value": value})

    results.append_rows(rows)
    print(f"\nrun_id: {run_id}")
    print(f"wrote {len(rows)} rows to {config.RESULTS_CSV}")
    print(f"environment: {results.platform_path(run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
