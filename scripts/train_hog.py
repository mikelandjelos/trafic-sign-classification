"""Task 5.3: train the final HOG + `LinearSVC` model and record its cost.

    poetry run python scripts/train_hog.py

The exact analogue of `scripts/train_pca.py`, at the configuration task 5.2 selected
(`raw_gray`, 6 px cells, 9 orientations, C = 0.01, balanced). Same protocol throughout: train
split only, test set untouched, `val_*` metrics, batched **and** single-image inference.

One structural difference from PCA worth seeing in the numbers: **HOG has no fitted stage**
(note 12 section 2.1), so its "training" is only the `LinearSVC` fit and its saved model is
only the SVM coefficients. PCA's model is 96 % basis. The timing below therefore measures
descriptor extraction + SVM fit, with nothing learned in between.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.svm import LinearSVC

from gtsrb import cache, config, data, evaluation, results, timing, tuning
from gtsrb.representations.hog import HOGRepresentation

METHOD = "hog_svm"


def model_path(preproc: str) -> Path:
    return config.MODELS_DIR / f"{METHOD}_{preproc}.joblib"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train final HOG + LinearSVC (task 5.3).")
    parser.add_argument("--sweep", type=Path,
                        default=config.RESULTS_DIR / "sweeps" / "hog_configs.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()

    selected = tuning.best_from_sweep(args.sweep)
    preproc = str(selected["preproc"])
    ppc = int(selected["pixels_per_cell"])
    orientations = int(selected["orientations"])
    C = float(selected["C"])
    class_weight = selected["class_weight"]
    print(f"selected at 5.2: preproc={preproc} pixels_per_cell={ppc} "
          f"orientations={orientations} C={C:g} class_weight={class_weight}")

    train, val = data.train_val_split()
    train_images = cache.load_images(train, preproc)
    val_images = cache.load_images(val, preproc)
    y_train = train["class_id"].to_numpy()
    y_val = val["class_id"].to_numpy()

    phi = HOGRepresentation(orientations=orientations, pixels_per_cell=(ppc, ppc),
                            preproc=preproc)
    classifier = LinearSVC(C=C, class_weight=class_weight, max_iter=5000,
                           random_state=config.SEED)

    def fit_everything() -> None:
        phi.fit(train_images[:1])          # no-op beyond recording the output width
        classifier.fit(phi.transform(train_images), y_train)

    train_timing = timing.time_training(fit_everything, label="train")
    print(f"train: {train_timing.median_s:.1f} s (descriptors + LinearSVC; "
          f"HOG itself learns nothing)")

    val_features = phi.transform(val_images)
    val_result = evaluation.evaluate(y_val, classifier.predict(val_features))
    print(val_result.summary())

    infer_timing = timing.time_inference(
        lambda: classifier.predict(phi.transform(val_images)),
        n_items=len(val_images), label="inference",
    )
    single = val_images[:1]
    single_timing = timing.time_inference(
        lambda: classifier.predict(phi.transform(single)),
        n_items=1, label="inference_single",
    )
    print(f"inference (batched): {infer_timing.ms_per_item:.4f} ms/img")
    print(f"inference (single) : {single_timing.ms_per_item:.4f} ms/img")

    path = model_path(preproc)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"representation": phi, "classifier": classifier, "preproc": preproc}, path)
    size_mb = path.stat().st_size / 1024**2
    print(f"model: {path} ({size_mb:.2f} MB)")

    if args.dry_run:
        print("\n--dry-run: nothing written to results.csv")
        return 0

    run_id = results.start_run()
    rows = [
        *[{**row, "metric": f"val_{row['metric']}"}
          for row in results.rows_from_evaluation(
              run_id, METHOD, preproc, "clean", results.NO_LEVEL, val_result, per_class=True)],
        *results.rows_from_timing(run_id, METHOD, preproc, train_timing),
        *results.rows_from_timing(run_id, METHOD, preproc, infer_timing),
        *results.rows_from_timing(run_id, METHOD, preproc, single_timing),
    ]
    for metric, value in (
        ("selected_pixels_per_cell", float(ppc)),
        ("selected_orientations", float(orientations)),
        ("selected_C", C),
        ("selected_class_weight_balanced", 1.0 if class_weight == "balanced" else 0.0),
        ("feature_dim", float(phi.n_features)),
        ("model_size_mb", size_mb),
        ("n_train_images", float(len(train_images))),
    ):
        rows.append({"run_id": run_id, "method": METHOD, "preproc": preproc,
                     "degradation": "clean", "level": results.NO_LEVEL,
                     "metric": metric, "value": float(value)})

    results.append_rows(rows)
    print(f"\nrun_id: {run_id}")
    print(f"wrote {len(rows)} rows to {config.RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
