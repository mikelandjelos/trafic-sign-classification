"""Tasks 6.5 / 6.5b: train the final BoVW models and record their cost.

    poetry run python scripts/train_bovw.py              # both rows
    poetry run python scripts/train_bovw.py --spm-only   # just bovw_spm_svm

The exact analogue of `scripts/train_pca.py` and `scripts/train_hog.py`, at the configuration
task 6.5 selected. Same protocol throughout: train split only, test set untouched, `val_*`
metrics, batched **and** single-image inference.

Two rows from one vocabulary
----------------------------
This script produces **both** `bovw_svm` and `bovw_spm_svm`, and it fits the vocabulary
**once**, sharing it between them. That is not an optimisation -- it is the experimental
control. The two rows must differ in exactly one respect (whether a descriptor's position is
recorded), so anything they could otherwise differ in has to be literally the same object.
Fitting two vocabularies, even with the same seed and the same data, would leave the claim
resting on a convention instead of on construction.

`BoVWSpatialPyramid` subclasses `BoVWRepresentation` for the same reason; see note 14 §9.

The `C` grids differ
--------------------
`C` is tuned per method (Q4), and these are two methods: plain BoVW has `n_words` features
where the pyramid has `n_words x 21`, and the 4.2/5.2/6.5 measurements all show the best `C`
tracking dimensionality. Each row therefore re-selects `C` from the sweep for its own
feature space rather than inheriting the other's.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import sklearn
from sklearn.svm import LinearSVC

from gtsrb import cache, config, data, evaluation, results, timing, tuning
from gtsrb.representations.bovw import BoVWRepresentation, BoVWSpatialPyramid

PLAIN_METHOD = "bovw_svm"
SPM_METHOD = "bovw_spm_svm"
SPM_LEVELS = 2


def model_path(method: str, preproc: str) -> Path:
    return config.MODELS_DIR / f"{method}_{preproc}.joblib"


def train_one(method: str, phi, train_images, y_train, val_images, y_val,
              c_grid, dry_run: bool) -> dict:
    """Tune `C` for this feature space, fit, evaluate, time, save, and record."""
    print(f"\n=== {method} ({phi.n_features} dims) ===", flush=True)

    started = time.perf_counter()
    features_train = phi.transform(train_images)
    features_val = phi.transform(val_images)
    encode_s = time.perf_counter() - started
    print(f"encoded train+val in {encode_s:.0f}s", flush=True)

    tuned = tuning.tune_linear_svc(features_train, y_train, features_val, y_val,
                                   c_grid=c_grid, verbose=True)
    C = float(tuned.best.params["C"])
    class_weight = tuned.best.params["class_weight"]
    print(f"selected: C={C:g} class_weight={class_weight}")

    classifier = LinearSVC(C=C, class_weight=class_weight, max_iter=5000,
                           random_state=config.SEED)

    # Training cost is the WHOLE method: vocabulary fit + encoding + SVM. Timing only the SVM
    # would flatter BoVW against HOG, which has no fitted stage at all (note 12 §2.1).
    def fit_everything() -> None:
        phi.fit(train_images)
        classifier.fit(phi.transform(train_images), y_train)

    train_timing = timing.time_training(fit_everything, label="train")
    print(f"train: {train_timing.median_s:.1f} s (vocabulary + encoding + LinearSVC)")

    val_result = evaluation.evaluate(y_val, classifier.predict(phi.transform(val_images)))
    print(val_result.summary())

    infer_timing = timing.time_inference(
        lambda: classifier.predict(phi.transform(val_images)),
        n_items=len(val_images), label="inference")
    single = val_images[:1]
    single_timing = timing.time_inference(
        lambda: classifier.predict(phi.transform(single)),
        n_items=1, label="inference_single")
    print(f"inference (batched): {infer_timing.ms_per_item:.4f} ms/img")
    print(f"inference (single) : {single_timing.ms_per_item:.4f} ms/img")

    path = model_path(method, phi.preproc)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"representation": phi, "classifier": classifier, "preproc": phi.preproc},
                path)
    size_mb = path.stat().st_size / 1024**2
    print(f"model: {path} ({size_mb:.2f} MB)")

    if dry_run:
        print("--dry-run: nothing written to results.csv")
        return {"macro_f1": val_result.macro_f1, "accuracy": val_result.accuracy}

    run_id = results.start_run()
    rows = [
        *[{**row, "metric": f"val_{row['metric']}"}
          for row in results.rows_from_evaluation(
              run_id, method, phi.preproc, "clean", results.NO_LEVEL, val_result,
              per_class=True)],
        *results.rows_from_timing(run_id, method, phi.preproc, train_timing),
        *results.rows_from_timing(run_id, method, phi.preproc, infer_timing),
        *results.rows_from_timing(run_id, method, phi.preproc, single_timing),
    ]
    extra = [
        ("selected_keypoint_size", float(phi.extractor.keypoint_size)),
        ("selected_step", float(phi.extractor.step)),
        ("selected_n_words", float(phi.n_words)),
        ("n_keypoints", float(phi.extractor.n_keypoints)),
        ("selected_C", C),
        ("selected_class_weight_balanced", 1.0 if class_weight == "balanced" else 0.0),
        ("feature_dim", float(phi.n_features)),
        ("model_size_mb", size_mb),
        ("n_train_images", float(len(train_images))),
    ]
    if isinstance(phi, BoVWSpatialPyramid):
        extra += [("spm_levels", float(phi.levels)), ("spm_cells", float(phi.n_cells))]
    for metric, value in extra:
        rows.append({"run_id": run_id, "method": method, "preproc": phi.preproc,
                     "degradation": "clean", "level": results.NO_LEVEL,
                     "metric": metric, "value": float(value)})

    results.append_rows(rows)
    print(f"run_id: {run_id}; wrote {len(rows)} rows to {config.RESULTS_CSV}")
    return {"macro_f1": val_result.macro_f1, "accuracy": val_result.accuracy}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train final BoVW + BoVW-SPM (6.5, 6.5b).")
    parser.add_argument("--sweep", type=Path,
                        default=config.RESULTS_DIR / "sweeps" / "bovw_configs.csv")
    parser.add_argument("--plain-only", action="store_true")
    parser.add_argument("--spm-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()
    # Bound sklearn's internal pairwise-distance chunking; see bovw.chunk_for_budget.
    sklearn.set_config(working_memory=128)

    selected = tuning.best_from_sweep(args.sweep)
    preproc = str(selected["preproc"])
    size, step = int(selected["keypoint_size"]), int(selected["step"])
    n_words = int(selected["n_words"])
    print(f"selected at 6.5: preproc={preproc} size={size} step={step} k={n_words}")

    train, val = data.train_val_split()
    train_images = cache.load_images(train, preproc)
    val_images = cache.load_images(val, preproc)
    y_train = train["class_id"].to_numpy()
    y_val = val["class_id"].to_numpy()

    # ONE vocabulary, fitted once and shared. This is the experimental control -- see the
    # module docstring.
    started = time.perf_counter()
    plain = BoVWRepresentation(n_words=n_words, step=step, keypoint_size=size,
                               preproc=preproc)
    plain.fit(train_images, progress=True)
    print(f"vocabulary: {time.perf_counter() - started:.0f}s (shared by both rows)")

    spm = BoVWSpatialPyramid(n_words=n_words, step=step, keypoint_size=size,
                             preproc=preproc, levels=SPM_LEVELS)
    spm._kmeans = plain._fitted()

    # The pyramid's feature space is 21x wider, and the best C tracks dimensionality, so the
    # grid is shifted down for it rather than reused.
    scores = {}
    if not args.spm_only:
        scores[PLAIN_METHOD] = train_one(
            PLAIN_METHOD, plain, train_images, y_train, val_images, y_val,
            c_grid=(0.1, 1.0, 10.0, 100.0), dry_run=args.dry_run)
    if not args.plain_only:
        scores[SPM_METHOD] = train_one(
            SPM_METHOD, spm, train_images, y_train, val_images, y_val,
            c_grid=(0.01, 0.1, 1.0, 10.0), dry_run=args.dry_run)

    if len(scores) == 2:
        delta = (scores[SPM_METHOD]["macro_f1"] - scores[PLAIN_METHOD]["macro_f1"]) * 100
        print(f"\nlayout is worth {delta:+.1f} pp macro-F1 "
              f"({scores[PLAIN_METHOD]['macro_f1']:.4f} -> "
              f"{scores[SPM_METHOD]['macro_f1']:.4f}) "
              f"-- same descriptors, same vocabulary, same classifier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
