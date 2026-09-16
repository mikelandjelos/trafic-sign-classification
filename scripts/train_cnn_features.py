"""Task 7.5: the CNN as a feature extractor -- penultimate layer into the shared LinearSVC.

    poetry run python scripts/train_cnn_features.py

**No retraining.** This loads the network task 7.3 already trained, takes its 128-d
penultimate layer, and hands it to the same `LinearSVC` that classifies PCA, HOG and BoVW
features. That is the only way to compare a learned representation like for like with the
other three: `cnn_e2e` beating them is ambiguous between "better representation" and "better
classifier", and this row removes the ambiguity.

Two properties that make the comparison valid, both enforced elsewhere and relied on here:

- `SmallCNN.embed()` forces `eval()` and `no_grad()` internally, so the features carry no
  dropout noise (note 13 section 3). Extracted with dropout live they would differ between
  two calls on the same image, and `cnn_feat_svm` would be quietly worse for no visible reason.
- `C` and `class_weight` are tuned on validation through `gtsrb.tuning`, exactly as for every
  other method (Q4). The classifier and the selection protocol are fixed; the dials are not.

Caveat that belongs with the result (note 13 section 1.1): these features were optimised for a
**softmax head**, not for an SVM. If this row trails `cnn_e2e`, part of the gap is that
objective mismatch rather than a property of the representation -- an asymmetry PCA, HOG and
BoVW do not have, since they were never optimised for any classifier.
"""

from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import torch
from sklearn.svm import LinearSVC

from gtsrb import cache, config, data, evaluation, results, timing, tuning
from gtsrb.representations import cnn

METHOD = "cnn_feat_svm"


def load_trained(preproc: str) -> cnn.SmallCNN:
    path = config.MODELS_DIR / f"cnn_e2e_{preproc}.pt"
    if not path.exists():
        raise SystemExit(f"no trained network at {path}\nrun: poetry run python scripts/train_cnn.py")
    blob = torch.load(path, weights_only=True)
    model = cnn.SmallCNN(in_channels=int(blob["in_channels"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def embed_all(model: cnn.SmallCNN, images: np.ndarray, batch: int = 256) -> np.ndarray:
    """Penultimate features for a whole split, chunked to bound memory."""
    out = np.empty((len(images), model.embedding_dim), dtype=np.float32)
    for start in range(0, len(images), batch):
        block = cnn.as_batch(images[start:start + batch])
        out[start:start + len(block)] = model.embed(block).numpy()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="CNN features + LinearSVC (task 7.5).")
    parser.add_argument("--preproc", default=None,
                        help="default: whichever preproc task 7.3 selected")
    parser.add_argument("--threads", type=int, default=config.NUM_THREADS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()
    torch.set_num_threads(args.threads)

    # Preprocessing is FIXED across the comparison (plan change 2026-09-16, PROJECT_TASKS.md
    # section 1), so this defaults to that config rather than to whichever preproc scored
    # best at 7.3. Selecting the best-scoring one -- which is what this did -- would silently
    # hand back `clahe_gray` and reintroduce the per-method preprocessing the comparison no
    # longer uses, with nothing in the output to say so.
    if args.preproc is None:
        args.preproc = config.DEFAULT_PREPROC
    print(f"preproc={args.preproc} (fixed for the comparison), threads={args.threads}")

    model = load_trained(args.preproc)
    train, val = data.train_val_split()
    train_images = cache.load_images(train, args.preproc)
    val_images = cache.load_images(val, args.preproc)
    y_train = train["class_id"].to_numpy()
    y_val = val["class_id"].to_numpy()

    started = time.perf_counter()
    z_train = embed_all(model, train_images)
    z_val = embed_all(model, val_images)
    embed_seconds = time.perf_counter() - started
    print(f"embedded {len(z_train)} + {len(z_val)} images in {embed_seconds:.0f} s "
          f"-> {z_train.shape[1]}-d features")

    # Determinism check: embed() must be free of dropout noise (note 13 section 3).
    again = embed_all(model, val_images[:64])
    assert np.array_equal(again, z_val[:64]), "embed() is not deterministic -- dropout active?"
    print("embed() determinism check: OK")

    print("\ntuning C x class_weight on validation:")
    tuned = tuning.tune_linear_svc(z_train, y_train, z_val, y_val,
                                   extra_params={"preproc": args.preproc}, verbose=True)
    print("\n" + tuned.summary())
    best = tuned.best

    classifier = LinearSVC(C=best.params["C"], class_weight=best.params["class_weight"],
                           max_iter=5000, random_state=config.SEED)
    train_timing = timing.time_training(lambda: classifier.fit(z_train, y_train), label="train")
    val_result = evaluation.evaluate(y_val, classifier.predict(z_val))
    print("\n" + val_result.summary())

    infer_timing = timing.time_inference(
        lambda: classifier.predict(embed_all(model, val_images)),
        n_items=len(val_images), label="inference")
    single = val_images[:1]
    single_timing = timing.time_inference(
        lambda: classifier.predict(embed_all(model, single)), n_items=1,
        label="inference_single")
    print(f"inference (batched): {infer_timing.ms_per_item:.4f} ms/img")
    print(f"inference (single) : {single_timing.ms_per_item:.4f} ms/img")

    path = config.MODELS_DIR / f"{METHOD}_{args.preproc}.joblib"
    joblib.dump({"classifier": classifier, "preproc": args.preproc,
                 "cnn_checkpoint": f"cnn_e2e_{args.preproc}.pt"}, path)
    size_mb = path.stat().st_size / 1024**2
    print(f"model: {path} ({size_mb:.2f} MB, SVM head only -- the network is shared with cnn_e2e)")

    if args.dry_run:
        print("\n--dry-run: nothing written to results.csv")
        return 0

    run_id = results.start_run()
    rows = [
        *[{**row, "metric": f"val_{row['metric']}"}
          for row in results.rows_from_evaluation(
              run_id, METHOD, args.preproc, "clean", results.NO_LEVEL, val_result,
              per_class=True)],
        *results.rows_from_timing(run_id, METHOD, args.preproc, train_timing),
        *results.rows_from_timing(run_id, METHOD, args.preproc, infer_timing),
        *results.rows_from_timing(run_id, METHOD, args.preproc, single_timing),
    ]
    for metric, value in (
        ("selected_C", float(best.params["C"])),
        ("selected_class_weight_balanced",
         1.0 if best.params["class_weight"] == "balanced" else 0.0),
        ("feature_dim", float(z_train.shape[1])),
        ("model_size_mb", size_mb),
        ("embed_seconds", embed_seconds),
        ("n_train_images", float(len(z_train))),
    ):
        rows.append({"run_id": run_id, "method": METHOD, "preproc": args.preproc,
                     "degradation": "clean", "level": results.NO_LEVEL,
                     "metric": metric, "value": float(value)})

    results.append_rows(rows)
    print(f"\nrun_id: {run_id}; wrote {len(rows)} rows to {config.RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
