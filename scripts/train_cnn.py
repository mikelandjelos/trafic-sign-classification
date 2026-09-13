"""Task 7.3: train the CNN end to end and record it.

    poetry run python scripts/train_cnn.py --preproc clahe_gray --threads 6

One run per preprocessing config (three in total, see 13-cnn.md section 6). Each run trains
from scratch, because for the CNN the representation *is* the weights -- unlike PCA, HOG and
BoVW, where a new preproc only means re-fitting the classifier.

Writes `cnn_e2e` rows to results.csv, the trained weights to results/models/, and the epoch
history to results/histories/ for the training-curve figure (task 7.6).

Threads
-------
`--threads` is recorded in results.csv and in the run's platform JSON, because it changes the
training *timing* and therefore what Table 1's cost column means. Two rules:

- **Constant within a run.** Changing thread count mid-training would alter reduction order
  partway through, and in training that does not stay in the last bits -- it perturbs a
  gradient, which perturbs the weights, which can move the early-stopping epoch. The run
  would then not be reproducible even on this machine.
- **Varying across runs is acceptable but must be read carefully.** A run sharing the machine
  with another job is slower for reasons that have nothing to do with the model, so training
  times are only comparable between runs at the same thread count and contention.

The project pins 8 (physical cores), not 16 (SMT) -- measured at task 0.3.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from gtsrb import cache, config, data, evaluation, results, timing, training
from gtsrb.representations import cnn

METHOD = "cnn_e2e"


def model_path(preproc: str) -> Path:
    return config.MODELS_DIR / f"{METHOD}_{preproc}.pt"


def history_path(preproc: str) -> Path:
    return config.RESULTS_DIR / "histories" / f"{METHOD}_{preproc}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the CNN end to end (task 7.3).")
    parser.add_argument("--preproc", default="clahe_gray")
    parser.add_argument("--threads", type=int, default=config.NUM_THREADS)
    parser.add_argument("--max-epochs", type=int, default=training.DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=training.DEFAULT_PATIENCE)
    parser.add_argument("--lr", type=float, default=training.DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=training.DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="train but write no results")
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()
    torch.set_num_threads(args.threads)
    print(f"preproc={args.preproc}  threads={torch.get_num_threads()}  "
          f"lr={args.lr}  batch={args.batch_size}  max_epochs={args.max_epochs}  "
          f"patience={args.patience}", flush=True)

    train_frame, val_frame = data.train_val_split()
    train_images = cache.load_images(train_frame, args.preproc)
    val_images = cache.load_images(val_frame, args.preproc)
    y_train = train_frame["class_id"].to_numpy()
    y_val = val_frame["class_id"].to_numpy()

    model = cnn.build(args.preproc)
    print(f"model: {model.n_parameters():,} parameters, "
          f"{model.in_channels} input channel(s)", flush=True)

    started = time.perf_counter()
    history = training.train(
        model, train_images, y_train, val_images, y_val,
        max_epochs=args.max_epochs, batch_size=args.batch_size,
        learning_rate=args.lr, patience=args.patience, verbose=True,
    )
    train_seconds = time.perf_counter() - started
    print("\n" + history.summary(), flush=True)

    # The model now holds the BEST epoch's weights (training.train restores them), so this
    # scores the model that will actually be reported -- not the last epoch's.
    val_loader = training._loader(val_images, y_val, args.batch_size, shuffle=False)
    _, predictions = training.evaluate_model(model, val_loader)
    val_result = evaluation.evaluate(y_val, predictions)
    print(val_result.summary(), flush=True)

    # --- inference cost: batched and single-image, as task 4.3 established ----------------
    batch = cnn.as_batch(val_images)
    single = batch[:1]

    def infer_all() -> None:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(batch), args.batch_size):
                model(batch[start:start + args.batch_size])

    def infer_one() -> None:
        model.eval()
        with torch.no_grad():
            model(single)

    infer_timing = timing.time_inference(infer_all, n_items=len(batch), label="inference")
    single_timing = timing.time_inference(infer_one, n_items=1, label="inference_single")
    print(f"inference (batched): {infer_timing.ms_per_item:.4f} ms/img", flush=True)
    print(f"inference (single) : {single_timing.ms_per_item:.4f} ms/img", flush=True)

    path = model_path(args.preproc)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "preproc": args.preproc,
                "in_channels": model.in_channels}, path)
    size_mb = path.stat().st_size / 1024**2
    print(f"model: {path} ({size_mb:.2f} MB)", flush=True)

    hpath = history_path(args.preproc)
    hpath.parent.mkdir(parents=True, exist_ok=True)
    hpath.write_text(json.dumps({
        "preproc": args.preproc, "threads": args.threads, "lr": args.lr,
        "batch_size": args.batch_size, "best_epoch": history.best_epoch,
        "stopped_early": history.stopped_early, "epochs": history.table(),
    }, indent=2) + "\n")
    print(f"history: {hpath}", flush=True)

    if args.dry_run:
        print("--dry-run: nothing written to results.csv")
        return 0

    run_id = results.start_run()
    rows = [
        *[{**row, "metric": f"val_{row['metric']}"}
          for row in results.rows_from_evaluation(
              run_id, METHOD, args.preproc, "clean", results.NO_LEVEL, val_result,
              per_class=True)],
        *results.rows_from_timing(run_id, METHOD, args.preproc, infer_timing),
        *results.rows_from_timing(run_id, METHOD, args.preproc, single_timing),
    ]
    for metric, value in (
        ("train_seconds", train_seconds),
        ("feature_dim", float(model.embedding_dim)),
        ("model_size_mb", size_mb),
        ("n_parameters", float(model.n_parameters())),
        ("n_train_images", float(len(train_images))),
        ("selected_epoch", float(history.best_epoch)),
        ("epochs_run", float(len(history.records))),
        ("torch_threads", float(args.threads)),
        ("learning_rate", args.lr),
        ("batch_size", float(args.batch_size)),
    ):
        rows.append({"run_id": run_id, "method": METHOD, "preproc": args.preproc,
                     "degradation": "clean", "level": results.NO_LEVEL,
                     "metric": metric, "value": float(value)})

    results.append_rows(rows)
    print(f"\nrun_id: {run_id}; wrote {len(rows)} rows to {config.RESULTS_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
