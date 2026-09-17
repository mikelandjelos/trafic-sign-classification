"""Task 8.1: run the full evaluation grid on the TEST set.

    poetry run python scripts/evaluate_grid.py                    # raw_gray, all 5 methods
    poetry run python scripts/evaluate_grid.py --methods hog_svm  # one method
    poetry run python scripts/evaluate_grid.py --dry-run          # no results.csv writes

5 methods x 16 conditions (clean + noise/blur/gamma at 5 levels each) = **80 cells**.
**Inference only -- nothing is retrained.** Every model was fitted once by its own
`train_*.py` and is reloaded here exactly as saved.

This is the first and only place the test set is touched. Rows written here carry the plain
`accuracy` / `macro_f1` names; the `val_*` rows already in `results.csv` came from selection
and must never be confused with these.

The Q5 guard: a representation travels with its preprocessing
--------------------------------------------------------------
A representation fitted on `clahe_gray` **cannot detect** being handed `raw_gray` images --
both are 2304-dimensional, no shape check fires, and the results look entirely plausible.
This is the one place in the project where that mistake can happen.

So `preproc` is never passed in as a flag here. **It is read out of each saved artifact**,
which is why every `train_*.py` writes it alongside the weights. The runner then loads the
cache that name selects and hands the model only those pixels. The pairing is constructed as
one unit and cannot come apart; `tests/test_evaluate_grid.py` asserts it.

Why the loop is ordered (preproc -> condition -> method)
--------------------------------------------------------
Degradations are seeded per image by `config.rng_for(degradation, level, path)`, so any two
methods would see identical pixels anyway. Ordering the loop this way makes that **structural
rather than merely seeded**: every method at a given preproc is handed *the same array
object*, so "the five methods were compared on different pixels" is not a failure mode that
can occur, however the seeding is later changed.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import joblib
import numpy as np
import sklearn

from gtsrb import cache, config, data, degradations, evaluation, preprocessing, results

#: The comparison, in report order -- the proposal's five configurations (section 4.3).
#:
#: `bovw_spm_svm` was added on 2026-09-16 and **removed again on 2026-09-17**: at the k=1000
#: the sweep selected, L=2 is 21,000 dimensions and ~6 GB resident to train, which does not
#: fit this machine alongside everything else. The layout measurement it was added to carry is
#: already recorded (note 14 section 9) and stays in the report as a diagnostic. See index Q6.
METHODS: tuple[str, ...] = (
    "pca_svm", "hog_svm", "bovw_svm", "cnn_feat_svm", "cnn_e2e",
)

#: `clean` plus every (degradation, level). Note gamma's identity is 1.0, in the MIDDLE of
#: its range, so `levels[0]` is not the baseline -- `identity_for` is.
GRID_DEGRADATIONS: tuple[str, ...] = ("noise", "blur", "gamma")


@dataclass
class LoadedMethod:
    """One trained model, bound to the preprocessing it was fitted on."""

    name: str
    preproc: str
    predict: Callable[[np.ndarray], np.ndarray]


def _load_sklearn_method(name: str) -> LoadedMethod:
    """PCA / HOG / BoVW / CNN-features: a representation plus a LinearSVC."""
    matches = sorted(config.MODELS_DIR.glob(f"{name}_*.joblib"))
    if not matches:
        raise SystemExit(
            f"no trained model for {name} in {config.MODELS_DIR}\n"
            f"run the matching scripts/train_*.py first"
        )
    if len(matches) > 1:
        raise SystemExit(
            f"{len(matches)} saved models for {name}: {[p.name for p in matches]}\n"
            f"preprocessing is fixed for the comparison, so exactly one is expected -- "
            f"delete the superseded ones rather than guessing which is current"
        )
    blob = joblib.load(matches[0])
    phi, classifier = blob["representation"], blob["classifier"]
    # From the ARTIFACT, never from a flag. This is the Q5 guard.
    preproc = str(blob["preproc"])
    return LoadedMethod(name, preproc,
                        lambda images: classifier.predict(phi.transform(images)))


def _load_cnn_feat_svm() -> LoadedMethod:
    """The SVM head is saved alone; the network it reads is shared with `cnn_e2e`."""
    import torch

    from gtsrb.representations import cnn as cnn_module

    matches = sorted(config.MODELS_DIR.glob("cnn_feat_svm_*.joblib"))
    if not matches:
        raise SystemExit("no trained cnn_feat_svm; run scripts/train_cnn_features.py")
    if len(matches) > 1:
        raise SystemExit(f"{len(matches)} saved cnn_feat_svm models: "
                         f"{[p.name for p in matches]}; expected exactly one")
    blob = joblib.load(matches[0])
    preproc = str(blob["preproc"])
    classifier = blob["classifier"]

    net_path = config.MODELS_DIR / f"cnn_e2e_{preproc}.pt"
    if not net_path.exists():
        raise SystemExit(
            f"cnn_feat_svm was fitted on the {preproc} network, but {net_path} is missing. "
            f"The SVM head cannot run without it -- this is why Table 1's honest model size "
            f"for this row is the head PLUS the shared network."
        )
    weights = torch.load(net_path, weights_only=True)
    model = cnn_module.SmallCNN(in_channels=int(weights["in_channels"]))
    model.load_state_dict(weights["state_dict"])
    model.eval()

    def predict(images: np.ndarray, batch: int = 256) -> np.ndarray:
        features = np.empty((len(images), model.embedding_dim), dtype=np.float32)
        for start in range(0, len(images), batch):
            block = cnn_module.as_batch(images[start:start + batch])
            features[start:start + len(block)] = model.embed(block).numpy()
        return classifier.predict(features)

    return LoadedMethod("cnn_feat_svm", preproc, predict)


def _load_cnn_e2e() -> LoadedMethod:
    import torch

    from gtsrb.representations import cnn as cnn_module

    matches = sorted(config.MODELS_DIR.glob("cnn_e2e_*.pt"))
    if not matches:
        raise SystemExit("no trained cnn_e2e; run scripts/train_cnn.py")
    # Several preprocessing variants exist from 7.3, so pick the one the comparison fixed on
    # rather than whichever sorts first -- and say so if it is absent.
    wanted = config.MODELS_DIR / f"cnn_e2e_{preprocessing.DEFAULT_PREPROC}.pt"
    if not wanted.exists():
        raise SystemExit(f"no cnn_e2e trained on {preprocessing.DEFAULT_PREPROC}; have "
                         f"{[p.name for p in matches]}")
    weights = torch.load(wanted, weights_only=True)
    preproc = str(weights["preproc"])
    model = cnn_module.SmallCNN(in_channels=int(weights["in_channels"]))
    model.load_state_dict(weights["state_dict"])
    model.eval()

    def predict(images: np.ndarray, batch: int = 256) -> np.ndarray:
        out = np.empty(len(images), dtype=np.int64)
        with torch.no_grad():
            for start in range(0, len(images), batch):
                block = cnn_module.as_batch(images[start:start + batch])
                out[start:start + len(block)] = model(block).argmax(dim=1).numpy()
        return out

    return LoadedMethod("cnn_e2e", preproc, predict)


def load_method(name: str) -> LoadedMethod:
    if name == "cnn_e2e":
        return _load_cnn_e2e()
    if name == "cnn_feat_svm":
        return _load_cnn_feat_svm()
    return _load_sklearn_method(name)


def conditions() -> list[tuple[str, float]]:
    """`clean` plus every (degradation, level) in the grid."""
    out: list[tuple[str, float]] = [("clean", results.NO_LEVEL)]
    for degradation in GRID_DEGRADATIONS:
        out.extend((degradation, level) for level in degradations.levels_for(degradation))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Full evaluation grid on test (task 8.1).")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate and print, but write nothing to results.csv")
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()
    sklearn.set_config(working_memory=128)

    loaded = [load_method(name) for name in args.methods]
    print("loaded models (preproc read from each artifact -- the Q5 guard):")
    for method in loaded:
        print(f"  {method.name:<14} <- {method.preproc}")

    test = data.load_annotations("test")
    y_true = test["class_id"].to_numpy()
    keys = [str(path) for path in test["path"]]

    # Group by preproc so every method sharing one sees the SAME degraded array object.
    by_preproc: dict[str, list[LoadedMethod]] = {}
    for method in loaded:
        by_preproc.setdefault(method.preproc, []).append(method)

    run_id = results.start_run()
    rows: list[dict] = []
    grid = conditions()
    print(f"\n{len(loaded)} methods x {len(grid)} conditions = "
          f"{len(loaded) * len(grid)} cells\n")

    for preproc, methods in by_preproc.items():
        images = cache.load_images(test, preproc)
        for degradation, level in grid:
            started = time.perf_counter()
            degraded = degradations.apply(images, degradation, level, keys)
            for method in methods:
                result = evaluation.evaluate(y_true, method.predict(degraded))
                rows.extend(results.rows_from_evaluation(
                    run_id, method.name, preproc, degradation, level, result,
                    per_class=(degradation == "clean")))
                print(f"  {method.name:<14} {degradation:<6} {level:<5g} "
                      f"acc={result.accuracy:.4f} macro_f1={result.macro_f1:.4f}",
                      flush=True)
            print(f"    [{degradation} {level:g}: {time.perf_counter() - started:.0f}s]",
                  flush=True)

    if args.dry_run:
        print(f"\n--dry-run: {len(rows)} rows NOT written")
        return 0

    results.append_rows(rows)
    print(f"\nrun_id: {run_id}; wrote {len(rows)} rows to {config.RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
