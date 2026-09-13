"""Task 4.5: measure what a random per-image split would have inflated (Q2).

    poetry run python scripts/measure_leakage.py

The project's central methodological claim is that GTSRB must be split by *track*, because a
random per-image split puts near-duplicate frames of the same physical sign on both sides.
Everywhere else that claim is **argued**. Here it becomes a number: train the identical model
twice, once on each split, and report the gap in validation accuracy.

What is held constant
---------------------
Everything except the split rule. Same preprocessing, same k, same C, same class_weight, same
val fraction, same seed for the model. The hyperparameters are the ones task 4.2 selected on
the track-disjoint split and are **not** re-tuned for the random split: re-tuning would
measure "how well can each split be exploited" rather than the split's effect on a fixed
model, and the point here is the inflation a researcher would unknowingly report.

The random split is run over several seeds, because a single draw could be lucky and the
claim deserves a spread rather than a point.

Why the naive split is implemented here
---------------------------------------
`gtsrb.data` deliberately offers no way to split by image -- the API only knows how to do the
correct thing. Reproducing the wrong behaviour inside the library so this script could import
it would be backwards, so it lives here, next to the only use that will ever exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

from gtsrb import cache, config, data, evaluation, results, tuning
from gtsrb.representations.pca import PCARepresentation

METHOD = "pca_svm"
RANDOM_SEEDS = (0, 1, 2)

TRACK_COLOUR = "#2b6cb0"
RANDOM_COLOUR = "#c05621"


def random_image_split(frame: pd.DataFrame, seed: int,
                       val_fraction: float = config.VAL_FRACTION) -> np.ndarray:
    """The wrong split: assign every image independently, ignoring tracks."""
    rng = config.rng_for("leakage_random_split", seed)
    return rng.random(len(frame)) < val_fraction


def score(train_frame: pd.DataFrame, val_frame: pd.DataFrame, preproc: str,
          k: int, C: float, class_weight) -> evaluation.ClassificationResult:
    """Fit the whole method on `train_frame`, score it on `val_frame`."""
    train_images = cache.load_images(train_frame, preproc)
    val_images = cache.load_images(val_frame, preproc)
    phi = PCARepresentation(k, preproc=preproc).fit(train_images)
    classifier = LinearSVC(C=C, class_weight=class_weight, max_iter=5000,
                           random_state=config.SEED)
    classifier.fit(phi.transform(train_images), train_frame["class_id"].to_numpy())
    predictions = classifier.predict(phi.transform(val_images))
    return evaluation.evaluate(val_frame["class_id"].to_numpy(), predictions)


def sibling_statistics(frame: pd.DataFrame, is_val: np.ndarray) -> dict:
    """How badly a given assignment mixes tracks -- the *cause* of the inflation."""
    val_tracks = set(frame.loc[is_val, "track_id"])
    train_tracks = set(frame.loc[~is_val, "track_id"])
    shared = val_tracks & train_tracks
    n_val_with_sibling = int(frame.loc[is_val, "track_id"].isin(train_tracks).sum())
    return {
        "tracks_on_both_sides": len(shared),
        "tracks_total": frame["track_id"].nunique(),
        "val_images": int(is_val.sum()),
        "val_images_with_a_sibling_in_train": n_val_with_sibling,
    }


def figure(track: evaluation.ClassificationResult, randoms: list, out_dir: Path) -> Path:
    """The inflation, with the spread over random draws."""
    random_acc = [r.accuracy for r in randoms]
    random_f1 = [r.macro_f1 for r in randoms]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    for ax, track_value, random_values, label in (
        (left, track.accuracy, random_acc, "accuracy"),
        (right, track.macro_f1, random_f1, "macro-F1"),
    ):
        ax.bar([0], [track_value], color=TRACK_COLOUR, width=0.55,
               label="track-disjoint (correct)")
        ax.bar([1], [float(np.mean(random_values))], color=RANDOM_COLOUR, width=0.55,
               yerr=[float(np.std(random_values))], capsize=6,
               label=f"random per-image ({len(random_values)} seeds)")
        gap = float(np.mean(random_values)) - track_value
        ax.annotate(
            f"+{gap * 100:.1f} pp\ninflation",
            xy=(1, float(np.mean(random_values))), xytext=(0.72, 0.95),
            textcoords="axes fraction",
            ha="center", va="top", fontsize=10, fontweight="bold", color=RANDOM_COLOUR,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["by track", "by image"])
        ax.set_ylabel(f"validation {label}")
        ax.set_ylim(0, 1.22)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "What a random per-image split would have inflated\n"
        "Identical model, identical hyperparameters — only the split rule differs. "
        "The gap is leakage,\nnot a better model: every validation image has a "
        "near-duplicate sibling frame in training.",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "split_leakage_measured.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Leakage measurement (task 4.5).")
    parser.add_argument("--sweep", type=Path,
                        default=config.RESULTS_DIR / "sweeps" / "pca_components.csv")
    parser.add_argument("--figures", type=Path, default=config.FIGURES_DIR / "split")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()

    selected = tuning.best_from_sweep(args.sweep)
    preproc, k = str(selected["preproc"]), int(selected["n_components"])
    C, class_weight = float(selected["C"]), selected["class_weight"]
    print(f"fixed configuration: preproc={preproc} k={k} C={C:g} class_weight={class_weight}")

    annotations = data.load_annotations("train")

    # --- the correct split ---------------------------------------------------------------
    train_frame, val_frame = data.train_val_split(annotations)
    track_result = score(train_frame, val_frame, preproc, k, C, class_weight)
    is_val_track = annotations["path"].isin(val_frame["path"]).to_numpy()
    track_stats = sibling_statistics(annotations, is_val_track)
    print(f"\ntrack-disjoint : acc={track_result.accuracy:.4f} "
          f"macro_f1={track_result.macro_f1:.4f}  "
          f"({len(train_frame)} train / {len(val_frame)} val)")
    print(f"  tracks on both sides: {track_stats['tracks_on_both_sides']} of "
          f"{track_stats['tracks_total']}")
    print(f"  val images with a sibling in train: "
          f"{track_stats['val_images_with_a_sibling_in_train']} of {track_stats['val_images']}")

    # --- the wrong split, several draws ---------------------------------------------------
    random_results = []
    for seed in RANDOM_SEEDS:
        is_val = random_image_split(annotations, seed)
        result = score(annotations[~is_val], annotations[is_val], preproc, k, C, class_weight)
        random_results.append(result)
        stats = sibling_statistics(annotations, is_val)
        print(f"\nrandom seed {seed} : acc={result.accuracy:.4f} "
              f"macro_f1={result.macro_f1:.4f}  "
              f"({int((~is_val).sum())} train / {int(is_val.sum())} val)")
        print(f"  tracks on both sides: {stats['tracks_on_both_sides']} of "
              f"{stats['tracks_total']}")
        print(f"  val images with a sibling in train: "
              f"{stats['val_images_with_a_sibling_in_train']} of {stats['val_images']} "
              f"({100 * stats['val_images_with_a_sibling_in_train'] / stats['val_images']:.1f} %)")

    mean_acc = float(np.mean([r.accuracy for r in random_results]))
    mean_f1 = float(np.mean([r.macro_f1 for r in random_results]))
    gap_acc = mean_acc - track_result.accuracy
    gap_f1 = mean_f1 - track_result.macro_f1

    print("\n" + "=" * 72)
    print(f"INFLATION  accuracy: {track_result.accuracy:.4f} -> {mean_acc:.4f}  "
          f"(+{gap_acc * 100:.2f} pp)")
    print(f"INFLATION  macro-F1: {track_result.macro_f1:.4f} -> {mean_f1:.4f}  "
          f"(+{gap_f1 * 100:.2f} pp)")
    print("=" * 72)

    print(f"\nwrote {figure(track_result, random_results, args.figures)}")

    if args.dry_run:
        print("--dry-run: nothing written to results.csv")
        return 0

    run_id = results.start_run()
    rows = [
        {"run_id": run_id, "method": METHOD, "preproc": preproc, "degradation": "clean",
         "level": results.NO_LEVEL, "metric": metric, "value": float(value)}
        for metric, value in (
            ("leakage_val_accuracy_random_split", mean_acc),
            ("leakage_val_macro_f1_random_split", mean_f1),
            ("leakage_gap_accuracy_pp", gap_acc * 100),
            ("leakage_gap_macro_f1_pp", gap_f1 * 100),
            ("leakage_random_split_seeds", float(len(RANDOM_SEEDS))),
        )
    ]
    results.append_rows(rows)
    print(f"run_id: {run_id}; wrote {len(rows)} rows to {config.RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
