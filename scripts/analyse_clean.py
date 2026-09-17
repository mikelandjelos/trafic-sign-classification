"""Tasks 9.2, 9.3, 9.4 -- everything that reads the clean-test predictions.

    poetry run python scripts/analyse_clean.py

One pass, three deliverables, because all three want the same predictions:

  9.2  figures/results/confusion_best_worst.png   confusion matrices, best and worst method
  9.3  stdout (markdown)                          top-10 confused pairs, per method
  9.4  figures/results/accuracy_by_size.png       accuracy vs ROI height

**Inference only, from the saved models**, through `evaluate_grid.load_method` -- so the Q5
pairing guard applies here too: each representation is handed the cache it was fitted on, read
from its own artifact rather than from a flag.

Why 9.3 reports all five methods, not just best and worst
----------------------------------------------------------
`PROJECT_TASKS` 9.3 asks for the top-10 pairs and expects speed limits to dominate. The
validation-time observation (note 13 section 7.3) was more interesting than that: **PCA and the
CNN, which share nothing structurally, fail on the SAME classes** -- same top confusion, four
of five worst classes in common, attenuated ~4x. A single method's table cannot show that, so
this prints all five and then quantifies the overlap between them.

If the hardest pairs are the same everywhere, the difficulty is **intrinsic to those classes**
rather than a property of any representation -- which is a different and better claim than
"speed limits confuse predictably", and it is checkable rather than assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_grid

from gtsrb import cache, config, data, evaluation

METHODS: tuple[tuple[str, str], ...] = (
    ("pca_svm", "PCA"),
    ("hog_svm", "HOG"),
    ("bovw_svm", "BoVW"),
    ("cnn_feat_svm", "CNN feat + SVM"),
    ("cnn_e2e", "CNN end-to-end"),
)


def predict_all(test, y_true):
    """{method: (label, predictions, ClassificationResult)} on the clean test set."""
    out = {}
    images_by_preproc: dict[str, np.ndarray] = {}
    for method, label in METHODS:
        loaded = evaluate_grid.load_method(method)
        if loaded.preproc not in images_by_preproc:
            images_by_preproc[loaded.preproc] = cache.load_images(test, loaded.preproc)
        predictions = loaded.predict(images_by_preproc[loaded.preproc])
        result = evaluation.evaluate(y_true, predictions)
        out[method] = (label, predictions, result)
        print(f"  {label:<16} acc={result.accuracy:.4f}  macro-F1={result.macro_f1:.4f}",
              flush=True)
    return out


def figure_confusions(scored: dict, out_dir: Path) -> Path:
    """Task 9.2 -- best and worst, side by side, on a shared log colour scale."""
    ranked = sorted(scored.items(), key=lambda kv: kv[1][2].macro_f1)
    (worst_key, (worst_label, _, worst)) = ranked[0]
    (best_key, (best_label, _, best)) = ranked[-1]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))
    for ax, (label, result) in zip(axes, [(best_label, best), (worst_label, worst)],
                                   strict=True):
        matrix = result.confusion.astype(float)
        # Row-normalise: classes differ 12.5x in test frequency, so raw counts would make the
        # large classes look worse purely by being large.
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(matrix, cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{label} — macro-F1 {result.macro_f1:.4f}", fontsize=11)
        ax.set_xlabel("predicted class", fontsize=9)
        ax.set_ylabel("true class", fontsize=9)
        ax.set_xticks(range(0, 43, 5))
        ax.set_yticks(range(0, 43, 5))
        fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of the true class")

    fig.suptitle(
        f"Confusion matrices, clean test — best ({best_label}) and worst ({worst_label})\n"
        "Row-normalised, because the test classes differ 12.5× in frequency and raw counts "
        "would make the large classes look worse simply for being large.\n"
        "A clean diagonal is a working method; the off-diagonal structure is what the "
        "discussion is about — see the pair table for which pairs they are.",
        fontsize=10.5, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "confusion_best_worst.png"
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  best={best_key}  worst={worst_key}")
    return path


def table_confused_pairs(scored: dict, k: int = 10) -> str:
    """Task 9.3 -- the top-k pairs per method, then the overlap between methods."""
    lines: list[str] = []
    worst_classes: dict[str, set[int]] = {}
    top_pairs: dict[str, set[tuple[int, int]]] = {}

    for method, label in METHODS:
        _, _, result = scored[method]
        pairs = result.most_confused_pairs(k)
        top_pairs[method] = {(t, p) for t, p, _, _ in pairs}
        worst_classes[method] = set(np.argsort(result.per_class_f1)[:5].tolist())

        lines += [f"\n### {label}\n",
                  "| true class | predicted as | n | % of true class |", "|---|---|---|---|"]
        for true_id, pred_id, count, fraction in pairs:
            lines.append(
                f"| {true_id} {config.CLASS_NAMES[true_id][:32]} | "
                f"{pred_id} {config.CLASS_NAMES[pred_id][:32]} | {count} | {fraction * 100:.1f} % |"
            )

    # How much do the methods agree about which classes are hard?
    lines += ["\n### Do the methods fail on the same classes?\n",
              "Overlap of each method's five worst classes by per-class F1:\n",
              "| | " + " | ".join(label for _, label in METHODS) + " |",
              "|---" * (len(METHODS) + 1) + "|"]
    for method_a, label_a in METHODS:
        cells = [
            str(len(worst_classes[method_a] & worst_classes[method_b])) + "/5"
            for method_b, _ in METHODS
        ]
        lines.append(f"| **{label_a}** | " + " | ".join(cells) + " |")

    everywhere = set.intersection(*worst_classes.values())
    lines.append(
        f"\n**Classes in the worst five of ALL five methods: "
        f"{sorted(everywhere) if everywhere else 'NONE'}**"
        + (f" — {', '.join(config.CLASS_NAMES[c] for c in sorted(everywhere))}"
           if everywhere else
           " — so there is no class that is simply hard for everything; difficulty is "
           "representation-specific.")
    )

    shared_pairs = set.intersection(*top_pairs.values())
    lines += ["\n### Do they make the same MISTAKES, not just on the same classes?\n",
              "Overlap of each method's top-10 confused pairs:\n",
              "| | " + " | ".join(label for _, label in METHODS) + " |",
              "|---" * (len(METHODS) + 1) + "|"]
    for method_a, label_a in METHODS:
        cells = [
            str(len(top_pairs[method_a] & top_pairs[method_b]))
            for method_b, _ in METHODS
        ]
        lines.append(f"| **{label_a}** | " + " | ".join(cells) + " |")
    lines.append(
        f"\n**Pairs in the top-10 of ALL five: "
        f"{sorted(shared_pairs) if shared_pairs else 'NONE'}**"
    )
    return "\n".join(lines)


def figure_accuracy_by_size(scored: dict, test, y_true, out_dir: Path) -> Path:
    """Task 9.4 -- free: the same clean predictions, grouped by ROI height."""
    buckets = evaluation.size_buckets(test["roi_h"])
    order = [b for b in dict.fromkeys(buckets) if isinstance(b, str)]
    order.sort(key=lambda b: int(b.strip("[)").split(",")[0]))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    counts = None
    by_group_all: dict[str, dict] = {}
    for (method, label), (_, colour, _) in zip(
        METHODS,
        [(None, c, None) for c in ("#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#1a202c")],
        strict=True,
    ):
        _, predictions, _ = scored[method]
        by_group = evaluation.accuracy_by_group(y_true, predictions, buckets)
        by_group_all[method] = by_group
        values = [by_group[b][0] for b in order]
        counts = [by_group[b][1] for b in order]
        ax.plot(range(len(order)), values, "o-", color=colour, lw=2.0, ms=6, label=label)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{b}\nn={n:,}" for b, n in zip(order, counts, strict=True)],
                       fontsize=9)
    ax.set_xlabel("ROI height (px)", fontsize=10)
    ax.set_ylabel("accuracy, clean test", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    # Y truncated to 0.70: every method sits in [0.74, 1.00], so a zero baseline spends 74 %
    # of the panel on empty space and hides the differences the figure exists to show. The
    # truncation is stated in the caption rather than left for the reader to notice.
    ax.set_ylim(0.70, 1.01)

    smallest = {label: by_group_all[method][order[0]][0] for method, label in METHODS}
    best_small = max(smallest.items(), key=lambda kv: kv[1])
    worst_small = min(smallest.items(), key=lambda kv: kv[1])
    fig.suptitle(
        "Accuracy by sign size — the same clean-test predictions, grouped by ROI height\n"
        "Free: figure 9.2's predictions, partitioned differently. **Y axis starts at 0.70** — "
        "all five methods lie in [0.74, 1.00].\n"
        "The smallest bucket is not a tail: 47.6 % of training images are under 32 px "
        "(note 02), so this is the MODAL case, and train/test size distributions agree to "
        "within 0.1 pp.\n"
        f"At <32 px: best {best_small[0]} {best_small[1]:.3f}, worst {worst_small[0]} "
        f"{worst_small[1]:.3f}.",
        fontsize=10.2, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "accuracy_by_size.png"
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean-test analysis (9.2, 9.3, 9.4).")
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR / "results")
    args = parser.parse_args()

    config.set_seeds()
    test = data.load_annotations("test")
    y_true = test["class_id"].to_numpy()

    print("predicting on clean test (inference only, from the saved models):")
    scored = predict_all(test, y_true)

    print(f"\nwrote {figure_confusions(scored, args.out)}")
    print(f"wrote {figure_accuracy_by_size(scored, test, y_true, args.out)}")

    print("\n## Task 9.4 — accuracy by ROI-height bucket")
    buckets = evaluation.size_buckets(test["roi_h"])
    order = sorted({b for b in buckets if isinstance(b, str)},
                   key=lambda b: int(b.strip("[)").split(",")[0]))
    print("| method | " + " | ".join(order) + " | drop, largest → smallest |")
    print("|---" * (len(order) + 2) + "|")
    for method, label in METHODS:
        _, predictions, _ = scored[method]
        by_group = evaluation.accuracy_by_group(y_true, predictions, buckets)
        values = [by_group[b][0] for b in order]
        cells = " | ".join(f"{v:.3f}" for v in values)
        print(f"| {label} | {cells} | {(values[-1] - values[0]) * 100:+.1f} pp |")
    print("\n" + "=" * 78)
    print("## Task 9.3 — most-confused pairs")
    print(table_confused_pairs(scored))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
