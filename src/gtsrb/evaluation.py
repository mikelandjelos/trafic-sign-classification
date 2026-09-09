"""Task 1.4: the evaluation harness every method is measured with.

One function computes every metric, so all five representations are scored identically:

    result = evaluation.evaluate(y_true, y_pred)
    print(result.summary())

Why accuracy alone is not enough here
-------------------------------------
GTSRB class frequencies differ by 10.7x (see docs/report-material/02). A classifier that
neglects class 0 entirely -- 210 of 39 209 training images -- loses 0.5 % accuracy and
would still look strong. Macro-F1 weights all 43 classes equally and exposes exactly that
failure, so it is reported alongside accuracy everywhere, and Table 1 carries both.

The fixed-labels detail
-----------------------
Every metric is computed with `labels=range(43)` pinned explicitly. scikit-learn otherwise
infers the label set from whatever appears in `y_true`/`y_pred`, which means a heavily
degraded run where some class is never predicted would silently produce a 42x42 confusion
matrix and a 42-element per-class F1 vector. Those cannot be compared across the 105-run
grid, and misalignment between runs would not raise anything -- it would just quietly
shift every per-class number by one index.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from gtsrb import config

LABELS = list(range(config.N_CLASSES))


@dataclass(frozen=True)
class ClassificationResult:
    """Metrics for one (method, condition) evaluation."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: np.ndarray  # shape (43,)
    confusion: np.ndarray  # shape (43, 43), rows = true, cols = predicted
    support: np.ndarray  # shape (43,), true count per class
    n_samples: int

    def worst_classes(self, k: int = 5) -> list[tuple[int, str, float, int]]:
        """The k classes with the lowest F1: (class_id, name, f1, support)."""
        order = np.argsort(self.per_class_f1)[:k]
        return [
            (int(c), config.CLASS_NAMES[c], float(self.per_class_f1[c]), int(self.support[c]))
            for c in order
        ]

    def most_confused_pairs(self, k: int = 10) -> list[tuple[int, int, int, float]]:
        """The k most frequent (true, predicted) confusions, excluding the diagonal.

        Returns (true_id, pred_id, count, fraction_of_true_class). Feeds task 9.3.
        """
        errors = self.confusion.copy()
        np.fill_diagonal(errors, 0)
        flat = np.argsort(errors, axis=None)[::-1][:k]
        pairs = []
        for index in flat:
            true_id, pred_id = np.unravel_index(index, errors.shape)
            count = int(errors[true_id, pred_id])
            if count == 0:
                break
            support = int(self.support[true_id])
            pairs.append((int(true_id), int(pred_id), count, count / support if support else 0.0))
        return pairs

    def summary(self) -> str:
        lines = [
            f"samples       : {self.n_samples}",
            f"accuracy      : {self.accuracy:.4f}",
            f"macro-F1      : {self.macro_f1:.4f}",
            f"weighted-F1   : {self.weighted_f1:.4f}",
            "worst classes :",
        ]
        for class_id, name, f1, support in self.worst_classes():
            lines.append(f"  {class_id:>2} {name:<40} F1={f1:.3f}  n={support}")
        lines.append("most confused :")
        for true_id, pred_id, count, fraction in self.most_confused_pairs(5):
            lines.append(
                f"  {config.CLASS_NAMES[true_id]:<32} -> "
                f"{config.CLASS_NAMES[pred_id]:<32} {count:>4} ({fraction:.1%})"
            )
        return "\n".join(lines)


def evaluate(y_true, y_pred) -> ClassificationResult:
    """Score predictions against ground truth. All metrics over the fixed 43 labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("empty prediction array")

    return ClassificationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
        ),
        per_class_f1=f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0),
        confusion=confusion_matrix(y_true, y_pred, labels=LABELS),
        support=np.bincount(y_true, minlength=config.N_CLASSES),
        n_samples=int(y_true.size),
    )


def accuracy_by_group(y_true, y_pred, groups) -> dict[object, tuple[float, int]]:
    """Accuracy within each distinct value of `groups`. Returns {group: (accuracy, n)}.

    Used for the accuracy-vs-size figure (task 9.4), where `groups` comes from
    `size_buckets(df["roi_h"])`. Size-stratified results are free: they are the existing
    clean-test predictions, grouped differently, with no extra inference.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError("y_true, y_pred and groups must have the same length")

    correct = y_true == y_pred
    return {
        group: (float(correct[groups == group].mean()), int((groups == group).sum()))
        for group in sorted(np.unique(groups), key=str)
    }


def size_buckets(roi_heights) -> np.ndarray:
    """Bucket ROI heights for the accuracy-vs-size figure (task 9.4).

    Returns an array of bucket labels like "[32,48)". Buckets come from
    `config.SIZE_BUCKETS`, and the last one is open-ended.
    """
    edges = list(config.SIZE_BUCKETS)
    heights = np.asarray(roi_heights)
    names = [f"[{lo},{hi})" for lo, hi in itertools.pairwise(edges)]
    names.append(f"[{edges[-1]},inf)")
    index = np.digitize(heights, edges[1:], right=False)
    return np.array(names, dtype=object)[index]
