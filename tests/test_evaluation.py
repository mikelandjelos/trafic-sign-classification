"""Tests for the evaluation harness (task 1.4).

The property worth protecting here is shape stability. Across the 105-run grid, some
heavily degraded runs will never predict certain classes. If metric arrays changed shape
or index meaning between runs, nothing would raise -- the per-class numbers would just
quietly refer to different classes in different rows of results.csv.
"""

from __future__ import annotations

import numpy as np
import pytest

from gtsrb import config, evaluation


def test_shapes_are_fixed_even_when_classes_are_absent():
    """Only 3 of 43 classes appear; metrics must still span all 43."""
    result = evaluation.evaluate([1, 1, 2, 2, 3], [1, 2, 2, 2, 3])
    assert result.confusion.shape == (config.N_CLASSES, config.N_CLASSES)
    assert result.per_class_f1.shape == (config.N_CLASSES,)
    assert result.support.shape == (config.N_CLASSES,)


def test_per_class_index_matches_class_id():
    """per_class_f1[i] must be the F1 of class i, not of the i-th class that appeared."""
    result = evaluation.evaluate([5, 5, 40], [5, 5, 40])
    assert result.per_class_f1[5] == 1.0
    assert result.per_class_f1[40] == 1.0
    assert result.per_class_f1[0] == 0.0  # absent, not present-and-perfect


def test_perfect_predictions():
    y = np.arange(config.N_CLASSES)
    result = evaluation.evaluate(y, y)
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert np.array_equal(result.confusion, np.eye(config.N_CLASSES, dtype=int))


def test_macro_f1_punishes_neglecting_a_rare_class():
    """The reason macro-F1 is reported: accuracy alone hides a neglected small class."""
    y_true = np.array([0] * 2 + [1] * 98)
    y_pred = np.array([1] * 100)  # never predicts the rare class
    result = evaluation.evaluate(y_true, y_pred)
    assert result.accuracy == pytest.approx(0.98)
    assert result.macro_f1 < 0.03


def test_support_counts_true_labels():
    result = evaluation.evaluate([1, 1, 1, 2], [1, 1, 2, 2])
    assert result.support[1] == 3
    assert result.support[2] == 1
    assert result.support.sum() == 4


def test_most_confused_pairs_excludes_diagonal_and_sorts_descending():
    y_true = np.array([1] * 10 + [2] * 10)
    y_pred = np.array([2] * 6 + [1] * 4 + [1] * 3 + [2] * 7)
    pairs = evaluation.evaluate(y_true, y_pred).most_confused_pairs(k=5)
    assert all(true_id != pred_id for true_id, pred_id, _, _ in pairs)
    counts = [count for _, _, count, _ in pairs]
    assert counts == sorted(counts, reverse=True)
    assert pairs[0] == (1, 2, 6, pytest.approx(0.6))


def test_most_confused_pairs_empty_when_perfect():
    y = np.arange(config.N_CLASSES)
    assert evaluation.evaluate(y, y).most_confused_pairs() == []


def test_size_buckets_boundaries():
    """Buckets are half-open [lo, hi); a value equal to an edge belongs to the upper one."""
    buckets = evaluation.size_buckets([15, 31, 32, 47, 48, 71, 72, 200])
    assert list(buckets) == [
        "[0,32)",
        "[0,32)",
        "[32,48)",
        "[32,48)",
        "[48,72)",
        "[48,72)",
        "[72,inf)",
        "[72,inf)",
    ]


def test_accuracy_by_group():
    y_true = np.array([1, 1, 2, 2])
    y_pred = np.array([1, 9, 2, 2])
    groups = np.array(["a", "a", "b", "b"])
    result = evaluation.accuracy_by_group(y_true, y_pred, groups)
    assert result["a"] == (0.5, 2)
    assert result["b"] == (1.0, 2)


def test_rejects_mismatched_and_empty_input():
    with pytest.raises(ValueError):
        evaluation.evaluate([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        evaluation.evaluate([], [])
    with pytest.raises(ValueError):
        evaluation.accuracy_by_group([1, 2], [1, 2], ["a"])
