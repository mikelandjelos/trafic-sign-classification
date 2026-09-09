"""Tests for the tidy results table (task 1.6).

results.csv is a deliverable: the report's tables and figures are all derived from it, so
a malformed or silently-truncated table costs far more than it looks. These tests pin the
schema, the append semantics, and the NaN refusal that task 8.3 depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from gtsrb import evaluation, results, timing


@pytest.fixture
def csv_path(tmp_path):
    return tmp_path / "results.csv"


def test_run_id_is_sortable_and_carries_the_commit():
    run_id = results.new_run_id()
    stamp, _, commit = run_id.partition("-")
    assert len(stamp) == 15 and stamp[8] == "T"  # 20260909T201215
    assert commit  # short commit, or "nogit" outside a repository


def test_run_ids_sort_chronologically():
    earlier = f"20260101T000000-{timing.git_commit()}"
    later = f"20260909T201215-{timing.git_commit()}"
    assert sorted([later, earlier]) == [earlier, later]


def test_append_writes_header_once(csv_path):
    for value in (0.1, 0.2, 0.3):
        results.append_result("r1", "pca_svm", "raw_gray", "clean", 0, "accuracy", value,
                              path=csv_path)
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 4, "expected 1 header + 3 rows"
    assert lines[0] == ",".join(results.COLUMNS)


def test_column_order_is_the_schema(csv_path):
    results.append_result("r1", "hog_svm", "clahe_gray", "noise", 20, "accuracy", 0.9,
                          path=csv_path)
    frame = results.load(csv_path)
    assert list(frame.columns) == results.COLUMNS


def test_appends_accumulate_across_runs(csv_path):
    results.append_result("r1", "pca_svm", "raw_gray", "clean", 0, "accuracy", 0.9,
                          path=csv_path)
    results.append_result("r2", "pca_svm", "raw_gray", "clean", 0, "accuracy", 0.91,
                          path=csv_path)
    frame = results.load(csv_path)
    assert len(frame) == 2
    assert set(frame["run_id"]) == {"r1", "r2"}


def test_refuses_nan_values(csv_path):
    """Task 8.3 requires a NaN-free table; catching it at write time localises the bug."""
    with pytest.raises(ValueError, match="NaN"):
        results.append_rows(
            [{
                "run_id": "r1", "method": "m", "preproc": "p", "degradation": "clean",
                "level": 0.0, "metric": "accuracy", "value": float("nan"),
            }],
            path=csv_path,
        )
    assert not csv_path.exists()


def test_rejects_rows_missing_columns(csv_path):
    with pytest.raises(ValueError, match="missing required columns"):
        results.append_rows([{"run_id": "r1", "method": "m"}], path=csv_path)


def test_empty_rows_is_a_noop(csv_path):
    results.append_rows([], path=csv_path)
    assert not csv_path.exists()


def test_rows_from_evaluation(csv_path):
    y = np.arange(43)
    result = evaluation.evaluate(y, y)
    rows = results.rows_from_evaluation("r1", "pca_svm", "raw_gray", "noise", 20, result)
    metrics = {row["metric"] for row in rows}
    assert metrics == {"accuracy", "macro_f1", "weighted_f1", "n_samples"}
    assert all(row["level"] == 20.0 for row in rows)
    assert all(row["run_id"] == "r1" for row in rows)


def test_rows_from_evaluation_per_class_emits_43(csv_path):
    y = np.arange(43)
    result = evaluation.evaluate(y, y)
    rows = results.rows_from_evaluation("r1", "m", "p", "clean", 0, result, per_class=True)
    per_class = [row for row in rows if row["metric"].startswith("f1_class_")]
    assert len(per_class) == 43


def test_rows_from_timing_includes_ms_per_image():
    result = timing.TimingResult(
        label="inference", median_s=2.0, min_s=2.0, iqr_s=0.0, n_items=1000,
        repeats=5, discarded_first_s=None,
    )
    rows = results.rows_from_timing("r1", "pca_svm", "raw_gray", result)
    metrics = {row["metric"]: row["value"] for row in rows}
    assert metrics["inference_seconds"] == pytest.approx(2.0)
    assert metrics["inference_ms_per_image"] == pytest.approx(2.0)


def test_rows_from_timing_omits_ms_per_image_for_training():
    result = timing.TimingResult(
        label="train", median_s=12.0, min_s=12.0, iqr_s=0.0, n_items=None,
        repeats=1, discarded_first_s=None,
    )
    metrics = {row["metric"] for row in results.rows_from_timing("r1", "m", "p", result)}
    assert metrics == {"train_seconds"}


def test_pivot_shapes_the_table(csv_path):
    for method in ("pca_svm", "hog_svm"):
        for degradation, level, value in (("clean", 0, 0.9), ("noise", 20, 0.7)):
            results.append_result("r1", method, "raw_gray", degradation, level, "accuracy",
                                  value, path=csv_path)
    table = results.pivot(results.load(csv_path))
    assert list(table.index) == ["hog_svm", "pca_svm"]
    assert set(table.columns) == {("clean", 0.0), ("noise", 20.0)}


def test_pivot_refuses_to_average_levels_together(csv_path):
    """Pivoting on degradation alone would silently mean sigma=20 with sigma=40."""
    for level, value in ((20, 0.8), (40, 0.6)):
        results.append_result("r1", "pca_svm", "raw_gray", "noise", level, "accuracy",
                              value, path=csv_path)
    with pytest.raises(ValueError, match="would be averaged together"):
        results.pivot(results.load(csv_path), columns="degradation")


def test_pivot_refuses_to_average_across_runs(csv_path):
    for run_id in ("r1", "r2"):
        results.append_result(run_id, "pca_svm", "raw_gray", "clean", 0, "accuracy", 0.9,
                              path=csv_path)
    with pytest.raises(ValueError, match="single run_id"):
        results.pivot(results.load(csv_path))


def test_pivot_rejects_unknown_metric(csv_path):
    results.append_result("r1", "m", "p", "clean", 0, "accuracy", 0.9, path=csv_path)
    with pytest.raises(ValueError, match="no rows with metric"):
        results.pivot(results.load(csv_path), metric="nonexistent")


def test_check_complete_reports_missing_cells(csv_path):
    results.append_result("r1", "pca_svm", "raw_gray", "clean", 0, "accuracy", 0.9,
                          path=csv_path)
    problems = results.check_complete(
        results.load(csv_path),
        methods=["pca_svm", "hog_svm"],
        degradations={"clean": [0], "noise": [20]},
    )
    assert any("hog_svm" in p for p in problems)
    assert any("noise" in p for p in problems)
    assert not any("pca_svm / clean" in p for p in problems)


def test_check_complete_passes_on_a_full_grid(csv_path):
    for method in ("pca_svm", "hog_svm"):
        for degradation, level in (("clean", 0), ("noise", 20)):
            results.append_result("r1", method, "raw_gray", degradation, level, "accuracy",
                                  0.9, path=csv_path)
    problems = results.check_complete(
        results.load(csv_path),
        methods=["pca_svm", "hog_svm"],
        degradations={"clean": [0], "noise": [20]},
    )
    assert problems == []


def test_load_missing_file_returns_empty_frame(tmp_path):
    frame = results.load(tmp_path / "nope.csv")
    assert len(frame) == 0
    assert list(frame.columns) == results.COLUMNS
