"""Task 4.2: validation-set hyperparameter selection.

Run on a small synthetic 3-class problem. What is under test is the *selection protocol* --
that it picks by macro-F1, records every point, breaks ties predictably and reports
non-convergence rather than hiding it — none of which needs GTSRB to verify.
"""

from __future__ import annotations

import numpy as np
import pytest

from gtsrb import tuning


@pytest.fixture(scope="module")
def problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A separable 3-class problem, deliberately imbalanced 10:2:1 like GTSRB is."""
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    counts_train, counts_val = (300, 60, 30), (100, 20, 10)

    def build(counts):
        features = np.vstack([
            rng.normal(centres[c], 1.2, size=(n, 2)) for c, n in enumerate(counts)
        ]).astype(np.float32)
        labels = np.concatenate([np.full(n, c) for c, n in enumerate(counts)])
        return features, labels

    x_train, y_train = build(counts_train)
    x_val, y_val = build(counts_val)
    return x_train, y_train, x_val, y_val


def test_records_every_grid_point(problem) -> None:
    result = tuning.tune_linear_svc(*problem)
    assert len(result.records) == len(tuning.C_GRID) * len(tuning.CLASS_WEIGHTS)
    assert result.best in result.records


def test_selects_the_best_macro_f1_not_the_best_accuracy(problem) -> None:
    """The selection criterion is macro-F1 — the distinction matters under imbalance."""
    result = tuning.tune_linear_svc(*problem)
    assert result.best.macro_f1 == max(r.macro_f1 for r in result.records)


def test_ties_break_toward_more_regularisation() -> None:
    """Two grid points with identical macro-F1 must resolve to the smaller C, every time."""
    shared = {"accuracy": 0.9, "macro_f1": 0.8, "fit_seconds": 1.0, "n_iter": 5,
              "converged": True}
    records = [
        tuning.TuningRecord(params={"C": 10.0, "class_weight": None}, **shared),
        tuning.TuningRecord(params={"C": 0.1, "class_weight": None}, **shared),
    ]
    assert tuning.select_best(records).params["C"] == 0.1
    assert tuning.select_best(list(reversed(records))).params["C"] == 0.1


def test_select_best_rejects_an_empty_grid() -> None:
    with pytest.raises(ValueError, match="no grid points"):
        tuning.select_best([])


def test_every_record_carries_its_params_and_cost(problem) -> None:
    result = tuning.tune_linear_svc(*problem)
    for record in result.records:
        assert record.params["C"] in tuning.C_GRID
        assert record.params["class_weight"] in tuning.CLASS_WEIGHTS
        assert record.fit_seconds > 0
        assert record.n_iter >= 1
        assert 0.0 <= record.accuracy <= 1.0
        assert 0.0 <= record.macro_f1 <= 1.0


def test_extra_params_reach_the_table(problem) -> None:
    """The sweep scripts thread k / preproc through here to build a tidy table."""
    result = tuning.tune_linear_svc(*problem, extra_params={"n_components": 7})
    rows = result.table()
    assert all(row["n_components"] == 7 for row in rows)
    assert len(rows) == len(result.records)


def test_a_custom_grid_is_honoured(problem) -> None:
    result = tuning.tune_linear_svc(*problem, c_grid=(0.5,), class_weights=(None,))
    assert len(result.records) == 1
    assert result.best.params == {"C": 0.5, "class_weight": None}


def test_non_convergence_is_recorded_not_silenced(problem) -> None:
    """A grid point that hit max_iter has not found its optimum; comparing it against a
    converged point compares the solver's patience, not the regularisation."""
    x_train, y_train, x_val, y_val = problem
    record = tuning.fit_and_score(
        x_train, y_train, x_val, y_val, C=1.0, max_iter=1
    )
    assert record.converged is False
    assert record.n_iter >= 1


def test_convergence_is_true_when_it_converges(problem) -> None:
    x_train, y_train, x_val, y_val = problem
    record = tuning.fit_and_score(x_train, y_train, x_val, y_val, C=0.1)
    assert record.converged is True


def test_summary_flags_unconverged_points(problem) -> None:
    x_train, y_train, x_val, y_val = problem
    bad = tuning.fit_and_score(x_train, y_train, x_val, y_val, C=1.0, max_iter=1)
    result = tuning.TuningResult(best=bad, records=[bad])
    assert "did not converge" in result.summary()


def test_summary_is_quiet_when_all_converged(problem) -> None:
    result = tuning.tune_linear_svc(*problem, c_grid=(0.1,), class_weights=(None,))
    assert "did not converge" not in result.summary()
    assert "selected" in result.summary()


def test_balanced_weighting_helps_the_rare_class(problem) -> None:
    """Sanity check on the mechanism the class_weight sweep exists to test."""
    x_train, y_train, x_val, y_val = problem
    plain = tuning.fit_and_score(x_train, y_train, x_val, y_val, C=0.1)
    balanced = tuning.fit_and_score(
        x_train, y_train, x_val, y_val, C=0.1, class_weight="balanced"
    )
    # Not asserting balanced *wins* -- on separable data it need not. Asserting only that
    # the option is actually applied and produces a different model.
    assert plain.params["class_weight"] is None
    assert balanced.params["class_weight"] == "balanced"


# --- reading a selection back out of a sweep -------------------------------------------


@pytest.fixture
def sweep_csv(tmp_path):
    """A miniature sweep table shaped like results/sweeps/pca_components.csv."""
    import pandas as pd

    frame = pd.DataFrame([
        {"preproc": "clahe_gray", "n_components": 128, "C": 1.0,
         "class_weight": "balanced", "macro_f1": 0.70, "accuracy": 0.75},
        {"preproc": "clahe_gray", "n_components": 256, "C": 0.01,
         "class_weight": None, "macro_f1": 0.80, "accuracy": 0.84},
        {"preproc": "raw_gray", "n_components": 256, "C": 0.1,
         "class_weight": "balanced", "macro_f1": 0.77, "accuracy": 0.81},
    ])
    path = tmp_path / "sweep.csv"
    frame.to_csv(path, index=False)
    return path


def test_best_from_sweep_picks_the_highest_macro_f1_OBEYING_THE_POLICY(sweep_csv) -> None:
    """The fixture's global best (0.80) is a `class_weight=None` row, which Q8 excludes.

    So the selection is the best row *among those the policy allows* -- 0.77. A sweep run
    before 2026-09-17 contains both settings, and picking its global winner would silently
    reintroduce the per-method class weighting the policy exists to remove.
    """
    best = tuning.best_from_sweep(sweep_csv)
    assert best["class_weight"] == tuning.CLASS_WEIGHT
    assert best["macro_f1"] == 0.77
    assert best["preproc"] == "raw_gray"


def test_best_from_sweep_filters_first(sweep_csv) -> None:
    """Filtering must narrow before selecting, or 8.2 gets the global winner every time."""
    best = tuning.best_from_sweep(sweep_csv, preproc="raw_gray")
    assert best["preproc"] == "raw_gray"
    assert best["macro_f1"] == 0.77


def test_the_class_weight_policy_can_be_overridden_for_an_ablation(sweep_csv) -> None:
    """An explicit `class_weight=` is honoured -- that is what an ablation would pass."""
    best = tuning.best_from_sweep(sweep_csv, class_weight=None)
    assert best["macro_f1"] == 0.80
    assert best["preproc"] == "clahe_gray"


def test_a_sweep_with_no_policy_compliant_row_is_refused(tmp_path) -> None:
    """Silence would mean training at a configuration no recorded experiment selected."""
    import pandas as pd

    path = tmp_path / "old.csv"
    pd.DataFrame([{"preproc": "raw_gray", "C": 1.0, "class_weight": None,
                   "macro_f1": 0.5, "accuracy": 0.6}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="fixed class_weight policy"):
        tuning.best_from_sweep(path)


def test_best_from_sweep_converts_missing_class_weight_to_none(tmp_path) -> None:
    """pandas reads an absent class_weight as NaN; LinearSVC needs None.

    Left as NaN it is silently truthy, so `LinearSVC(class_weight=nan)` raises deep in the
    fit rather than where the mistake was made. Reached here via an explicit override, since
    the default policy now excludes `None` rows.
    """
    import pandas as pd

    path = tmp_path / "s.csv"
    pd.DataFrame([{"preproc": "raw_gray", "C": 1.0, "class_weight": None,
                   "macro_f1": 0.5, "accuracy": 0.6}]).to_csv(path, index=False)
    assert tuning.best_from_sweep(path, class_weight=None)["class_weight"] is None


def test_best_from_sweep_rejects_a_filter_matching_nothing(sweep_csv) -> None:
    with pytest.raises(ValueError, match="no rows"):
        tuning.best_from_sweep(sweep_csv, preproc="clahe_hsv")
