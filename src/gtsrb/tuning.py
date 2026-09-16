"""Validation-set hyperparameter selection for the fixed classifier (tasks 4.2, 5.2, 6.3).

    best, records = tuning.tune_linear_svc(Z_train, y_train, Z_val, y_val)

Why this is shared code
-----------------------
Every representation ends in the same place: features into `LinearSVC`. Q4 (closed at task
4.1) decided that **`C` is tuned per method**, because PCA is the only representation whose
features are not internally normalised and a frozen `C` would hand each method a dial
calibrated for another's feature scale. "Fixed classifier" means the same estimator and the
same selection protocol -- which only holds if every method is tuned *the same way*, so the
protocol lives here rather than being re-typed in four scripts.

What it selects on
------------------
**Macro-F1, not accuracy.** GTSRB is imbalanced 10.7x (2250 images for class 2, 210 for
class 0), so accuracy is dominated by the large classes and a model can improve it by
neglecting small ones. Macro-F1 weights all 43 classes equally, which is also the metric
the report leads with -- selecting on one metric and reporting another would be indefensible.
Both are recorded for every grid point regardless.

Selection is on the **validation** split only. The test set is touched once, at task 8.1.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

from gtsrb import config, evaluation

#: Regularisation grid. Spans four orders of magnitude, which is enough to find the plateau
#: for any of the five representations without a finer search that validation noise on
#: 7830 images could not resolve anyway.
C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)

#: Whether to reweight classes by inverse frequency. Swept rather than assumed: the
#: PROJECT_TASKS gotcha list flags it as "consider", and it interacts with macro-F1 exactly
#: where the imbalance bites.
#:
#: **OPEN (Q8, raised 2026-09-17).** This comment used to read "whatever wins here must then
#: be applied to ALL five methods -- it is a policy about the imbalance, not a
#: per-representation nuisance parameter". **The code has never done that**, and the methods
#: now genuinely disagree: `pca_svm` selects `balanced` on `clahe_gray` but `None` on
#: `raw_gray`, while HOG and both CNN rows select `balanced`. The comment described an
#: intention, not the implementation, so it is corrected here to describe what actually
#: happens -- selection is per (method, preproc), exactly like `C`.
#:
#: Whether that is right is a live question, not a settled one. For: every method selects on
#: the same criterion (validation macro-F1), so `class_weight` is only a means to that
#: objective and the Q4 argument for per-method `C` applies unchanged. Against: unlike `C`, it
#: changes what the *fit* optimises, so two methods with different settings are not solving
#: quite the same problem. Note the effect is small and unstable -- at 4.2 `balanced` won by
#: 0.55 pp and *lost* at k=128 -- so no class-weighting claim should rest on it either way.
#: See `00-INDEX.md` Q8.
CLASS_WEIGHTS: tuple[str | None, ...] = (None, "balanced")


@dataclass(frozen=True)
class TuningRecord:
    """One grid point: what was tried, what it scored, what it cost."""

    params: dict
    accuracy: float
    macro_f1: float
    fit_seconds: float
    n_iter: int
    converged: bool

    def row(self) -> dict:
        return {**self.params, "accuracy": self.accuracy, "macro_f1": self.macro_f1,
                "fit_seconds": self.fit_seconds, "n_iter": self.n_iter,
                "converged": self.converged}


@dataclass
class TuningResult:
    """The selected configuration plus every grid point that was considered."""

    best: TuningRecord
    records: list[TuningRecord] = field(default_factory=list)

    def table(self) -> list[dict]:
        return [record.row() for record in self.records]

    def summary(self) -> str:
        lines = [
            (
                f"selected: {self.best.params}  macro_f1={self.best.macro_f1:.4f}  "
                f"accuracy={self.best.accuracy:.4f}"
            )
        ]
        unconverged = [r for r in self.records if not r.converged]
        if unconverged:
            lines.append(f"WARNING: {len(unconverged)} of {len(self.records)} grid points "
                         f"did not converge: {[r.params for r in unconverged]}")
        return "\n".join(lines)


def fit_and_score(
    features_train: np.ndarray,
    y_train: np.ndarray,
    features_val: np.ndarray,
    y_val: np.ndarray,
    C: float,
    class_weight: str | None = None,
    extra_params: dict | None = None,
    max_iter: int = 5000,
) -> TuningRecord:
    """Train one `LinearSVC` and score it on validation.

    `max_iter` is raised well above sklearn's default of 1000. Convergence is *recorded*
    rather than silenced: a grid point that hit the cap has not found the optimum for that
    `C`, and comparing it against converged points would compare the solver's patience
    instead of the regularisation.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        started = time.perf_counter()
        model = LinearSVC(
            C=C, class_weight=class_weight, max_iter=max_iter, random_state=config.SEED
        ).fit(features_train, y_train)
        elapsed = time.perf_counter() - started
        converged = not any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )

    result = evaluation.evaluate(y_val, model.predict(features_val))
    return TuningRecord(
        params={"C": C, "class_weight": class_weight, **(extra_params or {})},
        accuracy=result.accuracy,
        macro_f1=result.macro_f1,
        fit_seconds=elapsed,
        n_iter=int(np.max(model.n_iter_)),
        converged=converged,
    )


def select_best(records: list[TuningRecord]) -> TuningRecord:
    """The selection rule, in one place: highest macro-F1, ties to the smaller `C`.

    Ties break toward **more** regularisation (smaller `C`), which is the conventional
    choice and, more importantly, makes the selection a deterministic function of the
    records rather than of the order they happen to arrive in.
    """
    if not records:
        raise ValueError("no grid points to select from")
    return max(records, key=lambda r: (r.macro_f1, -r.params["C"]))


def tune_linear_svc(
    features_train: np.ndarray,
    y_train: np.ndarray,
    features_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: tuple[float, ...] = C_GRID,
    class_weights: tuple[str | None, ...] = CLASS_WEIGHTS,
    extra_params: dict | None = None,
    verbose: bool = False,
) -> TuningResult:
    """Sweep `C` x `class_weight` on validation and return the best by macro-F1.

    Selection is delegated to `select_best`, so the rule lives in exactly one place.
    """
    records: list[TuningRecord] = []
    for class_weight in class_weights:
        for C in c_grid:
            record = fit_and_score(
                features_train, y_train, features_val, y_val,
                C=C, class_weight=class_weight, extra_params=extra_params,
            )
            records.append(record)
            if verbose:
                flag = "" if record.converged else "  [did not converge]"
                # flush: these sweeps run for hours redirected to a log file, where
                # Python block-buffers stdout. Without it a sweep that is working fine
                # looks stalled for 25 minutes at a stretch (observed at task 5.2).
                print(f"    C={C:<7} class_weight={class_weight!s:<9} "
                      f"macro_f1={record.macro_f1:.4f} acc={record.accuracy:.4f} "
                      f"{record.fit_seconds:5.1f}s{flag}", flush=True)

    return TuningResult(best=select_best(records), records=records)


def best_from_sweep(csv_path, **filters) -> dict:
    """The winning grid point from a sweep CSV, as a plain dict.

    Training scripts read their hyperparameters from the sweep that chose them rather than
    repeating the numbers in source. A literal would drift silently the first time a sweep is
    re-run with a wider grid, and the model would then be trained at a configuration no
    recorded experiment selected.

    `filters` restricts before selecting, e.g. `preproc="clahe_gray"`.
    """
    import pandas as pd

    frame = pd.read_csv(csv_path)
    for column, value in filters.items():
        frame = frame[frame[column] == value]
    if frame.empty:
        raise ValueError(f"no rows in {csv_path} matching {filters}")

    best = frame.loc[frame["macro_f1"].idxmax()].to_dict()
    # pandas reads an absent class_weight as NaN; LinearSVC wants None.
    if pd.isna(best.get("class_weight")):
        best["class_weight"] = None
    return best
