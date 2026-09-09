"""Task 1.6: the tidy results table every run appends to.

    run = results.start_run()      # mints 20260909T201215-4f31ae3, writes platform_<id>.json
    results.append_result(run, "pca_svm", "raw_gray", "clean", 0, "accuracy", 0.93)
    results.append_rows(
        results.rows_from_evaluation(run, "pca_svm", "raw_gray", "noise", 20, eval_result)
    )

Schema
------
    run_id, method, preproc, degradation, level, metric, value

One row per (run, method, condition, metric). Long/tidy rather than wide: the grid is
5 methods x 21 conditions x several metrics, and new metrics get added as the project
proceeds. A wide table would need a schema change every time; a tidy one never does, and
pandas pivots it into any table or figure the report needs in one call.

Why `run_id` and not `git_commit`
---------------------------------
The file is append-only, so rows from different executions coexist. `git_commit` cannot
distinguish two runs at the same commit -- and a dirty working tree gives the same hash for
different code, which is the normal state mid-development. `run_id` identifies the
*execution*; the commit is recorded inside `platform_<run_id>.json` along with the CPU,
thread counts and library versions, so nothing is lost and each row points at the exact
environment it was measured in.

The id is `<sortable timestamp>-<short commit>`: sortable so results.csv can be filtered
chronologically without a join, and commit-suffixed so the code version is legible without
opening the JSON.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from gtsrb import config, timing

COLUMNS = ["run_id", "method", "preproc", "degradation", "level", "metric", "value"]

#: Conditions with no intensity (clean data) use this level, so the column stays numeric.
NO_LEVEL = 0.0


def new_run_id() -> str:
    """A unique, sortable identifier for one execution. Falls back to `nogit` cleanly."""
    stamp = time.strftime("%Y%m%dT%H%M%S")
    commit = timing.git_commit() or "nogit"
    return f"{stamp}-{commit}"


def platform_path(run_id: str) -> Path:
    """Where this run's environment snapshot lives, beside results.csv."""
    return config.RESULTS_DIR / f"platform_{run_id}.json"


def start_run(run_id: str | None = None) -> str:
    """Begin a run: mint an id and record the environment it will be measured in."""
    run_id = run_id or new_run_id()
    timing.save_platform_info(platform_path(run_id))
    return run_id


def append_rows(rows: list[dict], path: Path | None = None) -> Path:
    """Append tidy rows to results.csv, writing the header only when creating the file."""
    path = path or config.RESULTS_CSV
    if not rows:
        return path

    frame = pd.DataFrame(rows)
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"rows missing required columns: {sorted(missing)}")
    if frame["value"].isna().any():
        raise ValueError("refusing to write NaN values -- task 8.3 requires a clean table")

    frame = frame[COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path


def append_result(
    run_id: str,
    method: str,
    preproc: str,
    degradation: str,
    level: float,
    metric: str,
    value: float,
    path: Path | None = None,
) -> Path:
    """Append a single metric value."""
    return append_rows(
        [
            {
                "run_id": run_id,
                "method": method,
                "preproc": preproc,
                "degradation": degradation,
                "level": float(level),
                "metric": metric,
                "value": float(value),
            }
        ],
        path=path,
    )


def rows_from_evaluation(
    run_id: str,
    method: str,
    preproc: str,
    degradation: str,
    level: float,
    result,
    per_class: bool = False,
) -> list[dict]:
    """Tidy rows for a `ClassificationResult`.

    `per_class=True` additionally emits one `f1_class_<i>` row for each of the 43 classes.
    Off by default: it multiplies the table by ~43 and is only needed for the clean
    condition, where the per-class analysis actually happens.
    """
    base = {
        "run_id": run_id,
        "method": method,
        "preproc": preproc,
        "degradation": degradation,
        "level": float(level),
    }
    rows = [
        {**base, "metric": "accuracy", "value": float(result.accuracy)},
        {**base, "metric": "macro_f1", "value": float(result.macro_f1)},
        {**base, "metric": "weighted_f1", "value": float(result.weighted_f1)},
        {**base, "metric": "n_samples", "value": float(result.n_samples)},
    ]
    if per_class:
        rows += [
            {**base, "metric": f"f1_class_{i}", "value": float(f1)}
            for i, f1 in enumerate(result.per_class_f1)
        ]
    return rows


def rows_from_timing(
    run_id: str, method: str, preproc: str, result, degradation: str = "clean"
) -> list[dict]:
    """Tidy rows for a `TimingResult` -- training seconds or inference ms/img."""
    base = {
        "run_id": run_id,
        "method": method,
        "preproc": preproc,
        "degradation": degradation,
        "level": NO_LEVEL,
    }
    rows = [{**base, "metric": f"{result.label}_seconds", "value": float(result.median_s)}]
    if result.ms_per_item is not None:
        rows.append(
            {**base, "metric": f"{result.label}_ms_per_image", "value": float(result.ms_per_item)}
        )
    return rows


def load(path: Path | None = None) -> pd.DataFrame:
    """Read results.csv."""
    path = path or config.RESULTS_CSV
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def pivot(
    frame: pd.DataFrame,
    metric: str = "accuracy",
    index: str = "method",
    columns: str | list[str] = ("degradation", "level"),
) -> pd.DataFrame:
    """Pivot the tidy table into a report-shaped one (Table 1, robustness curves).

    Raises if any cell would aggregate more than one row, rather than silently averaging.

    This matters: `pivot_table` defaults to `aggfunc="mean"`, so pivoting on `degradation`
    alone quietly averages noise sigma=20 with sigma=40 into a single "noise" column, and
    the result looks entirely plausible. Degradation and level therefore default to being
    pivoted *together*, and any remaining collision is an error the caller must resolve --
    usually by filtering to one run_id.
    """
    columns = [columns] if isinstance(columns, str) else list(columns)
    subset = frame[frame["metric"] == metric]
    if subset.empty:
        raise ValueError(f"no rows with metric={metric!r}; available: {sorted(frame['metric'].unique())}")

    duplicated = subset.duplicated(subset=[index, *columns])
    if duplicated.any():
        offenders = subset[subset.duplicated(subset=[index, *columns], keep=False)]
        raise ValueError(
            f"{int(duplicated.sum())} rows would be averaged together by this pivot "
            f"(index={index}, columns={columns}). Add 'level' to columns, or filter to a "
            f"single run_id. Offending keys: "
            f"{offenders[[index, *columns, 'run_id']].head(4).to_dict('records')}"
        )

    return subset.pivot_table(index=index, columns=columns, values="value", aggfunc="mean")


def check_complete(
    frame: pd.DataFrame, methods: list[str], degradations: dict[str, list[float]],
    metric: str = "accuracy",
) -> list[str]:
    """Task 8.3: report which (method, degradation, level) cells are missing or NaN."""
    problems = []
    if frame["value"].isna().any():
        problems.append(f"{int(frame['value'].isna().sum())} NaN values present")

    have = set(
        map(tuple, frame[frame["metric"] == metric][["method", "degradation", "level"]].values)
    )
    for method in methods:
        for degradation, levels in degradations.items():
            for level in levels:
                if (method, degradation, float(level)) not in have:
                    problems.append(f"missing: {method} / {degradation} / {level}")
    return problems
