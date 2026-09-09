# 07 — The results table

**Feeds:** Methodology → reproducibility; the source of every table and figure in Results.
**Status:** complete (task 1.6)

Implemented as `gtsrb.results`; tested in `tests/test_results.py`.

---

## Schema

```
run_id, method, preproc, degradation, level, metric, value
```

One row per (run, method, condition, metric). **Long/tidy rather than wide**, because the
grid is 5 methods × 21 conditions × several metrics and new metrics keep being added as
the project proceeds. A wide table needs a schema change every time a metric is added; a
tidy one never does, and pandas pivots it into whatever shape a table or figure needs.

Conditions with no intensity (`clean`) use `level = 0.0`, so the column stays numeric and
sorts correctly rather than mixing floats with a sentinel string.

## `run_id`, and why not `git_commit`

The file is **append-only**, so rows from different executions coexist in it. That makes
the execution, not the code version, the right key:

- `git_commit` cannot distinguish two runs at the same commit — which is exactly when the
  column is needed ("which of these rows came from the run I need to drop?").
- A **dirty working tree** yields the same hash for different code, and that is the normal
  state mid-development.
- `platform.json` is per-run; without a run key, rows cannot be matched to the environment
  they were measured in.

Nothing is lost, because the commit is recorded *inside* `platform_<run_id>.json` together
with the CPU, achieved thread counts, load average and library versions. So
`run_id → platform_<run_id>.json → git_commit`, in one column instead of two.

The id is `<sortable timestamp>-<short commit>`, e.g. `20260909T202647-cf31ae3`: sortable
so the table can be filtered chronologically without a join, commit-suffixed so the code
version is legible at a glance, and degrading to `nogit` outside a repository rather than
failing.

## Two silent-failure guards

Both exist because the failure they prevent produces a *plausible-looking* number rather
than an error.

### 1. The pivot trap

`pandas.pivot_table` defaults to `aggfunc="mean"`. Pivoting the tidy table on
`degradation` alone therefore **silently averages noise σ=20 with σ=40** into a single
"noise" column — and the output looks entirely reasonable. This was caught in the
end-to-end smoke test of the harness, not by inspection:

```
BEFORE                          AFTER
degradation   clean   noise     degradation   clean   noise
                                level          0.0     20.0    40.0
hog_svm      0.9388  0.7948     hog_svm      0.9388  0.8602  0.7295
pca_svm      0.8669  0.7177     pca_svm      0.8669  0.7860  0.6494
```

The left-hand table would have flattened every robustness curve in figure 9.5 into a
single meaningless point per stressor. `results.pivot()` now pivots on
`(degradation, level)` by default and **raises** if any cell would aggregate more than one
row, naming the offending keys. Averaging across runs is refused on the same grounds.

### 2. NaN refusal at write time

`append_rows()` refuses to write NaN. Task 8.3 requires a complete, NaN-free table; catching
a NaN at the moment it is produced localises the bug to one method and one condition,
whereas discovering it during Day 4 analysis means re-running the grid to find out where it
came from.

`check_complete()` closes the loop by reporting exactly which
(method, degradation, level) cells are absent, so 8.3 is a check rather than an inspection.

---

## Consequence for the report

Every number in the report is traceable: a figure cites `results.csv`, a row cites its
`run_id`, and the `run_id` cites the machine, thread counts, library versions and commit it
was produced on. The claim "reproducible from a single script" is therefore verifiable
rather than aspirational.
