"""Task 9.1: Table 1 -- the headline comparison, with its measurement conditions attached.

    poetry run python scripts/table1.py            # markdown, for the notes
    poetry run python scripts/table1.py --latex    # LaTeX, for the report

Reads `results/results.csv` and the per-run `platform_<run_id>.json` files. Nothing is
recomputed here, so the table cannot disagree with the recorded results.

Why the cost columns carry a conditions column
----------------------------------------------
The accuracy columns are trustworthy: one test set, one pass, 80 cells, no NaNs (task 8.3).
**The cost columns are not comparable in the same way**, and printing them without saying so
would be the least defensible thing in the report.

Each method's timing was recorded by its own `train_*.py`, on a different day, at a different
system load -- **1-minute load average ranged from 1.27 to 13.12** across the five runs on a
16-thread machine. On top of that, PROJECT_TASKS section 10 records that the CPU power profile
alone moved CNN epoch time by ~20 %, and that `scaling_governor` reads `powersave` in both
profiles on amd-pstate, so the change is invisible in the logs.

So this script prints the load average beside every duration. The honest options for the
report are then (a) print the table with this column and the caveats, or (b) re-measure all
five in one pass on an idle machine and print clean numbers. It does not decide between them;
it makes the choice visible.

Three further caveats the table itself carries
-----------------------------------------------
1. **`model_size_mb` is not like-for-like.** 96 % of PCA's 2.35 MB is the learned *basis*;
   HOG has no fitted stage at all, so its 0.58 MB is entirely SVM coefficients; and
   `cnn_feat_svm`'s 0.04 MB is the SVM head ALONE -- it cannot run without the 3.38 MB network
   it shares with `cnn_e2e`, so its honest figure is **3.42 MB**.
2. **Batched and single-image inference differ by up to 49x**, and the ratio is
   method-specific (PCA 52x, BoVW 2.0x, HOG 1.3x, CNN 1.4x). A camera presents one frame at a
   time, so the single-image column is the one a latency claim may use.
3. **`train_seconds` measures the whole method**, vocabulary/basis fit included -- not just
   the classifier. Timing only the SVM would flatter PCA and BoVW against HOG, which has no
   fitted stage. **The exception is `cnn_feat_svm`**, whose 1.1 s is the SVM head alone and
   would make it look like the cheapest method in the study; it cannot run without the
   2,596 s network, so its honest figure is ~2,597 s. Printed per row below the table.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from gtsrb import config

METHODS: tuple[str, ...] = (
    "pca_svm", "hog_svm", "bovw_svm", "cnn_feat_svm", "cnn_e2e",
)

LABELS = {
    "pca_svm": "PCA + LinearSVC",
    "hog_svm": "HOG + LinearSVC",
    "bovw_svm": "BoVW + LinearSVC",
    "cnn_feat_svm": "CNN features + LinearSVC",
    "cnn_e2e": "CNN end-to-end",
}

#: What each `train_seconds` actually covers. `cnn_feat_svm` is the trap: 1.1 s is the SVM
#: head ALONE, which would make it the cheapest method in the study to train. It cannot exist
#: without the 2,596 s network it reads, so its honest figure is ~2,597 s -- the same
#: not-like-for-like problem as its 0.04 MB model size, and easier to miss.
TRAIN_NOTE = {
    "pca_svm": "basis fit + SVM",
    "hog_svm": "descriptors + SVM; HOG itself fits nothing",
    "bovw_svm": "vocabulary + encoding + SVM",
    "cnn_feat_svm": "SVM head ONLY -- add the 2,596 s network it reads => ~2,597 s",
    "cnn_e2e": "24 epochs, CPU-only, 7 torch threads",
}

#: What dominates each model file -- the reason `model_size_mb` is not comparable.
SIZE_NOTE = {
    "pca_svm": "96 % learned basis",
    "hog_svm": "all SVM coefficients; HOG fits nothing",
    "bovw_svm": "vocabulary + SVM",
    "cnn_feat_svm": "SVM head only; +3.38 MB shared network = 3.42 MB",
    "cnn_e2e": "the network",
}


def load_average(run_id: str) -> float | None:
    path = config.RESULTS_DIR / f"platform_{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("load_average", [None])[0]


def latest_per_cell(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent run for each (method, degradation, level, metric).

    `results.csv` is append-only by design, so a re-run does not overwrite -- it adds. Filtering
    by a hard-coded `run_id` prefix is the obvious shortcut and it is a trap: when `pca_svm` was
    refitted under the Q8 policy and its grid row re-run, the new rows fell OUTSIDE the prefix
    this function used to carry, and the table silently kept printing the superseded numbers
    while every other document had been corrected. Sorting by `run_id` (which is timestamp-
    prefixed) and taking the last is stable under any number of re-runs.
    """
    return (frame.sort_values("run_id")
                 .groupby(["method", "degradation", "level", "metric"], as_index=False)
                 .last())


def collect() -> pd.DataFrame:
    frame = pd.read_csv(config.RESULTS_CSV)
    # Test rows are the ones without the `val_` prefix; `clean` at NO_LEVEL is the headline.
    test = latest_per_cell(frame[frame.metric.isin(("accuracy", "macro_f1"))])

    rows = []
    for method in METHODS:
        row: dict = {"method": method, "label": LABELS[method]}

        clean = test[(test.method == method) & (test.degradation == "clean")]
        for metric in ("accuracy", "macro_f1"):
            hit = clean[clean.metric == metric]
            row[metric] = float(hit.value.iloc[0]) if not hit.empty else None

        # Cost metrics come from each method's own training run, not from the grid.
        own = frame[(frame.method == method) & (frame.preproc == "raw_gray")]
        for metric in ("train_seconds", "inference_ms_per_image",
                       "inference_single_ms_per_image", "model_size_mb", "feature_dim"):
            hit = own[own.metric == metric]
            row[metric] = float(hit.value.iloc[-1]) if not hit.empty else None

        train_rows = own[own.metric == "train_seconds"]
        run_id = train_rows.run_id.iloc[-1] if not train_rows.empty else None
        row["run_id"] = run_id
        row["load1"] = load_average(run_id) if run_id else None
        rows.append(row)
    return pd.DataFrame(rows).set_index("method").loc[list(METHODS)]


def markdown(df: pd.DataFrame) -> str:
    out = [
        ("| method | accuracy | macro-F1 | feature dim | train (s) | infer batched (ms) | "
         "infer single (ms) | model (MB) |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        out.append(
            f"| {r.label} | {r.accuracy:.4f} | **{r.macro_f1:.4f}** | {int(r.feature_dim)} | "
            f"{r.train_seconds:.1f} | {r.inference_ms_per_image:.4f} | "
            f"{r.inference_single_ms_per_image:.4f} | {r.model_size_mb:.2f} |"
        )
    out += ["", "**Measurement conditions for the cost columns** — these are NOT one pass:", "",
            "| method | run | 1-min load average when timed |", "|---|---|---|"]
    for _, r in df.iterrows():
        load = f"{r.load1:.2f}" if r.load1 is not None else "—"
        out.append(f"| {r.label} | `{r.run_id}` | **{load}** |")
    out += ["", "**What `train (s)` covers:**", ""]
    for method, r in df.iterrows():
        out.append(f"- *{r.label}* — {TRAIN_NOTE[method]}")
    out += ["", "**What dominates each model file:**", ""]
    for method, r in df.iterrows():
        out.append(f"- *{r.label}* — {SIZE_NOTE[method]}")
    return "\n".join(out)


def latex(df: pd.DataFrame) -> str:
    out = [
        r"\begin{tabular}{lrrrrrrr}", r"\hline",
        (r"Metoda & Tačnost & makro-F1 & Dim. & Trening (s) & Batch (ms) & 1 slika (ms) & "
         r"Model (MB) \\"), r"\hline",
    ]
    for _, r in df.iterrows():
        out.append(
            f"{r.label} & {r.accuracy:.4f} & {r.macro_f1:.4f} & {int(r.feature_dim)} & "
            f"{r.train_seconds:.1f} & {r.inference_ms_per_image:.4f} & "
            f"{r.inference_single_ms_per_image:.4f} & {r.model_size_mb:.2f} \\\\"
        )
    out += [r"\hline", r"\end{tabular}"]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Table 1 (task 9.1).")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    df = collect()
    print(latex(df) if args.latex else markdown(df))

    loads = [r.load1 for _, r in df.iterrows() if r.load1 is not None]
    if loads:
        print(f"\nload average across the five timing runs: "
              f"{min(loads):.2f} .. {max(loads):.2f} on a 16-thread machine "
              f"({max(loads) / min(loads):.1f}x)")
        print("=> the cost columns are indicative only; re-measure in one pass on an idle "
              "machine before quoting them as a comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
