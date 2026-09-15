"""Collect the report-bound figures into `figures/report/`, which IS committed.

    poetry run python scripts/collect_report_figures.py          # copy what exists
    poetry run python scripts/collect_report_figures.py --check  # list status, copy nothing

Why this exists
---------------
Generated figures are gitignored to avoid binary churn -- they are all reproducible from
`scripts/`. But the figures that go into the final report must survive in the repository:
the report is written on Day 5, possibly on a different machine, and "just re-run the
scripts" is not a plan when a script depends on a 1 GB dataset that is also gitignored.

So `figures/report/` is the one committed figure directory, and MANIFEST below is the single
explicit list of what belongs in it. That makes "which figures are in the report" one
reviewable list rather than an accident of which files happened to get committed, and it
doubles as the deliverables checklist in PROJECT_TASKS section 8.

Adding a figure to the report is a one-line change here.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from gtsrb import config


@dataclass(frozen=True)
class ReportFigure:
    """One figure destined for the report."""

    source: str  #: path under figures/, as the generating script writes it
    name: str  #: filename inside figures/report/
    task: str  #: which task produces it
    why: str  #: one line -- what it is doing in the report
    deliverable: bool  #: True = named "Figure:" in PROJECT_TASKS; False = flagged as worth including


#: The report's figures. `deliverable=True` entries are the ones PROJECT_TASKS names
#: explicitly; `deliverable=False` are ones judged important enough to argue for.
MANIFEST: tuple[ReportFigure, ...] = (
    # --- named deliverables (PROJECT_TASKS section 8) ---
    ReportFigure(
        "diagrams/pipeline.png", "pipeline.png", "design",
        "Where degradations enter the pipeline — settles the question the whole design rests on.",
        deliverable=True),
    ReportFigure(
        "demo/degradation/degradation_contact_sheet.png", "degradation_contact_sheet.png", "3.5",
        "Every degradation x level on one sample. Shows what the stressors actually do.",
        deliverable=True),
    # --- flagged: not named as deliverables, argued for here ---
    ReportFigure(
        "demo/split/split_leakage.png", "split_leakage.png", "1.7",
        "STRONGLY RECOMMENDED. One track's 30 frames and where each split rule sends them. "
        "The project's central methodological claim, made self-evident in one image.",
        deliverable=False),
    ReportFigure(
        "split/split_leakage_measured.png", "split_leakage_measured.png", "4.5",
        "STRONGLY RECOMMENDED. The leakage claim as a number: +5.08 pp accuracy and "
        "+9.58 pp macro-F1 from a random per-image split. Pairs with split_leakage.png, "
        "which shows the cause.",
        deliverable=False),
    ReportFigure(
        "demo/split/split_similarity.png", "split_similarity.png", "1.7",
        "Same-track correlation 0.61 vs 0.17 across tracks of the same class. Converts "
        "'near-duplicate' from an assertion into a magnitude, independent of any model.",
        deliverable=False),
    ReportFigure(
        "pca/pca_component_sweep.png", "pca_component_sweep.png", "4.2",
        "Justifies k=256 and shows variance and accuracy disagreeing — the reason selection "
        "is on validation macro-F1 and not on a variance threshold.",
        deliverable=False),
    ReportFigure(
        "hog/hog_config_sweep.png", "hog_config_sweep.png", "5.2",
        "Justifies HOG's configuration, and shows the two findings the ablation rests on: "
        "HOG selects raw_gray where PCA selected clahe_gray, and cell size dominates "
        "orientation count.",
        deliverable=False),
    ReportFigure(
        "demo/pca/pca_reconstruction.png", "pca_reconstruction.png", "4.1",
        "What the subspace keeps at k=2..256. At low k every sign, triangles included, "
        "collapses toward a speed-limit disc: 'holistic, no notion of a part', visibly.",
        deliverable=False),
    ReportFigure(
        "demo/pca/pca_pc1_brightness.png", "pca_pc1_brightness.png", "4.1",
        "PC1 holds 52.3 % of the variance and correlates +0.9975 with brightness. The key "
        "fact for interpreting PCA's gamma behaviour in the discussion.",
        deliverable=False),
    ReportFigure(
        "cnn/cnn_architecture.png", "cnn_architecture.png", "7.1",
        "The CNN's two outputs on one trunk: softmax head (cnn_e2e) and the penultimate "
        "layer branching into the shared LinearSVC (cnn_feat_svm). Explains why one network "
        "produces two of the five rows — the central point of the experimental design.",
        deliverable=False),
    ReportFigure(
        "demo/preprocessing/preprocessing_configs.png", "preprocessing_configs.png", "2.2",
        "The three preprocessing configs side by side — needed to read the ablation (9.7).",
        deliverable=False),
    # --- pending: produced by tasks not yet done ---
    ReportFigure(
        "pca/pca_eigensigns.png", "pca_eigensigns.png", "4.4",
        "Top-16 eigenvectors as images. Ties directly to the Eigenfaces lecture.",
        deliverable=True),
    ReportFigure(
        "hog/hog_visualization.png", "hog_visualization.png", "5.4",
        "HOG visualisation, one sample per super-category.", deliverable=True),
    ReportFigure(
        "results/confusion_best_worst.png", "confusion_best_worst.png", "9.2",
        "Confusion matrices for the best and worst method.", deliverable=True),
    ReportFigure(
        "results/accuracy_by_size.png", "accuracy_by_size.png", "9.4",
        "Accuracy vs ROI height, one line per method.", deliverable=True),
    ReportFigure(
        "results/robustness_curves.png", "robustness_curves.png", "9.5",
        "Robustness curves, 3 panels, relative to each method's own clean baseline. "
        "The headline figure of the study.", deliverable=True),
)


def report_dir() -> Path:
    return config.FIGURES_DIR / "report"


def collect(check_only: bool = False) -> int:
    destination = report_dir()
    if not check_only:
        destination.mkdir(parents=True, exist_ok=True)

    missing = 0
    for figure in MANIFEST:
        source = config.FIGURES_DIR / figure.source
        kind = "deliverable" if figure.deliverable else "flagged    "
        if not source.exists():
            missing += 1
            print(f"  PENDING     {kind}  {figure.name:<34} (task {figure.task})")
            continue
        if check_only:
            print(f"  present     {kind}  {figure.name:<34} (task {figure.task})")
            continue
        shutil.copy2(source, destination / figure.name)
        print(f"  copied      {kind}  {figure.name:<34} (task {figure.task})")

    print(f"\n{len(MANIFEST) - missing} of {len(MANIFEST)} report figures available; "
          f"{missing} pending on unfinished tasks.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect report figures (committed).")
    parser.add_argument("--check", action="store_true", help="report status, copy nothing")
    args = parser.parse_args()
    return collect(check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
