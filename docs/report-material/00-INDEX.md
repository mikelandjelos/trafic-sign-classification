# Report material — index

Notes written **as the work happens**, to be consumed when writing the final report
(`docs/report/`, Serbian). Nothing here is reconstructed after the fact: every decision,
number, and figure gets its rationale recorded at the time it is made.

Distinct from:

- `docs/PROJECT_TASKS.md` — the task tracker (what to do, what is done)
- `docs/proposition/` — the project proposal (`predlog_projekta.tex`). **Never edited
  without an explicit request** — see CLAUDE.md.
- `docs/report/` — the final Serbian writeup (does not exist yet)

## Conventions

- One file per report section, numbered in the order they will likely appear.
- Working language English; the report is translated at the end. Terms that have an
  established Serbian rendering are noted inline as `English (srpski)`.
- Every claim that will appear in the report cites either a results.csv row, a figure
  file, or a source in `docs/proposition/references.bib`.
- **Surprises get written down even when they are inconvenient.** A prediction that
  failed is more valuable to the discussion section than one that held.

## Notes

| Note | Feeds report section | Status |
|---|---|---|
| [01-environment-and-setup.md](01-environment-and-setup.md) | Methodology → reproducibility; Appendix | complete (task 0.1) |
| [02-dataset-structure.md](02-dataset-structure.md) | Data; Methodology → train/val protocol | complete (task 0.2) |
| [03-reproducibility.md](03-reproducibility.md) | Methodology → protocol; Limitations | complete (task 0.3) |
| [04-splitting-protocol.md](04-splitting-protocol.md) | Methodology → experimental protocol | complete (tasks 1.2, 1.3, 1.7) |
| [05-evaluation-metrics.md](05-evaluation-metrics.md) | Methodology → evaluation; Table 1; figs 9.2–9.4 | complete (task 1.4) |
| [06-cost-measurement.md](06-cost-measurement.md) | Methodology → cost; Table 1; **Limitations** | complete (task 1.5) |
| [07-results-table.md](07-results-table.md) | Methodology → reproducibility; source of all Results | complete (task 1.6) |
| [08-preprocessing.md](08-preprocessing.md) | Methodology → preprocessing; ablation table 9.7 | complete (tasks 2.1–2.3) |
| [09-jitter-and-datasets.md](09-jitter-and-datasets.md) | Scope; **Limitations**; Future work | decided (extension §11) |
| [10-degradations.md](10-degradations.md) | Methodology → degradations; fig. 9.5; contact sheet | complete (tasks 3.1–3.3, 3.5) |
| [11-pca.md](11-pca.md) | Methodology → representations; Table 1; figs 4.4, 9.5, 9.7 | complete (task 4.1) |

Pipeline diagram: `docs/diagrams/pipeline.puml` → `figures/diagrams/pipeline.png` (source tracked, render ignored).

<!-- Add a row per note as it is created. Keep status honest: in progress / complete / stale. -->

## Open questions to resolve before the report

Running list; each gets closed with a decision and a rationale, not dropped.

| # | Question | Raised at | Status |
|---|---|---|---|
| Q5 | **Preproc/representation pairing is unenforced.** A representation fitted on `clahe_gray` cannot detect being handed `raw_gray` images — same 2304 dims, so no error, silently wrong results. Hazard for the 8.2 ablation, which loops over all three configs. | task 4.1 | **CLOSED** — guard in the **task 8 runner**, not in the representations: it constructs `(representation, images)` as one paired unit, with a test asserting the pairing. Catches the mistake at the one place it can occur, without plumbing the preproc name through five `transform` implementations. See [11](11-pca.md) §6.3 |
| Q4 | **Is one `C` across all five methods part of "fixed classifier", or a confound?** PCA is the only representation whose features are not internally normalised (HOG block-normalises, BoVW L2-normalises, the CNN has batch norm), so a single `LinearSVC(C=…)` meets it on a different footing. Measured at 4.1: feature σ spread is 29× within PCA, though standardising changes val accuracy by only 0.09 pp. | task 4.1 | **CLOSED** — **tune `C` per method** on validation, and record the chosen `C` per method in `results.csv`. "Fixed classifier" means the same estimator and the same training protocol, not the same nuisance hyperparameter: holding `C` fixed would hand each representation a dial calibrated for someone else's feature scale, and any difference that produced would be an artifact, not a property of φ. Stated explicitly in the report's protocol section. See [11](11-pca.md) §6.2 |
| Q3 | ~~Gap in the predictions table~~ → **rewritten**, not extended: the framing changed twice (jitter left the MVP; gamma was never predicted), so only 2 of 4 original rows still described MVP conditions. | task 3.5 | **CLOSED** — table approved and recorded in `PROJECT_TASKS.md` §2 and `predictions.md`, timestamped **2026-09-13**. P.1 was pulled forward to *before task 4.1*, i.e. before any model existed — the original "before task 8" would have meant predicting after seeing PCA, HOG, BoVW and CNN results |
| Q2 | **Opportunity, not a problem.** The claim that a random split inflates validation accuracy is currently argued rather than measured. Training one method twice — track-disjoint split vs. random per-image split — turns it into a number. ~15 min once task 4.3 exists. | task 1.2 | **approved** — scheduled as task **4.5**; see [04](04-splitting-protocol.md) |
| Q1 | GTSRB crops carry only a ~17 % margin around the sign, so the planned "±40 % jitter, re-crop, no padding" is not achievable. | setup, before task 0.1 | **CLOSED** — measured: +40 % impossible for 72 % of images (28,166 clip); median headroom 1.30×. Jitter removed from the MVP entirely (training *and* core grid) and moved to full-frame datasets as extension §11. The measurement is itself a reported finding. See [09](09-jitter-and-datasets.md) |
