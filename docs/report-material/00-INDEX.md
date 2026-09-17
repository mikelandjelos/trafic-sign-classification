# Report material — index

Notes written **as the work happens**, to be consumed when writing the final report
(`docs/report/`, Serbian). Nothing here is reconstructed after the fact: every decision,
number, and figure gets its rationale recorded at the time it is made.

Distinct from:

- `docs/PROJECT_TASKS.md` — the task tracker (what to do, what is done)
- `docs/proposition/` — the project proposal (`predlog_projekta.tex`). **Never edited
  without an explicit request** — see CLAUDE.md.
- `docs/report/` — the final Serbian writeup. `main.tex` and `references.bib` are
  committed but **still empty**; scaffolding only, written on Day 5.

## Conventions

- One file per report section, numbered in the order they will likely appear.
- Working language English; the report is translated at the end. Terms that have an
  established Serbian rendering are noted inline as `English (srpski)`.
- Every claim that will appear in the report cites either a results.csv row, a figure
  file, or a source in `docs/proposition/references.bib`.
- **Surprises get written down even when they are inconvenient.** A prediction that
  failed is more valuable to the discussion section than one that held. *(Scored at 8.1:
  6 of 7 held, and the one miss — note 15 §4.1 — turned out to explain a second
  unpredicted result as well.)*

## Notes

| Note | Feeds report section | Status |
|---|---|---|
| [01-environment-and-setup.md](01-environment-and-setup.md) | Methodology → reproducibility; Appendix | complete (task 0.1) |
| [02-dataset-structure.md](02-dataset-structure.md) | Data; Methodology → train/val protocol | complete (task 0.2) |
| [03-reproducibility.md](03-reproducibility.md) | Methodology → protocol; Limitations | complete (tasks 0.3, 4.1) |
| [04-splitting-protocol.md](04-splitting-protocol.md) | Methodology → experimental protocol; **Results** (the leakage number) | complete (tasks 1.2, 1.3, 1.7, 4.5) |
| [05-evaluation-metrics.md](05-evaluation-metrics.md) | Methodology → evaluation **and selection**; Table 1; figs 9.2–9.4 | complete (tasks 1.4, 4.2) |
| [06-cost-measurement.md](06-cost-measurement.md) | Methodology → cost; Table 1; **Limitations** | complete (tasks 1.5, 4.3) |
| [07-results-table.md](07-results-table.md) | Methodology → reproducibility; source of all Results | complete (task 1.6) |
| [08-preprocessing.md](08-preprocessing.md) | Methodology → preprocessing; ablation table 9.7; **Limitations** | complete (tasks 2.1–2.3) + addendum 2026-09-16 (preproc fixed at `raw_gray`) |
| [09-jitter-and-datasets.md](09-jitter-and-datasets.md) | Scope; **Limitations**; Future work | decided (extension §11) |
| [10-degradations.md](10-degradations.md) | Methodology → degradations; fig. 9.5; contact sheet | complete (tasks 3.1–3.3, 3.5) |
| [11-pca.md](11-pca.md) | Methodology → representations; **classifier protocol**; Table 1; figs 4.2, 4.4, 9.5, 9.7 | complete (4.1–4.5) + addenda 2026-09-16/17 — **§ADD3 noise result + §ADD4 whitening ablation (mechanism confirmed, −3.3 pp, but second-order)** |
| [12-hog.md](12-hog.md) | Methodology → representations; Table 1; fig 5.4; noise & gamma panels (9.5) | complete (5.1–5.5) + addenda 2026-09-16/17 — **§ADD2: the sharpest failure in the study, 14.8 % retained at σ=40** |
| [13-cnn.md](13-cnn.md) | Methodology → representations; Table 1; **Limitations** (no GPU, untuned LR, patience-limited stop) | **complete (7.1–7.6)** + addenda 2026-09-16/17 — **§ADD3: learned ≠ robust; the two CNN rows are indistinguishable under stress** |
| [14-bovw.md](14-bovw.md) | Methodology → representations; Table 1; **the layout measurement (§9), in the DISCUSSION**; **preprocessing-vs-normalisation pattern (§11.1, feeds 9.7)**; **a falsified blur prediction (§13.2, feeds 9.6)**; §11 premise | **complete (6.1–6.6)**; reported config in §11; 6.5b reverted (SPM = diagnostic); **§14: most gamma-robust method, and the one falsified prediction** |

| [15-results.md](15-results.md) | **Results (the whole section)**; Table 1 (9.1); confusions (9.2/9.3); size (9.4); robustness curves (9.5); **predictions vs outcomes (9.6)**; preprocessing (9.7); **the five findings (9.8)** | **complete (8.1, 8.3, 9.1–9.8)** — 80/80 cells, 0 NaNs. §8 a Q8 violation caught late; §10 predictions scored **8/10**; §11 the preprocessing axis; §12 the five findings |

Pipeline diagram: `docs/diagrams/pipeline.puml` → `figures/diagrams/pipeline.png`; the
render is copied into `figures/report/` (committed) by
`scripts/collect_report_figures.py`. Updated at task 4.2 to show validation driving
hyperparameter selection — it previously implied val fed the fit.

<!-- Add a row per note as it is created. Keep status honest: in progress / complete / stale. -->

## Open questions to resolve before the report

Running list; each gets closed with a decision and a rationale, not dropped.

| # | Question | Raised at | Status |
|---|---|---|---|
| Q8 | **Is `class_weight` a per-method nuisance parameter like `C`, or one policy for the whole study?** `tuning.py` asserted the latter ("whatever wins must be applied to ALL five methods") but **the code has never implemented it**, and since the `raw_gray` switch the methods genuinely disagree: `pca_svm` selects `None`, while `hog_svm` and both CNN rows select `balanced`. The comment described an intention, not the implementation. | task 6.5, **2026-09-17** | **CLOSED — `class_weight="balanced"` is FIXED for every method.** The deciding distinction: `C` is a regularisation *strength*, so two methods with different `C` still minimise the same loss, whereas `class_weight` changes **what the loss is** — per-method settings mean the methods were not trained toward the same objective, which is what "the classifier is held fixed" exists to rule out. `"balanced"` because macro-F1 weights all 43 classes equally, so a class-balanced objective matches the metric the study selects and reports on; it is also what 3 of 4 methods chose freely. `best_from_sweep` now drops policy-violating rows — **it fired on its first real use**, where the BoVW sweep's unconstrained best was a `None` row at 0.8808 against the compliant 0.8801. **That 0.07 pp gap is itself the reason no class-weighting effect may be claimed anywhere.** Cost: one PCA refit. See `tuning.py` `CLASS_WEIGHT`, [14](14-bovw.md) §11.4 |
| Q7 | **Is per-method preprocessing selection a confound in the headline comparison?** Decided at 4.2 that each method selects its own config, on the grounds that borrowing another's costs ~1 pp. But that makes every method-vs-method gap a mixture of representation *and* preprocessing. | task 6.5, **2026-09-16** | **CLOSED** — **preprocessing is held FIXED at `raw_gray`** for all five configurations. The proposal (§3) already required it: *"uz **isti pretprocesing** … razlike u rezultatima mogu pripisati isključivo samoj reprezentaciji"*, so the per-method protocol was a divergence and this restores it. `raw_gray` is the neutral, unenhanced input and the config HOG selected independently. Costs PCA 2.6 pp and `cnn_e2e` 0.16 pp of absolute accuracy, which is the correct price for an unconfounded comparison. **The per-method sweeps are kept and reported at 9.7** — they are what *measures* the preprocessing effect; 8.2 is demoted to optional. See `PROJECT_TASKS.md` §1 |
| Q6 | **Should Spatial Pyramid Matching enter the comparison?** Excluded at 6.5 (before any SPM number existed) because it moves BoVW onto HOG's position on the layout axis. Then measured at **+10.4 pp macro-F1**, landing within 0.1 pp of HOG. | task 6.5, **2026-09-16** | **CLOSED — added 2026-09-16 as a sixth row, then REVERTED 2026-09-17: it is reported as a diagnostic, not a method.** The science was right and the engineering was not. At the k=1000 the sweep selected, L=2 is 21,000 dims / ~6 GB resident, which does not fit this machine; three training runs died. **The +10.4 pp was measured at k=500 (10,500 dims), and the later k=1000 decision doubled the pyramid's width — that connection was missed until three runs had failed.** Lesson recorded: a method's feasibility must be re-checked when a parameter it depends on changes. Nothing measured is lost — the layout result, the L=1 intermediate, and the descriptor-quality/layout substitution all stay in the discussion, and `BoVWSpatialPyramid` stays in the codebase with its tests so the number remains reproducible. The comparison is back to the proposal's five configurations, so the §4.3 divergence is also withdrawn. See [14](14-bovw.md) §9.4 |
| Q5 | **Preproc/representation pairing is unenforced.** A representation fitted on `clahe_gray` cannot detect being handed `raw_gray` images — same 2304 dims, so no error, silently wrong results. Hazard for the 8.2 ablation, which loops over all three configs. | task 4.1 | **CLOSED** — guard in the **task 8 runner**, not in the representations: it constructs `(representation, images)` as one paired unit, with a test asserting the pairing. Catches the mistake at the one place it can occur, without plumbing the preproc name through five `transform` implementations. See [11](11-pca.md) §6.3 |
| Q4 | **Is one `C` across all five methods part of "fixed classifier", or a confound?** PCA is the only representation whose features are not internally normalised (HOG block-normalises, BoVW L2-normalises, the CNN has batch norm), so a single `LinearSVC(C=…)` meets it on a different footing. Measured at 4.1: feature σ spread is 29× within PCA, though standardising changes val accuracy by only 0.09 pp. | task 4.1 | **CLOSED** — **tune `C` per method** on validation, and record the chosen `C` per method in `results.csv`. "Fixed classifier" means the same estimator and the same training protocol, not the same nuisance hyperparameter: holding `C` fixed would hand each representation a dial calibrated for someone else's feature scale, and any difference that produced would be an artifact, not a property of φ. Stated explicitly in the report's protocol section. See [11](11-pca.md) §6.2 |
| Q3 | ~~Gap in the predictions table~~ → **rewritten**, not extended: the framing changed twice (jitter left the MVP; gamma was never predicted), so only 2 of 4 original rows still described MVP conditions. | task 3.5 | **CLOSED** — table approved and recorded in `PROJECT_TASKS.md` §2 and `predictions.md`, timestamped **2026-09-13**. P.1 was pulled forward to *before task 4.1*, i.e. before any model existed — the original "before task 8" would have meant predicting after seeing PCA, HOG, BoVW and CNN results |
| Q2 | **Opportunity, not a problem.** The claim that a random split inflates validation accuracy is currently argued rather than measured. Training one method twice — track-disjoint split vs. random per-image split — turns it into a number. ~15 min once task 4.3 exists. | task 1.2 | **CLOSED** — measured at task 4.5: a random per-image split inflates val accuracy by **+5.08 pp** and macro-F1 by **+9.58 pp** (1,305 of 1,307 tracks land on both sides; 100 % of val images keep a sibling in train). Macro-F1 inflates ~2× as much, because leakage flatters the rare classes most. See [04](04-splitting-protocol.md) |
| Q1 | GTSRB crops carry only a ~17 % margin around the sign, so the planned "±40 % jitter, re-crop, no padding" is not achievable. | setup, before task 0.1 | **CLOSED** — measured: +40 % impossible for 72 % of images (28,166 clip); median headroom 1.30×. Jitter removed from the MVP entirely (training *and* core grid) and moved to full-frame datasets as extension §11. The measurement is itself a reported finding. See [09](09-jitter-and-datasets.md) |
