# 15 — Results: the degradation grid (tasks 8.1, 8.3)

*Feeds: **Results** (the whole section); Table 1 (9.1); the robustness curves (9.5); and
**9.6, predictions vs outcomes — which is the core of the discussion**.*

Produced by `scripts/evaluate_grid.py`, run `20260917T022546-bc880c9` (four methods) and
`20260917T023936-bc880c9` (BoVW). **Status: complete.** 5 methods × 16 conditions = **80
cells**, 535 rows, **0 NaNs, 0 missing cells** (`results.check_complete`, task 8.3).

**This is the first and only time the test set was touched.** Everything before this note is
validation. Rows here carry the plain `accuracy` / `macro_f1` names; the `val_*` rows that
drove selection are never mixed with them.

---

## 1. Clean test performance, and a check the split passed

| method | accuracy | macro-F1 | val macro-F1 | val → test gap |
|---|---|---|---|---|
| `cnn_e2e` | 0.9785 | **0.9684** | 0.9856 | 1.72 pp |
| `cnn_feat_svm` | **0.9804** | 0.9651 | 0.9774 | 1.22 pp |
| `hog_svm` | 0.9238 | 0.9062 | 0.9175 | 1.14 pp |
| `bovw_svm` | 0.9141 | 0.8596 | 0.8840 | 2.45 pp |
| `pca_svm` | 0.7986 | 0.7525 | 0.7713 | 1.88 pp |

**Every method scores lower on test than on validation, by 1.1–2.5 pp.** That is the right
direction and a modest magnitude, and it is worth stating explicitly as evidence the
track-disjoint protocol did its job: had the split leaked, validation would have been
*inflated* relative to test by something closer to the +9.58 pp measured at task 4.5. The
selection was honest.

BoVW has the largest gap (2.45 pp), PCA the second (1.88 pp) — consistent with both being the
methods whose hyperparameters were selected on the most grid points.

**The two metrics already disagree, in two places.** `cnn_feat_svm` has the best *accuracy*
(0.9804) while `cnn_e2e` has the best *macro-F1* (0.9684) — they swap depending on the column.
And HOG leads BoVW by only **1.0 pp on accuracy** but **4.7 pp on macro-F1**: the gap between
them is almost entirely in the rare classes, which is the 12.5× test imbalance showing through.
This is why the report leads with macro-F1 and prints both.

**One ordering flipped between validation and test, and it is worth reporting.** On validation
BoVW had the *higher accuracy* of the two (0.9330 vs HOG's 0.9298); on test HOG leads
(0.9238 vs 0.9141). A 0.3 pp validation lead did not survive to a 1.0 pp test deficit — a
concrete, in-project demonstration that differences of a few tenths of a point on one split
are not real, and the reason §7 insists on ordinal claims. Their macro-F1 ordering, where the
gap is 4.7 pp, is stable across both splits.

---

## 2. The grid, as retention against each method's own clean baseline

This is what figure 9.5 plots, and the reason is in the numbers below: absolute scores are
dominated by where each method *starts*, which is not the question. Retention isolates **rate
of degradation**.

macro-F1 as % of that method's own clean macro-F1:

| condition | PCA | HOG | BoVW | CNN-feat | CNN-e2e |
|---|---|---|---|---|---|
| **noise σ=5** | 99.2 | 80.6 | 98.5 | 97.6 | 97.1 |
| **noise σ=10** | 97.6 | 61.9 | 93.5 | 89.2 | 89.2 |
| **noise σ=20** | 90.9 | 37.0 | 82.8 | 71.8 | 71.4 |
| **noise σ=40** | **77.5** | **14.8** | 65.6 | 44.3 | 43.8 |
| **blur k=5** | 96.4 | 89.8 | 92.8 | 97.1 | 97.2 |
| **blur k=9** | 77.4 | 53.7 | 61.4 | 71.6 | 73.8 |
| **blur k=15** | **43.1** | 23.9 | **20.7** | 36.8 | 39.6 |
| **gamma 0.4** | **79.1** | 99.9 | 98.9 | 99.0 | 98.4 |
| **gamma 1.5** | 90.8 | 99.0 | 99.6 | 98.8 | 98.5 |
| **gamma 2.5** | **72.3** | 89.2 | **95.3** | 86.2 | 86.1 |

---

## 3. The headline result: the ranking inverts

`predictions.md`, locked 2026-09-13 before any model existed:

> **No single representation wins everywhere, and the ranking inverts between stressors.**
> The sharpest falsifiable case: **PCA is predicted best under noise and worst under gamma,
> while HOG is predicted worst under noise and best under gamma.** If that crossing appears,
> the project's central claim is demonstrated in one figure.

**It appears, in both directions, at the extremes of both stressors.**

| ranking (best → worst) | Spearman vs clean |
|---|---|
| **clean** — `cnn_e2e` > `cnn_feat` > `hog` > `bovw` > `pca` | — |
| **noise σ=40** — `pca` > `bovw` > `cnn_feat` > `cnn_e2e` > `hog` | **−0.70** |
| blur k=15 — `cnn_e2e` > `cnn_feat` > `pca` > `hog` > `bovw` | +0.70 |
| gamma 2.5 — `cnn_e2e` > `cnn_feat` > `bovw` > `hog` > `pca` | +0.90 |

**Under heavy noise the ordering is close to reversed.** PCA moves last → **first**, HOG third
→ **last**. And PCA's win is not a retention artifact: in absolute macro-F1 it scores
**0.5834** against both CNNs' ~0.425 and HOG's 0.1341. A holistic linear subspace outperforms
a convolutional network by 37 %, relative, on the same pixels.

**This is the single most important number in the project**, because it is the case where the
conventional answer ("use the CNN") is measurably wrong for a stated condition — which is
precisely the decision the proposal's §1 says this study exists to inform.

The blur and gamma columns are *positively* correlated with clean performance, so the
inversion is **stressor-specific, not general**. That is a finding in its own right and should
be said plainly: the premise "ranking depends on the stressor" is confirmed, but it is
confirmed by **one** stressor out of three.

---

## 4. Predictions vs outcomes (feeds 9.6)

**Six of seven MVP rows held.** Scored against the 2026-09-13 table only; the SPM addendum was
never evaluated because that method was dropped (note 14 §9.4).

| prediction | outcome | |
|---|---|---|
| Noise: **PCA most robust** | 77.5 % retained, best | ✅ |
| Noise: **HOG least robust** | 14.8 % retained, worst by 4.4× | ✅ |
| Noise: BoVW "between PCA and HOG" | 65.6 %, 2nd of five | ✅ |
| Blur: **PCA most robust** | 43.1 % retained, best | ✅ |
| Blur: **BoVW least robust** | 20.7 % retained, worst | ✅ |
| Blur ⚠: "HOG should beat BoVW" (revised) | 23.9 % vs 20.7 % | ✅ |
| Gamma: **PCA least robust** | 72.3 % retained, worst | ✅ |
| Gamma: **HOG most robust** | **BoVW 95.3 % > HOG 89.2 %** | ❌ |

### 4.1 The miss, and it is the most informative row in the table

The gamma prediction said, in full:

> HOG's block-wise L2 normalisation cancels contrast scaling almost entirely, and **SIFT's
> descriptor normalisation gives BoVW similar (slightly weaker) protection.**

**Measured, BoVW's protection is not slightly weaker — it is the strongest in the study.** At
γ=2.5 BoVW retains **95.3 %** against HOG's 89.2 %, and in absolute *accuracy* BoVW is the
outright best method in the condition (0.8873, ahead of HOG's 0.8409 **and** `cnn_e2e`'s
0.8113).

**Why the reasoning was wrong.** The prediction treated the two normalisations as the same
kind of thing at different strengths. They are not:

- **HOG's L2-Hys** normalises over a **2×2 block of cells** — a 12×12 px region at the
  selected configuration — so it divides out contrast that is roughly constant over that
  region.
- **SIFT** normalises **per 128-dim patch descriptor**, and additionally **clips every
  component at 0.2 and renormalises** (Lowe's recipe, note 14 §3). That clip is a
  *non-linearity*, and it suppresses exactly the large gradient components a gamma curve
  inflates.

So SIFT's normalisation is both more local and non-linear, and a monotone point transform is
close to the best case for it. A linear normaliser cancels a linear contrast change exactly
(HOG's measured 0.0000 at task 5.5) but only partly cancels gamma (0.0339) — whereas the clip
degrades gracefully across both.

### 4.2 And it connects to the other thing BoVW did unexpectedly

Task 6.6 found that blur does **not** collapse BoVW's vocabulary — distinct words fall only
18 %, between-class similarity is flat — and attributed that to the same L2 normalisation:
blur cuts gradient *magnitude* while descriptor *direction* still varies, and k-means assigns
by direction.

**One property of the descriptor, two unpredicted effects, in the same direction.** The report
should present these together rather than as two separate surprises: SIFT's normalise-clip-
renormalise makes BoVW robust to anything that scales gradient magnitude without moving
edges — which is gamma exactly, and blur partially.

**It also explains the blur row that DID hold.** BoVW is still worst under blur (20.7 %)
despite its descriptors surviving — because blur *does* move and merge edges, and what BoVW
then lacks is any layout to fall back on. That is the prediction's *second* mechanism, and
§9's +10.4 pp layout measurement is the direct evidence for it. So the blur result is right
for the second reason, not the first.

### 4.3 A caveat that must accompany the gamma row

HOG's "win" in the prediction was about robustness, and on **retention** HOG does beat the
CNNs (89.2 % vs 86.1 %). But in **absolute** macro-F1 at γ=2.5 the CNNs score higher (0.8338
vs 0.8084), and on **accuracy** BoVW beats everything. Three metrics, three different winners.

The report must name the quantity every time it makes a robustness claim, or a reader checking
a different column will reasonably conclude the opposite.

---

## 5. Mechanisms, per stressor

**Noise (adds high-frequency energy).** The ordering tracks how much high-frequency detail a
method relies on, exactly as predicted. PCA keeps 256 of 2304 dimensions and most noise energy
falls outside that subspace; HOG *differentiates*, a high-pass operation that amplifies noise
directly — the per-cell measurement at task 5.5 showed low-contrast background cells reaching
**6× their clean gradient energy** at σ=40, so their orientation votes go random. The CNNs sit
between, which is itself notable: **learned features are not automatically robust features.**

**Blur (removes high-frequency energy).** The opposite operation, and the ordering is nearly
the opposite too — the CNNs and PCA lead, the two local hand-designed methods trail. PCA's
lead is the predicted mechanism: blur preserves the low-frequency structure the leading
eigenvectors encode. BoVW is worst because it has no layout to fall back on once local
structure merges (§4.2).

**Gamma (monotone point transform, geometry untouched).** Everything except PCA is nearly
immune below γ=2.5, which is the expected consequence of internal normalisation. PCA is the
only method with none, and it is also **asymmetric**: γ=0.4 costs it 20.9 pp of retention
while γ=1.5 costs 9.2 pp. That fits the PC1-is-brightness measurement (52.3 % of variance,
r=+0.9975 with mean intensity, note 11): darkening compresses toward a region of the subspace
the training data populated, brightening pushes the projection off the training distribution.

---

## 6. Secondary findings

**The two CNN rows are near-identical under every stressor** — within 0.5 pp at almost every
cell, and they swap places repeatedly. The objective mismatch (note 13 §1.1) costs ~0.8 pp of
clean macro-F1 and **nothing at all in robustness**. So it is a property of the *head*, not of
the representation, which is the cleanest possible version of that caveat.

**BoVW is the second most noise-robust method** (65.6 %), ahead of both CNNs. Combined with
its gamma win, BoVW is the most robust *hand-designed* method overall on two of three
stressors — a very different picture from its clean ranking (4th of 5), and the strongest
single argument in the report for not choosing a representation on clean accuracy.

**Nothing survives blur k=15.** The best method retains 43.1 % and the worst 20.7 %. At that
strength the question stops being "which representation" and becomes "this input is not
classifiable" — worth saying, because a robustness curve that is compared only *between*
methods can obscure that all of them have failed.

---

## 7. Caveats on these numbers

- **One seed, no repeats.** Differences of a few tenths of a point are not interpretable.
  Claims are made **ordinally** wherever possible, and the large effects (PCA vs HOG under
  noise is 5.2×) are far outside any plausible noise band.
- **One preprocessing config** (`raw_gray`, fixed — index Q7). The grid was not re-run at
  `clahe_gray` / `clahe_hsv`; 8.2 is optional and unrun, so ranking stability across
  preprocessing is *argued* from the per-method sweeps rather than measured on the test grid.
- **Degradations are applied to the preprocessed 48×48 model input**, so what is measured is
  the residual preprocessing did not remove (task 3.1). This is not a claim about cameras.
- **BoVW's number is a lower bound** — `k` was capped at 1000 by decision, not shown to
  plateau (note 14 §11.3, extrapolated cost ~2 pp).
- **The CNN's learning rate was never tuned** (note 13 addendum 2), so its rows are also a
  lower bound — and it is the method that already wins on clean data.
