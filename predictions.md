# Predictions

**Recorded: 2026-09-13**, before task 4.1 — before PCA, HOG, BoVW or the CNN existed, and
before any accuracy number had been produced. Nothing below was written or amended after
seeing a result.

Latest commit at time of recording: task 3.5 (degradations complete, no model trained).

> A prediction made after seeing results is worthless. This file is the record that these
> were made in advance; task 9.6 compares it against outcomes, and entries that turn out
> **wrong are more valuable to the discussion than entries that hold** — a failed prediction
> localises a wrong belief about the representations.

> ⚠ **One addendum was added later, on 2026-09-17**, for a method that did not exist when
> this file was locked. It is at the bottom, separately dated, and states exactly what was
> known when it was written. **Everything above the addendum is untouched and predates every
> result.** The addendum predates every *degradation* result, which is what it predicts.

---

## MVP stressors

| Stressor | Most robust | Least robust | Reasoning |
|---|---|---|---|
| **Gaussian noise** σ 0→40 | PCA | HOG | Noise is high-frequency and isotropic; PCA keeps only k ≈ 128 of 2304 dimensions, so most noise energy falls outside the retained subspace and is averaged away. HOG differentiates — a high-pass operation that amplifies noise directly and randomises orientation bins. BoVW is also gradient-based but SIFT's 4×4 spatial pooling averages somewhat, so it should land between. |
| **Motion blur** k 0→15 px | PCA | BoVW | Blur is low-pass and preserves exactly the low-frequency structure the leading eigenvectors encode. BoVW suffers twice: local 12 px patches become near-uniform, so descriptors collapse onto a few codewords, *and* the orderless histogram has no spatial layout left to fall back on. HOG keeps the rigid grid, so even weakened gradients in the right cells still carry signal — it should beat BoVW here. |
| **Gamma** γ 0.4 ↔ 2.5 | HOG | PCA | Gamma is a monotone point transform: edge locations are untouched, only contrast changes. HOG's block-wise L2 normalisation cancels contrast scaling almost entirely, and SIFT's descriptor normalisation gives BoVW similar (slightly weaker) protection. PCA operates on raw intensities, so a global remap shifts every projection coefficient and moves the test point away from the training distribution. At γ = 2.5, 3.71 % of pixels crush to zero — irreversible loss that hurts the intensity-based method most. |
| **Small signs** <32 px ROI height | CNN | BoVW | Upsampled small signs carry no genuine high-frequency detail, so dense SIFT has too little local support to form meaningful descriptors. The CNN is trained on the same size distribution (47.6 % of training images are <32 px — the modal case, not a tail), so it can learn whatever coarse cues exist. This measures scale-dependence, *not* distribution shift: train and test size distributions agree to within 0.1 pp. |

## Extension stressors (§11 — GTSDB, not part of the MVP)

| Stressor | Most robust | Least robust | Reasoning |
|---|---|---|---|
| **Bbox jitter** ±40 % | BoVW | PCA | Orderless encoding discards layout, so a shifted or rescaled box perturbs the histogram far less than it perturbs an alignment-critical subspace. HOG's rigid grid should also suffer, as content crosses cell boundaries; CNN pooling gives partial tolerance. |
| **Cross-dataset** GTSRB → GTSDB @ jitter 0 | HOG | CNN (e2e) | This measures domain shift alone. The CNN has the most capacity to exploit GTSRB-specific statistics — camera, compression, track structure — so it has the most to lose. Contrast-normalised hand-designed features encode structure that transfers; only the linear classifier on top is fitted to GTSRB. |

---

## Headline prediction

**No single representation wins everywhere, and the ranking inverts between stressors.**

The sharpest falsifiable case:

> **PCA is predicted best under noise and worst under gamma, while HOG is predicted worst
> under noise and best under gamma.**

If that crossing appears, the project's central claim is demonstrated in a single figure.

**If the ranking is stable across all stressors, the premise is wrong** — and that is a more
interesting result than confirming it, because it would mean the structural differences
between these representations do not translate into different failure modes at this scale.

## Secondary expectations

Lower confidence than the table above; recorded so they cannot be retrofitted.

- **Clean accuracy order:** CNN (e2e) > CNN-feat+SVM > HOG > BoVW > PCA. The CNN gap on
  clean data is expected to be modest — GTSRB is saturated, and the comparison is
  deliberately not about who wins on clean data.
- **Accuracy vs. macro-F1:** the gap should be widest for the weakest method, since rare
  classes fail first. PCA's macro-F1 is expected to trail its accuracy by the largest margin.
- **Cost:** PCA cheapest at inference (one dense matmul), BoVW most expensive (dense SIFT
  plus vocabulary assignment per image). Expected to span more than an order of magnitude.
- **Preprocessing ablation:** `clahe_gray` > `raw_gray` for the gradient methods (contrast
  normalisation helps the under-exposed frames), with `clahe_hsv` buying little for its 3×
  dimensionality — and possibly hurting BoVW, which can only use one channel.

---

# ADDENDUM — BoVW + Spatial Pyramid Matching

**Recorded: 2026-09-17**, before task 8.1 — i.e. **before any degradation result exists for
any method**. Latest commit at time of recording: the task 8.1 grid runner, which has not been
run.

## Why this is a late addition, and what was already known

`bovw_spm_svm` was promoted to a sixth configuration on 2026-09-16 (note 14 §9.4, index Q6).
It did not exist when this file was locked, so it has no row in the table above.

**Disclosed in full, because it bears on whether these predictions are legitimate.** At the
time of writing I knew:

- every method's **clean validation** score, including SPM's (0.9184 macro-F1 on `clahe_gray`,
  landing within 0.1 pp of HOG's 0.9175);
- that layout is worth +10.4 pp on clean data, and that the benefit *shrinks* as the base
  representation improves (+14.1 pp at a poor configuration, +10.4 pp at a good one).

I did **not** know any degradation result, for any method, at any level. Nothing below is
retrodiction: the entire robustness grid was unrun.

The clean scores do constrain these predictions — knowing SPM ≈ HOG on clean data is what
makes "SPM should track HOG's curves" a natural guess. That is stated rather than hidden, and
9.6 should read these rows with that caveat attached. They are weaker evidence than the
2026-09-13 table, and should not be presented as equal to it.

## Predictions

The interesting question is not whether SPM beats plain BoVW — it does on clean data, and
will almost certainly continue to. It is **whether the gap between them is constant across
stressors**. The pair differs in exactly one variable, so any *change* in their gap localises
what layout is actually buying.

| Stressor | Prediction | Reasoning |
|---|---|---|
| **Motion blur** k 0→15 | **SPM degrades more slowly than plain BoVW** — the gap WIDENS | This is the sharpest case. The locked table predicts blur hurts BoVW *twice*: descriptors collapse onto few codewords, **and** there is no layout to fall back on. SPM removes the second half of that mechanism. If the gap does not widen under blur, the "twice" claim was wrong and the damage is entirely in the descriptors. |
| **Gaussian noise** σ 0→40 | **SPM degrades slightly FASTER — the gap NARROWS** | The opposite direction, and the reason is sparsity, not layout. A 4×4 level gives each cell ~36 descriptors over a 500-word vocabulary, so per-cell histograms are extremely sparse. Noise randomises codeword assignment, and a given *fraction* of misassignments perturbs a sparse histogram proportionally more than a dense pooled one. Layout survives noise fine; the statistics estimated within each cell do not. |
| **Gamma** γ 0.4↔2.5 | **Near-identical curves — the gap is UNCHANGED** | Deliberately the null row. Gamma is a monotone point transform: it moves no edge and no keypoint, so it cannot touch layout, and SIFT's descriptor normalisation handles the rest. **If these two curves separate under gamma, the reasoning behind the other two rows is suspect** — that is what this row is for. |
| **Small signs** <32 px | **SPM better than plain BoVW** | Upsampled small signs carry no genuine high-frequency detail, so descriptors degrade toward each other — but the *arrangement* of a sign's parts survives upsampling intact. Same mechanism as the blur row. |

## Headline addendum prediction

> **Layout is worth more when structure is destroyed (blur, small signs) and worth less when
> assignments are randomised (noise).** So the BoVW→SPM gap should *widen* under blur and
> *narrow* under noise, crossing nowhere but visibly changing slope.

**If the gap is instead flat across all three stressors**, the honest reading is that SPM is
simply a strictly better representation and "layout" is not doing anything stressor-specific —
which would undercut the structural framing of the whole comparison and is the more
interesting outcome.

## Secondary expectation

**SPM's robustness curves should track HOG's more closely than plain BoVW's do.** Both are now
rigid grids of local histograms at nearly the same clean accuracy, so if the layout axis is
real they should also *fail* alike. If SPM keeps behaving like BoVW under stress despite
having layout, then layout is not what separates BoVW from HOG, and the +10.4 pp has some
other explanation — most likely the 21× dimensionality.
