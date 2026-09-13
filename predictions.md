# Predictions

**Recorded: 2026-09-13**, before task 4.1 — before PCA, HOG, BoVW or the CNN existed, and
before any accuracy number had been produced. Nothing below was written or amended after
seeing a result.

Latest commit at time of recording: task 3.5 (degradations complete, no model trained).

> A prediction made after seeing results is worthless. This file is the record that these
> were made in advance; task 9.6 compares it against outcomes, and entries that turn out
> **wrong are more valuable to the discussion than entries that hold** — a failed prediction
> localises a wrong belief about the representations.

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
