# 14 — BoVW: the orderless representation (tasks 6.1–6.2)

*Feeds: Methodology → representations; Table 1 (9.1); BoVW demo (6.6); the blur panel (9.5);
the §11 jitter extension, whose premise rests on this method's orderlessness.*

Implementation: `src/gtsrb/representations/bovw.py`. Tests: `tests/test_bovw.py` (33).
**Status:** extraction complete (tasks 6.1, 6.2); vocabulary and encoding at 6.3–6.4.

---

## 1. What the representation is

Local descriptors are extracted from the image, each is assigned to one of *k* **visual
words** learned by k-means, and the image becomes a histogram of word counts.

**The spatial position of every descriptor is discarded.** That is the defining property and
the reason BoVW is in this study: it is the only representation that is *orderless*. HOG
places its cells on a rigid grid and keeps layout; PCA is alignment-critical to the point of
being holistic; the CNN pools locally but preserves arrangement. BoVW throws layout away
entirely.

Two predictions rest on that, from opposite directions:

- **Blur should hurt it twice** (`predictions.md`): local 12 px patches become near-uniform so
  descriptors collapse onto a few codewords, *and* there is no spatial layout left to fall
  back on.
- **Bbox jitter should hurt it least** (§11): a shifted or rescaled box perturbs a histogram
  far less than it perturbs an alignment-critical subspace.

---

## 2. Why a fixed grid and not a detector

**Detector-based SIFT returns almost nothing on these crops** — the §10 gotcha — and it fails
*silently*: an image contributes no descriptors and its histogram is all zeros, with no error
anywhere.

The sharper version of the problem, which the tests pin: a detector returns a **variable**
number of keypoints per image. Even where it does fire, the histograms would not be
comparable, because two images would be summarising different amounts of evidence. A fixed
grid gives every image exactly the same number of descriptors, and task 6.2 asserts it.

---

## 3. The parameters, and what was measured

`step=6`, `keypoint_size=12` on 48×48 → an **8×8 grid = 64 keypoints** per image, each
describing a 12 px neighbourhood. Neighbourhoods therefore overlap by half, which keeps the
encoding from being a hard tiling.

| property | measured |
|---|---|
| keypoints dropped by OpenCV's border pruning | **0**, at every (step, size) tried |
| descriptor dimensionality | 128 |
| descriptors per image | 64, invariant |
| extraction cost | **0.57 ms/img** → ~18 s per pass over the training split |
| descriptor L2 norm | **511.1–512.8** (OpenCV normalises; the spread is Lowe's 0.2 clip and renormalise) |

Descriptors therefore arrive **already on a sphere**, which is the right footing for k-means.
No further scaling is applied here; the histogram normalisation at 6.4 is a separate decision.

### 3.1 `upright=True` — a decision, not a default

The keypoint angle is fixed at 0 rather than letting SIFT estimate a dominant orientation per
patch. Two reasons pointing the same way:

- **Traffic signs are upright by construction**, so absolute gradient orientation *is* signal.
  Rotation-normalising each patch throws it away — a "30" and an upside-down "30" would get
  similar descriptors, and nothing in GTSRB is upside down.
- **The orientation estimate is unstable on low-contrast patches**, so it would inject noise
  precisely where the descriptor is already least reliable.

This matches VLFeat's dense-SIFT convention. The two settings are measurably different, so
this is recorded as a choice rather than left implicit.

### 3.2 Zero descriptors — where the claim stops being true

A zero descriptor would be a degenerate point for k-means and an undefined direction for
assignment, so it was worth checking rather than assuming — especially since **heavy blur
flattens local structure**, which is exactly when it might appear.

| condition | all-zero descriptors |
|---|---|
| clean (`raw_gray`, `clahe_gray`) | **0 of 128,000** |
| blur k=9 | 0 of 38,400 |
| **blur k=15** | **0 of 38,400** |
| noise σ=40 | 0 of 38,400 |
| gamma γ=2.5 | 0 of 38,400 |

**It never happens on real signs, even at full degradation.** But it is *not* impossible: a
perfectly uniform patch does produce a zero vector, verified on a synthetic constant image.

The distinction is recorded because the code relies only on the first statement. No special
handling is added — adding a guard for a case that never occurs would be untested code on a
hot path — but "never happens on GTSRB" and "cannot happen" are different claims, and a note
that says the stronger one would be wrong.

---

## 4. A bug the tests caught: the sample was short

`sample_descriptors(images, 500)` returned **480**. The per-image quota was
`round(n_samples / n_images)` — 8 for 60 images — and 60 × 8 = 480, with nothing topping it
up. `round` undershoots whenever the ratio has a fractional part below .5.

Caught by a test asserting the exact returned count, and it matters: the vocabulary at 6.3 is
fitted on "~200k descriptors", and silently getting fewer would have changed the k-means
result with nothing to indicate it. Now `ceil`, then trimmed to exactly `n_samples`.

---

## 5. Design notes worth keeping

**`keypoints` is rebuilt on every access, deliberately.** `cv2.SIFT.compute` may mutate the
`KeyPoint` objects it is handed — it can overwrite `angle` — so caching and sharing one list
would let one call silently change what the next call describes. The allocation is trivial
next to the descriptor computation.

**Colour input uses the last channel, not a mean.** For `clahe_hsv` that is V, the channel
CLAHE actually operated on. Averaging H, S and V would blend hue into a gradient operator,
which is not a meaningful quantity.

**`describe_batch` on the full training split is ~1 GB** (31,379 × 64 × 128 float32). That is
why `sample_descriptors` streams instead: 6.3 needs ~200k of roughly 2M descriptors and never
has to materialise the rest.

---

## 6. Open for tasks 6.3–6.4

- `MiniBatchKMeans` on ~200k sampled descriptors, k ∈ {200, 500}.
- Histogram encoding, then **L2 or power normalisation** — a decision to make explicitly, not
  by default: raw counts would make the histogram scale with the number of descriptors, which
  is constant here (64), so the choice is about the *distribution* of mass across words rather
  than about total magnitude.
- Per the 4.2 result, **sweep all three preprocessing configs** and tune `C` jointly with
  *k* — the vocabulary size changes the feature dimensionality (200 vs 500), and the best `C`
  tracked dimensionality closely for PCA.
