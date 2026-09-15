# 14 — BoVW: the orderless representation (tasks 6.1–6.4)

*Feeds: Methodology → representations; Table 1 (9.1); BoVW demo (6.6); the blur panel (9.5);
the §11 jitter extension, whose premise rests on this method's orderlessness.*

Implementation: `src/gtsrb/representations/bovw.py`. Tests: `tests/test_bovw.py` (33).
**Status:** complete (tasks 6.1–6.4); 6.5 blocked on the sampling-density decision (§6.3).

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

## 6. Vocabulary and encoding (tasks 6.3–6.4)

`BoVWRepresentation` completes the method: dense SIFT → `MiniBatchKMeans` vocabulary →
orderless histogram → normalisation.

**`MiniBatchKMeans`, not full `KMeans`** — ~200k × 128 descriptors would fit in RAM, but full
Lloyd iterations over them are needlessly slow and the vocabulary is a means to an encoding,
not an object of study. Seeded from `config.SEED`; two fits with the same seed produce the
same vocabulary, which is asserted.

**Encoding is chunked.** Assigning the whole test split at once means 12,630 × 64 = 808k
descriptors against the centroids, and the distance matrix alone runs to hundreds of MB.
Chunking bounds that, and cannot change the result — each descriptor's nearest word depends
only on that descriptor. A test proves it: `chunk=7` and `chunk=1000` give byte-identical
histograms.

### 6.1 The normalisation choice, and why it is narrower than it looks

The usual reason to normalise a BoVW histogram is that images yield *different numbers* of
keypoints, so raw counts are not comparable. **That reason does not apply here.** Every image
contributes exactly 64 descriptors and each is assigned to exactly one word, so every raw
histogram already sums to 64 — pinned by `test_raw_counts_sum_to_the_keypoint_count`.

So `l1` is a pure rescale that a linear classifier cannot see, and the real question is the
**distribution of mass across words** — specifically *burstiness*, where one repeated texture
fires the same codeword many times and that bin dominates the vector.

Only the square root changes relative weights. On a bursty histogram `[36, 16, 4, 8]`:

| scheme | result | top/second ratio |
|---|---|---|
| none | `[36, 16, 4, 8]` | 2.25 |
| l1 | `[.562, .250, .062, .125]` | 2.25 |
| l2 | `[.891, .396, .099, .198]` | 2.25 |
| **power_l2** | `[.750, .500, .250, .354]` | **1.50** |

`l1` and `l2` leave the ratio exactly as it was; `power_l2` compresses it to its square root.
That is Perronnin's burstiness correction, and it is why **`power_l2` is the default**.

Measured on a class-stratified subsample (5,160 train / 1,690 val, all 43 classes, k=200,
`C`=1, untuned):

| scheme | macro-F1 | accuracy |
|---|---|---|
| **power_l2** | **0.2769** | 0.2976 |
| l2 | 0.2695 | 0.2911 |
| none | 0.2426 | 0.2598 |

The predicted ordering holds. *(Absolute values are low because the subsample is a sixth of
the training split with an untuned classifier — this validates the default, it is not a
result.)*

Following the precedent set for HOG's `block_norm`, the scheme is **fixed with a stated
reason rather than swept**: it is part of the representation's definition. It remains a
parameter, so it is available as an ablation.

### 6.2 A trap this probe walked into

The first smoke test reported macro-F1 **0.023** — chance level for 43 classes — which looked
like a broken encoder. It was a broken *probe*: it subsampled with `frame.iloc[:3000]`, and
the annotations are class-ordered, so the sample contained **3 of 43 classes**.

Worth recording because the same mistake would be invisible in a results table: accuracy was
0.446, which looks plausible, while macro-F1 was at chance. **A large accuracy/macro-F1 gap is
the signature of a label-space problem**, not of a weak model — the same reason
`labels=range(43)` is pinned in `gtsrb.evaluation` (note 05).

---

## 6.3 OPEN ISSUE — the planned sampling density is too coarse

Measured on the full training split, k=500, best of C ∈ {0.1, 1}:

| keypoint size | step | keypoints/img | macro-F1 | accuracy |
|---|---|---|---|---|
| **12** (the plan's value) | 6 | 64 | **0.3402** | 0.4162 |
| 8 | 4 | 144 | 0.4583 | 0.5295 |
| **6** | 3 | 256 | **0.5248** | 0.5936 |

**+18.5 pp macro-F1 from sampling density alone**, which is far more than `k` or `C` moved
anything. The cause is visible in the descriptors: at size 12 a keypoint covers a quarter of
a 48×48 crop, so the 64 patches overlap heavily and describe nearly the same content. Mean
within-image descriptor correlation:

| size | 6 | 8 | 12 | 16 |
|---|---|---|---|---|
| mean within-image corr | +0.077 | +0.119 | **+0.228** | +0.362 |

**The implementation is not broken**, which was checked before concluding: the vocabulary is
fully used (200/200 words), each image spreads its 64 descriptors over ~36–39 distinct words,
and larger `C` is monotonically *worse* (1 → 1000 falls 0.3402 → 0.3131), so the grid is not
mis-centred either.

**Two things are being conflated and must stay separate in the report:**

1. **The parameters are badly chosen.** `step=6, size=12` come from `PROJECT_TASKS` 6.1 and
   are too coarse for 48×48 input. This is fixable and should be swept at 6.5.
2. **Orderless encoding is genuinely weak on aligned rigid objects**, and always will be.
   What separates "30" from "50" is *where* strokes sit; a bag of patches cannot see that.
   **Spatial Pyramid Matching exists precisely to fix this** — 2×2 and 4×4 spatial bins were
   introduced because pure orderless BoVW underperforms on exactly this kind of task.

   We deliberately do **not** add a spatial pyramid. It would re-introduce layout and collapse
   BoVW onto HOG's position on the layout axis — the axis this study exists to measure. So
   BoVW is expected to trail the others, and that is a *result*, not a defect. The report must
   say so explicitly, or a reader will read a weak BoVW row as a bad implementation.

**Decision needed before 6.5:** whether to add `(step, keypoint_size)` to the sweep. The
evidence says it is the higher-leverage parameter — more than `k` — but it deviates from the
plan's "sweep k ∈ {200, 500}" and multiplies the grid. Parked pending that decision.

---

## 7. Open for task 6.5

- Sweep k ∈ {200, 500} jointly with `C` and `class_weight`, across all three preprocessing
  configs — the 4.2 protocol. Vocabulary size changes the feature dimensionality (200 vs
  500), and the best `C` tracked dimensionality closely for PCA.
- Cost: the vocabulary and encoding are cheap (~5 s and ~4 s per 3,000 images in the probe);
  as with HOG, the grid is dominated by the `LinearSVC` fits — but at 200–500 dimensions
  those are far cheaper than HOG's 900–2,352.
