# 14 — BoVW and BoVW+SPM: the layout axis (tasks 6.1–6.6)

*Feeds: Methodology → representations; Table 1 (9.1); BoVW demo (6.6); the blur panel (9.5);
**the layout measurement (§9), which is the study's central structural claim**; the §11
jitter extension, whose premise rests on this method's orderlessness.*

Implementation: `src/gtsrb/representations/bovw.py`. Tests: `tests/test_bovw.py` (33).
**Status:** 6.1–6.4 complete. **6.5 tuned but being re-run on `raw_gray`** (plan change
2026-09-16) — §7–§9 record the tuning journey on `clahe_gray`; the reported numbers come from
the re-run. **6.6 open.** §9.4 promotes SPM to a sixth configuration in the comparison.

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

## 6.3 RESOLVED — the planned sampling geometry was far too coarse

The plan's `step=6, keypoint_size=12` (task 6.1) turned out to be the single largest thing
wrong with this method. Measured on the full training split, k=500:

| keypoint size | step | keypoints/img | macro-F1 | accuracy |
|---|---|---|---|---|
| **12** (the plan's value) | 6 | 64 | **0.3492** | 0.4317 |
| 8 | 4 | 144 | 0.4583 | 0.5295 |
| **6** | 3 | 256 | **0.5248** | 0.5936 |

The cause is visible in the descriptors: at size 12 a keypoint covers a quarter of a 48×48
crop, so the 64 patches overlap heavily and describe nearly the same content. Mean
within-image descriptor correlation:

| size | 6 | 8 | 12 | 16 |
|---|---|---|---|---|
| mean within-image corr | +0.077 | +0.119 | **+0.228** | +0.362 |

**The implementation was not broken**, which was checked before concluding: the vocabulary is
fully used (200/200 words), each image spreads its 64 descriptors over ~36–39 distinct words,
and larger `C` is monotonically *worse* (1 → 1000 falls 0.3402 → 0.3131), so the grid was not
mis-centred either. The parameters were simply wrong for 48×48 input.

This is the finding that motivated putting `(step, keypoint_size)` on the 6.5 sweep. See §8
for what the sweep then showed, including two claims made here that it falsified.

---

## 7. Task 6.5 — the configuration sweep

`scripts/sweep_bovw.py`. Sweeps `(keypoint_size, step) × k × C × class_weight`, selecting on
validation macro-F1 through `gtsrb.tuning` — the same protocol as `sweep_pca.py` and
`sweep_hog.py`.

**Geometry is on the grid, and `size` and `step` are decoupled.** Every earlier probe varied
them *together*, so it was not known whether the gain came from finer **spacing** (more
descriptors) or finer **scale** (each descriptor covering less of the sign). The grid
deliberately includes cells that hold one fixed while moving the other.

Descriptors are extracted **once per geometry** and reused across every `k` — they do not
depend on the vocabulary, and this is the only reason the full grid is affordable
(extraction is 80–120 s per pass and dominates everything else).

### 7.1 Results — `clahe_gray`, `results/sweeps/bovw_stage1{,b}.csv`

| size | step | kp/img | k | macro-F1 | accuracy |
|---|---|---|---|---|---|
| 12 | 6 | 64 | 500 | 0.3492 | 0.4317 |
| 6 | 3 | 256 | 500 | 0.5285 | 0.5969 |
| 6 | 2 | 576 | 500 | 0.5611 | 0.6243 |
| 4 | 3 | 256 | 500 | 0.6511 | 0.7074 |
| 4 | 2 | 576 | 500 | 0.6723 | 0.7370 |
| 4 | 2 | 576 | 1000 | 0.7322 | 0.7782 |
| 3 | 2 | 576 | 500 | 0.7347 | 0.7946 |
| **3** | **2** | **576** | **1000** | **0.7939** | **0.8573** |

`C = 10` wins almost everywhere — at the **top** of `tuning.C_GRID`, so the grid had to be
extended for this method. Recorded as a fourth instance of the dimensionality/`C` coupling
seen for PCA and HOG, pointing the other way: BoVW's features are the lowest-dimensional and
want the *largest* `C`.

### 7.2 The decoupling result: it is **scale**, not density

Because `step` fixes the keypoint count independently of `size`, the two effects separate
cleanly:

| held fixed | varied | effect |
|---|---|---|
| density (256 kp, step 3) | scale 6 → 4 | **+12.3 pp** (0.5285 → 0.6511) |
| scale (size 4) | density 256 → 576 kp (2.25×) | **+2.1 pp** (0.6511 → 0.6723) |

**Scale outweighs density by roughly 6×.** What matters is not how many patches there are but
how much of the sign each one covers: a 12 px patch on a 48 px crop is summarising a quarter
of the object, which is not a *local* descriptor in any useful sense. Shrinking it to 3–4 px
is what turns dense SIFT into a local feature here.

> **Correction.** §6.3 above, `PROJECT_TASKS.md` 6.5 and the first version of
> `sweep_bovw.py` all called this "sampling density" and the sweep figure was titled *"Density
> is the lever"*. That was imprecise — the two were confounded in every probe that produced
> it, and when separated, density is the minor term. The report says **scale**.

### 7.3 Vocabulary size *was* a lever

500 → 1000 words is worth a consistent **+6 pp macro-F1**, at both geometries where both were
run:

| geometry | k=500 | k=1000 | Δ |
|---|---|---|---|
| size 4 / step 2 | 0.6723 | 0.7322 | **+6.0 pp** |
| size 3 / step 2 | 0.7347 | 0.7939 | **+5.9 pp** |

> **Correction, and the more instructive of the two.** The previous revision of this note
> (§8.1, committed 2026-09-15) concluded from a literature check that "vocabulary size is
> probably not the missing ingredient", because a published
> pLSA system succeeds with a 300-word codebook. That inference was wrong, and wrong in a way
> worth recording: **that system's 300 words feed a probabilistic topic model, not a linear
> SVM.** A topic model can recover structure from a coarse codebook that a linear classifier
> on raw counts cannot. The codebook size was transferred across a difference in what sits on
> top of it, which is exactly the kind of transfer the 4.2 hyperparameter measurement had
> already shown to be unsafe within this project. Reasoning from one number in one paper,
> without the surrounding architecture, cost ~6 pp of accuracy and nearly closed off the
> search.

---

## 8. The recovery, and three claims that had to be withdrawn

BoVW went from **0.3492** macro-F1 at the plan's parameters to **0.8791** — a gain of
**+53 pp**, larger than any other effect measured anywhere in this project. The method was
never broken; it was badly parameterised, and the parameter that mattered was not the one the
plan swept (`k`) nor the one the first diagnosis named (density), but **descriptor scale**.

### 8.1 Withdrawn: the estimated performance cap

An earlier note estimated a ceiling of "~75–82 % accuracy" for orderless BoVW on this task,
extrapolating from Lazebnik's 72.2 % on Caltech-101. **Plain BoVW reaches 92.3 % accuracy
here.** The extrapolation crossed a different task, different descriptors, different sampling
and a different number of classes, and it should not have been offered as a bound.

Recorded because it had a real effect on the work: a cap in that range makes further tuning
look pointless, and had it been believed, the search would have stopped around 0.73.

### 8.2 Withdrawn: the literature-derived SPM claim

A search summary attributed "dense SIFT + HOG + LBP + **spatial pyramid matching** → 99.67 %"
to two GTSRB papers. Both were downloaded and read: **neither mentions spatial pyramid
matching at all.** The figure was the search engine's synthesis. Recorded because it nearly
became a citation.

### 8.3 Withdrawn: two of four OOM diagnoses

Four runs were killed by the OOM reaper. The first was genuinely ours — `encode()` used a
flat 4,000-image chunk, which costs 0.12 GB at 64 keypoints and **1.10 GB** at 576, so the
finest geometry blew up. Chunks are now sized by *bytes*, accounting for both the descriptor
block and the k-means assignment (`n_keypoints × (128×4 + k×8)`).

Peak RSS was then instrumented at **~1.2 GB** against 11.5 GB free, so the two later kills
were environmental (a background memory policy), not ours — running in the foreground avoided
them. Two diagnoses were therefore withdrawn.

**But one was real and had been missed:** `experiment_spatial_pyramid.py` still carried the
*old* chunking after `sweep_bovw.py` was fixed. The fix was applied to the file in front of
me without grepping for the pattern elsewhere. That is the propagation failure to remember —
the environmental explanation was partly covering for a genuine un-propagated bug.

---

## 9. What orderlessness costs — measured (`scripts/experiment_spatial_pyramid.py`)

This is the most directly useful measurement the method produced, and it is the study's
central structural claim reduced to a single controlled manipulation.

**Spatial Pyramid Matching** (Lazebnik) partitions the image into increasingly fine grids —
level 0 is the whole image (= plain BoVW), level 1 is 2×2, level 2 is 4×4 — builds a histogram
per cell and concatenates them, weighted 1/2^(L−l). The keypoints lie on a regular grid, so
which cell a descriptor belongs to is known **from its index alone**: the position information
was available all along and plain BoVW simply throws it away.

All three rows below share **one vocabulary, one set of descriptors, one classifier and one
protocol**. The only difference is whether position is recorded.

At `size 2 / step 2`, k=500, `clahe_gray`:

| levels | cells | dims | macro-F1 | accuracy |
|---|---|---|---|---|
| 0 — plain BoVW | 1 | 500 | 0.8143 | 0.8793 |
| 1 — 2×2 | 5 | 2,500 | 0.8973 | 0.9295 |
| **2 — 4×4** | 21 | 10,500 | **0.9184** | **0.9441** |

**Layout is worth +10.4 pp macro-F1 / +6.5 pp accuracy** on this task.

### 9.1 Why this is a better experiment than BoVW vs. HOG

The report's structural claim — that discarding spatial arrangement is expensive on rigid,
aligned objects — was going to be argued from the BoVW/HOG gap. But those two differ in six
ways at once: descriptor, pooling, quantisation, normalisation, dimensionality *and* layout.
BoVW vs. BoVW+SPM differs in **exactly one**. It is the controlled version of the same claim.

And it lands almost exactly on HOG (0.9184 vs **0.9175** macro-F1), which says the BoVW/HOG
gap *is* the layout information, with the other five differences contributing ~nothing net.

### 9.2 The pyramid helps less as the base representation improves

Run earlier at the poor configuration (k=200, size 4): L0 **0.5276** → L1 0.6315 → L2
**0.6686**, a gain of **+14.1 pp**. At the good configuration the same manipulation is worth
**+10.4 pp**. Layout and descriptor quality are **partial substitutes**: a finer-scale
descriptor with a larger vocabulary already recovers some of what spatial binning was
supplying. Worth stating, because it means "SPM is worth +X pp" is not a constant.

### 9.3 Internal consistency check

SPM L=0 at k=500 is by construction plain BoVW, and scores 0.8143. Plain BoVW at the same
geometry with k=1000 scores 0.8791 — a +6.5 pp vocabulary effect, matching the independent
+6.0/+5.9 pp measured at two other geometries in §7.3. Two separately-run experiments agree
on a parameter effect neither was designed to measure.

### 9.4 DECISION (2026-09-16): SPM becomes a sixth configuration, not a replacement

**Both go in the comparison.** Plain BoVW stays the method the proposal defines; SPM is added
alongside it as `bovw_spm_svm`.

**Why SPM does not simply replace BoVW.** The proposal's §4.3 table defines BoVW as *"bez
rasporeda, raspored odbačen"* — without layout, layout discarded — and it is the only method
on that side of the axis. Substituting SPM would leave the layout axis with no occupant: HOG
is a rigid grid of local histograms and so is SPM L=2. Two rows at the same point, and the
contrast the study exists to measure disappears. It would also make the blur prediction
untestable, since that prediction turns on there being *no layout to fall back on*.

Note the exclusion was decided and written down **before** any of these numbers existed
(§8.2 of the previous revision, committed 2026-09-15), and it passes the neutrality test in
`CLAUDE.md`: had SPM come out *worse*, the structural reason to keep BoVW orderless would be
unchanged. It is not protecting a result.

**Why it is nonetheless added.** Reporting BoVW at 0.88 when the standard deployed form of the
method reaches 0.92 understates it, and a report that omits SPM silently looks like it
handicapped the method that lost. Adding a row costs 16 inference-only grid cells and buys:

- the controlled layout measurement of §9.1, in the results table rather than a footnote;
- a directly falsifiable pair of opposite predictions — orderlessness should **hurt** under
  blur (nothing to fall back on) and **help** under bbox jitter (§11, nothing to misalign).
  One method family, two settings, opposite directions. That is the sharpest claim available.

**Costs, recorded honestly:** the comparison grows to six rows; SPM's 10,500 dims at k=500 is
6× HOG's, so the Table 1 feature-dimension and model-size columns need the same "not
like-for-like" caveat PCA already carries; and it **diverges from the proposal's "pet
konfiguracija"** (five configurations, §4.3). Per `CLAUDE.md` the proposal is not edited — the
divergence is recorded here and was raised with the author, who approved it.

---

## 10. Cost, and whether any of this is usable in real time

Measured at `size 2 / step 2`, k=1000 — the slowest configuration of the five methods:

| stage | cost |
|---|---|
| dense SIFT extraction | ~1.9 ms/img |
| vocabulary assignment + histogram | ~0.5 ms/img |
| **total** | **~2.4 ms/sign (~420 signs/s)** |

**Slowest of the five methods, and still not the binding constraint.** At 30 fps a frame
budget is 33 ms, so classification takes ~7 % of it. SPM adds only the pooling, which is
negligible next to extraction — the descriptors are already computed — so it is in the same
range.

The honest framing for the report: for this module, *all* the candidate representations run in
real time on a laptop CPU, and the representation choice should be made on robustness, not
on speed. The cost that would actually dominate a deployed system is detection (M1), which
this project explicitly does not implement. Timings remain relative costs in one recorded
environment, per note 06.

---

## 11. Open for task 6.5 (re-run) and 6.6

**Everything in §7–§9 was measured on `clahe_gray` and must be re-run on `raw_gray`**, which
is now the single fixed preprocessing config for the whole comparison (plan change,
2026-09-16 — see `PROJECT_TASKS.md` §1). The numbers above are expected to move; they are
recorded as the tuning *journey*, and the reported figures come from the re-run.

Also outstanding:

- **`k` is capped at 1000 by decision, not by measurement** (2026-09-17). `k=2000` at size 2
  remains untested and is the one demonstrably promising cell left: `k` was worth +6 pp at
  500 → 1000 and has *not* been shown to plateau. **BoVW's reported number is therefore a
  lower bound**, and the report must say so in the same breath as PCA's k=256 caveat — both
  are selections pinned to the edge of a grid that was not widened. The budget goes to the
  degradation grid instead, which is the project's actual subject, and at size 2 a k=2000
  vocabulary would put SPM L=2 at ~42,000 dimensions.
- **The SPM row is L = 2 only** (2026-09-17). L=1 stays here as the intermediate measurement
  — it is what shows the effect is graded and that layout and descriptor quality are partial
  substitutes — but it does not get a Table 1 row.
- **Persist everything.** The size-2 results and every pyramid row above were printed to a
  terminal and never written to a CSV — they exist only in a session transcript. This is the
  documentation failure of this task and the reason 6.5 is being re-run rather than
  transcribed.
- **6.6 demo.** The centrepiece should be the **permutation test**: shuffle the keypoint
  positions before pooling and show the histogram is byte-identical. That makes
  orderlessness a demonstrated property rather than an asserted one, and it is the natural
  visual companion to §9 — the same figure can show the SPM histogram *changing* under the
  same shuffle.
