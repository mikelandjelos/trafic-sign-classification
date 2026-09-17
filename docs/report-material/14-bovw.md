# 14 — BoVW, and what orderlessness costs (tasks 6.1–6.6)

*Feeds: Methodology → representations; Table 1 (9.1); BoVW demo (6.6); the blur panel (9.5);
**the layout measurement (§9) — reported in the DISCUSSION as a diagnostic, not as a table
row**; the §11 jitter extension, whose premise rests on this method's orderlessness.*

Implementation: `src/gtsrb/representations/bovw.py` (`BoVWRepresentation`,
`BoVWSpatialPyramid`). Tests: `tests/test_bovw.py` (58).
**Status:** 6.1–6.5 complete. **6.5b reverted 2026-09-17** — SPM is a diagnostic (§9.4), not a sixth method. 6.6 written, not yet run.
**The reported row** (`results.csv`, run `20260917T014750`): `raw_gray`, size 2 / step 2,
k = 1000, C = 1, `balanced` → val macro-F1 **0.8840**, accuracy **0.9330**. The sweep cell that
selected it scored 0.8801/0.9317 (§11); the small gap is the final fit using a full vocabulary
fit rather than the sweep's sampled one.
**A +53.5 pp recovery** from the parameters the plan specified.
§7–§9 record the tuning journey on `clahe_gray` and are kept as history, not as results;
**§9.4 records why SPM is a diagnostic rather than a sixth row.**

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

### 9.4 DECISION (2026-09-16), REVERSED (2026-09-17): SPM is a diagnostic, not a method

**Final position: the comparison is the proposal's five configurations. This measurement is
reported in the discussion, not as a Table 1 row.**

#### What was decided, and why it was reversed

On 2026-09-16 SPM was promoted to a sixth configuration, on the strength of the numbers above.
The argument was sound and is worth keeping, because it is what the measurement is *for*:

> BoVW vs. HOG differs in six ways at once — descriptor, pooling, quantisation, normalisation,
> dimensionality **and** layout. BoVW vs. BoVW+SPM differs in **exactly one**. It is the
> controlled form of the study's central claim, and it lands within 0.1 pp of HOG, which says
> the BoVW/HOG gap *is* the layout information.

It was reversed a day later for a reason that has nothing to do with that argument: **it does
not fit the machine.** At the **k = 1000** that the 6.5 sweep selected, L=2 is **21,000
dimensions** — a 4.91 GB float64 training matrix, ~6 GB resident, on a laptop with ~7 GB free.
Three training runs died, two of them silently at the header because `LinearSVC` upcasts
float32 to float64 internally, so each of the six fits transiently wanted 2.45 + 4.91 =
**7.36 GB**.

#### The mistake, recorded plainly

**The cost was invisible at the moment the row was proposed.** The +10.4 pp was measured at
**k = 500** — 10,500 dimensions, perfectly comfortable. The k = 1000 decision came *later*, for
plain BoVW, and doubled the pyramid's width as a side effect. Nobody connected the two until
three runs had failed, and each failure was diagnosed as the local symptom (chunk size, then
`nohup`, then dtype) rather than as the scope decision underneath.

> **The general lesson, which is the reportable part: a method's feasibility has to be
> re-checked when a parameter it depends on changes.** A shared vocabulary made the two rows
> a controlled comparison — and it also coupled their costs, so a decision taken for one
> silently resized the other.

#### Why plain BoVW would never have been replaced by SPM in any case

Even had it fitted, SPM was always an *addition*. Plain BoVW is the only method on the "layout
discarded" side of the proposal's §4.3 axis; substituting would have left that axis with no
occupant — HOG is a rigid grid of local histograms and so is SPM L=2 — and would have made the
blur prediction untestable, since that prediction turns on there being no layout to fall back
on. That reasoning is unaffected by the reversal.

#### What survives

Everything measured. §9's table, the L=1 intermediate, the "pyramid helps less as the base
improves" substitution effect, and the HOG coincidence all stand and are reported as a
**diagnostic in the discussion**. `gtsrb.representations.bovw.BoVWSpatialPyramid` stays in the
codebase with its 9 tests — including the two identities that make the measurement valid
(`levels=0` reduces exactly to plain BoVW; permuting keypoint positions leaves the plain
histogram byte-identical while changing the pyramid) — so the number remains reproducible by
`scripts/experiment_spatial_pyramid.py`. **The §4.3 divergence is withdrawn with the row:** the
report describes five configurations, exactly as the proposal does.

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

## 11. The reported configuration — `raw_gray` (task 6.5, 2026-09-17)

`results/sweeps/bovw_configs.csv`. 112 grid points, **all converged**. §7–§9 above were
measured on `clahe_gray` and are kept as the tuning *journey*; **these are the reported
numbers.**

| size | step | kp/img | k | macro-F1 | accuracy |
|---|---|---|---|---|---|
| **12** (the plan's value) | 6 | 64 | 1000 | **0.3455** | 0.4142 |
| 6 | 3 | 256 | 1000 | 0.5693 | 0.6271 |
| 6 | 2 | 576 | 1000 | 0.5814 | 0.6450 |
| 4 | 3 | 256 | 1000 | 0.6862 | 0.7332 |
| 4 | 2 | 576 | 1000 | 0.7079 | 0.7640 |
| 3 | 2 | 576 | 1000 | 0.7863 | 0.8457 |
| **2** | **2** | **576** | **1000** | **0.8801** | **0.9317** |

**Selected: size 2 / step 2, k = 1000, C = 1, `class_weight="balanced"`.**
**A +53.5 pp recovery** from the parameters the plan specified — the largest single effect
measured anywhere in this project.

`C = 1` wins, not `C = 10` as on `clahe_gray`, so **the selection is no longer pinned to a
grid edge** — extending the grid to 100 was justified and 100 never wins. That retires the
caveat `C` carried at 6.5's first pass.

### 11.1 BoVW barely notices CLAHE — and that is a mechanism, not a coincidence

| preproc | best macro-F1 |
|---|---|
| `clahe_gray` | 0.8791 |
| `raw_gray` | **0.8801** |

**0.1 pp apart.** So the fixed-`raw_gray` decision costs BoVW essentially nothing, and the
reason is the one already on record for HOG and the CNN: **SIFT descriptors arrive from
OpenCV already L2-normalised** (§3), so contrast enhancement has nothing left to do.

This completes a clean four-way pattern for the 9.7 discussion — *how much preprocessing
matters is a property of the representation*, specifically of whether it normalises
internally:

| method | internal normalisation | cost of dropping CLAHE |
|---|---|---|
| PCA | **none** | **−2.64 pp** |
| HOG | block L2-Hys | −0.4 pp (CLAHE *hurts*) |
| BoVW | SIFT descriptor L2 | **−0.1 pp** |
| CNN | BatchNorm | −0.16 pp |

PCA is the outlier, and it is the only method without internal normalisation. That is the
whole finding, and it is now measured on four methods rather than argued.

### 11.2 Scale over density, reproduced on a second preprocessing config

| held fixed | varied | `raw_gray` | (`clahe_gray`) |
|---|---|---|---|
| density (256 kp, step 3) | scale 6 → 4 | **+11.7 pp** | +12.3 pp |
| scale (size 4) | density 256 → 576 kp | **+2.2 pp** | +2.1 pp |

Roughly **5.3×** apart, against ~6× before. An independent replication of §7.2 on different
pixels, which is worth more than the original single measurement.

### 11.3 The vocabulary gain shrinks as scale improves — the k = 1000 cap is cheaper than feared

500 → 1000 words, by geometry:

| geometry | k=500 | k=1000 | Δ |
|---|---|---|---|
| size 6 / step 3 | 0.4910 | 0.5693 | +7.8 pp |
| size 4 / step 3 | 0.6140 | 0.6862 | +7.2 pp |
| size 4 / step 2 | 0.6351 | 0.7079 | +7.3 pp |
| size 3 / step 2 | 0.7302 | 0.7863 | +5.6 pp |
| **size 2 / step 2** | 0.8434 | **0.8801** | **+3.7 pp** |

**The gain halves as the descriptor gets finer** — +7.3 pp at size 4, +3.7 pp at size 2.

This is the *same substitution pattern* the spatial pyramid shows (§9.2): a better descriptor
recovers what a larger vocabulary was otherwise supplying, just as it recovers what spatial
binning was supplying. Three separate parameters — scale, vocabulary size, spatial layout —
are all partly buying the same thing, which is why their individual effects are not additive
and why quoting any of them as a constant would be wrong.

**Consequence for the reported caveat.** BoVW's number remains a **lower bound**, since k was
capped by decision rather than shown to plateau — but the bound is tighter than the earlier
flat "+6 pp per doubling" reading suggested. The extrapolation to k=2000 at size 2 is
plausibly ~+2 pp, not +6. State it that way rather than leaving the larger figure standing.

### 11.4 The Q8 guard fired on its first real use

The sweep ran with the pre-Q8 grid (both `class_weight` settings), and its own unconstrained
best was `class_weight=None` at 0.8808. The policy-compliant selection is **0.8801** — a
**0.07 pp** difference.

Two things follow. First, `best_from_sweep`'s policy filter did exactly the job it was added
for: without it, the final model would silently have violated a protocol decision made the
same day. Second, the size of the difference is itself the argument for §Q8's closing
sentence — **no class-weighting effect may be claimed anywhere in this report.** At the
selected configuration the two settings are indistinguishable.

---

## 12. Still open

- **`k = 2000` is untested**, capped by decision (2026-09-17). See §11.3: the extrapolated
  cost is now ~2 pp rather than ~6.
- **The SPM row is L = 2 only.** L=1 stays in §9 as the intermediate measurement — it is what
  shows the effect is graded and that layout and descriptor quality are partial substitutes —
  but it does not get a Table 1 row.
- **6.6 demo** — `scripts/demo/bovw_mechanics.py`, written, not yet run. Centrepiece is the
  permutation test: shuffle the keypoint positions before pooling, and the plain histogram is
  byte-identical while the pyramid's changes. Four figures are in the report manifest.
- **A documentation failure worth keeping.** The first size-2 result and every pyramid row in
  §9 were printed to a terminal and never written to a CSV — they existed only in a session
  transcript and had to be re-run. Everything in §11 is persisted in
  `results/sweeps/bovw_configs.csv` and `results.csv`.

---

## 13. Task 6.6 — the demo, and the prediction it falsified

`scripts/demo/bovw_mechanics.py` → `figures/demo/bovw/` (4 figures, all four in the report
manifest). Built entirely from `gtsrb.representations.bovw` and `gtsrb.degradations`; nothing
here re-derives the encoding.

### 13.1 Orderlessness, demonstrated rather than asserted

`bovw_permutation.png`. Shuffle which grid position each descriptor came from, re-pool, and
compare — same descriptors, same vocabulary, only the positions permuted:

| | max abs difference |
|---|---|
| **plain BoVW** | **0.000e+00** — the same vector, bit for bit |
| BoVW + SPM (L=2) | 0.0833 |

The codeword map makes it visible: the original shows the sign's circular structure, the
shuffled panel is pure noise — and the plain histogram *does not notice*. Everywhere else in
this project "orderless" is an assertion about the design; here it is a measurement, and it is
the clearest single statement of the axis the study is about.

### 13.2 The failure test falsified the prediction it was written to illustrate

`bovw_blur_collapse.png`. `predictions.md` says blur should hurt BoVW **twice**: descriptors
collapse onto a few codewords, *and* there is no layout to fall back on. The first half is
testable at the representation level, so the demo tested it.

| blur kernel | 0 | 15 | change |
|---|---|---|---|
| distinct words per image | 231.4 | 188.9 | **−18 %** |
| between-class cosine similarity | 0.232 | 0.244 | **+0.012** |

**The predicted collapse does not happen.** Vocabulary usage narrows only mildly, and the
histograms of different classes stay as far apart as they started.

**Likely mechanism, and the project already relies on it elsewhere:** SIFT descriptors are
**L2-normalised** (§3). Blur cuts gradient *magnitude*, but the descriptor *direction* still
varies from patch to patch, and k-means assigns by direction. The same normalisation that
gives BoVW its gamma robustness also prevents codeword collapse. That is a satisfying
consistency — one property explaining behaviour under two different stressors — but it was
**not** anticipated in the locked prediction.

**What this does and does not license.** It rules out *one proposed mechanism* for a blur
effect; it does **not** say blur is harmless to BoVW. This measures the representation, not
accuracy: histograms can remain mutually distant and still stop being discriminative in the
way the classifier needs. **Task 8.1 decides whether blur hurts BoVW.** Sample here is ~22
images, one per two classes.

**For 9.6:** the blur row's *second* mechanism (no layout to fall back on) is untouched by
this and remains live — and §9's +10.4 pp is direct evidence for it. So if blur does hurt
BoVW at 8.1, the reading should be "because layout is missing", not "because the codewords
collapsed". That is a sharper claim than the prediction made, and it was reached by the
prediction being half wrong.

### 13.3 The other two figures

`bovw_grid_and_words.png` — the dense grid on four sign shapes and the codeword map, the
methodology section's explanatory figure. `bovw_normalisation.png` — `power_l2` against
`l1`/`l2`/`none` on a real histogram, showing that only the square root changes the ratio
between bins (§6.1's argument, on real data rather than a toy vector).

**A caption bug caught on the first run, worth recording.** The blur figure's panel titles
originally read *"Vocabulary usage collapses as blur rises"* and *"…and different signs start
to look alike"* — written from the prediction, before the data existed. The measurement says
neither. They now describe what was measured. This is exactly the failure mode `CLAUDE.md`
warns about: a figure that asserts the expected result rather than the observed one, and it
survived until the figure was actually looked at.
