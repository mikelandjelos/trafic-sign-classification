# 08 — Preprocessing

**Feeds:** Methodology → preprocessing; the ablation table (task 9.7); supports fig. 9.4.
**Status:** complete (tasks 2.1, 2.2, 2.3)

Implemented as `gtsrb.preprocessing`; tested in `tests/test_preprocessing.py`.

---

## Primitives

| Function | Purpose |
|---|---|
| `load_image` | read GTSRB's P6 PPM as BGR uint8 |
| `to_gray` | BGR → grayscale |
| `to_hsv` | BGR → HSV |
| `clahe` | local contrast normalisation |
| `gaussian_blur` | mild denoising |
| `resize` | to 48×48, adaptive interpolation |
| `crop_roi` | crop to the annotated box, clamped |

Every function takes and returns **uint8**, so results cache directly as uint8 `.npy`
(task 2.3) with no lossy round-trip through float.

**Colour convention.** `cv2.imread` returns BGR, not RGB. Matplotlib expects RGB, so
figures must convert before displaying — otherwise red prohibition signs render blue, which
is the sort of error that survives until the report is printed. A test pins it: pure red in
BGR must give hue 0.

---

## Decision 1: interpolation is chosen per image

GTSRB crops span 25–266 px against a 48×48 target, so the *same* call both enlarges and
shrinks. Measured over the training set:

| Adaptive choice | Images | Share |
|---|---|---|
| `INTER_AREA` (shrinking) | 15 387 | **39.2 %** |
| `INTER_LINEAR` (enlarging) | 23 822 | **60.8 %** |

Neither branch is a rare case, so any single fixed setting is wrong for a large fraction of
the dataset. `INTER_AREA` averages over the source footprint and avoids aliasing when
shrinking, but degenerates towards nearest-neighbour when enlarging; `INTER_LINEAR` is the
reverse.

**How much does it matter?** Mean absolute difference between the two methods' outputs:

| Subset | Source height | mean \|AREA − LINEAR\| |
|---|---|---|
| 200 smallest crops (upscaled) | 25–26 px | 1.64 gray levels |
| 200 largest crops (downscaled) | 148–225 px | **4.00 gray levels** |

**The 4.00 figure is the important one.** It is comparable in magnitude to σ = 5 Gaussian
noise — the mildest stressor in the entire robustness study. So a fixed interpolation would
inject an artifact as large as the weakest degradation being measured, and, critically,
**that artifact would be correlated with sign size**. It would land directly on figure 9.4
(accuracy vs. sign size) and be misread as a property of the representations rather than of
the resampling. Choosing per image removes the confound at no cost.

## Decision 2: CLAHE runs *after* the resize

At native resolution, a fixed tile grid means something different for every image: 4×4
tiles cover ~6 px each on a 25 px crop and ~66 px each on a 266 px one. The operation would
therefore vary systematically with sign size, confounding the preprocessing ablation
(task 8.2) with the size analysis.

Applied at 48×48 the operation is identical for every sample: 4×4 tiles of 12×12 px.

**Tile grid is (4, 4), not OpenCV's default (8, 8).** At 48×48 the default gives 6×6 px
tiles — small enough that CLAHE amplifies sensor noise into apparent structure. 12×12 px
tiles retain enough context to normalise illumination rather than texture.

**CLAHE on colour input equalises V only.** Three-channel input is treated as HSV and only
the value channel is equalised. Equalising H would rotate hues and destroy exactly the
colour coding that makes traffic signs separable (red prohibition, blue mandatory, yellow
priority). A test asserts H and S are untouched and V is not.

**Why CLAHE is the one preprocessing step with a clear physical motivation here:** GTSRB
tracks are captured while driving through changing light, so the dataset genuinely contains
heavily under- and over-exposed frames of the same physical sign. Local contrast
normalisation addresses a real property of the data rather than being applied by habit.

---

## The three ablation configs (task 2.2)

| Config | Pipeline | Output | Flat dim | What it isolates |
|---|---|---|---|---|
| `raw_gray` | gray → resize | 48×48 | 2 304 | baseline, no normalisation |
| `clahe_gray` | gray → resize → CLAHE | 48×48 | 2 304 | the effect of contrast normalisation alone |
| `clahe_hsv` | HSV → resize → CLAHE(V) | 48×48×3 | 6 912 | the additional value of colour |

The three form a **ladder**, each adding exactly one factor to the previous, so the ablation
attributes any difference to a single change rather than to a bundle of them:
`raw_gray → clahe_gray` isolates contrast normalisation; `clahe_gray → clahe_hsv` isolates
colour.

`clahe_hsv` triples the input dimensionality (2 304 → 6 912), which is itself a result worth
reporting: if colour buys little accuracy, it is a poor trade for 3× the feature dimension
and the corresponding cost in Table 1.

**Note for BoVW (task 6):** dense SIFT is single-channel, so under `clahe_hsv` it must
operate on one channel (V is the natural choice). This means BoVW cannot exploit colour the
way PCA and the CNN can, and that asymmetry has to be stated when reading the ablation table
rather than presented as BoVW simply doing worse.

## Decision 3: GTSRB images keep the framing the dataset provides

`load_and_preprocess()` does **not** crop to the annotated ROI (`roi=False`). Each image is
used as GTSRB frames it — the sign plus its ~17 % margin. Only the pixels are processed
(colour conversion, CLAHE, resize); the box is untouched, and no jitter enters training or
the core grid at all.

**This reverses an earlier decision, and the reversal is worth recording.** The first version
cropped to the tight ROI so that the clean condition would coincide exactly with bbox jitter
at level 0, avoiding a step at zero in figure 9.5. That reasoning was sound but it was
solving a problem that no longer exists: jitter has been moved out of the core grid entirely
and onto full-frame datasets, because GTSRB's pre-cropped images cannot support outward
jitter without inventing pixels (see
[09-jitter-and-datasets.md](09-jitter-and-datasets.md)). With no jitter panel on GTSRB,
there is no level-0 point to align with, and the argument for tight cropping disappears.

Using the provided framing is also how GTSRB is conventionally used, which keeps the
headline accuracy numbers comparable with published results, and it preserves the margin as
genuine context rather than discarding it.

The choice is not cosmetic either way: for the first training image, ROI-crop versus provided
framing changes the 48×48 output by **36.8 gray levels on average**.

`roi=True` is retained deliberately — it is how the GTSDB evaluation will reproduce GTSRB's
framing (annotated box plus an equivalent margin) so that jitter level 0 is a like-for-like
baseline rather than a framing mismatch masquerading as domain shift.

The ROI columns remain in use for one thing regardless: **`roi_h` is the size measure for
the accuracy-vs-size buckets (task 9.4)**. That is metadata, not cropping.

## The cache (task 2.3), and proving it changes nothing

`gtsrb.cache` preprocesses each split once and stores it as uint8 `.npy`:

| | raw_gray | clahe_gray | clahe_hsv |
|---|---|---|---|
| train (39 209) | 86.2 MB | 86.2 MB | 258.5 MB |
| test (12 630) | 27.8 MB | 27.8 MB | 83.3 MB |

A train/val subset costs no extra storage — both index into the one cached array for the
split, selected by path.

### The transparency requirement

A cache that silently altered pixels would corrupt every result in the grid while
presenting as a speedup. So the uncached path is not a fallback — it is the **reference
implementation**, and the cache is checked against it bit-for-bit:

```
  OK   train/raw_gray     300 images compared, 0 mismatched
  OK   train/clahe_gray   300 images compared, 0 mismatched
  OK   train/clahe_hsv    300 images compared, 0 mismatched
  OK   test/raw_gray      300 images compared, 0 mismatched
  OK   test/clahe_gray    300 images compared, 0 mismatched
  OK   test/clahe_hsv     300 images compared, 0 mismatched
```

`verify()` uses `np.array_equal` with **no tolerance parameter**: both paths run the same
function over the same rows in the same order, so any difference at all is a bug rather
than rounding. `poetry run python -m gtsrb.cache --verify` re-runs it, and the equality is
also pinned by tests for all three configs.

Staleness is handled by a manifest per cache file recording a fingerprint of the pipeline
(preproc name, image size, CLAHE clip limit and tile grid, plus a `CACHE_VERSION`) and a
hash of the ordered path list. Any mismatch rebuilds rather than serving stale pixels.

### A bug the staleness test caught

Writing the invalidation test exposed a real flaw: `clahe()` bound `CLAHE_CLIP_LIMIT` as a
**default argument**, frozen at import. The cache fingerprint records that constant to
decide staleness — so the fingerprint could change while the pixels did not, and anything
inspecting the constant would be reading a value the pipeline no longer honoured. The
defaults are now resolved at call time, which makes the fingerprint an honest description
of the pipeline rather than a coincidence.

### What the speedup is actually worth

Loading the full test split (12 630 images), `clahe_gray`:

| | median |
|---|---|
| uncached (decode PPM + convert + CLAHE + resize) | 1.4345 s |
| cached (`.npy` read + row selection) | 0.0172 s |
| **speedup** | **83×** |

**Stated honestly: across the 80-run grid this saves about 2 minutes** (1.9 min → 1.4 s),
which is real but not decisive on its own. The value is mostly elsewhere:

- **Development iteration.** The hyperparameter sweeps (4.2 PCA components, 5.2 HOG cell
  sizes, 6.3 vocabulary sizes) reload the *training* set — three times larger — repeatedly.
  That is where the minutes accumulate.
- **Determinism by construction.** Every run in the grid reads the same bytes, so no
  variation in the input pixels is even possible between methods.

## Note on the two blurs

`gaussian_blur` here is *preprocessing* (mild denoising, part of the input pipeline). The
motion blur in task 3.2 is a *stressor* (a degradation applied at test time to measure
robustness). They are different operations serving opposite purposes and must not be
conflated in the report — one is part of the method, the other is part of the measurement.
