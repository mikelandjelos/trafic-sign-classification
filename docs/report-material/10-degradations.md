# 10 — Controlled degradations

**Feeds:** Methodology → controlled degradations; the robustness curves (fig. 9.5); the
contact sheet (3.5).
**Status:** complete (tasks 3.1, 3.2, 3.3, 3.5).

Implemented as `gtsrb.degradations`; tested in `tests/test_degradations.py`.

---

## The decision that governs all three stressors

**Degradations are applied to the preprocessed 48×48 model input, not to the source
image.** This trades physical realism for experimental control, and the report should say
so explicitly rather than let a reader assume a camera simulation.

Real sensor noise enters at capture, before any resizing. A faithful simulation would
therefore perturb the source crop and then downsample. The problem is that GTSRB crops
span **25–266 px** against a 48×48 target, so the same operation means different things
across the dataset: downsampling a 200 px crop averages noise away, while a 25 px crop is
*upsampled* and retains all of it. "σ = 20" would then denote a different effective noise
level depending on sign size.

Applying to the model input instead buys four properties the study needs:

| Property | Why it matters |
|---|---|
| One meaning per level, dataset-wide | σ = 20 is a single well-defined condition, not an average over sign sizes |
| Independent of sign size | keeps the robustness curves (9.5) and the size analysis (9.4) separable rather than confounded |
| Independent of preprocessing config | the ablation (8.2) does not interact with the degradation axis |
| Identical pixels for all five methods | the point of the entire controlled design |

There is a practical benefit too: degrading cached 48×48 arrays keeps the whole grid fast,
whereas degrading at source resolution would defeat the cache and re-decode every PPM for
every condition.

**The honest framing for the report:** these are *controlled perturbations of the
classifier's input*, not simulations of a camera. That is the right instrument for the
question being asked — which representation degrades fastest — but it is not evidence
about any specific sensor.

## Determinism: keyed by image path, not row position

Each image's perturbation is drawn from `config.rng_for(degradation, level, path)`.

Using the **path** rather than the row index is what makes the guarantee hold in practice:
subsetting the test set, reordering it, or evaluating a different number of methods cannot
change the pixels any given image receives. Pinned by a test that reproduces a batch from
an arbitrary subset and requires it to match the full batch exactly, and by one that
perturbs the global numpy seed between calls and requires no change.

The identity level is returned **unchanged** rather than recomputed, so the clean condition
and the identity level are the same array, not merely similar ones. Which level that is
varies: 0 for noise and blur, **1.0 for gamma** — see the structural point in §3.3.

---

## 3.1 Gaussian noise — σ ∈ {0, 5, 10, 20, 40}

Zero-mean additive Gaussian noise in 0–255 units, clipped to range. Measured over 1 000
test images (`clahe_gray`):

| σ | realised std | mean shift | saturated px |
|---|---|---|---|
| 0 | 0.00 | +0.000 | 3.16 % |
| 5 | 4.95 | −0.073 | 2.14 % |
| 10 | 9.86 | −0.156 | 2.66 % |
| 20 | 19.34 | −0.235 | 5.14 % |
| 40 | 36.57 | +0.482 | 12.18 % |

Two things in that table are worth a sentence each in the report:

- **The realised std falls below σ at high levels** (36.57 against 40). That is clipping,
  and it is correct behaviour rather than a defect: at σ = 40 a bright sign face genuinely
  saturates, which is what an over-exposed sensor does. A test pins it as expected.
- **3.16 % of pixels are already saturated at σ = 0.** That is CLAHE, not noise — local
  contrast normalisation pushes pixels to the extremes by design. Worth knowing before
  attributing all saturation to the degradation. (The dip to 2.14 % at σ = 5 is noise
  nudging already-saturated pixels back inside the range.)

### A bug the tests caught: truncation instead of rounding

The first implementation ended with `np.clip(noisy, 0, 255).astype(np.uint8)`.
`astype(np.uint8)` **truncates toward zero**, so every perturbed pixel lost its fractional
part:

```
truncate: mean -0.4886   std 20.025    <- systematic darkening
round   : mean +0.0109   std 20.026
```

A **−0.49 level systematic darkening at every σ**, applied on top of noise that is supposed
to be zero-mean. Small in absolute terms, but it is a brightness shift injected by the
degradation itself and applied to every image at every level — exactly the sort of
confound the study exists to avoid, since a representation sensitive to mean intensity
(PCA on raw pixels, most obviously) would have been measured against a slightly different
condition than intended.

Fixed with `np.rint()` before the cast. The test that caught it asserted zero-mean noise;
a second test now pins the absence of the shift directly.

Worth noting in the discussion: the property under test was *the statistical behaviour the
degradation claims to have*, not the code path. Asserting "noise is zero-mean" is what
surfaced a defect that no shape or dtype check would have found.

---

## 3.2 Motion blur — kernel ∈ {0, 3, 5, 9, 15} px, random angle

Convolution with a normalised line kernel at a uniformly random orientation, simulating
camera/vehicle motion during exposure. Measured over 1 000 test images (`clahe_gray`):

| kernel | mean shift | std ratio | edge energy retained |
|---|---|---|---|
| 0 | +0.000 | 1.000 | 100.0 % |
| 3 | −0.006 | 0.965 | 81.2 % |
| 5 | −0.008 | 0.936 | 68.0 % |
| 9 | −0.001 | 0.886 | 52.9 % |
| 15 | −0.006 | 0.834 | 42.7 % |

Edge energy (mean absolute horizontal gradient) falls monotonically and more than halves by
the strongest level, while mean intensity is preserved to within 0.01 levels.

### Three choices that keep the stressor clean

**The kernel is normalised to sum 1.** Blur must not also darken or brighten the image —
brightness is what gamma (3.3) measures, and a blur that shifted intensity would confound
two axes of the grid. Pinned by a test across every size × angle combination.

**Borders use `BORDER_REFLECT_101`, not zero padding.** Zero padding would darken every
edge in proportion to the kernel size: a vignette that *grows with the degradation level*
and would be measured as part of the blur's effect. A test blurs a constant image at k = 15
and requires it to come back unchanged, borders included.

**The angle is sampled over [0, 180), not [0, 360).** A line kernel at θ and θ + 180 is the
same kernel, so the wider range would merely sample each orientation twice. Drawn per image
from `rng_for("blur", k, path)`, so it is reproducible and varies between images — a single
shared angle would make the stressor systematic rather than random.

### Known property: pixel count varies with angle, extent does not

A 15 px kernel rasterises to 15 nonzero pixels at 0° but 11 at 45°. That is correct rather
than a defect: the diagonal line spans the same *geometric* extent (≈14 px in both cases)
because diagonal steps cover more distance. The blur therefore averages over the same
displacement at every angle, but over slightly fewer samples on the diagonals. Worth knowing
rather than worth fixing.

### Severity of the strongest level, in context

k = 15 on a 48×48 image is a blur across **31 % of the image width** — severe, and
deliberately so; the top level should be strong enough to separate the representations. It
is also a direct consequence of the "degrade the model input" decision: at source
resolution, 15 px would be mild on a 200 px crop and destructive on a 25 px one. Applying
it after the resize is what makes "k = 15" one condition rather than a family of them.

---

## 3.3 Gamma — γ ∈ {0.4, 0.7, 1.0, 1.5, 2.5}

`out = 255 · (in/255)^γ`. γ < 1 brightens and lifts shadows, γ > 1 darkens, γ = 1 is the
identity. Measured over 1 000 test images (`clahe_gray`, mean 99.4):

| γ | direction | mean after | pixels at 0 | pixels at 255 |
|---|---|---|---|---|
| 0.4 | brightens | 165.2 | 0.00 % | 3.24 % |
| 0.7 | brightens | 125.6 | 0.00 % | 3.16 % |
| 1.0 | identity | 99.4 | 0.00 % | 3.16 % |
| 1.5 | darkens | 72.5 | 0.00 % | 3.16 % |
| 2.5 | darkens | 47.2 | **3.71 %** | 3.16 % |

Mean intensity moves monotonically across the grid, which is what makes the robustness
curve readable. The 3.16 % at 255 is the pre-existing CLAHE saturation (note 08), not an
effect of gamma — only γ = 0.4 raises it, and only slightly.

**At γ = 2.5, 3.71 % of pixels are crushed to zero.** That is genuine information loss, not
a reversible transform: darkening then re-brightening does not recover the original. A
round-trip (γ = 2.5 followed by γ = 0.4) leaves a mean absolute error of **1.42 levels**.

### It is the one degradation with no randomness

Gamma is a fixed function of the pixel value, so it takes no random draw at all. `rng` is
accepted for interface uniformity and ignored, and the registry records
`stochastic=False`. Two consequences worth noting:

- The "all five methods see identical pixels" guarantee is *trivially* satisfied here,
  rather than relying on the path-keyed seeding that noise and blur need.
- A test asserts two different images receive the same mapping, which is the correct
  behaviour for gamma and would be a bug for either of the others.

Implemented as a **256-entry lookup table**: the transform has only 256 possible inputs, so
the table is exact, and it is memoised because it depends on γ alone — rebuilding it per
image would dominate the cost of an otherwise trivial operation. The LUT is monotone and
fixes the endpoints (0 → 0, 255 → 255), so intensity ordering is preserved and pure black
and pure white are unmoved.

### A structural point: the identity level is not `levels[0]`

Gamma perturbs in **two directions**, so its identity sits in the *middle* of its range at
γ = 1.0, while noise and blur have theirs at 0, first in the list.

The original implementation assumed `levels[0]` was the no-op — correct for the first two
degradations and silently wrong for gamma, where it would have treated γ = 0.4 as "clean"
and γ = 1.0 as a perturbation. Every robustness curve for gamma would then have been
plotted against the wrong baseline.

The registry now makes each degradation declare its own `identity`, exposed as
`identity_for(name)`, and a test pins that `identity_for("gamma") != levels_for("gamma")[0]`.
This matters beyond gamma: task 9.5 normalises each curve to its own baseline, and that
code must ask which level is the baseline rather than assume position.

---

## 3.5 Contact sheet

`docs/demo/degradation_contact_sheet.py` → `figures/demo/degradation_contact_sheet.png`.
Three rows (noise, blur, gamma) × five levels on one sign, with the identity level boxed in
green.

It shows the **48×48 preprocessed model input**, degraded by `gtsrb.degradations` itself —
same function, same path-keyed seed as evaluation — so it is the actual classifier input,
not an illustration of it.

**Sample selection, and a mistake worth recording.** The first version ranked candidates by
*edge energy*, reasoning that a sharp image shows degradation best. It selected a sign in
front of a fence: the background clutter dominated every panel and made the degradation
harder to read, not easier. Ranking by global contrast and *penalising* high-frequency
clutter picks a clean sign against a plain background, which is what the figure needs.

### What the figure makes visible that a table cannot

- **Gamma preserves spatial structure; blur destroys it.** Gamma is a monotone point
  transform, so edges stay exactly where they are and only their contrast changes. Motion
  blur moves image energy across pixels. This predicts a genuine split in the results:
  gradient-orientation methods with block normalisation (HOG) should be far more robust to
  gamma than to blur, while a holistic intensity method (PCA) should be hurt by gamma's
  global intensity shift.
- **At γ = 2.5 the sign is close to black**, which is the 3.71 % zero-clipping from §3.3
  made visible.
- **At k = 15 the blur direction is plainly diagonal**, confirming the random-angle draw is
  doing what it should, and the sign is barely legible — the strongest level is genuinely
  strong.
- **At σ = 40 the sign remains recognisable to a human** while being heavily corrupted,
  which is the useful regime for separating methods: a level that destroyed the sign
  entirely would push every method to chance and discriminate nothing.
