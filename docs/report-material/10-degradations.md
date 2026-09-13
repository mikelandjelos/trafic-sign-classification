# 10 — Controlled degradations

**Feeds:** Methodology → controlled degradations; the robustness curves (fig. 9.5); the
contact sheet (3.5).
**Status:** in progress — Gaussian noise (3.1) and motion blur (3.2) done; gamma (3.3) and
contact sheet (3.5) pending.

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

Level 0 is returned **unchanged** rather than recomputed, so the clean condition and level
0 are the same array, not merely similar ones.

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

### The severity of the strongest level, in context

k = 15 on a 48×48 image is a blur across **31 % of the image width** — severe, and
deliberately so; the top level should be strong enough to separate the representations. It
is also a direct consequence of the "degrade the model input" decision: at source
resolution, 15 px would be mild on a 200 px crop and destructive on a 25 px one. Applying
it after the resize is what makes "k = 15" one condition rather than a family of them.
