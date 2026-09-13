# 11 — PCA: the holistic representation (task 4.1)

*Feeds: Methodology → representations; Table 1 (9.1); eigensigns figure (4.4); discussion
of the gamma panel (9.5).*

Implementation: `src/gtsrb/representations/pca.py`. Tests: `tests/test_pca.py` (31).
Interface: `src/gtsrb/representations/__init__.py`.

---

## 1. What the representation is

The Eigenfaces construction applied to traffic signs *(eigenlica → „sopstveni znakovi")*.
Each preprocessed 48×48 image is a point in **R²³⁰⁴**; PCA finds the orthonormal directions
of greatest variance across the training set and keeps the leading *k*. A sign is then
represented by its *k* coordinates in that basis.

It anchors one end of the comparison, being the only representation that is

- **holistic** — every output coordinate depends on every input pixel, so there is no such
  thing as a local feature, and
- **alignment-critical** — the basis is learned in absolute pixel coordinates, so a shifted
  sign is a different point, not the same point described differently.

Those two properties are what the four-way structural table in `PROJECT_TASKS.md` §1
contrasts BoVW and HOG against.

---

## 2. Decisions, with reasons

### 2.1 `whiten=False` — the load-bearing one

`sklearn`'s PCA can rescale every retained component to unit variance. It is **not** used
here. Two reasons; the second is why it matters.

**(a) Fidelity to the construction.** "PCA representation" in the Eigenfaces sense is an
orthogonal projection onto the leading subspace. Whitening composes that projection with a
diagonal rescaling, so the coordinates the classifier sees would no longer be the
projections that the eigenimages of task 4.4 depict. The figure and the features would
describe different things.

**(b) It would suppress the effect under test.** The recorded prediction
(`predictions.md`, locked 2026-09-13) is that **PCA is the most robust representation under
additive noise**, on this argument:

> isotropic noise distributes its energy evenly over all 2304 dimensions, but only *k* are
> retained, so most of it falls outside the subspace and is discarded.

That argument requires the retained components to stay weighted by their variance. Whitening
divides each component by its own standard deviation, which **amplifies precisely the
low-variance retained directions** — the ones carrying the least signal and the largest
share of noise. Choosing `whiten=True` would therefore weaken the predicted mechanism as a
side effect of a setting that looks like routine preprocessing.

Whitening is a *hypothesis about the mechanism*, not a free knob. Baking it in would
prejudge the result the study is trying to measure. It remains available as `whiten=True`
and is a legitimate ablation if the noise curve turns out interesting — it is tested
(`test_whitening_equalises_the_variances`), just not used.

**Reportable as a limitation:** results are for unwhitened PCA. A whitened variant might
score differently, particularly under noise, and that is not measured here.

### 2.2 Fitted on the training split only

31,379 images; never train+val, never test. This is easy to get wrong for PCA specifically,
because it uses no labels — which makes fitting it on everything *feel* harmless. It is not:
both the mean and the basis would carry information about rows the model is later scored on.
`pca.fit_on_train()` is the single entry point that enforces it, so no caller has to
remember.

### 2.3 Fitted on clean images only

Degradations are a **test-time** stressor. A degraded image is projected through the basis
learned from clean data; the representation never adapts to the stressor. Re-fitting per
condition would answer a different question — "how well can this method be *retrained* for
a stressor" — and would make the robustness curves incomparable with the CNN's, which is
likewise trained once.

### 2.4 `svd_solver='randomized'`, `random_state=config.SEED`

A full SVD of a 31,379 × 2304 matrix computes all 2304 singular directions to obtain the
256 that are wanted. The randomized solver (Halko et al.) is **stochastic**, so the seed is
what makes two runs comparable at all; it is pinned to the project seed like everything
else. Measured effect of the seed: the leading components are reproduced to ~1e-4 relative,
the trailing ones only to ~1e-3 — a reason not to over-read the tail of the scree curve at
task 4.2.

---

## 3. Measured

Training split, 31,379 images, fit at *k* = 256. "Reconstruction error" is mean absolute
error in gray levels (0–255), deliberately on the same scale as the degradation magnitudes
of task 3.x — "PCA discards 4.8 levels" is directly comparable with "σ = 5 noise".

| preproc | input dim | *k* for 80 % | 90 % | 95 % | variance at *k*=256 | recon err (train / val) | fit time |
|---|---|---|---|---|---|---|---|
| `raw_gray` | 2304 | **8** | 31 | 81 | 0.985 | 4.78 / 4.86 | 1.6 s |
| `clahe_gray` | 2304 | **25** | 73 | 155 | 0.972 | 7.36 / 7.41 | 2.7 s |
| `clahe_hsv` | 6912 | **106** | > 256 | > 256 | 0.870 | 13.96 / 14.21 | 9.0 s |

Two things to take from this table.

**Train and val reconstruction errors agree to within 0.1 gray levels.** The basis is not
memorising the training rows — it generalises to held-out *tracks*, which is the stronger
statement given the split is track-disjoint. Worth stating in the report: it means the
subspace captures sign structure rather than the specific frames.

**Cost is negligible.** Under 3 seconds for the grayscale configs. PCA will be the cheapest
entry in the Table 1 cost column by a wide margin, and its inference cost is a single
2304 × k matrix multiply.

### 3.1 Finding — the leading direction is brightness, not sign identity

Measured on `raw_gray`:

| | value |
|---|---|
| variance in PC1 alone | **52.3 %** |
| correlation of the PC1 score with per-image mean brightness | **+0.9975** |
| variance in PC1 after CLAHE | 39.8 % (correlation still +0.99) |

So **over half of the variance PCA is asked to model on raw grayscale is illumination**, a
nuisance factor that carries no class information at all. Locked in as a test
(`test_real_signs_are_dominated_by_one_direction`) because the gamma panel at task 9.5 is
read through it.

**What it implies for the gamma prediction — genuinely two-sided, recorded now so the
reading is not chosen after seeing the answer:**

- *Supporting:* a gamma shift is a global intensity remap, so it displaces every test point
  along exactly the direction PCA spends most of its budget on. The training mean subtracted
  during projection is the clean mean, and gamma moves the whole test distribution away from
  it.
- *Against:* if PC1 is non-discriminative, `LinearSVC` will have learned to give it little
  weight, and a large displacement along an ignored coordinate costs nothing. On that
  reading PCA could prove *more* gamma-robust than predicted.

The two readings differ in what they expect from the **non-linearity** of the gamma curve:
a purely linear brightness scale would move points along PC1 only, while a gamma curve also
warps the relationships the lower components encode. Which dominates is exactly what task
9.5 measures. **Do not resolve this here.**

### 3.2 Finding — CLAHE makes the data harder to compress

The 80 %-variance column rises monotonically along the preprocessing ladder: **8 → 25 → 106**
components. Each added factor raises the intrinsic dimensionality of the dataset.

The mechanism is visible in §3.1: CLAHE's job is to normalise away global illumination, and
illumination is exactly what the dominant direction encoded. Removing it does not reduce the
data's complexity — it removes the one cheap direction that was absorbing half the variance,
leaving the remainder spread more evenly.

This has a direct consequence for the preprocessing ablation (8.2 / 9.7): **at a fixed *k*,
PCA retains less of the signal under CLAHE than without it.** If `clahe_gray` underperforms
`raw_gray` for PCA, that is a candidate explanation, and it is a *PCA-specific* effect — HOG
and BoVW are contrast-normalised internally and should not show it. Worth checking whether
the ablation bears this out, since it would be a clean example of a preprocessing choice
interacting with one representation and not the others.

### 3.3 Gotcha — memory layout changed the projections

`PCA.fit` leaves `components_` **F-contiguous**; a `joblib` save/load round-trip restores it
**C-contiguous**. The values are bit-for-bit identical, but BLAS selects a different GEMM
kernel for a different layout, which sums in a different order — so a reloaded model's
projections differed from the in-memory model's by up to **9.5e-7 absolute** on scores of
scale ~6.

Nothing is *wrong* with either result, but "the model changed when I reloaded it" is a
miserable thing to debug during the 80-run grid. `fit` now normalises the layout with
`np.ascontiguousarray`, making save/load exact, and `test_save_load_round_trip` is the
regression test.

The same cause survives in one place that **cannot** be fixed: a float32 GEMM blocks
according to its number of rows, so projecting a 10-image subset and projecting the full
batch disagree in the last bits (measured: same 1e-6 absolute, ~1e-7 relative). That is
orders of magnitude below any margin `LinearSVC` decides on — for a prediction to flip, an
image would have to sit within 1e-6 of a decision boundary — so it is documented rather than
engineered around. `test_transform_is_independent_of_batch_composition` asserts the bound
rather than exact equality.

---

## 4. Interface note

`src/gtsrb/representations/__init__.py` defines the `Representation` protocol that tasks
5–7 will also implement: `fit(images)` on clean training images, `transform(images)` to
`(n, n_features)` float32, plus `name` and `n_features`. Holding it fixed is what lets task
8.1 loop over all five methods rather than special-casing each, and it is the code-level
expression of the experimental design — the classifier is constant, only φ varies.

---

## 5. Open for task 4.2

- Choose *k* from {32, 64, 128, 256} on validation accuracy, not on explained variance —
  the table above says 95 % of variance needs 155 components under `clahe_gray`, but
  variance is not the objective; class separability is.
- **Note for the implementation:** the top-*k* components of a 256-component randomized fit
  are *not* bit-identical to a fresh *k*-component fit, because the random projection
  depends on the requested rank. Sweeping by slicing one large fit is therefore cheaper but
  not equivalent; fit each *k* separately (≈3 s each — the saving is not worth the caveat).
- The scree / cumulative-variance figure belongs with that decision, not here.
