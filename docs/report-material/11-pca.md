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

### 2.1 `whiten=False`

`sklearn`'s PCA can rescale every retained component to unit variance. It is **not** used
here.

**The reason is fidelity to the construction.** "PCA representation" in the Eigenfaces sense
is an orthogonal projection onto the leading subspace. Whitening composes that projection
with a diagonal rescaling, so the coordinates the classifier sees would no longer be the
projections that the eigenimages of task 4.4 depict — the figure and the features would
describe different things. This reason stands on its own and does not depend on any expected
result.

**A fact about the operator, recorded separately from the choice.** Whitening divides each
component by its own standard deviation, which amplifies the low-variance retained
directions — the ones where an isotropic perturbation has the largest share relative to
signal. Whitened PCA is therefore plausibly a *differently robust representation*, not
merely a rescaled one.

> **This was initially written down backwards, and the correction matters more than the
> conclusion.** The first draft of this note gave "it would suppress the mechanism the noise
> prediction rests on" as a *reason* for `whiten=False`. That is choosing a configuration to
> protect a prediction, which is exactly the reasoning this project must not use. The
> conclusion is unchanged — `whiten=False` is still right, on the fidelity argument alone —
> but the second consideration is a **reason to measure whitening as a separate condition**,
> never a reason to avoid it. If whitened PCA is run and proves more robust, that is a
> finding to report.

`whiten=True` is supported and tested (`test_whitening_equalises_the_variances`), simply not
what the headline results use.

**Reportable as a limitation:** results are for unwhitened PCA. A whitened variant might
behave differently, particularly under noise, and that is not measured here.

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

This has a direct consequence for the preprocessing ablation (8.2 / 9.7): at a fixed *k*,
PCA retains less *variance* under CLAHE than without it.

> **Checked, and the obvious inference is wrong.** The first version of this note predicted
> that `clahe_gray` would therefore *underperform* `raw_gray` for PCA. A diagnostic sweep
> (`LinearSVC`, C = 1, val accuracy) says the opposite at every *k*:
>
> | *k* | `raw_gray` | `clahe_gray` |
> |---|---|---|
> | 32 | 0.4696 | **0.5594** |
> | 128 | 0.7756 | **0.8013** |
> | 256 | 0.8095 | **0.8259** |
>
> **Retained variance is not retained class information.** The variance CLAHE removes is
> illumination, which carries no class signal, so discarding it *improves* the information
> density of the remaining components even though the cumulative-variance curve looks worse.
> The scree curve is a statement about reconstruction, not about discriminability — a useful
> caution for reading the 4.2 figure, and the reason 4.2 selects *k* on validation accuracy
> rather than on a variance threshold.
>
> *(Provisional: diagnostic probe, not harness-recorded. Tasks 4.2 / 8.2 produce the
> reportable numbers.)*

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

## 5. Demo figures

`scripts/demo/pca_mechanics.py` → `figures/demo/pca/`. These explain the *machinery*; the
eigensigns figure of task 4.4 is the polished report artifact.

| Figure | Shows |
|---|---|
| `pca_reconstruction.png` | four signs rebuilt from *k* = 2 … 256, annotated with the split-wide error |
| `pca_spectrum.png` | scree + cumulative variance for all three preproc configs; the 8 → 25 → 106 finding |
| `pca_pc1_brightness.png` | PC1 vs mean intensity (r = +0.9975) + the training set sorted by PC1 |
| `pca_degraded.png` | degraded input through the clean basis: reconstruction, residual, feature shift |

### 5.1 What the reconstruction ladder shows that the numbers do not

At *k* = 2 and *k* = 8 **every sign reconstructs toward a round speed-limit disc with
digit-like blobs** — a Yield triangle included. The leading components encode the dataset's
modal shape, and with 10.7× class imbalance (note 02) that shape is a speed-limit sign.
Useful for the report: it makes "holistic and alignment-critical" visible, and it is the
clearest illustration of why PCA has no notion of a *part*.

Also visible, and worth a sentence in the limitations: **PCA reconstructs the background
too** — brickwork, foliage, the pole. GTSRB crops carry a ~17 % margin (note 09), so a
meaningful share of the retained variance is spent modelling scenery that carries no class
information. HOG and BoVW see the same pixels, but PCA is the method that spends *variance
budget* on them.

### 5.2 `pca_degraded.png` — read the two numbers correctly

`|res|` is what the subspace cannot express; `shift` is how far the 128-d feature vector
moved, in units of the median distance between two random clean training images.

| condition | \|res\| (gray levels) | shift |
|---|---|---|
| noise σ=40 | 34.7 / 36.4 | **0.10 / 0.11** |
| blur k=15 | 6.6 / 6.7 | 0.36 / 0.37 |
| gamma γ=2.5 | 18.2 / 15.5 | **0.65 / 0.72** |

**A small residual is not good news.** Blur has the *smallest* residual of the three — the
blurred image sits almost entirely inside the subspace, so the projection reproduces the
damage faithfully and passes it on. Noise has the largest residual and the smallest shift:
the subspace rejects it.

**Two cautions, recorded before task 9.5 runs:**

1. `shift` is not accuracy. What costs accuracy is displacement *relative to the decision
   boundaries*, and a displacement that is **common-mode** — the same direction for every
   class — can be largely absorbed by a linear classifier. Gamma's shift is exactly the kind
   most likely to be common-mode (everything darkens together), so the largest shift here
   need not produce the largest accuracy drop.
2. This diagnostic orders the three stressors the same way `predictions.md` does for PCA.
   That is **not** confirmation — it is one mechanism measurement on two images through an
   operator whose behaviour under isotropic perturbation is analytically predictable. It
   would be circular to treat it as evidence for the prediction it was designed to
   visualise. The decomposition that *would* be evidence — splitting the shift into
   common-mode and class-confusing parts — is not built; noted as an option if the 9.5
   curves need explaining.

---

## 6. Risks ahead — checked, not assumed

### 6.1 Resolved: unscaled PCA features into `LinearSVC`

The anticipated problem: unwhitened PCA features have wildly unequal scales, while
`LinearSVC` applies one L2 penalty to every weight. Measured feature spread at *k* = 128,
`clahe_gray`: PC1 σ = 8.03, PC128 σ = 0.27 — a **29× spread in σ, 865× in variance**. The
concern was distorted regularisation, slow convergence, and a distorted `C` sweep at 4.2.

Measured instead of argued (`C = 1.0`, train 31,379 / val 7,830):

| features | val accuracy | fit time | max n_iter | convergence |
|---|---|---|---|---|
| unscaled | **0.8013** | 11.6 s | 28 | converged |
| standardised | 0.8004 | 5.8 s | 57 | converged |

**A 0.09 pp difference — the risk is not real at this scale.** Both converge well inside
`max_iter`. Unscaled is ~2× slower to fit but needs fewer iterations. No feature-scaling
step is warranted, which also avoids the awkwardness of standardising PCA scores (which is
whitening under another name) after deciding not to whiten.

*(Provisional numbers from a diagnostic probe, not a recorded result — task 4.3 produces the
reportable figures through the timing harness.)*

### 6.2 Decided (Q4): `C` is tuned per method

PCA is the only representation whose features are **not** internally normalised — HOG
block-normalises, SIFT/BoVW normalise descriptors and histogram, the CNN has batch norm. A
single `LinearSVC` with one `C` therefore meets PCA on a different footing than the others.
§6.1 shows scaling barely matters *for PCA in isolation*; it does not show that one `C`
suits all five.

**Decision: tune `C` per method on validation, and record the chosen value per method in
`results.csv`.** "Fixed classifier" means the same estimator and the same training protocol
— not the same nuisance hyperparameter. Freezing `C` would hand every representation a dial
calibrated for another's feature scale, and a difference produced that way would be an
artifact of the dial rather than a property of φ, which is the one thing this study must not
confuse. The report's protocol section states this explicitly, and the per-method `C` values
are reported alongside Table 1 so the choice is auditable.

Applies to tasks **4.3, 5.3, 6.5, 7.5** and the grid at **8.1**.

### 6.3 Decided (Q5): the pairing guard lives in the task 8 runner

`PCARepresentation.preproc` is metadata. `transform` cannot detect being handed `raw_gray`
images when fitted on `clahe_gray` — same 2304 dims, so sklearn's feature-count check passes
and the result is silently wrong. The gray↔`clahe_hsv` mismatch *is* caught (2304 vs 6912).
This is a driver-level hazard for the 8.2 ablation, which loops over all three configs.

**Decision: guard in the task 8 runner, not in the representations.** The runner constructs
`(representation, images)` as one paired unit — a representation is never obtainable without
the cache it was fitted from — and a test asserts the pairing. This catches the mistake at
the single place it can realistically occur, instead of threading a preproc name through
five separate `transform` implementations to defend against a call that nothing in the
codebase makes.

### 6.4 Not a risk

- **Degraded input through a clean basis** — works unchanged; `degradations.apply` returns
  uint8 of the same shape, which `as_matrix` consumes directly.
- **Cost** — fit ≈ 2.7 s, SVM ≈ 12 s. PCA will be the cheapest row in Table 1.
- **`dual`** — n_samples (31,379) ≫ n_features (≤ 256), so liblinear's primal solver is the
  right one; sklearn 1.9's `dual="auto"` already selects it.

---

## 7. Open for task 4.2

- Choose *k* from {32, 64, 128, 256} on validation accuracy, not on explained variance —
  the table above says 95 % of variance needs 155 components under `clahe_gray`, but
  variance is not the objective; class separability is.
- **Note for the implementation:** the top-*k* components of a 256-component randomized fit
  are *not* bit-identical to a fresh *k*-component fit, because the random projection
  depends on the requested rank. Sweeping by slicing one large fit is therefore cheaper but
  not equivalent; fit each *k* separately (≈3 s each — the saving is not worth the caveat).
- The scree / cumulative-variance figure belongs with that decision, not here.
