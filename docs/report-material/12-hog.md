# 12 — HOG: the rigid-grid, locally-normalised representation (task 5.1)

*Feeds: Methodology → representations; Table 1 (9.1); HOG figure (5.4); discussion of the
noise and gamma panels (9.5).*

Implementation: `src/gtsrb/representations/hog.py`. Tests: `tests/test_hog.py` (28).

---

## 1. What the representation is

The image is divided into a fixed grid of **cells**; inside each cell the gradient
orientations are histogrammed; overlapping **blocks** of cells are then normalised together.
A sign becomes the concatenation of those normalised histograms.

It sits opposite PCA on two of the project's structural axes:

- **local** — each output coordinate depends on one small patch, not on every pixel;
- **layout-preserving** — cell *(i, j)* always describes the same region of the image, which
  is precisely what BoVW throws away.

`skimage.feature.hog`, not `cv2.HOGDescriptor`: OpenCV's defaults are built for 64×128
pedestrian windows and are awkward to reconfigure for a 48×48 square crop.

---

## 2. Decisions, with reasons

### 2.1 HOG learns nothing — `fit` is a no-op

There is no basis, no codebook, no weights. The descriptor is a fixed function of the pixels;
`fit` exists only to satisfy the shared `Representation` interface and to record the output
width. Two consequences belong in Table 1 rather than in a footnote:

- **training cost is only the `LinearSVC` fit** — PCA also pays for an SVD, BoVW for k-means,
  the CNN for everything;
- **the "model" is only the SVM coefficients** — where PCA carries a 256×2304 basis that is
  96 % of its 2.35 MB (note 06).

Pinned by `test_fit_is_a_no_op`: fitting on two disjoint halves of the data must produce
byte-identical descriptors. If `fit` ever started learning something, the cost column would
quietly stop meaning what the report says it means.

`fit` still takes images, because the output width depends on the input shape — deriving it
by descriptor arithmetic would be a second implementation of skimage's blocking rules, one
that could disagree with it silently. Measuring one image is cheaper and cannot drift.

### 2.2 `block_norm="L2-Hys"` — the mechanism the gamma prediction rests on

Each block is L2-normalised, clipped at 0.2, renormalised (Dalal–Triggs). A contrast change
scales every gradient in a block by the same factor, and L2 normalisation divides that factor
straight back out. **This is not incidental — it is the entire reason HOG is predicted to be
the gamma-robust method.** It is therefore pinned by a test rather than left as a comment: if
`block_norm` were ever changed to a variant without L2 normalisation, the mechanism would
disappear and every other test would still pass.

### 2.3 `cells_per_block=(2, 2)`, against skimage's default of (3, 3)

At 48×48 with 8 px cells the grid is only 6×6 cells; 3×3 blocks would leave 4×4 block
positions and over-smooth an already small grid. (2, 2) is also the Dalal–Triggs
recommendation. Not swept at task 5.2 — which varies cell size and orientation count — so it
is fixed and stated rather than silently chosen.

### 2.4 `transform_sqrt=False`

skimage's default, and Dalal–Triggs reported little benefit from gamma compression for their
detector. Worth naming explicitly because it is **literally a gamma transform** (sqrt is
γ = 0.5) applied to the input before gradients are taken, so switching it on would interact
directly with the task 3.3 stressor. It is exposed as a parameter and left off: a candidate
ablation, **not** a default chosen to flatter a predicted result.

---

## 3. Measured

Output width and cost for the four configurations task 5.2 will sweep:

| pixels/cell | orientations | cell grid | feature dim | ms/img | full train split |
|---|---|---|---|---|---|
| (8, 8) | 9 | 6×6 | **900** | 0.53 | 17 s |
| (8, 8) | 12 | 6×6 | 1200 | 0.56 | 17 s |
| (6, 6) | 9 | 8×8 | 1764 | 0.82 | 26 s |
| (6, 6) | 12 | 8×8 | 2352 | 0.83 | 26 s |

All four dimensions are pinned in tests, so a skimage upgrade that changed the blocking rules
would fail loudly rather than silently shift the feature space.

### 3.1 Verified — a linear contrast change is cancelled *exactly*

Descriptors are **bit-identical** for uint8 input in [0, 255] and float input in [0, 1], and
a 0.5× contrast reduction moves the descriptor by **0.000** in the units of §3.2 below. Not
"approximately": L2 normalisation removes a global gradient scaling completely.

That is the clean, unarguable half of the gamma story.

### 3.2 Measured — but gamma is *not* a linear scale, and is only partly cancelled

Median descriptor displacement under each stressor, in units of the median distance between
two random clean descriptors (the same diagnostic used for PCA in `11-pca.md` §5.2), over 400
training images:

| condition | HOG | PCA |
|---|---|---|
| linear contrast ×0.5 | **0.000** | (not measured) |
| noise σ=40 | **0.988** | 0.10 |
| blur k=15 | **1.242** | 0.36 |
| gamma γ=2.5 | **0.358** | 0.65 |

**Read these with three caveats, all of which matter:**

1. **Displacement is not accuracy.** What costs accuracy is movement *relative to the
   decision boundaries*; a displacement shared by every class can be largely absorbed by a
   linear classifier. Task 9.5 measures accuracy; this measures geometry.
2. **The two columns are not strictly commensurable.** Each is normalised by its own space's
   median inter-image distance, which makes both dimensionless but does not make them the
   same ruler. HOG's 0.988 and PCA's 0.10 are measured in different geometries. The
   *within-column* ordering is the trustworthy part.
3. **This diagnostic was designed with the predictions already written down.** It orders the
   stressors the way `predictions.md` does for both methods, and it would be circular to
   present that as confirmation. It is a mechanism measurement whose value is that it makes
   the 9.5 result *interpretable*, not that it anticipates it.

With that said, the within-column reading is worth recording now: **blur displaces HOG more
than noise does** (1.242 vs 0.988), and a blur displacement above 1.0 means a blurred sign's
descriptor is further from its own clean self than two unrelated signs are from each other.
Gamma is the mildest of the three for HOG, at roughly a third of blur's effect — consistent
with §2.2, and consistent with the partial-cancellation test.

`test_gamma_is_only_partially_cancelled` asserts the *ordering* (gamma perturbs more than an
equivalent linear contrast change) rather than a magnitude, which keeps the 9.5 gamma panel a
real measurement instead of a foregone conclusion.

---

## 4. Open for task 5.2

- Sweep `pixels_per_cell ∈ {(6,6), (8,8)}` × `orientations ∈ {9, 12}`, jointly with `C` and
  `class_weight` via `gtsrb.tuning`, exactly as PCA was — the 4.2 finding that the best `C`
  shifts with feature dimensionality (10 → 0.01 as *k* grew) applies here too, and the four
  configurations span 900 to 2352 dimensions.
- Per the 4.2 result on preprocessing, **sweep all three preproc configs**: reusing one
  config's hyperparameters on another cost ~1 pp there, about 39 % of the preprocessing
  effect being measured.
- Expect HOG to need less regularisation headroom than PCA at comparable dimensionality,
  because its features are already block-normalised — but that is a hypothesis, not a reason
  to narrow the grid.
