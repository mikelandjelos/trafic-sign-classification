# 12 — HOG: the rigid-grid, locally-normalised representation (tasks 5.1–5.5)

*Feeds: Methodology → representations; Table 1 (9.1); HOG figure (5.4); discussion of the
noise and gamma panels (9.5).*

Implementation: `src/gtsrb/representations/hog.py`. Tests: `tests/test_hog.py` (28).
**Status:** complete (tasks 5.1–5.5)

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

## 4. Task 5.2 — the configuration sweep

`scripts/sweep_hog.py` → `results/sweeps/hog_configs.csv`,
`figures/report/hog_config_sweep.png`. 96 grid points (3 preproc × 4 geometries × 4 `C` ×
2 `class_weight`), ~2 h 20 m. **All 96 converged.**

**Selected: `raw_gray`, 6 px cells, 9 orientations, C = 0.01, `class_weight="balanced"` —
validation macro-F1 0.9175, accuracy 0.9298.**

| preproc | best geometry | dim | best C | macro-F1 | accuracy |
|---|---|---|---|---|---|
| **`raw_gray`** | 6px / 9o | 1764 | 0.01 | **0.9175** | 0.9298 |
| `clahe_gray` | 6px / 9o | 1764 | 0.01 | 0.9138 | 0.9264 |
| `clahe_hsv` | 6px / 12o | 2352 | 0.01 | 0.8205 | 0.8456 |

### 4.1 Finding — HOG and PCA select *different* preprocessing

PCA chose `clahe_gray` (note 11 §7.1b); **HOG chooses `raw_gray`**. The margin is small
(+0.0037) but the *direction* is opposite, and that is what matters.

This is the 4.2 decision paying off concretely. Had `clahe_gray` been imposed on HOG because
PCA preferred it, HOG would have been measured at a handicap and the difference would have
been attributed to the representation. **A per-method preprocessing choice is not bookkeeping
— the methods genuinely disagree about what preprocessing helps them.**

The mechanism is consistent with §2.2: HOG block-normalises internally, so CLAHE's contrast
normalisation is largely redundant for it — it is doing a job HOG already does. PCA has no
such internal normalisation, which is why CLAHE helps PCA and not HOG.

### 4.2 Finding — cell size dominates orientation count

Best macro-F1 per geometry on the winning preproc:

| geometry | dim | macro-F1 |
|---|---|---|
| **6px / 9o** | 1764 | **0.9175** |
| 6px / 12o | 2352 | 0.9140 |
| 8px / 9o | 900 | 0.8819 |
| 8px / 12o | 1200 | 0.8765 |

Halving the cell area (8 px → 6 px) is worth **~3.5 pp**; going from 9 to 12 orientation bins
is worth **nothing** — it is slightly *negative* at both cell sizes. Spatial resolution is the
binding constraint on a 48×48 crop, not angular resolution, which makes sense: at 8 px cells
the grid is only 6×6 for a whole sign.

Useful for the report's cost discussion: the selected config is **not** the highest-dimensional
one. 2352 dims buys nothing over 1764.

### 4.3 The dimensionality/regularisation coupling, again

Best `C` is **0.01** at 1764–2352 dims and **0.10** at 900–1200 dims — the same coupling
measured for PCA, where the best `C` fell from 10 to 0.01 as *k* grew 32 → 256. Third
independent confirmation that `C` cannot be fixed while feature dimensionality varies.

### 4.4 `clahe_hsv` hurts HOG badly (−9.7 pp) — and a likely reason worth checking

Colour costs HOG almost ten points, far more than it cost PCA. A plausible mechanism, **stated
as a hypothesis because it has not been verified**:

`skimage.feature.hog` with `channel_axis` computes gradients in every channel and keeps the
largest-magnitude one per pixel. In HSV, **hue is an angular quantity that wraps** (0–179 in
OpenCV), and red — the most common sign colour — sits exactly at the wrap point. A spatial
gradient computed across that discontinuity is spurious and *large*, so it would win the
max-magnitude selection precisely where it is meaningless.

**VERIFIED (task 7.3).** Measured on the cached data: *Stop* has 56.5 % of its saturated
pixels at hue < 10 and 4.4 % at hue > 169; *No entry* 39.2 % and 14.2 % — red genuinely
straddles the wrap. **3.98 % of adjacent-pixel hue gradients exceed 90**, i.e. claim more than
half the colour wheel between neighbours, and mean hue gradients (17.17) are as large as
intensity gradients (17.06), so they routinely win the max-magnitude selection.

The CNN independently corroborates it: `clahe_hsv` cost it 2.3 pp *and* made it peak at epoch
5 instead of 18 — a strong spurious signal learned early that does not generalise. The
ordering across methods follows the mechanism: the more a method relies on spatial
derivatives, the more the wrap costs it (HOG −9.7, PCA −3.0, which never differentiates).

Full measurement and the consequences for the 9.7 ablation: `08-preprocessing.md`, addendum.

### 4.5 The three methods treat `clahe_hsv` differently — and must, but say so

Worth recording because the preprocessing ablation (9.7) compares across methods:

| method | what it does with the 3 channels |
|---|---|
| PCA | flattens all three (6912 dims) |
| HOG | gradients per channel, keeps max magnitude (`channel_axis=-1`) |
| BoVW | **V channel only** — SIFT is defined on intensity (note 14 §5) |
| CNN | all three as input planes |

Each is the natural choice for that representation, and there is no single convention that
would suit all four. But it means "`clahe_hsv` is worse" is **not one statement** — it is four
different operations on the same pixels. The ablation must say which.

---

## 5. Task 5.3 — the trained model and its cost

`scripts/train_hog.py` → `results/results.csv` (59 rows),
`results/models/hog_svm_raw_gray.joblib`.

| | value |
|---|---|
| val accuracy | **0.9298** |
| val macro-F1 | **0.9175** |
| training time | descriptors + `LinearSVC` (nothing is learned in between) |
| inference, batched | 0.7592 ms/img |
| inference, single image | 0.9830 ms/img |
| model size | **0.58 MB** |
| feature dim | 1764 |

Reproduces the 5.2 selected cell exactly, the same end-to-end consistency check PCA passed.

### 5.1 The cost columns behave as the structural claim predicts

**Model size: 0.58 MB against PCA's 2.35 MB** — and the composition is the point, not the
number. HOG's model is *entirely* `LinearSVC` coefficients (1764 × 43 doubles ≈ 0.6 MB); it
has no learned stage to store. PCA's 2.35 MB is 96 % basis. This is exactly the caveat note 06
records: **the column does not measure the same thing across rows**, and Table 1 must say what
dominates each.

**Batching ratio: 1.3×** (0.7592 batched vs 0.9830 single). The three methods now span the
full range — PCA **49×**, HOG **1.3×**, CNN **0.87×** — tracking how much per-image work each
does. PCA's cost is one matrix multiply that amortises beautifully; HOG spends most of its
time in per-image descriptor extraction; the CNN is nearly all per-image work.

### 5.2 Finding — HOG fails on *completely different classes* than PCA and the CNN

This is the sharpest result so far, and it was not predicted.

| | worst classes | top confusion |
|---|---|---|
| **PCA** | 41, 32, 40, 29 | End of no passing → End of all speed limits (60.0 %) |
| **CNN** | 41, 32, 40, 29 | End of no passing → End of all speed limits (15.0 %) |
| **HOG** | **29, 24, 30, 5** | **Speed limit (80) → Speed limit (50) (16.9 %)** |

**Four of HOG's five top confusions are speed limits.** PCA's and the CNN's hardest pairs are
the *end-of-restriction* signs, which barely trouble HOG.

The mechanism follows the structural axes directly:

- **End-of-restriction signs** are near-identical grey circles distinguished by a diagonal
  strikethrough — a strong, well-localised **oriented gradient**, precisely what HOG encodes
  and what a holistic intensity subspace cannot separate. So HOG solves what PCA cannot.
- **Speed-limit digits** sit at the same position in every sign and differ only in fine
  stroke shape *within* a cell. HOG's 8×8 px cells pool exactly that detail away, while a
  holistic or hierarchical encoding keeps it.

**This is the representation × difficulty interaction the project set out to find**, appearing
on *clean* data before any degradation is applied. It also revises the 9.3 premise twice over:
"speed limits confuse predictably" is true **for HOG** and false for PCA and the CNN.

**Consequence for task 9.2/9.3:** reporting confusion matrices only for the best and worst
method would hide this entirely — the two extremes (CNN and PCA) share a failure mode, and the
method in the middle has a different one. The tables must cover all five.

---

## 6. Task 5.4 — the visualisation figure

`scripts/figure_hog.py` → `figures/report/hog_visualization.png`. Four shapes rather than four
arbitrary classes — octagonal (*Stop*), triangular warning (*Road work*), round mandatory
(*Keep right*), round prohibitory (*Speed limit 80*) — because what matters is what the grid
does to **shape**. Configuration read from the sweep CSV, not hard-coded.

**Two things the figure must state, and does:**

- **It is drawn from the *un-normalised* cell histograms.** The classifier sees the
  block-normalised vector, in which a contrast change has been divided out (§2.2). A reader
  who takes the rendering for the descriptor will draw the wrong conclusion about the gamma
  panel at 9.5.
- **The rendering is brightened for display only** (`exposure.rescale_intensity` at 35 % of
  max). Raw, a few strong cells dominate and everything else is near-black. Cosmetic; the
  descriptor is untouched.

The cell grid is drawn over the HOG panel, so the pooling unit is visible rather than implied.

### 6.1 The figure corroborates the 5.2 result visually

**Speed limit (80)** is the useful panel. The circular border produces strong, confident
strokes around the rim, while the **digits produce only short, weak segments** — the fine
stroke detail that separates 80 from 50 is pooled into near-identical cell histograms. That is
§5.2's confusion finding made visible: HOG sees the *shape* clearly and the *digits* barely.

Contrast *Road work*, where the triangle's three edges are unmistakable, and *Keep right*,
where the arrow's diagonal dominates a whole band of cells. Both are shape distinctions, and
both are classes HOG handles well.

---

## 7. Task 5.5 — the mechanics demo

`scripts/demo/hog_mechanics.py` → `figures/demo/hog/`. Three figures, built from
`gtsrb.representations.hog` and `gtsrb.degradations` themselves. The sample is *Speed limit
(80)* throughout — the class HOG confuses most (§5.2), so the demo is pointed at its own
weakest case rather than a flattering one.

### 7.1 `hog_block_normalisation.png` — the gamma mechanism, quantified on one sign

| variant | mean \|Δdescriptor\| |
|---|---|
| contrast ×0.5 (linear) | **0.0000** |
| gamma 2.5 (non-linear) | **0.0339** |

The two descriptor traces for the linear change are **exactly superimposed** — L2-Hys divides
a global gradient scaling straight out. The gamma trace visibly departs: peaks shift in
relative height, because a gamma curve changes the *ratios* between gradients rather than
scaling them all equally.

This is §2.2 and §3.1 made visible on a single image, and it is the figure to point at when
the 9.5 gamma panel is discussed: **HOG's protection against contrast is exact, and its
protection against gamma is partial.**

### 7.2 `hog_noise_response.png` — the demo's failure test

Per-cell gradient energy at σ ∈ {0, 5, 10, 20, 40}, with the bottom row showing energy
**relative to clean**. This is the quantity that would expose a problem, per the demo rule,
and it shows the predicted mechanism directly:

- the **corner cells** — low-contrast sky and road, no real gradient — reach **6× or more**
  of their clean energy;
- the **sign centre**, where genuine high-contrast structure lives, stays near 1×.

So noise does not raise all cells equally. It swamps precisely the cells that had nothing to
report, and their orientation votes become close to random while the informative cells are
barely touched. **That is why differentiation is the wrong operation under noise** — it is
high-pass, so it amplifies rather than averages. Mean |Δdescriptor| at σ=40: 0.1261, roughly
4× the gamma displacement.

### 7.3 `hog_cell_size.png` — why 6 px beat 8 px

The same sign at both cell sizes with the grid drawn on. At 8 px a whole sign gets only a 6×6
grid and the digits fall *inside* single cells; at 6 px the 8×8 grid begins to resolve them.
The visual counterpart of the +3.5 pp measured at 5.2, and of why the digits are what HOG
loses (§5.2).

---

## 8. Task 5 is complete

5.1–5.5 done. Open items belonging elsewhere: the `clahe_hsv` hue-wrap check is verified and
recorded in note 08; `transform_sqrt` remains an untested ablation (§2.4).
- **5.4** the HOG visualisation figure, one sample per super-category.
- **5.5** the mechanics demo — cell grid, block normalisation before/after, and gradient
  magnitude per cell under noise, where HOG is predicted to suffer most (§3.2 measured a
  displacement of 0.988 under σ=40).

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


---

## ADDENDUM (2026-09-16) — HOG is unaffected by the preprocessing plan change

The comparison is now fixed at `raw_gray` (`PROJECT_TASKS.md` §1, index Q7). **HOG had already
selected `raw_gray` on its own at 5.2**, so its reported row is unchanged: 6 px cells, 9
orientations, C=0.01, `balanced`, val macro-F1 **0.9175**, accuracy **0.9298**.

Worth stating in the report rather than passing over: the config chosen for the whole study is
the one HOG selected independently, which is a point in favour of `raw_gray` as the neutral
choice rather than an arbitrary one. The mechanism is in §4 — HOG block-normalises internally,
so CLAHE is redundant for it and very slightly harmful (−0.4 pp).

One consequence to carry into 9.1: **HOG is now the only hand-designed method whose reported
number required no compromise**, while PCA gives up 2.6 pp. If HOG's margin over PCA is
discussed, that 2.6 pp belongs in the sentence.
