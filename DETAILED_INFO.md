# A Comparison of Image Representations for Traffic Sign Classification under Controlled Degradations

A controlled comparison of four ways to represent a cropped traffic sign image,
evaluated on GTSRB (43 classes, ~52k images).

**Input:** one RGB crop containing a single traffic sign (15×15 to 250×250 px)
**Output:** one of 43 class labels

## The question

GTSRB is a saturated benchmark — modern CNNs exceed 99%, and another accuracy
number adds nothing. This project asks a different question: **which representation
fails under which conditions?**

The four representations differ structurally, and those differences should predict
different failure modes:

| Representation | Spatial structure | Locality | Learned? |
|---|---|---|---|
| PCA | holistic, alignment-critical | global | no |
| HOG | rigid grid, layout preserved | local | no |
| BoVW | orderless, layout discarded | local | vocabulary only |
| CNN | hierarchical, pooling-invariant | local | fully |

The expected result is not a leaderboard but an **interaction**: the ranking should
change depending on the stressor. Predictions were recorded in `predictions.md`
before any experiment was run.

## Method

Each representation φ maps an image to a fixed-length vector, and the **same linear
SVM** classifies it, so any difference in results is attributable to the representation
rather than to the classifier.

"Same" means the same estimator and the same selection protocol — not the same
hyperparameters. `C` and `class_weight` are tuned per method on validation, because PCA is
the only representation here whose features are not internally normalised (HOG
block-normalises, BoVW L2-normalises, the CNN has batch norm). Freezing `C` would hand each
representation a dial calibrated for another's feature scale, and the difference that
produced would be an artifact of the dial rather than a property of φ. The selected values
are reported alongside the results. The CNN therefore appears
twice — once as a feature extractor feeding the same SVM, and once end-to-end with
its own softmax head. Five configurations total.

Models are evaluated on clean data and under three controlled degradations at five
levels each: Gaussian noise, motion blur, and gamma shift.

**Degradations are injected after preprocessing**, on the model input. That is deliberate:
what gets injected is then, by construction, the degradation the preprocessing *failed to
remove* — the residual the representation actually has to cope with. The question is
therefore "given a degradation your pipeline didn't remove, which representation copes
best?", which needs no claim about camera behaviour. Simulating degradation at capture
time is a different question, scoped as an extension.

## Two methodological notes

**Track-disjoint splits — and the cost of getting it wrong is measured, not asserted.**
GTSRB training images come in tracks of 30 frames of the same physical sign, so a random
per-image split puts near-duplicates on both sides. Splits here are by track ID, with an
assertion enforcing it.

How much that matters was measured by training the identical model twice — same
hyperparameters, same validation fraction, only the split rule differing:

| split rule | val accuracy | val macro-F1 |
|---|---|---|
| track-disjoint | 0.8442 | 0.7977 |
| random per-image | 0.8950 | 0.8936 |
| **inflation** | **+5.08 pp** | **+9.58 pp** |

Under the random split, 1,305 of 1,307 tracks land on both sides and **100 % of validation
images keep a sibling frame in training**. The inflated number is stable across seeds
(σ ≈ 0.3 pp) and is produced by one line of `train_test_split` — it is reproducibly wrong,
which is why nothing looks broken.

**Macro-F1 inflates about twice as much as accuracy**, because leakage flatters the rare
classes most: a class with 7 tracks has almost no diversity to generalise across, so once
its siblings are in training its held-out frames are close to a lookup. The metric chosen
*because* of class imbalance is therefore the one leakage corrupts worst.

**Bounding-box jitter is deliberately absent — and the reason is a result.** Feeding a
classifier *detected* rather than *annotated* boxes is the question that matters for a
real system, so the original plan perturbed the annotated ROI by up to 40% and re-cropped.
GTSRB cannot support that: its images are already cropped to the sign plus a median ~17%
margin, and everything beyond was discarded when the dataset was built. Measured across
all 39,209 training images, **+40% expansion is impossible for 72% of them** (28,166 would
run off the edge); median headroom is 1.30×. Producing the curve anyway requires padding
with invented pixels, which measures the padding strategy rather than the representation.

Jitter therefore appears nowhere in training or in the core grid. It moves to an extension
that evaluates the *unmodified* models on GTSDB full road scenes, where the pixels actually
exist in every direction — see `docs/report-material/09-jitter-and-datasets.md`.

## Scope

This implements the **classification** stage only, on crops whose boundaries come from
dataset annotation. Detection on full frames, temporal tracking via optical flow,
and distance estimation are out of scope — see `PROJECT_TASKS.md` §1.

("Classification" rather than "recognition" deliberately: in the traffic-sign literature
*recognition* routinely covers the whole detection-plus-classification pipeline, which
would overstate what this delivers.)

## Stack

Python 3.11 · OpenCV · scikit-image · scikit-learn · PyTorch (CPU) · NumPy/Pandas/Matplotlib

CPU-only throughout; no GPU required.

## Context

Coursework project for *Računarski vid* (Computer Vision), Faculty of Electronic
Engineering, University of Niš.