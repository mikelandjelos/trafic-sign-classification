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

Each representation φ maps an image to a fixed-length vector, and a **fixed linear
SVM** classifies it. Holding the classifier constant means any difference in results
is attributable to the representation, not the classifier. The CNN therefore appears
twice — once as a feature extractor feeding the same SVM, and once end-to-end with
its own softmax head. Five configurations total.

Models are evaluated on clean data and under three controlled degradations at five
levels each: Gaussian noise, motion blur, and gamma shift.

## Two methodological notes

**Track-disjoint splits.** GTSRB training images come in tracks of 30 frames of the
same physical sign. A random split leaks near-duplicate frames across train and
validation and inflates accuracy. Splits here are by track ID, with an assertion
enforcing it.

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