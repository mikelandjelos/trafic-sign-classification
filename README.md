# Traffic Sign Recognition: A Comparison of Image Representations

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

Models are evaluated on clean data and under four controlled degradations at five
levels each: Gaussian noise, motion blur, gamma shift, and **bounding-box jitter**.

## Two methodological notes

**Track-disjoint splits.** GTSRB training images come in tracks of 30 frames of the
same physical sign. A random split leaks near-duplicate frames across train and
validation and inflates accuracy. Splits here are by track ID, with an assertion
enforcing it.

**Bounding-box jitter** perturbs the annotated ROI by up to 40% in scale and
position, re-cropping from the source image rather than padding — so real background
enters the crop, as it would with output from an actual detector. This measures how
much accuracy would be lost when feeding this classifier detected rather than
annotated regions.

## Scope

This implements the **recognition** stage only, on crops whose boundaries come from
dataset annotation. Detection on full frames, temporal tracking via optical flow,
and distance estimation are out of scope — see `PROJECT_TASKS.md` §1.

## Stack

Python 3.11 · OpenCV · scikit-image · scikit-learn · PyTorch (CPU) · NumPy/Pandas/Matplotlib

CPU-only throughout; no GPU required.

## Context

Coursework project for *Računarski vid* (Computer Vision), Faculty of Electronic
Engineering, University of Niš.