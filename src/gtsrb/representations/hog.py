"""Task 5.1: Histogram of Oriented Gradients -- the rigid-grid, locally-normalised feature.

    phi = hog.HOGRepresentation().fit(train_images)   # fit is a no-op; see below
    Z   = phi.transform(val_images)                   # (n, 900) float32

HOG divides the image into a fixed grid of cells, builds an orientation histogram of the
image gradient inside each cell, and normalises those histograms over overlapping blocks. It
sits opposite PCA on two of the project's structural axes: it is **local** (each output
coordinate depends on one small patch, not on every pixel) and it **preserves layout** (cell
*i, j* always describes the same region), where BoVW discards layout entirely.

`skimage.feature.hog`, not `cv2.HOGDescriptor`: OpenCV's defaults are tuned for 64x128
pedestrian windows and are awkward to reconfigure for a 48x48 square crop.

HOG learns nothing
------------------
**`fit` is a no-op.** There is no basis, no codebook, no weights -- the descriptor is a fixed
function of the pixels. `fit` exists only to satisfy the shared `Representation` interface
and to record the output width. Two consequences that belong in Table 1 rather than in a
footnote:

- its training cost is *only* the `LinearSVC` fit, where PCA also pays for an SVD and BoVW
  for k-means;
- its "model" is only the SVM coefficients, where PCA carries a 256x2304 basis (96 % of that
  method's 2.35 MB, see note 06).

Why the defaults are what they are
----------------------------------
`block_norm="L2-Hys"` (Dalal-Triggs): each block is L2-normalised, clipped at 0.2, and
renormalised. **This is the mechanism the gamma prediction rests on** -- a contrast change
scales every gradient in a block by the same factor, which L2 normalisation divides straight
back out. It is not incidental; it is why HOG is expected to be the gamma-robust method.

`cells_per_block=(2, 2)`, against skimage's default of (3, 3). At 48x48 with 8 px cells the
grid is only 6x6 cells, so 3x3 blocks would leave 4x4 block positions and over-smooth an
already small grid. (2, 2) is also the Dalal-Triggs recommendation. Not swept at task 5.2,
which varies cell size and orientation count; fixed and stated instead.

`transform_sqrt=False` (skimage's default, and Dalal-Triggs found gamma compression gave
little benefit for their detector). Worth naming because it is **literally a gamma transform**
(sqrt is gamma = 0.5) applied to the input before gradients are taken, so enabling it would
interact directly with the task 3.3 stressor. It is exposed as a parameter and left off: a
candidate ablation, not a default chosen to flatter a predicted result.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import hog as skimage_hog

from gtsrb import config, preprocessing

DEFAULT_ORIENTATIONS = 9
DEFAULT_PIXELS_PER_CELL: tuple[int, int] = (8, 8)
DEFAULT_CELLS_PER_BLOCK: tuple[int, int] = (2, 2)


def as_images(images: np.ndarray) -> np.ndarray:
    """`(n, H, W[, C])` uint8 -> float32 in [0, 1], shape unchanged.

    Kept as images rather than flattened: HOG needs the 2-D layout, which is the whole point
    of it. Scaled to [0, 1] for consistency with every other representation, though HOG is
    invariant to it -- gradients scale linearly and L2-Hys divides the factor back out. That
    invariance is asserted in the tests rather than assumed.
    """
    images = np.asarray(images)
    if images.ndim < 3:
        raise ValueError(f"expected (n, H, W[, C]) images, got shape {images.shape}")
    if images.dtype == np.uint8:
        return images.astype(np.float32) / 255.0
    return np.ascontiguousarray(images, dtype=np.float32)


class HOGRepresentation:
    """Fixed-grid gradient-orientation histograms with block normalisation."""

    name = "hog"

    def __init__(
        self,
        orientations: int = DEFAULT_ORIENTATIONS,
        pixels_per_cell: tuple[int, int] = DEFAULT_PIXELS_PER_CELL,
        cells_per_block: tuple[int, int] = DEFAULT_CELLS_PER_BLOCK,
        block_norm: str = "L2-Hys",
        transform_sqrt: bool = False,
        preproc: str = preprocessing.DEFAULT_PREPROC,
    ) -> None:
        if orientations < 2:
            raise ValueError(f"orientations must be >= 2, got {orientations}")
        for name, value in (("pixels_per_cell", pixels_per_cell),
                            ("cells_per_block", cells_per_block)):
            if len(value) != 2 or min(value) < 1:
                raise ValueError(f"{name} must be two positive ints, got {value}")
        self.orientations = int(orientations)
        self.pixels_per_cell = tuple(int(v) for v in pixels_per_cell)
        self.cells_per_block = tuple(int(v) for v in cells_per_block)
        self.block_norm = block_norm
        self.transform_sqrt = bool(transform_sqrt)
        self.preproc = preproc
        self._n_features: int | None = None

    # --- the shared interface -------------------------------------------------------

    def fit(self, images: np.ndarray) -> HOGRepresentation:
        """No-op beyond recording the output width. HOG has no learned parameters.

        It still takes `images`, because the output width depends on the input shape and
        deriving it by descriptor arithmetic would be a second implementation of skimage's
        blocking rules -- one that could disagree with it silently. Measuring one image is
        cheaper and cannot drift.
        """
        images = as_images(images)
        if len(images) == 0:
            raise ValueError("cannot fit on an empty batch")
        self._n_features = int(self._describe(images[0]).size)
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        images = as_images(images)
        if len(images) == 0:
            return np.empty((0, self.n_features), dtype=np.float32)
        first = self._describe(images[0])
        out = np.empty((len(images), first.size), dtype=np.float32)
        out[0] = first
        for i in range(1, len(images)):
            out[i] = self._describe(images[i])
        self._n_features = int(out.shape[1])
        return out

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        return self.fit(images).transform(images)

    @property
    def n_features(self) -> int:
        if self._n_features is None:
            raise RuntimeError("HOGRepresentation is not fitted; call fit(images) first")
        return self._n_features

    # --- the descriptor -------------------------------------------------------------

    def _describe(self, image: np.ndarray, visualize: bool = False):
        return skimage_hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            transform_sqrt=self.transform_sqrt,
            visualize=visualize,
            feature_vector=True,
            channel_axis=-1 if image.ndim == 3 else None,
        )

    def describe(self, image: np.ndarray) -> np.ndarray:
        """The descriptor for a single image, `(H, W[, C])` uint8 or float."""
        return self._describe(as_images(image[None])[0]).astype(np.float32)

    def visualize(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`(descriptor, hog_image)` for one image -- the task 5.4 figure.

        The rendered image draws, in each cell, a star of line segments whose brightness is
        the histogram weight for that orientation. It is a *visualisation* of the descriptor,
        not the descriptor: it is drawn from the un-normalised cell histograms, so it shows
        where gradient energy is, not what the block normalisation hands the classifier.
        """
        prepared = as_images(image[None])[0]
        descriptor, rendered = self._describe(prepared, visualize=True)
        return descriptor.astype(np.float32), rendered

    def cell_grid(self, image_shape: tuple[int, int] | None = None) -> tuple[int, int]:
        """How many cells the grid has -- `(rows, cols)`."""
        height, width = image_shape or config.IMAGE_SIZE
        return (height // self.pixels_per_cell[0], width // self.pixels_per_cell[1])

    def __repr__(self) -> str:
        state = "fitted" if self._n_features is not None else "unfitted"
        return (
            f"HOGRepresentation(orientations={self.orientations}, "
            f"pixels_per_cell={self.pixels_per_cell}, "
            f"cells_per_block={self.cells_per_block}, preproc={self.preproc!r}, {state})"
        )
