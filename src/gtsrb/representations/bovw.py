"""Tasks 6.1-6.2: dense SIFT -- the local descriptors BoVW is built from.

    extractor = bovw.DenseSIFT()
    descriptors = extractor.describe(image)        # (64, 128) float32
    sample = extractor.sample_descriptors(images, 200_000)   # for k-means at task 6.3

BoVW is the **orderless** corner of the comparison: local descriptors are extracted, each is
assigned to a visual word, and the image becomes a histogram of word counts. The spatial
position of every descriptor is discarded, which is exactly what distinguishes it from HOG's
rigid grid -- and the property the section 11 jitter extension is designed to probe.

This module covers the extraction half. The vocabulary and histogram encoding follow at
tasks 6.3-6.4.

Why a fixed grid and not a detector
-----------------------------------
**Detector-based SIFT returns zero keypoints on small crops** -- the documented gotcha in
PROJECT_TASKS section 10, and it fails *silently*: an image simply contributes nothing and
the histogram is all zeros. GTSRB crops are 15x15 to 250x250 px, resized to 48x48, and at
that size a blob detector finds almost nothing. So the keypoints are placed on a fixed grid,
every image gets the same number of them, and task 6.2 asserts it.

The parameters, and what was measured
-------------------------------------
`step=6`, `keypoint_size=12` on 48x48 gives an **8x8 grid = 64 keypoints** per image, each
describing a 12 px neighbourhood -- so neighbourhoods overlap by half, which is what keeps
the encoding from being a hard tiling. Measured over 2,000 images per config:

- **no keypoint is ever dropped.** OpenCV prunes keypoints too close to the border; at this
  step and size none are, and the same held for every (step, size) tried. Asserted anyway,
  because a pruned keypoint would make one image's histogram quietly incomparable.
- **no descriptor is all-zero on real signs** -- 0 of 128,000 on `raw_gray` and
  `clahe_gray`, and 0 of 38,400 under every degradation at full strength (blur 15, noise 40,
  gamma 2.5). This was worth checking rather than assuming: heavy blur flattens local
  structure, and a zero descriptor is a degenerate point for k-means and an undefined
  direction for assignment.

  It is **not** impossible in general: a *perfectly* uniform patch does produce a zero
  descriptor (verified on a synthetic constant image). Real GTSRB crops never contain one,
  even after blur, so no special handling is added -- but the boundary is recorded here
  rather than left as an implicit assumption.

`upright=True` (fixed angle 0) rather than letting SIFT estimate a dominant orientation per
patch. Two reasons, and they point the same way: traffic signs are upright by construction,
so the absolute gradient orientation *is* signal and rotation-normalising each patch throws
it away; and the orientation estimate is unstable on low-contrast patches, which would inject
noise exactly where the descriptor is least reliable. This matches VLFeat's dense-SIFT
convention. The two are measurably different, not a formality.

Descriptor scaling
------------------
OpenCV returns float32 descriptors already L2-normalised to a norm of ~512 (measured
511.1-512.8; the spread is the 0.2 clipping and renormalisation in Lowe's recipe). They
therefore lie on a sphere, which is the right footing for k-means -- no further scaling is
applied here, and task 6.4's histogram normalisation is a separate decision.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from gtsrb import config

DEFAULT_STEP = 6
DEFAULT_KEYPOINT_SIZE = 12
DESCRIPTOR_DIM = 128


@lru_cache(maxsize=8)
def _sift() -> cv2.SIFT:
    """One shared detector. `cv2.SIFT_create()` is not free and holds no per-image state."""
    return cv2.SIFT_create()


class DenseSIFT:
    """SIFT descriptors on a fixed grid -- no detector, same count for every image."""

    def __init__(
        self,
        step: int = DEFAULT_STEP,
        keypoint_size: int = DEFAULT_KEYPOINT_SIZE,
        upright: bool = True,
        image_size: tuple[int, int] = config.IMAGE_SIZE,
    ) -> None:
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        if keypoint_size < 2:
            raise ValueError(f"keypoint_size must be >= 2, got {keypoint_size}")
        self.step = int(step)
        self.keypoint_size = int(keypoint_size)
        self.upright = bool(upright)
        self.image_size = (int(image_size[0]), int(image_size[1]))

    @property
    def keypoints(self) -> list[cv2.KeyPoint]:
        """The grid, built fresh each access.

        Deliberately not cached: `SIFT.compute` may mutate the `cv2.KeyPoint` objects it is
        given (it can overwrite `angle`), so handing out a shared list would let one call
        change what the next call describes.
        """
        height, width = self.image_size
        angle = 0.0 if self.upright else -1.0
        half = self.step // 2
        return [
            cv2.KeyPoint(float(x), float(y), float(self.keypoint_size), angle)
            for y in range(half, height, self.step)
            for x in range(half, width, self.step)
        ]

    @property
    def grid_shape(self) -> tuple[int, int]:
        height, width = self.image_size
        half = self.step // 2
        return (len(range(half, height, self.step)), len(range(half, width, self.step)))

    @property
    def n_keypoints(self) -> int:
        rows, cols = self.grid_shape
        return rows * cols

    def describe(self, image: np.ndarray) -> np.ndarray:
        """`(H, W)` uint8 -> `(n_keypoints, 128)` float32.

        Raises if OpenCV returned a different number of descriptors than keypoints given --
        the silent-failure mode this whole design exists to avoid (task 6.2).
        """
        image = np.asarray(image)
        if image.ndim == 3:
            # SIFT is defined on intensity; for clahe_hsv the V channel is the one CLAHE
            # acted on, and taking a mean would blend hue into a gradient operator.
            image = image[:, :, -1]
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0 if image.max() <= 1.0 else image,
                            0, 255).astype(np.uint8)

        _, descriptors = _sift().compute(image, self.keypoints)
        if descriptors is None or len(descriptors) != self.n_keypoints:
            got = 0 if descriptors is None else len(descriptors)
            raise RuntimeError(
                f"dense SIFT returned {got} descriptors for {self.n_keypoints} keypoints; "
                f"OpenCV pruned keypoints near the border, so this image's histogram would "
                f"not be comparable with the others"
            )
        return descriptors.astype(np.float32)

    def describe_batch(self, images: np.ndarray, progress: bool = False) -> np.ndarray:
        """`(n, H, W[, C])` -> `(n, n_keypoints, 128)` float32.

        Note the size: 31,379 training images at 64x128 float32 is about **1 GB**. Use
        `sample_descriptors` when only a subsample is needed (task 6.3 fits the vocabulary on
        ~200k of roughly 2M), which never materialises the full array.
        """
        rows = range(len(images))
        if progress:
            from tqdm import tqdm

            rows = tqdm(rows, desc="  dense SIFT", unit="img")
        out = np.empty((len(images), self.n_keypoints, DESCRIPTOR_DIM), dtype=np.float32)
        for i in rows:
            out[i] = self.describe(images[i])
        return out

    def sample_descriptors(
        self, images: np.ndarray, n_samples: int, seed_parts: tuple = ("bovw", "vocab"),
        progress: bool = False,
    ) -> np.ndarray:
        """A deterministic subsample of all descriptors, without materialising them all.

        Samples a fixed number of keypoints *per image* rather than uniformly over the whole
        descriptor pool. The two are equivalent here because every image contributes exactly
        `n_keypoints` descriptors, and the per-image form streams in constant memory.

        Seeded through `config.rng_for`, so the vocabulary at task 6.3 is a pure function of
        (images, n_samples, seed) and does not depend on how many images were processed
        before -- the same discipline the degradations use.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        total = len(images) * self.n_keypoints
        if n_samples >= total:
            return self.describe_batch(images, progress=progress).reshape(-1, DESCRIPTOR_DIM)

        rng = config.rng_for(*seed_parts, n_samples, len(images))
        # ceil, not round: round() undershoots whenever n_samples/len(images) has a
        # fractional part below .5, and the pool is then trimmed down to exactly
        # n_samples -- topping up after the fact is not possible without a second pass.
        per_image = max(1, -(-n_samples // len(images)))
        chosen: list[np.ndarray] = []
        rows = range(len(images))
        if progress:
            from tqdm import tqdm

            rows = tqdm(rows, desc="  dense SIFT (sampling)", unit="img")
        for i in rows:
            descriptors = self.describe(images[i])
            take = rng.choice(self.n_keypoints, size=min(per_image, self.n_keypoints),
                              replace=False)
            chosen.append(descriptors[np.sort(take)])
        pool = np.concatenate(chosen)
        if len(pool) > n_samples:
            keep = rng.choice(len(pool), size=n_samples, replace=False)
            pool = pool[np.sort(keep)]
        return pool

    def __repr__(self) -> str:
        rows, cols = self.grid_shape
        return (
            f"DenseSIFT(step={self.step}, keypoint_size={self.keypoint_size}, "
            f"upright={self.upright}, grid={rows}x{cols}={self.n_keypoints} keypoints)"
        )
