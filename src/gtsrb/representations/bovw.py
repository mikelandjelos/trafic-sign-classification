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


# =====================================================================================
# Tasks 6.3-6.4: vocabulary and histogram encoding
# =====================================================================================

#: Vocabulary sizes swept at task 6.5.
VOCABULARY_SIZES: tuple[int, ...] = (200, 500)

#: Descriptors drawn to fit the vocabulary. ~200k of the ~2M the training split produces
#: (31,379 images x 64) -- enough for a stable k-means at k <= 500, and 10x cheaper.
DEFAULT_VOCAB_SAMPLES = 200_000


def chunk_for_budget(n_keypoints: int, n_words: int,
                     budget_bytes: int = 128 * 1024**2) -> int:
    """How many images to encode at once, so the peak allocation stays under budget.

    **Sized in bytes, not images**, and it must account for *two* growing terms. Getting only
    the first is what the OOM reaper kept killing:

    1. the **descriptor block**, `chunk x n_keypoints x 128 x 4` bytes. At 576 keypoints a
       4,000-image chunk is 1.1 GB.
    2. the **assignment**, which hands `chunk x n_keypoints` rows to `KMeans.predict`. That
       computes distances to every centroid, so the working set grows with `n_words` too:
       524,160 rows against 1,000 centroids is 3.9 GB dense, and sklearn only chunks it down
       to `working_memory` (1 GB by default).

    Both scale with `chunk x n_keypoints`, so one bound covers them if the per-row cost
    includes an allowance for the distance computation.

    **This lives at module level because it was got wrong three times in three files.** The
    fix was applied to whichever file was open, without grepping for the pattern; the same
    stale formula then sat in `scripts/experiment_spatial_pyramid.py` and again in
    `scripts/sweep_bovw.py`, each rediscovered only by being killed. Anything that encodes in
    chunks calls this -- there is no second copy to go stale.
    """
    per_row = DESCRIPTOR_DIM * 4 + n_words * 8
    return max(1, budget_bytes // (n_keypoints * per_row))


def normalise_histograms(histograms: np.ndarray, scheme: str = "power_l2") -> np.ndarray:
    """Normalise count histograms, row-wise.

    **What this choice is and is not about.** Every image contributes exactly
    `n_keypoints` descriptors, so every raw histogram already sums to the same number --
    unlike the usual BoVW setting where images yield different numbers of keypoints and
    normalisation is what makes them comparable at all. Here total mass is *already*
    constant, so `l1` is a pure rescale that changes nothing a linear classifier can see.

    The choice is therefore about the **distribution of mass across words**, specifically
    about *burstiness*: a repeated texture (a stretch of uniform sign face, a repeated
    edge) fires the same codeword many times and that one bin dominates the vector.

    - ``power_l2`` (default) -- square root, then L2. The square root compresses large
      bins relative to small ones, which is Perronnin's fix for burstiness, and L2 puts
      every image on the unit sphere. This is the standard pairing for BoVW with a linear
      SVM and is why it is the default here.
    - ``l2`` -- L2 only. Keeps burstiness; useful as the contrast that shows whether the
      square root is doing anything.
    - ``l1`` -- mass to 1. Included for completeness; a pure rescale in this setting.
    - ``none`` -- raw counts.
    """
    histograms = np.asarray(histograms, dtype=np.float32)
    if scheme == "none":
        return histograms
    if scheme == "l1":
        totals = histograms.sum(axis=1, keepdims=True)
        return histograms / np.maximum(totals, 1e-12)
    if scheme in ("l2", "power_l2"):
        values = np.sqrt(histograms) if scheme == "power_l2" else histograms
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return (values / np.maximum(norms, 1e-12)).astype(np.float32)
    raise ValueError(
        f"unknown normalisation {scheme!r}; expected one of "
        f"'power_l2', 'l2', 'l1', 'none'"
    )


class BoVWRepresentation:
    """Dense SIFT -> visual-word vocabulary -> orderless histogram."""

    name = "bovw"

    def __init__(
        self,
        n_words: int = 500,
        step: int = DEFAULT_STEP,
        keypoint_size: int = DEFAULT_KEYPOINT_SIZE,
        normalisation: str = "power_l2",
        n_vocab_samples: int = DEFAULT_VOCAB_SAMPLES,
        preproc: str = "clahe_gray",
        random_state: int = config.SEED,
    ) -> None:
        if n_words < 2:
            raise ValueError(f"n_words must be >= 2, got {n_words}")
        normalise_histograms(np.zeros((1, 2), np.float32), normalisation)  # validate early
        self.n_words = int(n_words)
        self.normalisation = normalisation
        self.n_vocab_samples = int(n_vocab_samples)
        self.preproc = preproc
        self.random_state = random_state
        self.extractor = DenseSIFT(step=step, keypoint_size=keypoint_size)
        self._kmeans = None

    # --- task 6.3: the vocabulary -----------------------------------------------------

    def fit(self, images: np.ndarray, progress: bool = False) -> BoVWRepresentation:
        """Learn the visual-word vocabulary from **clean training images only**.

        `MiniBatchKMeans`, not full `KMeans`: ~200k x 128 descriptors would fit in RAM but
        full Lloyd iterations over them are needlessly slow, and the vocabulary is a means
        to an encoding rather than an object of study in its own right.
        """
        from sklearn.cluster import MiniBatchKMeans

        descriptors = self.extractor.sample_descriptors(
            images, self.n_vocab_samples, seed_parts=("bovw", "vocab", self.preproc),
            progress=progress,
        )
        self._kmeans = MiniBatchKMeans(
            n_clusters=self.n_words,
            random_state=self.random_state,
            n_init=3,
            batch_size=4096,
            max_iter=100,
        ).fit(descriptors)
        return self

    # --- task 6.4: the encoding -------------------------------------------------------

    def chunk_for(self, budget_bytes: int = 128 * 1024**2) -> int:
        """How many images to describe at once, so the peak allocation stays under budget."""
        return chunk_for_budget(self.extractor.n_keypoints, self.n_words, budget_bytes)

    def transform(self, images: np.ndarray, chunk: int | None = None,
                  progress: bool = False) -> np.ndarray:
        """`(n, H, W[, C])` -> `(n, n_words)` float32 histograms.

        Chunked deliberately: assigning every descriptor of the test split at once means
        12,630 x 64 = 808k vectors of 128 floats against the centroids, and the distance
        matrix alone would be hundreds of megabytes. Chunking bounds it without changing
        the result -- each descriptor's nearest centroid depends only on that descriptor.

        `chunk` defaults to `chunk_for()`, which sizes it by memory rather than by image
        count. Passing an explicit value is still honoured, and the tests assert that the
        choice cannot change the output.
        """
        chunk = self.chunk_for() if chunk is None else chunk
        kmeans = self._fitted()
        out = np.zeros((len(images), self.n_words), dtype=np.float32)
        starts = range(0, len(images), chunk)
        if progress:
            from tqdm import tqdm

            starts = tqdm(list(starts), desc="  BoVW encode", unit="chunk")
        for start in starts:
            block = images[start:start + chunk]
            descriptors = self.extractor.describe_batch(block)
            flat = descriptors.reshape(-1, DESCRIPTOR_DIM)
            words = kmeans.predict(flat).reshape(len(block), self.extractor.n_keypoints)
            for i, row in enumerate(words):
                out[start + i] = np.bincount(row, minlength=self.n_words)
        return normalise_histograms(out, self.normalisation)

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        return self.fit(images).transform(images)

    def _fitted(self):
        if self._kmeans is None:
            raise RuntimeError(
                "BoVWRepresentation is not fitted; call fit(train_images) first"
            )
        return self._kmeans

    @property
    def n_features(self) -> int:
        return self.n_words

    @property
    def vocabulary_(self) -> np.ndarray:
        """`(n_words, 128)` -- the visual words themselves, for the task 6.6 demo."""
        return self._fitted().cluster_centers_

    def __repr__(self) -> str:
        state = "fitted" if self._kmeans is not None else "unfitted"
        return (
            f"BoVWRepresentation(n_words={self.n_words}, "
            f"normalisation={self.normalisation!r}, preproc={self.preproc!r}, {state})"
        )
