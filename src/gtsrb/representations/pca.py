"""Task 4.1: PCA on the flattened 48x48 image -- the "eigensigns" representation.

    phi = pca.PCARepresentation(n_components=128).fit(train_images)
    Z   = phi.transform(val_images)          # (n, 128) float32

The classical Eigenfaces construction applied to traffic signs: treat each image as a
point in R^2304, find the orthonormal directions of greatest variance across the training
set, and keep the leading k. It is the only representation here that is **holistic** -- the
value of every output coordinate depends on every input pixel -- and the only one for which
spatial alignment is load-bearing, which is why it anchors one end of the comparison.

Why `whiten=False`
------------------
`sklearn`'s PCA can rescale each retained component to unit variance (`whiten=True`). That
often helps a downstream linear SVM, because it stops the two or three highest-variance
directions from dominating the margin. It is **not** used here.

The reason is fidelity to the construction: "PCA representation" in the Eigenfaces sense is
an orthogonal projection onto the leading subspace. Whitening composes that projection with
a diagonal rescaling, so the coordinates the classifier sees would no longer be the
projections the eigenimages of task 4.4 depict -- the figure and the features would describe
different things. That reason stands on its own and does not depend on any expected result.

Note what whitening would *also* do, as a fact about the operator rather than a reason for
the choice: dividing each component by its own standard deviation amplifies the low-variance
retained directions, which are where an isotropic perturbation has the largest share
relative to signal. So whitened PCA is plausibly a **differently robust representation**,
not merely a rescaled one.

That is a reason to treat it as a separate condition worth measuring, and explicitly **not**
a reason to prefer unwhitened because unwhitened is what some prediction assumed. If
whitened PCA is ever run and proves more robust, that is a finding to report, not a
contamination to avoid. `whiten=True` is supported and tested; it is simply not the
configuration the headline results use, and that limitation is stated in the report.

What is fitted, and on what
---------------------------
The basis and the mean come from the **training split only** (31 379 images), never from
val or test, and always from **clean** images. Under a degraded test condition the degraded
image is projected through the clean basis -- the representation does not get to adapt to
the stressor. This is not incidental: it is what makes the gamma prediction falsifiable. A
gamma shift moves the whole test distribution away from the training mean that is subtracted
here, and there is no mechanism by which a fixed basis can absorb that.

Cost
----
`svd_solver='randomized'` (Halko et al.) rather than a full SVD: the full decomposition of a
31 379 x 2304 matrix computes all 2304 singular directions to obtain the 256 that are
wanted. The randomized solver is stochastic, so `random_state` is pinned to `config.SEED`
like everything else in the project.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

from gtsrb import cache, config, preprocessing

#: Retained dimensionality, **selected at task 4.2** by joint (k, C, class_weight) sweep on
#: validation macro-F1: k=256 scored 0.7977 against 0.7610 at k=128.
#: Caveat recorded in docs/report-material/11-pca.md: 256 is the top of the grid the task
#: specified, and the curve is still rising there -- this is the best of the four offered,
#: not a located optimum.
DEFAULT_N_COMPONENTS = 256


def as_matrix(images: np.ndarray) -> np.ndarray:
    """`(n, H, W[, C])` uint8 or `(n, D)` float -> `(n, D)` float32 in [0, 1].

    Accepts either form so callers can pass the cache's uint8 images directly. The uint8
    path goes through `cache.flatten`, so the scaling is the same one every other
    representation sees rather than a second, subtly different convention.
    """
    images = np.asarray(images)
    if images.ndim == 2 and images.dtype != np.uint8:
        return np.ascontiguousarray(images, dtype=np.float32)
    if images.ndim < 3:
        raise ValueError(
            f"expected (n, H, W[, C]) images or an (n, D) float matrix, got shape "
            f"{images.shape} of dtype {images.dtype}"
        )
    return cache.flatten(images)


class PCARepresentation:
    """Linear projection onto the leading `n_components` principal directions.

    Implements the `representations.Representation` interface: `fit` on clean training
    images, `transform` on anything.
    """

    name = "pca"

    def __init__(
        self,
        n_components: int = DEFAULT_N_COMPONENTS,
        preproc: str = preprocessing.DEFAULT_PREPROC,
        whiten: bool = False,
        random_state: int = config.SEED,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        self.n_components = int(n_components)
        self.preproc = preproc
        self.whiten = bool(whiten)
        self.random_state = random_state
        self._pca: PCA | None = None
        self._image_shape: tuple[int, ...] | None = None

    # --- fitting --------------------------------------------------------------------

    def fit(self, images: np.ndarray) -> PCARepresentation:
        """Learn the mean and the basis from **clean training images only**."""
        matrix = as_matrix(images)
        n_samples, n_features = matrix.shape
        if self.n_components > min(n_samples, n_features):
            raise ValueError(
                f"n_components={self.n_components} exceeds min(n_samples={n_samples}, "
                f"n_features={n_features}); PCA cannot produce more components than that"
            )
        self._image_shape = (
            tuple(np.asarray(images).shape[1:]) if np.asarray(images).ndim >= 3 else None
        )
        self._pca = PCA(
            n_components=self.n_components,
            svd_solver="randomized",
            whiten=self.whiten,
            random_state=self.random_state,
        ).fit(matrix)

        # Force C order. `fit` leaves `components_` F-contiguous, but a joblib round-trip
        # restores it C-contiguous -- same values, bit for bit, different memory layout. BLAS
        # then selects a different GEMM kernel, which sums in a different order, and a saved
        # model's projections differ from the in-memory one's by ~1e-6 absolute (measured on
        # scores of scale ~6). Nothing is *wrong* with either result, but "the model changed
        # when I reloaded it" is a miserable thing to debug at task 8.1. Normalising the
        # layout at fit time makes save/load exact.
        self._pca.components_ = np.ascontiguousarray(self._pca.components_)
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        """Project onto the fitted basis. `(n, H, W[, C])` -> `(n, n_components)`."""
        return self._fitted().transform(as_matrix(images)).astype(np.float32)

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        return self.fit(images).transform(images)

    def _fitted(self) -> PCA:
        if self._pca is None:
            raise RuntimeError("PCARepresentation is not fitted; call fit(train_images)")
        return self._pca

    # --- fitted quantities ----------------------------------------------------------

    @property
    def n_features(self) -> int:
        return self.n_components

    @property
    def components_(self) -> np.ndarray:
        """`(n_components, D)` -- the orthonormal basis, rows in descending variance."""
        return self._fitted().components_

    @property
    def mean_(self) -> np.ndarray:
        """The training mean image, flattened. Subtracted before every projection."""
        return self._fitted().mean_

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self._fitted().explained_variance_ratio_

    def cumulative_variance(self) -> np.ndarray:
        """Cumulative share of training variance explained by the first k components."""
        return np.cumsum(self.explained_variance_ratio_)

    def components_for_variance(self, fraction: float) -> int | None:
        """Smallest k reaching `fraction` of the training variance, or None if unreached.

        Returns None rather than `n_components` when the target lies beyond what was
        fitted -- reporting the cap as if it were the answer would understate how many
        dimensions the data actually needs.
        """
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        cumulative = self.cumulative_variance()
        reached = np.searchsorted(cumulative, fraction) + 1
        return int(reached) if reached <= len(cumulative) else None

    # --- inverse direction, for the task 4.4 figure ----------------------------------

    def _to_images(self, matrix: np.ndarray) -> np.ndarray:
        """Flat rows -> uint8 images of the shape `fit` was given."""
        if self._image_shape is None:
            raise RuntimeError("fitted on a flat matrix; the image shape is unknown")
        clipped = np.clip(matrix * 255.0, 0, 255)
        return np.rint(clipped).astype(np.uint8).reshape(len(matrix), *self._image_shape)

    def eigenimages(self, k: int | None = None) -> np.ndarray:
        """The first `k` components reshaped as images -- the "eigensigns" of task 4.4.

        Returned as float, **not** rescaled to uint8: components are signed (entries run
        either side of zero) and clipping them to a display range would discard half of
        each one. Rendering is the figure script's business.
        """
        components = self.components_ if k is None else self.components_[:k]
        if self._image_shape is None:
            return components
        return components.reshape(len(components), *self._image_shape)

    def mean_image(self) -> np.ndarray:
        """The training mean, as a uint8 image."""
        return self._to_images(self.mean_[None, :])[0]

    def reconstruct(self, images: np.ndarray) -> np.ndarray:
        """Project and project back -- the residual is what the representation discards.

        Output matches the input's image shape and dtype convention (uint8), so a
        reconstruction can be shown next to its original.
        """
        approx = self._fitted().inverse_transform(self.transform(images))
        return self._to_images(approx)

    def reconstruction_error(self, images: np.ndarray) -> float:
        """Mean absolute reconstruction error in gray levels (0-255).

        Reported in gray levels rather than as a normalised score so it is directly
        comparable with the degradation magnitudes of task 3.x -- "PCA discards 4.6 levels"
        sits on the same scale as "sigma = 5 noise".
        """
        original = as_matrix(images)
        approx = self._fitted().inverse_transform(self.transform(images))
        return float(np.abs(original - approx).mean() * 255.0)

    # --- persistence ----------------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path | str) -> PCARepresentation:
        return joblib.load(Path(path))

    def __repr__(self) -> str:
        state = "fitted" if self._pca is not None else "unfitted"
        return (
            f"PCARepresentation(n_components={self.n_components}, preproc={self.preproc!r}, "
            f"whiten={self.whiten}, {state})"
        )


def default_path(n_components: int, preproc: str) -> Path:
    return config.MODELS_DIR / f"pca_{preproc}_{n_components}.joblib"


def fit_on_train(
    n_components: int = DEFAULT_N_COMPONENTS,
    preproc: str = preprocessing.DEFAULT_PREPROC,
    whiten: bool = False,
) -> tuple[PCARepresentation, np.ndarray]:
    """Fit on the training split and return `(representation, train_images)`.

    The single place the "train split only, clean images only" rule is enforced, so no
    caller has to remember it.
    """
    from gtsrb import data

    train, _ = data.train_val_split()
    images = cache.load_images(train, preproc)
    representation = PCARepresentation(
        n_components=n_components, preproc=preproc, whiten=whiten
    ).fit(images)
    return representation, images


def main() -> int:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Fit PCA on the training split (task 4.1).")
    parser.add_argument("--components", type=int, default=DEFAULT_N_COMPONENTS)
    parser.add_argument("--preproc", default=preprocessing.DEFAULT_PREPROC)
    parser.add_argument("--whiten", action="store_true", help="ablation only; see docstring")
    parser.add_argument("--save", action="store_true", help="write to results/models/")
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()

    from gtsrb import data

    train, val = data.train_val_split()
    train_images = cache.load_images(train, args.preproc)
    val_images = cache.load_images(val, args.preproc)

    started = time.perf_counter()
    phi = PCARepresentation(
        n_components=args.components, preproc=args.preproc, whiten=args.whiten
    ).fit(train_images)
    elapsed = time.perf_counter() - started

    print(f"{phi!r}")
    print(f"  fitted on      : {len(train_images)} train images, "
          f"{as_matrix(train_images).shape[1]} dims -> {phi.n_features}")
    print(f"  fit time       : {elapsed:.2f} s")
    print(f"  variance kept  : {phi.cumulative_variance()[-1]:.4f}")
    for fraction in (0.80, 0.90, 0.95, 0.99):
        k = phi.components_for_variance(fraction)
        print(f"    {fraction:.0%} of variance : "
              + (f"{k} components" if k else f"> {phi.n_components} components"))
    print(f"  recon error    : {phi.reconstruction_error(train_images):.2f} gray levels "
          f"(train), {phi.reconstruction_error(val_images):.2f} (val)")

    if args.save:
        path = phi.save(default_path(args.components, args.preproc))
        print(f"  saved          : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
