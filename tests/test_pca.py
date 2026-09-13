"""Task 4.1: PCA representation.

Most tests run on small synthetic images -- the properties under test (orthonormality,
centering, batch-independence, monotone reconstruction) are properties of the construction,
not of GTSRB, and asserting them on 31 379 real images would make the suite slow without
making it stronger. Two tests at the end do use the real data, because what they assert is
a fact about traffic signs rather than about PCA.
"""

from __future__ import annotations

import numpy as np
import pytest

from gtsrb import config, preprocessing
from gtsrb.representations import Representation
from gtsrb.representations.pca import PCARepresentation, as_matrix


@pytest.fixture(scope="module")
def images() -> np.ndarray:
    """200 synthetic 48x48 uint8 images with genuine low-rank structure plus noise."""
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(5, 48 * 48))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    # Scaled so each component contributes tens of gray levels per pixel -- with unit-norm
    # basis rows over 2304 dims, a coefficient of 1 is worth ~0.02 levels and the structure
    # would sit below the noise floor.
    scales = np.array([1200.0, 800.0, 500.0, 250.0, 120.0])
    flat = 128 + (rng.normal(size=(200, 5)) * scales) @ basis
    flat += rng.normal(0, 5, size=flat.shape)
    return np.clip(flat, 0, 255).astype(np.uint8).reshape(200, 48, 48)


@pytest.fixture(scope="module")
def fitted(images: np.ndarray) -> PCARepresentation:
    return PCARepresentation(n_components=16).fit(images)


# --- the shared interface -------------------------------------------------------------


def test_satisfies_the_representation_protocol(fitted: PCARepresentation) -> None:
    assert isinstance(fitted, Representation)
    assert fitted.name == "pca"


def test_transform_shape(fitted: PCARepresentation, images: np.ndarray) -> None:
    assert fitted.transform(images).shape == (len(images), 16)
    assert fitted.n_features == 16


def test_transform_is_float32(fitted: PCARepresentation, images: np.ndarray) -> None:
    # LinearSVC copies float64 input; keeping float32 halves the memory of the grid.
    assert fitted.transform(images).dtype == np.float32


# --- input handling -------------------------------------------------------------------


def test_as_matrix_scales_uint8_to_unit_range(images: np.ndarray) -> None:
    matrix = as_matrix(images)
    assert matrix.shape == (len(images), 2304)
    assert matrix.dtype == np.float32
    assert 0.0 <= matrix.min() and matrix.max() <= 1.0


def test_as_matrix_passes_through_a_float_matrix() -> None:
    flat = np.zeros((4, 2304), dtype=np.float64)
    assert as_matrix(flat).shape == (4, 2304)
    assert as_matrix(flat).dtype == np.float32


def test_as_matrix_rejects_a_single_image() -> None:
    with pytest.raises(ValueError, match="expected"):
        as_matrix(np.zeros((48, 48), dtype=np.uint8))


def test_colour_images_are_accepted(images: np.ndarray) -> None:
    colour = np.repeat(images[:, :, :, None], 3, axis=3)
    phi = PCARepresentation(n_components=8, preproc="clahe_hsv").fit(colour)
    assert phi.transform(colour).shape == (len(colour), 8)
    assert phi.eigenimages(2).shape == (2, 48, 48, 3)


# --- the construction itself ----------------------------------------------------------


def test_components_are_orthonormal(fitted: PCARepresentation) -> None:
    gram = fitted.components_ @ fitted.components_.T
    np.testing.assert_allclose(gram, np.eye(16), atol=1e-5)


def test_projection_is_centred(fitted: PCARepresentation, images: np.ndarray) -> None:
    """The training mean maps to the origin -- the mean is part of the model."""
    scores = fitted.transform(images)
    np.testing.assert_allclose(scores.mean(axis=0), 0.0, atol=1e-4)


def test_explained_variance_is_descending(fitted: PCARepresentation) -> None:
    ratios = fitted.explained_variance_ratio_
    assert np.all(np.diff(ratios) <= 1e-12)
    assert 0 < ratios.sum() <= 1.0 + 1e-9


def test_cumulative_variance_is_monotone(fitted: PCARepresentation) -> None:
    cumulative = fitted.cumulative_variance()
    assert np.all(np.diff(cumulative) >= 0)
    assert cumulative[-1] == pytest.approx(fitted.explained_variance_ratio_.sum())


def test_components_for_variance(fitted: PCARepresentation) -> None:
    k = fitted.components_for_variance(0.5)
    assert k is not None
    assert fitted.cumulative_variance()[k - 1] >= 0.5
    assert k == 1 or fitted.cumulative_variance()[k - 2] < 0.5


def test_components_for_variance_returns_none_when_unreached(images: np.ndarray) -> None:
    """A target beyond the fitted cap reports None, not the cap dressed up as an answer."""
    phi = PCARepresentation(n_components=2).fit(images)
    assert phi.components_for_variance(0.999999) is None


def test_components_for_variance_rejects_a_bad_fraction(fitted: PCARepresentation) -> None:
    with pytest.raises(ValueError, match="fraction"):
        fitted.components_for_variance(1.5)


# --- the guarantee the evaluation grid depends on --------------------------------------


def test_transform_is_independent_of_batch_composition(
    fitted: PCARepresentation, images: np.ndarray
) -> None:
    """Row i of a subset's projection matches row i of the full batch's.

    PCA is fitted, not adapted, so this should hold trivially -- and it is the same
    guarantee the degradation seeding makes (see `config.rng_for`), on which task 8.1's
    method comparison rests.

    It holds numerically, **not** bit-for-bit, and the tolerance is the point of the test.
    A float32 GEMM sums in an order that depends on how BLAS blocks the matrix, which
    depends on its number of rows; projecting 10 images and projecting 200 therefore
    disagree in the last bits. Measured at ~1e-6 absolute on scores of scale ~6, i.e. ~1e-7
    relative -- orders of magnitude below any margin `LinearSVC` will decide on.
    """
    full = fitted.transform(images)
    subset = fitted.transform(images[10:20])
    np.testing.assert_allclose(subset, full[10:20], atol=1e-5)


def test_fit_is_reproducible(images: np.ndarray) -> None:
    """randomized SVD is stochastic; the seed is what makes two runs comparable."""
    first = PCARepresentation(n_components=16).fit(images).transform(images)
    second = PCARepresentation(n_components=16).fit(images).transform(images)
    np.testing.assert_array_equal(first, second)


def test_a_different_seed_is_still_numerically_close(images: np.ndarray) -> None:
    """The solver is approximate, but not so approximate that the seed changes the answer.

    Split by signal and noise: the fixture has 5 genuine directions, so the leading 5 are
    recovered to 1e-4 whatever the seed. The trailing components fit residual noise and are
    only reproduced to ~1e-3 -- worth knowing before reading anything into the tail of the
    scree curve at task 4.2.
    """
    default = PCARepresentation(n_components=8).fit(images)
    other = PCARepresentation(n_components=8, random_state=config.SEED + 1).fit(images)
    np.testing.assert_allclose(
        default.explained_variance_ratio_[:5], other.explained_variance_ratio_[:5], rtol=1e-4
    )
    np.testing.assert_allclose(
        default.explained_variance_ratio_, other.explained_variance_ratio_, rtol=1e-2
    )


# --- reconstruction (task 4.4 depends on this) ------------------------------------------


def test_more_components_reconstruct_better(images: np.ndarray) -> None:
    errors = [
        PCARepresentation(n_components=k).fit(images).reconstruction_error(images)
        for k in (2, 8, 32)
    ]
    assert errors[0] > errors[1] > errors[2]


def test_reconstruct_returns_images_of_the_input_shape(
    fitted: PCARepresentation, images: np.ndarray
) -> None:
    approx = fitted.reconstruct(images)
    assert approx.shape == images.shape
    assert approx.dtype == np.uint8


def test_eigenimages_are_signed_and_not_clipped(fitted: PCARepresentation) -> None:
    """Components run either side of zero; rescaling them to uint8 would halve them."""
    eigen = fitted.eigenimages(4)
    assert eigen.shape == (4, 48, 48)
    assert eigen.min() < 0 < eigen.max()


def test_mean_image_is_a_uint8_image(fitted: PCARepresentation) -> None:
    mean = fitted.mean_image()
    assert mean.shape == (48, 48)
    assert mean.dtype == np.uint8


# --- whitening is available but off by default ------------------------------------------


def test_whitening_is_off_by_default() -> None:
    """See the module docstring: whitening would amplify the low-variance retained
    directions, which is exactly the mechanism the noise prediction rests on."""
    assert PCARepresentation().whiten is False


def test_unwhitened_scores_keep_their_variance_ordering(
    fitted: PCARepresentation, images: np.ndarray
) -> None:
    variances = fitted.transform(images).var(axis=0)
    assert np.all(np.diff(variances) <= 1e-6)
    assert variances[0] / variances[-1] > 2  # a real spread, not already whitened


def test_whitening_equalises_the_variances(images: np.ndarray) -> None:
    whitened = PCARepresentation(n_components=8, whiten=True).fit(images)
    variances = whitened.transform(images).var(axis=0)
    np.testing.assert_allclose(variances, 1.0, rtol=0.02)


# --- errors and persistence -------------------------------------------------------------


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        PCARepresentation().transform(np.zeros((2, 48, 48), dtype=np.uint8))


def test_too_many_components_raises(images: np.ndarray) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        PCARepresentation(n_components=5000).fit(images)


def test_non_positive_components_raises() -> None:
    with pytest.raises(ValueError, match="n_components"):
        PCARepresentation(n_components=0)


def test_save_load_round_trip(fitted: PCARepresentation, images: np.ndarray, tmp_path) -> None:
    """A reloaded model must project **identically**, not merely closely.

    This is exact only because `fit` normalises `components_` to C order. sklearn leaves it
    F-contiguous and joblib restores it C-contiguous, so without that the values round-trip
    bit-for-bit while the projections do not -- BLAS picks a different kernel for a
    different layout. Regression test for that fix.
    """
    path = fitted.save(tmp_path / "pca.joblib")
    reloaded = PCARepresentation.load(path)
    np.testing.assert_array_equal(reloaded.components_, fitted.components_)
    np.testing.assert_array_equal(reloaded.transform(images), fitted.transform(images))
    assert reloaded.whiten == fitted.whiten
    assert reloaded.preproc == fitted.preproc


# --- on the real dataset ----------------------------------------------------------------


def test_fit_on_train_uses_only_the_training_split() -> None:
    from gtsrb import data
    from gtsrb.representations import pca

    train, _ = data.train_val_split()
    phi, fitted_images = pca.fit_on_train(n_components=16)
    assert len(fitted_images) == len(train)
    assert phi.transform(fitted_images).shape == (len(train), 16)


def test_real_signs_are_dominated_by_one_direction() -> None:
    """On raw grayscale, PC1 is brightness and holds over half the variance.

    Recorded as a test because task 9.5's gamma panel is interpreted through it: a global
    intensity remap moves a test point along the single direction PCA spends most of its
    budget on. See docs/report-material/11-pca.md.
    """
    from gtsrb import cache, data
    from gtsrb.representations import pca

    train, _ = data.train_val_split()
    train_images = cache.load_images(train, "raw_gray")
    phi = pca.PCARepresentation(n_components=8, preproc="raw_gray").fit(train_images)
    brightness = as_matrix(train_images).mean(axis=1)
    correlation = np.corrcoef(phi.transform(train_images)[:, 0], brightness)[0, 1]

    assert phi.explained_variance_ratio_[0] > 0.5
    assert abs(correlation) > 0.99


def test_default_preproc_is_the_project_default() -> None:
    assert PCARepresentation().preproc == preprocessing.DEFAULT_PREPROC
