"""Task 5.1: HOG representation.

The tests that matter here are the **invariance** ones. HOG's expected behaviour under the
task 3.x stressors is not a guess about the implementation, it is a property of block
normalisation, and these pin it: if `block_norm` were ever changed to a variant without
L2 normalisation, the gamma prediction's mechanism would quietly disappear and every other
test would still pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from gtsrb import preprocessing
from gtsrb.representations import Representation
from gtsrb.representations.hog import HOGRepresentation, as_images


@pytest.fixture(scope="module")
def images() -> np.ndarray:
    """40 synthetic 48x48 uint8 images with real oriented structure to describe."""
    rng = np.random.default_rng(0)
    out = np.zeros((40, 48, 48), dtype=np.uint8)
    ys, xs = np.mgrid[0:48, 0:48]
    for i in range(40):
        angle = rng.uniform(0, np.pi)
        stripes = 128 + 90 * np.sin((xs * np.cos(angle) + ys * np.sin(angle)) / 3.0)
        disc = ((xs - 24) ** 2 + (ys - 24) ** 2) < rng.integers(8, 20) ** 2
        out[i] = np.clip(np.where(disc, stripes, 40), 0, 255).astype(np.uint8)
    return out


@pytest.fixture(scope="module")
def fitted(images: np.ndarray) -> HOGRepresentation:
    return HOGRepresentation().fit(images)


# --- the shared interface -------------------------------------------------------------


def test_satisfies_the_representation_protocol(fitted: HOGRepresentation) -> None:
    assert isinstance(fitted, Representation)
    assert fitted.name == "hog"


def test_transform_shape_and_dtype(fitted: HOGRepresentation, images: np.ndarray) -> None:
    features = fitted.transform(images)
    assert features.shape == (len(images), fitted.n_features)
    assert features.dtype == np.float32


@pytest.mark.parametrize(
    ("pixels_per_cell", "orientations", "expected_grid", "expected_dim"),
    [((8, 8), 9, (6, 6), 900), ((6, 6), 9, (8, 8), 1764),
     ((8, 8), 12, (6, 6), 1200), ((6, 6), 12, (8, 8), 2352)],
)
def test_output_width_for_the_sweep_configurations(
    images, pixels_per_cell, orientations, expected_grid, expected_dim
) -> None:
    """The four configurations task 5.2 sweeps, pinned so a skimage change cannot move them."""
    phi = HOGRepresentation(orientations=orientations,
                            pixels_per_cell=pixels_per_cell).fit(images)
    assert phi.cell_grid() == expected_grid
    assert phi.n_features == expected_dim
    assert phi.transform(images[:3]).shape == (3, expected_dim)


def test_default_preproc_is_the_project_default() -> None:
    assert HOGRepresentation().preproc == preprocessing.DEFAULT_PREPROC


# --- HOG learns nothing ----------------------------------------------------------------


def test_fit_is_a_no_op(images: np.ndarray) -> None:
    """Fitting on different data must not change a single descriptor.

    This is the structural claim that puts HOG opposite PCA in Table 1 -- no basis, no
    codebook, no training cost beyond the SVM. If `fit` ever started learning something,
    the cost column would silently stop meaning what the report says it means.
    """
    on_first = HOGRepresentation().fit(images[:20])
    on_second = HOGRepresentation().fit(images[20:])
    np.testing.assert_array_equal(on_first.transform(images), on_second.transform(images))


def test_transform_works_on_a_fresh_instance_after_fit_on_one_image(images) -> None:
    phi = HOGRepresentation().fit(images[:1])
    assert phi.transform(images).shape == (len(images), 900)


def test_n_features_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = HOGRepresentation().n_features


def test_fit_on_an_empty_batch_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        HOGRepresentation().fit(np.zeros((0, 48, 48), dtype=np.uint8))


# --- the invariances the gamma prediction rests on --------------------------------------


def test_input_scaling_does_not_change_the_descriptor(
    fitted: HOGRepresentation, images: np.ndarray
) -> None:
    """uint8 in [0,255] and float in [0,1] must give the same descriptor, exactly."""
    as_uint8 = fitted.transform(images[:5])
    as_float = fitted.transform(images[:5].astype(np.float32) / 255.0)
    np.testing.assert_array_equal(as_uint8, as_float)


def test_a_linear_contrast_change_is_normalised_away(
    fitted: HOGRepresentation, images: np.ndarray
) -> None:
    """Halving contrast scales every gradient by 0.5; L2 normalisation divides it back out.

    **This is the mechanism behind the recorded gamma prediction.** Note what it does and
    does not cover: gamma is a *non-linear* point transform, so it is only partially
    cancelled -- see `test_gamma_is_only_partially_cancelled`.
    """
    original = as_images(images[:5])
    lower_contrast = 0.5 * original + 0.25  # same structure, half the gradient magnitude
    np.testing.assert_allclose(
        fitted.transform(lower_contrast), fitted.transform(original), atol=1e-5
    )


def test_gamma_is_only_partially_cancelled(
    fitted: HOGRepresentation, images: np.ndarray
) -> None:
    """A gamma curve is not a linear scale, so block normalisation cannot fully remove it.

    Asserted as an ordering rather than a magnitude: a gamma change must perturb the
    descriptor *more* than an equivalent linear contrast change, which is what makes the
    9.5 gamma panel a real measurement rather than a foregone conclusion.
    """
    from gtsrb import degradations

    clean = images[:5]
    gamma = degradations.apply(clean, "gamma", 2.5, [f"img{i}" for i in range(len(clean))])
    linear = np.clip(as_images(clean) * 0.5, 0, 1)

    base = fitted.transform(clean)
    gamma_shift = float(np.abs(fitted.transform(gamma) - base).mean())
    linear_shift = float(np.abs(fitted.transform(linear) - base).mean())
    assert gamma_shift > linear_shift


# --- determinism and batching -----------------------------------------------------------


def test_transform_is_independent_of_batch_composition(
    fitted: HOGRepresentation, images: np.ndarray
) -> None:
    """Each descriptor depends on one image only; a subset must match the full batch."""
    full = fitted.transform(images)
    np.testing.assert_array_equal(fitted.transform(images[7:12]), full[7:12])


def test_transform_is_deterministic(fitted: HOGRepresentation, images: np.ndarray) -> None:
    np.testing.assert_array_equal(fitted.transform(images), fitted.transform(images))


def test_empty_batch_returns_an_empty_matrix(fitted: HOGRepresentation) -> None:
    empty = fitted.transform(np.zeros((0, 48, 48), dtype=np.uint8))
    assert empty.shape == (0, fitted.n_features)


# --- colour and visualisation -------------------------------------------------------------


def test_colour_images_are_accepted(images: np.ndarray) -> None:
    """`clahe_hsv` is (48, 48, 3); skimage needs channel_axis for it."""
    colour = np.repeat(images[:, :, :, None], 3, axis=3)
    phi = HOGRepresentation(preproc="clahe_hsv").fit(colour)
    assert phi.transform(colour[:3]).shape == (3, phi.n_features)


def test_visualize_returns_a_descriptor_and_a_rendered_image(
    fitted: HOGRepresentation, images: np.ndarray
) -> None:
    descriptor, rendered = fitted.visualize(images[0])
    assert descriptor.shape == (fitted.n_features,)
    assert rendered.shape == images[0].shape
    assert rendered.max() > 0


def test_describe_matches_transform_for_one_image(
    fitted: HOGRepresentation, images: np.ndarray
) -> None:
    np.testing.assert_allclose(fitted.describe(images[3]), fitted.transform(images[3:4])[0],
                               atol=1e-6)


# --- argument validation --------------------------------------------------------------------


@pytest.mark.parametrize("orientations", [0, 1, -3])
def test_too_few_orientations_raises(orientations) -> None:
    with pytest.raises(ValueError, match="orientations"):
        HOGRepresentation(orientations=orientations)


@pytest.mark.parametrize("bad", [(0, 8), (8,), (8, 8, 8), (-1, 4)])
def test_bad_cell_geometry_raises(bad) -> None:
    with pytest.raises(ValueError, match="pixels_per_cell|cells_per_block"):
        HOGRepresentation(pixels_per_cell=bad)


def test_as_images_rejects_a_single_image() -> None:
    with pytest.raises(ValueError, match="expected"):
        as_images(np.zeros((48, 48), dtype=np.uint8))
