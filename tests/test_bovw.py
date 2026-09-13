"""Tasks 6.1-6.2: dense SIFT extraction.

The central test here is `test_every_image_yields_the_same_descriptor_count` -- task 6.2
exists as a task precisely because the failure it guards against is silent: a detector, or a
pruned border keypoint, contributes nothing and the resulting histogram is simply wrong
rather than missing.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from gtsrb.representations.bovw import DESCRIPTOR_DIM, DenseSIFT


@pytest.fixture(scope="module")
def images() -> np.ndarray:
    """60 synthetic 48x48 signs: a disc with oriented internal structure, plus a flat one."""
    rng = np.random.default_rng(0)
    ys, xs = np.mgrid[0:48, 0:48]
    out = np.zeros((60, 48, 48), dtype=np.uint8)
    for i in range(59):
        angle = rng.uniform(0, np.pi)
        stripes = 128 + 90 * np.sin((xs * np.cos(angle) + ys * np.sin(angle)) / 3.0)
        disc = ((xs - 24) ** 2 + (ys - 24) ** 2) < rng.integers(10, 22) ** 2
        out[i] = np.clip(np.where(disc, stripes, 40), 0, 255).astype(np.uint8)
    out[59] = 128  # a completely flat image -- the degenerate case
    return out


@pytest.fixture(scope="module")
def extractor() -> DenseSIFT:
    return DenseSIFT()


# --- the grid ---------------------------------------------------------------------------


def test_default_grid_is_eight_by_eight(extractor: DenseSIFT) -> None:
    assert extractor.grid_shape == (8, 8)
    assert extractor.n_keypoints == 64


@pytest.mark.parametrize(("step", "expected"), [(4, (12, 12)), (6, (8, 8)), (8, (6, 6))])
def test_grid_shape_follows_the_step(step, expected) -> None:
    assert DenseSIFT(step=step).grid_shape == expected


def test_keypoints_are_rebuilt_each_access(extractor: DenseSIFT) -> None:
    """`SIFT.compute` can overwrite a KeyPoint's angle; a shared list would let one call
    change what the next one describes."""
    assert extractor.keypoints is not extractor.keypoints


def test_keypoints_carry_the_configured_size_and_angle() -> None:
    upright = DenseSIFT(keypoint_size=16, upright=True).keypoints
    assert all(kp.size == 16.0 for kp in upright)
    assert all(kp.angle == 0.0 for kp in upright)
    assert all(kp.angle == -1.0 for kp in DenseSIFT(upright=False).keypoints)


def test_keypoints_lie_inside_the_image(extractor: DenseSIFT) -> None:
    height, width = extractor.image_size
    assert all(0 <= kp.pt[0] < width and 0 <= kp.pt[1] < height for kp in extractor.keypoints)


# --- task 6.2: the same nonzero count, every image ------------------------------------------


def test_every_image_yields_the_same_descriptor_count(extractor, images) -> None:
    """Task 6.2. The failure mode is silent -- a dropped keypoint makes one image's
    histogram incomparable with the rest, and nothing errors."""
    counts = {len(extractor.describe(image)) for image in images}
    assert counts == {extractor.n_keypoints}


def test_no_descriptor_is_all_zero_on_textured_images(extractor, images) -> None:
    """A zero descriptor is a degenerate point for k-means and an undefined direction for
    assignment. Measured on real GTSRB: 0 of 128,000 clean, and 0 of 38,400 under blur 15,
    noise 40 and gamma 2.5 -- heavy blur flattens structure, so this was checked, not assumed."""
    for image in images[:59]:  # index 59 is the deliberately flat one
        norms = np.linalg.norm(extractor.describe(image), axis=1)
        assert (norms > 0).all()


def test_a_perfectly_flat_image_does_produce_zero_descriptors(extractor, images) -> None:
    """The boundary of the claim above, asserted rather than left implicit.

    A constant patch has no gradient, so SIFT returns a zero vector. Real signs never do
    this -- not even blurred at k=15 -- so no special handling exists; but "never happens on
    GTSRB" and "cannot happen" are different statements and the code relies only on the first.
    """
    descriptors = extractor.describe(images[59])
    assert descriptors.shape == (extractor.n_keypoints, DESCRIPTOR_DIM)
    assert (np.linalg.norm(descriptors, axis=1) == 0).all()


@pytest.mark.parametrize(("step", "size"), [(6, 12), (6, 16), (4, 12), (8, 12), (6, 24)])
def test_no_border_pruning_across_configurations(step, size, images) -> None:
    """OpenCV prunes keypoints too close to the border; none of these configurations trip it."""
    extractor = DenseSIFT(step=step, keypoint_size=size)
    assert len(extractor.describe(images[0])) == extractor.n_keypoints


def test_a_pruned_keypoint_would_raise(monkeypatch, extractor, images) -> None:
    """The guard itself: if OpenCV ever returns fewer descriptors, it must be loud."""
    import gtsrb.representations.bovw as module

    class Truncating:
        def compute(self, image, keypoints):
            return keypoints[:-1], np.zeros((len(keypoints) - 1, DESCRIPTOR_DIM), np.float32)

    monkeypatch.setattr(module, "_sift", lambda: Truncating())
    with pytest.raises(RuntimeError, match="pruned keypoints near the border"):
        extractor.describe(images[0])


# --- descriptor properties -------------------------------------------------------------------


def test_descriptor_shape_and_dtype(extractor: DenseSIFT, images: np.ndarray) -> None:
    descriptors = extractor.describe(images[0])
    assert descriptors.shape == (64, DESCRIPTOR_DIM)
    assert descriptors.dtype == np.float32


def test_sample_descriptors_handles_a_non_divisible_count(extractor, images) -> None:
    """n_samples/len(images) rarely divides evenly; the result must still be exact."""
    for n in (137, 500, 999):
        assert extractor.sample_descriptors(images, n).shape == (n, DESCRIPTOR_DIM)


def test_descriptors_are_l2_normalised_by_opencv(extractor, images) -> None:
    """OpenCV returns them already normalised to ~512, so they lie on a sphere -- the right
    footing for k-means. Recorded as a test so a future rescaling cannot pass unnoticed."""
    norms = np.linalg.norm(extractor.describe(images[0]), axis=1)
    assert 500.0 < norms.min() and norms.max() < 520.0


def test_upright_and_auto_orientation_differ(images: np.ndarray) -> None:
    """Not a formality: fixing the angle keeps absolute gradient orientation as signal."""
    upright = DenseSIFT(upright=True).describe(images[0])
    oriented = DenseSIFT(upright=False).describe(images[0])
    assert not np.array_equal(upright, oriented)


def test_describe_is_deterministic(extractor: DenseSIFT, images: np.ndarray) -> None:
    np.testing.assert_array_equal(extractor.describe(images[0]),
                                  extractor.describe(images[0]))


def test_colour_input_uses_the_last_channel(extractor: DenseSIFT, images) -> None:
    """For `clahe_hsv` the V channel is the one CLAHE acted on; averaging would blend hue
    into a gradient operator."""
    colour = np.stack([np.zeros_like(images[0]), np.zeros_like(images[0]), images[0]], axis=-1)
    np.testing.assert_array_equal(extractor.describe(colour), extractor.describe(images[0]))


def test_float_input_in_unit_range_is_accepted(extractor: DenseSIFT, images) -> None:
    np.testing.assert_allclose(
        extractor.describe(images[0].astype(np.float32) / 255.0),
        extractor.describe(images[0]), atol=1e-4)


# --- batching and sampling ---------------------------------------------------------------------


def test_describe_batch_shape(extractor: DenseSIFT, images: np.ndarray) -> None:
    batch = extractor.describe_batch(images[:5])
    assert batch.shape == (5, 64, DESCRIPTOR_DIM)


def test_describe_batch_matches_describe(extractor: DenseSIFT, images: np.ndarray) -> None:
    batch = extractor.describe_batch(images[:4])
    for i in range(4):
        np.testing.assert_array_equal(batch[i], extractor.describe(images[i]))


def test_sample_descriptors_returns_the_requested_count(extractor, images) -> None:
    sample = extractor.sample_descriptors(images, 500)
    assert sample.shape == (500, DESCRIPTOR_DIM)


def test_sample_descriptors_is_deterministic(extractor, images) -> None:
    """The vocabulary at 6.3 must be a pure function of (images, n, seed)."""
    np.testing.assert_array_equal(extractor.sample_descriptors(images, 300),
                                  extractor.sample_descriptors(images, 300))


def test_sample_descriptors_returns_everything_when_asked_for_more(extractor, images) -> None:
    sample = extractor.sample_descriptors(images[:3], 10_000)
    assert sample.shape == (3 * 64, DESCRIPTOR_DIM)


def test_sample_descriptors_draws_from_across_the_batch(extractor, images) -> None:
    """Per-image sampling, not "the first N images" -- otherwise the vocabulary would be
    built from one corner of the dataset."""
    sample = extractor.sample_descriptors(images, len(images))
    assert len(sample) == len(images)


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"step": 0}, "step"), ({"keypoint_size": 1}, "keypoint_size"),
])
def test_invalid_geometry_raises(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        DenseSIFT(**kwargs)


def test_sample_descriptors_rejects_a_bad_count(extractor, images) -> None:
    with pytest.raises(ValueError, match="n_samples"):
        extractor.sample_descriptors(images, 0)


def test_detector_sift_finds_almost_nothing_on_these_crops(images: np.ndarray) -> None:
    """The reason this module exists at all (PROJECT_TASKS section 10).

    A detector is not merely worse here -- on 48x48 crops it returns few or no keypoints, and
    a variable number per image, so the histograms would not be comparable even where it
    does fire.
    """
    detector = cv2.SIFT_create()
    counts = [len(detector.detect(image, None)) for image in images]
    assert min(counts) < DenseSIFT().n_keypoints
    assert len(set(counts)) > 1, "a detector gives a variable count; that is the problem"
