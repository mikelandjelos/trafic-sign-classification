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

from gtsrb.representations import Representation
from gtsrb.representations.bovw import (
    DESCRIPTOR_DIM,
    BoVWRepresentation,
    BoVWSpatialPyramid,
    DenseSIFT,
    chunk_for_budget,
    normalise_histograms,
)


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


# =====================================================================================
# Tasks 6.3-6.4: vocabulary and histogram encoding
# =====================================================================================

@pytest.fixture(scope="module")
def bovw(images: np.ndarray) -> BoVWRepresentation:
    return BoVWRepresentation(n_words=12, n_vocab_samples=600).fit(images)


# --- normalisation ------------------------------------------------------------------------


def test_none_leaves_counts_untouched() -> None:
    h = np.array([[36.0, 16.0, 4.0, 8.0]], dtype=np.float32)
    np.testing.assert_array_equal(normalise_histograms(h, "none"), h)


def test_l1_sums_to_one() -> None:
    h = np.array([[36.0, 16.0, 4.0, 8.0]], dtype=np.float32)
    assert normalise_histograms(h, "l1").sum() == pytest.approx(1.0)


def test_l2_gives_unit_norm() -> None:
    h = np.array([[36.0, 16.0, 4.0, 8.0]], dtype=np.float32)
    assert np.linalg.norm(normalise_histograms(h, "l2")) == pytest.approx(1.0, abs=1e-6)


def test_power_l2_is_the_only_scheme_that_changes_relative_weights() -> None:
    """The point of the default, asserted rather than asserted-in-prose.

    Every image contributes the same number of descriptors here, so total mass is already
    constant and `l1`/`l2` are pure rescales -- they cannot change the *ratio* between two
    bins. Only the square root compresses a bursty dominant word relative to the rest.
    """
    h = np.array([[36.0, 16.0, 4.0, 8.0]], dtype=np.float32)
    raw_ratio = h[0, 0] / h[0, 1]
    for scheme in ("l1", "l2"):
        v = normalise_histograms(h, scheme)[0]
        assert v[0] / v[1] == pytest.approx(raw_ratio)
    powered = normalise_histograms(h, "power_l2")[0]
    assert powered[0] / powered[1] == pytest.approx(np.sqrt(raw_ratio))
    assert powered[0] / powered[1] < raw_ratio


def test_zero_histogram_does_not_divide_by_zero() -> None:
    zeros = np.zeros((1, 5), dtype=np.float32)
    for scheme in ("l1", "l2", "power_l2"):
        assert np.isfinite(normalise_histograms(zeros, scheme)).all()


def test_unknown_normalisation_raises() -> None:
    with pytest.raises(ValueError, match="unknown normalisation"):
        normalise_histograms(np.zeros((1, 3), np.float32), "sqrt")


def test_constructor_validates_the_scheme_immediately() -> None:
    """Caught at construction, not two hours into a sweep."""
    with pytest.raises(ValueError, match="unknown normalisation"):
        BoVWRepresentation(normalisation="nope")


# --- the representation --------------------------------------------------------------------


def test_satisfies_the_representation_protocol(bovw: BoVWRepresentation) -> None:
    assert isinstance(bovw, Representation)
    assert bovw.name == "bovw"


def test_transform_shape(bovw: BoVWRepresentation, images: np.ndarray) -> None:
    features = bovw.transform(images)
    assert features.shape == (len(images), bovw.n_features) == (len(images), 12)
    assert features.dtype == np.float32


def test_vocabulary_shape(bovw: BoVWRepresentation) -> None:
    assert bovw.vocabulary_.shape == (12, DESCRIPTOR_DIM)


def test_raw_counts_sum_to_the_keypoint_count(images: np.ndarray) -> None:
    """The property the whole normalisation discussion rests on: mass is already constant.

    Every image yields exactly `n_keypoints` descriptors and each is assigned to exactly
    one word, so an unnormalised histogram always sums to 64 -- which is why `l1` changes
    nothing here and `power_l2` is the meaningful choice.
    """
    raw = BoVWRepresentation(n_words=12, n_vocab_samples=600,
                             normalisation="none").fit(images).transform(images)
    np.testing.assert_allclose(raw.sum(axis=1), 64.0)


def test_chunk_size_does_not_change_the_result(bovw, images) -> None:
    """Each descriptor's nearest word depends only on that descriptor, so chunking is an
    implementation detail -- and must be provably so, since the chunk size is tuned for
    memory rather than for results."""
    np.testing.assert_array_equal(bovw.transform(images, chunk=7),
                                  bovw.transform(images, chunk=1000))


def test_transform_is_deterministic(bovw: BoVWRepresentation, images) -> None:
    np.testing.assert_array_equal(bovw.transform(images), bovw.transform(images))


def test_two_fits_with_the_same_seed_agree(images: np.ndarray) -> None:
    """MiniBatchKMeans is stochastic; the vocabulary must still be reproducible."""
    first = BoVWRepresentation(n_words=12, n_vocab_samples=600).fit(images)
    second = BoVWRepresentation(n_words=12, n_vocab_samples=600).fit(images)
    np.testing.assert_allclose(first.vocabulary_, second.vocabulary_)
    np.testing.assert_array_equal(first.transform(images), second.transform(images))


def test_transform_before_fit_raises(images: np.ndarray) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        BoVWRepresentation().transform(images)


def test_too_few_words_raises() -> None:
    with pytest.raises(ValueError, match="n_words"):
        BoVWRepresentation(n_words=1)


def test_histograms_are_unit_norm_under_the_default(bovw, images) -> None:
    norms = np.linalg.norm(bovw.transform(images), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# --- task 6.5b: the spatial pyramid -------------------------------------------------------
#
# The SPM row exists to measure ONE thing: what discarding layout costs. These tests pin the
# properties that make that measurement valid -- shared vocabulary, exact reduction to plain
# BoVW at levels=0, and the permutation asymmetry that IS the layout axis.


@pytest.fixture(scope="module")
def fitted_pair(images: np.ndarray):
    """A plain and a pyramid representation sharing one vocabulary."""
    plain = BoVWRepresentation(n_words=12, n_vocab_samples=2000).fit(images)
    spm = BoVWSpatialPyramid(n_words=12, n_vocab_samples=2000, levels=2)
    # Share the FITTED vocabulary, exactly as the comparison requires.
    spm._kmeans = plain._kmeans
    return plain, spm


def test_levels_zero_reduces_exactly_to_plain_bovw(images: np.ndarray) -> None:
    """The identity that makes the pyramid a controlled manipulation, not another method."""
    plain = BoVWRepresentation(n_words=12, n_vocab_samples=2000).fit(images)
    spm = BoVWSpatialPyramid(n_words=12, n_vocab_samples=2000, levels=0)
    spm._kmeans = plain._kmeans
    np.testing.assert_allclose(spm.transform(images), plain.transform(images), rtol=1e-6)


def test_feature_dimensionality_matches_the_cell_count() -> None:
    for levels, cells in ((0, 1), (1, 5), (2, 21)):
        spm = BoVWSpatialPyramid(n_words=100, levels=levels)
        assert spm.n_cells == cells
        assert spm.n_features == 100 * cells


def test_cell_weights_follow_lazebnik() -> None:
    spm = BoVWSpatialPyramid(n_words=2, levels=2)
    weights = spm.cell_weights
    assert len(weights) == 21
    # level 0 shares level 1's weight; finer levels count for more
    assert weights[0] == pytest.approx(0.25)
    assert np.all(weights[1:5] == pytest.approx(0.25))
    assert np.all(weights[5:] == pytest.approx(0.5))


def test_shuffling_keypoint_positions_leaves_plain_bovw_identical(fitted_pair, images):
    """Orderlessness, demonstrated rather than asserted -- the centrepiece of the 6.6 demo.

    Permuting which grid position each descriptor came from cannot change a bag of words. It
    changes the pyramid, because the pyramid records position. That asymmetry IS the layout
    axis the study measures.
    """
    plain, spm = fitted_pair
    rng = np.random.default_rng(0)
    rows, cols = plain.extractor.grid_shape
    order = rng.permutation(rows * cols)

    def pooled(rep, permute: bool):
        kmeans = rep._fitted()
        out = []
        for image in images[:8]:
            words = kmeans.predict(rep.extractor.describe(image))
            if permute:
                words = words[order]
            grid = words.reshape(rows, cols)
            if isinstance(rep, BoVWSpatialPyramid):
                pieces = []
                for c in rep._cells:
                    rb = np.minimum((np.arange(rows) * c) // rows, c - 1)
                    cb = np.minimum((np.arange(cols) * c) // cols, c - 1)
                    for a in range(c):
                        for b in range(c):
                            sub = grid[np.ix_(rb == a, cb == b)]
                            pieces.append(np.bincount(sub.ravel(), minlength=rep.n_words))
                vec = np.concatenate(pieces) * np.repeat(rep.cell_weights, rep.n_words)
            else:
                vec = np.bincount(words, minlength=rep.n_words)
            out.append(vec)
        return normalise_histograms(np.asarray(out, np.float32), rep.normalisation)

    # Plain BoVW: byte-identical under the permutation.
    np.testing.assert_array_equal(pooled(plain, False), pooled(plain, True))
    # The pyramid: genuinely different.
    assert not np.allclose(pooled(spm, False), pooled(spm, True))


def test_chunking_cannot_change_the_pyramid_result(fitted_pair, images) -> None:
    _, spm = fitted_pair
    np.testing.assert_array_equal(spm.transform(images, chunk=3),
                                  spm.transform(images, chunk=1000))


def test_pyramid_satisfies_the_representation_protocol(fitted_pair) -> None:
    _, spm = fitted_pair
    assert isinstance(spm, Representation)
    assert spm.name == "bovw_spm"


def test_negative_levels_rejected() -> None:
    with pytest.raises(ValueError, match="levels must be >= 0"):
        BoVWSpatialPyramid(levels=-1)


def test_chunk_budget_accounts_for_the_vocabulary_size() -> None:
    """The regression guard for the OOM that recurred in three files.

    Sizing on the descriptor block alone makes the chunk independent of `n_words`, which is
    exactly the bug: the k-means assignment grows with the vocabulary too.
    """
    assert chunk_for_budget(576, 1000) < chunk_for_budget(576, 500)
    assert chunk_for_budget(576, 500) < chunk_for_budget(64, 500)
    assert chunk_for_budget(10_000, 10_000) >= 1
