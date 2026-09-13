"""Tests for the preprocessing primitives (task 2.1).

Every representation consumes the output of these functions, so a silent dtype or shape
change here would propagate into all five methods at once and be attributed to the
representations rather than to the preprocessing.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from gtsrb import config, data, preprocessing


@pytest.fixture(scope="module")
def sample_bgr():
    """A real GTSRB crop, not a synthetic array -- PPM decoding is part of what is tested."""
    frame = data.load_annotations("train").iloc[0]
    return preprocessing.load_image(frame["path"])


def test_load_image_reads_ppm_as_bgr_uint8(sample_bgr):
    assert sample_bgr.dtype == np.uint8
    assert sample_bgr.ndim == 3 and sample_bgr.shape[2] == 3


def test_load_image_matches_annotation_dimensions():
    frame = data.load_annotations("train").iloc[0]
    image = preprocessing.load_image(frame["path"])
    assert image.shape[:2] == (frame["height"], frame["width"])


def test_load_image_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        preprocessing.load_image("data/does/not/exist.ppm")


def test_to_gray_shape_and_dtype(sample_bgr):
    gray = preprocessing.to_gray(sample_bgr)
    assert gray.ndim == 2
    assert gray.shape == sample_bgr.shape[:2]
    assert gray.dtype == np.uint8


def test_to_gray_is_idempotent(sample_bgr):
    gray = preprocessing.to_gray(sample_bgr)
    assert np.array_equal(preprocessing.to_gray(gray), gray)


def test_to_hsv_shape_and_dtype(sample_bgr):
    hsv = preprocessing.to_hsv(sample_bgr)
    assert hsv.shape == sample_bgr.shape
    assert hsv.dtype == np.uint8


def test_to_hsv_rejects_grayscale(sample_bgr):
    with pytest.raises(ValueError):
        preprocessing.to_hsv(preprocessing.to_gray(sample_bgr))


def test_to_hsv_hue_of_pure_red_bgr():
    """Guards the BGR-vs-RGB convention: pure red must land at hue 0, not 120."""
    red_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    red_bgr[:, :, 2] = 255  # BGR -> red is the LAST channel
    hue = preprocessing.to_hsv(red_bgr)[:, :, 0]
    assert np.all(hue == 0)


def test_clahe_preserves_shape_and_dtype(sample_bgr):
    gray = preprocessing.resize(preprocessing.to_gray(sample_bgr))
    out = preprocessing.clahe(gray)
    assert out.shape == gray.shape
    assert out.dtype == np.uint8


def test_clahe_increases_contrast_on_a_flat_gradient():
    low_contrast = np.linspace(100, 140, 48 * 48).reshape(48, 48).astype(np.uint8)
    out = preprocessing.clahe(low_contrast)
    assert out.std() > low_contrast.std()


def test_clahe_on_hsv_touches_only_the_value_channel():
    """Equalising H would rotate hues and destroy the colour cue the signs depend on."""
    rng = np.random.default_rng(0)
    hsv = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
    out = preprocessing.clahe(hsv)
    assert np.array_equal(out[:, :, 0], hsv[:, :, 0]), "hue was modified"
    assert np.array_equal(out[:, :, 1], hsv[:, :, 1]), "saturation was modified"
    assert not np.array_equal(out[:, :, 2], hsv[:, :, 2]), "value was not equalised"


def test_clahe_does_not_mutate_its_input():
    rng = np.random.default_rng(0)
    hsv = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
    before = hsv.copy()
    preprocessing.clahe(hsv)
    assert np.array_equal(hsv, before)


def test_resize_to_configured_size(sample_bgr):
    out = preprocessing.resize(preprocessing.to_gray(sample_bgr))
    assert out.shape == config.IMAGE_SIZE
    assert out.dtype == np.uint8


def test_resize_keeps_channels_for_colour_input(sample_bgr):
    out = preprocessing.resize(sample_bgr)
    assert out.shape == (*config.IMAGE_SIZE, 3)


def test_resize_picks_inter_area_when_shrinking():
    """A 200x200 source is downscaled: INTER_AREA, which averages, must be used."""
    big = np.zeros((200, 200), dtype=np.uint8)
    big[::2] = 255  # alternating rows -- aliases badly under INTER_LINEAR
    adaptive = preprocessing.resize(big)
    expected = cv2.resize(big, (48, 48), interpolation=cv2.INTER_AREA)
    assert np.array_equal(adaptive, expected)


def test_resize_picks_inter_linear_when_enlarging():
    """A 20x20 source is upscaled, where INTER_AREA degenerates to nearest-neighbour."""
    small = np.arange(400, dtype=np.uint8).reshape(20, 20)
    adaptive = preprocessing.resize(small)
    expected = cv2.resize(small, (48, 48), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(adaptive, expected)


def test_resize_interpolation_can_be_overridden():
    small = np.arange(400, dtype=np.uint8).reshape(20, 20)
    out = preprocessing.resize(small, interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(out, cv2.resize(small, (48, 48), interpolation=cv2.INTER_NEAREST))


def test_gaussian_blur_smooths_and_preserves_dtype():
    image = np.zeros((48, 48), dtype=np.uint8)
    image[24, 24] = 255
    out = preprocessing.gaussian_blur(image, ksize=3)
    assert out.dtype == np.uint8
    assert out[24, 24] < 255, "centre should be attenuated"
    assert out[23, 24] > 0, "energy should spread to neighbours"


def test_gaussian_blur_rejects_even_kernel():
    with pytest.raises(ValueError):
        preprocessing.gaussian_blur(np.zeros((8, 8), np.uint8), ksize=4)


def test_gaussian_blur_zero_kernel_is_a_noop():
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    assert np.array_equal(preprocessing.gaussian_blur(image, ksize=0), image)


def test_crop_roi_extracts_the_box(sample_bgr):
    frame = data.load_annotations("train").iloc[0]
    crop = preprocessing.crop_roi(
        sample_bgr, frame["roi_x1"], frame["roi_y1"], frame["roi_x2"], frame["roi_y2"]
    )
    assert crop.shape[0] == frame["roi_h"]
    assert crop.shape[1] == frame["roi_w"]


def test_crop_roi_clamps_out_of_bounds_boxes():
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    crop = preprocessing.crop_roi(image, -10, -10, 100, 100)
    assert crop.shape[:2] == (30, 30)


def test_crop_roi_never_returns_an_empty_array():
    image = np.zeros((30, 30), dtype=np.uint8)
    assert preprocessing.crop_roi(image, 20, 20, 5, 5).size > 0


def test_full_chain_produces_cacheable_uint8(sample_bgr):
    """Task 2.3 caches these arrays as uint8 .npy -- the chain must not promote to float."""
    out = preprocessing.clahe(preprocessing.resize(preprocessing.to_gray(sample_bgr)))
    assert out.dtype == np.uint8
    assert out.shape == config.IMAGE_SIZE


# --- named configs (task 2.2) ----------------------------------------------------------


def test_the_three_ablation_configs_exist():
    assert set(preprocessing.PREPROC_CONFIGS) == {"raw_gray", "clahe_gray", "clahe_hsv"}


@pytest.mark.parametrize("name", ["raw_gray", "clahe_gray", "clahe_hsv"])
def test_config_output_matches_its_declared_shape(sample_bgr, name):
    """Declared shape/flat_dim feed Table 1's feature-dimension column -- they must agree."""
    cfg = preprocessing.get_config(name)
    out = cfg.apply(sample_bgr)
    assert out.shape == cfg.shape
    assert out.dtype == np.uint8
    assert out.size == cfg.flat_dim


def test_flat_dims_are_as_expected():
    assert preprocessing.get_config("raw_gray").flat_dim == 48 * 48
    assert preprocessing.get_config("clahe_gray").flat_dim == 48 * 48
    assert preprocessing.get_config("clahe_hsv").flat_dim == 48 * 48 * 3


def test_get_config_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown preproc config"):
        preprocessing.get_config("nope")


def test_clahe_gray_differs_from_raw_gray(sample_bgr):
    """If these matched, the ablation would be comparing a config against itself."""
    raw = preprocessing.preprocess(sample_bgr, "raw_gray")
    equalised = preprocessing.preprocess(sample_bgr, "clahe_gray")
    assert not np.array_equal(raw, equalised)


def test_clahe_hsv_keeps_hue_from_the_plain_hsv_pipeline(sample_bgr):
    plain = preprocessing.resize(preprocessing.to_hsv(sample_bgr))
    equalised = preprocessing.preprocess(sample_bgr, "clahe_hsv")
    assert np.array_equal(plain[:, :, 0], equalised[:, :, 0])


def test_load_and_preprocess_uses_the_provided_framing_by_default():
    """GTSRB images are used as the dataset frames them -- sign plus its ~17 % margin."""
    row = data.load_annotations("train").iloc[0]
    default = preprocessing.load_and_preprocess(row, "raw_gray")
    whole = preprocessing.load_and_preprocess(row, "raw_gray", roi=False)
    assert np.array_equal(default, whole)


def test_load_and_preprocess_can_still_crop_to_roi():
    """roi=True is how the GTSDB evaluation reproduces GTSRB's framing at jitter level 0."""
    row = data.load_annotations("train").iloc[0]
    cropped = preprocessing.load_and_preprocess(row, "raw_gray", roi=True)
    whole = preprocessing.load_and_preprocess(row, "raw_gray", roi=False)
    assert not np.array_equal(cropped, whole)


def test_load_and_preprocess_returns_declared_shape():
    row = data.load_annotations("train").iloc[0]
    for name, cfg in preprocessing.PREPROC_CONFIGS.items():
        assert preprocessing.load_and_preprocess(row, name).shape == cfg.shape


def test_default_preproc_is_a_known_config():
    assert preprocessing.DEFAULT_PREPROC in preprocessing.PREPROC_CONFIGS
