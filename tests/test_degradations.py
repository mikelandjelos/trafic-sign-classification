"""Tests for the controlled degradations (task 3.x).

The property the whole robustness study rests on is that **every method sees identical
degraded pixels**. If the noise a given image receives depended on batch position, batch
size, or evaluation order, the five methods would be compared on different data and the
representation x stressor interaction -- the project's contribution -- would be
contaminated. That invariance is what most of these tests pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from gtsrb import config, degradations


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    return rng.integers(40, 200, size=(12, 48, 48), dtype=np.uint8)


@pytest.fixture
def keys(images):
    return [f"data/fake/{i:05d}.ppm" for i in range(len(images))]


# --- identity ---------------------------------------------------------------------------


def test_level_zero_is_exactly_identity(images, keys):
    """The clean condition and level 0 must be the same pixels, not merely similar."""
    out = degradations.apply(images, "noise", 0, keys)
    assert np.array_equal(out, images)


def test_clean_degradation_is_identity(images, keys):
    assert np.array_equal(degradations.apply(images, "clean", 0, keys), images)


def test_sigma_zero_returns_input_unchanged():
    image = np.full((8, 8), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    assert np.array_equal(degradations.gaussian_noise(image, 0, rng), image)


# --- the invariance the study depends on ------------------------------------------------


def test_same_image_gets_same_noise_regardless_of_batch_position(images, keys):
    """Keyed on path, not position -- so a subset matches the full batch exactly."""
    subset_order = [5, 0, 11, 2]
    from_subset = degradations.apply(
        images[subset_order], "noise", 20, [keys[i] for i in subset_order]
    )
    from_full = degradations.apply(images, "noise", 20, keys)[subset_order]
    assert np.array_equal(from_subset, from_full)


def test_repeated_calls_are_identical(images, keys):
    first = degradations.apply(images, "noise", 20, keys)
    second = degradations.apply(images, "noise", 20, keys)
    assert np.array_equal(first, second)


def test_noise_does_not_depend_on_global_rng_state(images, keys):
    """A single global stream would make the result depend on what ran earlier."""
    np.random.seed(1)
    first = degradations.apply(images, "noise", 20, keys)
    np.random.seed(999)
    _ = np.random.normal(size=10_000)
    second = degradations.apply(images, "noise", 20, keys)
    assert np.array_equal(first, second)


def test_different_levels_give_different_pixels(images, keys):
    assert not np.array_equal(
        degradations.apply(images, "noise", 10, keys),
        degradations.apply(images, "noise", 20, keys),
    )


def test_different_images_get_different_noise(images, keys):
    """Otherwise the same noise field would be reused across the dataset."""
    flat = np.full((2, 48, 48), 128, dtype=np.uint8)
    out = degradations.apply(flat, "noise", 20, ["a.ppm", "b.ppm"])
    assert not np.array_equal(out[0], out[1])


# --- statistical behaviour ---------------------------------------------------------------


@pytest.mark.parametrize("sigma", [5, 10, 20])
def test_measured_std_is_close_to_sigma(sigma):
    """Away from the clipping limits the perturbation should match the requested sigma."""
    mid_grey = np.full((64, 64, 64), 128, dtype=np.uint8)
    keys = [f"{i}.ppm" for i in range(len(mid_grey))]
    out = degradations.apply(mid_grey, "noise", sigma, keys)
    delta = out.astype(int) - mid_grey.astype(int)
    assert delta.std() == pytest.approx(sigma, rel=0.05)
    assert abs(delta.mean()) < 0.5, "noise should be zero-mean"


def test_clipping_reduces_measured_std_at_extremes():
    """Documented behaviour, not a bug: saturation pulls the realised std below sigma."""
    bright = np.full((32, 48, 48), 250, dtype=np.uint8)
    keys = [f"{i}.ppm" for i in range(len(bright))]
    out = degradations.apply(bright, "noise", 40, keys)
    delta = out.astype(int) - bright.astype(int)
    assert delta.std() < 40
    assert out.max() <= 255


def test_output_stays_in_range_and_dtype(images, keys):
    out = degradations.apply(images, "noise", 40, keys)
    assert out.dtype == np.uint8
    assert out.shape == images.shape
    assert out.min() >= 0 and out.max() <= 255


def test_works_on_three_channel_input(keys):
    rng = np.random.default_rng(0)
    colour = rng.integers(40, 200, size=(12, 48, 48, 3), dtype=np.uint8)
    out = degradations.apply(colour, "noise", 20, keys)
    assert out.shape == colour.shape and out.dtype == np.uint8


def test_input_is_not_mutated(images, keys):
    before = images.copy()
    degradations.apply(images, "noise", 20, keys)
    assert np.array_equal(images, before)


# --- guards -------------------------------------------------------------------------------


def test_rejects_unknown_degradation(images, keys):
    with pytest.raises(KeyError, match="unknown degradation"):
        degradations.apply(images, "sparkle", 1, keys)


def test_rejects_off_grid_level(images, keys):
    """Off-grid levels would produce rows of results.csv that cannot be compared."""
    with pytest.raises(ValueError, match="not one of"):
        degradations.apply(images, "noise", 7, keys)


def test_rejects_mismatched_keys(images, keys):
    with pytest.raises(ValueError, match="keys has"):
        degradations.apply(images, "noise", 20, keys[:3])


def test_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma must be"):
        degradations.gaussian_noise(
            np.zeros((4, 4), np.uint8), -1, np.random.default_rng(0)
        )


def test_levels_for_exposes_the_grid():
    assert degradations.levels_for("noise") == (0, 5, 10, 20, 40)
    assert degradations.levels_for("blur") == (0, 3, 5, 9, 15)
    with pytest.raises(KeyError):
        degradations.levels_for("nope")


def test_identity_level_is_declared_not_positional():
    """gamma's identity is 1.0, in the MIDDLE of its range -- levels[0] is not the no-op.

    Anything iterating the grid must ask `identity_for()` rather than assume `levels[0]`.
    """
    assert degradations.identity_for("noise") == 0
    assert degradations.identity_for("blur") == 0
    assert degradations.identity_for("gamma") == 1.0
    assert degradations.levels_for("gamma")[0] == 0.4  # NOT the identity
    assert degradations.identity_for("gamma") != degradations.levels_for("gamma")[0]


def test_every_identity_is_one_of_its_levels():
    for name in ("noise", "blur", "gamma"):
        assert degradations.identity_for(name) in degradations.levels_for(name)


# --- gamma (task 3.3) ---------------------------------------------------------------------


def test_gamma_identity_returns_input_unchanged(images, keys):
    assert np.array_equal(degradations.apply(images, "gamma", 1.0, keys), images)


def test_gamma_below_one_brightens_and_above_one_darkens(images, keys):
    baseline = images.astype(float).mean()
    assert degradations.apply(images, "gamma", 0.4, keys).astype(float).mean() > baseline
    assert degradations.apply(images, "gamma", 0.7, keys).astype(float).mean() > baseline
    assert degradations.apply(images, "gamma", 1.5, keys).astype(float).mean() < baseline
    assert degradations.apply(images, "gamma", 2.5, keys).astype(float).mean() < baseline


def test_gamma_is_monotone_in_level(images, keys):
    """Brightness must move monotonically across the grid, or the curve is unreadable."""
    means = [
        degradations.apply(images, "gamma", g, keys).astype(float).mean()
        for g in degradations.levels_for("gamma")
    ]
    assert means == sorted(means, reverse=True)


def test_gamma_lut_is_monotone_and_fixes_the_endpoints():
    """Black stays black and white stays white; ordering of intensities is preserved."""
    for gamma in (0.4, 0.7, 1.5, 2.5):
        lut = degradations._gamma_lut(gamma)
        assert lut.shape == (256,) and lut.dtype == np.uint8
        assert (np.diff(lut.astype(int)) >= 0).all()
        assert lut[0] == 0 and lut[255] == 255


def test_gamma_is_deterministic_and_seed_independent(images, keys):
    """It is the one stressor with no randomness -- the rng is ignored entirely."""
    np.random.seed(1)
    first = degradations.apply(images, "gamma", 2.5, keys)
    np.random.seed(999)
    second = degradations.apply(images, "gamma", 2.5, keys)
    assert np.array_equal(first, second)
    assert degradations.get_degradation("gamma").stochastic is False


def test_gamma_ignores_the_key(images):
    """Unlike noise and blur, every image gets the same mapping."""
    flat = np.full((2, 8, 8), 100, dtype=np.uint8)
    out = degradations.apply(flat, "gamma", 2.5, ["a.ppm", "b.ppm"])
    assert np.array_equal(out[0], out[1])


def test_gamma_rejects_non_positive_values():
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="gamma must be"):
            degradations.gamma_correction(np.zeros((4, 4), np.uint8), bad)


def test_gamma_works_on_three_channel_input(keys):
    rng = np.random.default_rng(0)
    colour = rng.integers(40, 200, size=(12, 48, 48, 3), dtype=np.uint8)
    out = degradations.apply(colour, "gamma", 0.4, keys)
    assert out.shape == colour.shape and out.dtype == np.uint8


def test_gamma_output_stays_in_range(images, keys):
    for gamma in degradations.levels_for("gamma"):
        out = degradations.apply(images, "gamma", gamma, keys)
        assert out.dtype == np.uint8 and out.min() >= 0 and out.max() <= 255


def test_gamma_rejects_off_grid_level(images, keys):
    with pytest.raises(ValueError, match="not one of"):
        degradations.apply(images, "gamma", 3.0, keys)


# --- motion blur (task 3.2) ---------------------------------------------------------------


@pytest.mark.parametrize("size", [3, 5, 9, 15])
@pytest.mark.parametrize("angle", [0, 30, 45, 90, 135])
def test_blur_kernel_is_normalised(size, angle):
    """A kernel that does not sum to 1 would also change brightness, confounding 3.3."""
    kernel = degradations.motion_blur_kernel(size, angle)
    assert kernel.shape == (size, size)
    assert kernel.sum() == pytest.approx(1.0)
    assert (kernel >= 0).all()


def test_blur_kernel_rejects_even_and_zero_sizes():
    for bad in (0, 2, 4, -1):
        with pytest.raises(ValueError, match="odd"):
            degradations.motion_blur_kernel(bad, 0)


def test_blur_kernel_is_horizontal_at_zero_degrees():
    kernel = degradations.motion_blur_kernel(5, 0)
    assert (kernel[2] > 0).all(), "the centre row should carry the line"
    assert kernel[0].sum() == 0 and kernel[4].sum() == 0


def test_blur_level_zero_is_identity(images, keys):
    assert np.array_equal(degradations.apply(images, "blur", 0, keys), images)


def test_blur_preserves_mean_intensity(images, keys):
    """Blur must not darken or brighten -- that is a separate, separately measured effect."""
    for ksize in (3, 5, 9, 15):
        out = degradations.apply(images, "blur", ksize, keys)
        assert out.astype(float).mean() == pytest.approx(
            images.astype(float).mean(), abs=1.0
        )


def test_blur_reduces_high_frequency_energy_monotonically(images, keys):
    """The degradation should get strictly stronger with kernel size."""
    energies = []
    for ksize in (0, 3, 5, 9, 15):
        out = degradations.apply(images, "blur", ksize, keys)
        energies.append(np.abs(np.diff(out.astype(float), axis=2)).mean())
    assert energies == sorted(energies, reverse=True)
    assert energies[-1] < 0.6 * energies[0]


def test_blur_does_not_darken_the_borders(keys):
    """BORDER_REFLECT_101, not zero padding -- which would vignette with kernel size."""
    flat = np.full((12, 48, 48), 200, dtype=np.uint8)
    out = degradations.apply(flat, "blur", 15, keys)
    # A constant image must survive a normalised blur unchanged, edges included.
    assert out.min() >= 199, f"border darkened to {out.min()} (zero padding?)"


def test_blur_angle_varies_between_images():
    """A single shared angle would make the stressor systematic rather than random."""
    impulse = np.zeros((2, 48, 48), dtype=np.uint8)
    impulse[:, 24, 24] = 255
    out = degradations.apply(impulse, "blur", 15, ["a.ppm", "b.ppm"])
    assert not np.array_equal(out[0], out[1])


def test_blur_is_deterministic_and_position_independent(images, keys):
    order = [7, 2, 0]
    from_subset = degradations.apply(images[order], "blur", 9, [keys[i] for i in order])
    from_full = degradations.apply(images, "blur", 9, keys)[order]
    assert np.array_equal(from_subset, from_full)


def test_blur_works_on_three_channel_input(keys):
    rng = np.random.default_rng(0)
    colour = rng.integers(40, 200, size=(12, 48, 48, 3), dtype=np.uint8)
    out = degradations.apply(colour, "blur", 9, keys)
    assert out.shape == colour.shape and out.dtype == np.uint8


def test_blur_rejects_off_grid_level(images, keys):
    with pytest.raises(ValueError, match="not one of"):
        degradations.apply(images, "blur", 7, keys)


def test_rng_for_is_the_seeding_mechanism():
    """Guards the contract in CLAUDE.md: keyed by (degradation, level, image)."""
    expected = config.rng_for("noise", 20, "data/fake/00000.ppm").normal(0, 20, (48, 48))
    image = np.full((1, 48, 48), 128, dtype=np.uint8)
    out = degradations.apply(image, "noise", 20, ["data/fake/00000.ppm"])
    assert np.array_equal(out[0], np.rint(np.clip(128 + expected, 0, 255)).astype(np.uint8))


def test_noise_is_rounded_not_truncated():
    """Truncation would darken every perturbed pixel by ~0.5 levels at every sigma."""
    mid_grey = np.full((64, 64, 64), 128, dtype=np.uint8)
    keys = [f"{i}.ppm" for i in range(len(mid_grey))]
    delta = degradations.apply(mid_grey, "noise", 20, keys).astype(int) - 128
    assert abs(delta.mean()) < 0.05, f"mean shift {delta.mean():+.3f} suggests truncation"
