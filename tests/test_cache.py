"""Tests for the preprocessing cache (task 2.3).

The property that matters is **transparency**: a cached array must be bit-for-bit identical
to recomputing from the PPM files. A cache that quietly altered the pixels would corrupt
every result in the grid while presenting as a speedup, so the uncached path is treated as
the reference implementation and the cache is checked against it.

These tests redirect CACHE_DIR to a tmp_path so they never touch the real cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from gtsrb import cache, config, data, preprocessing


def _clear_annotation_memo() -> None:
    """`_split_annotations` may currently be monkeypatched to a plain function."""
    getattr(cache._split_annotations, "cache_clear", lambda: None)()


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    _clear_annotation_memo()
    yield tmp_path
    _clear_annotation_memo()


@pytest.fixture(scope="module")
def sample_frame():
    """A small slice of the test split -- enough rows to catch ordering bugs."""
    return data.load_annotations("test").iloc[:24]


# --- the transparency guarantee --------------------------------------------------------


@pytest.mark.parametrize("preproc", ["raw_gray", "clahe_gray", "clahe_hsv"])
def test_cached_equals_uncached_bit_for_bit(tmp_cache, sample_frame, preproc, monkeypatch):
    """The whole point of the cache: it must change nothing."""
    monkeypatch.setattr(cache, "_split_annotations", lambda split: sample_frame)
    cached = cache.load_images(sample_frame, preproc, use_cache=True)
    reference = cache.load_images(sample_frame, preproc, use_cache=False)
    assert np.array_equal(cached, reference)
    assert cached.dtype == reference.dtype == np.uint8


def test_verify_reports_identical(tmp_cache, sample_frame, monkeypatch):
    monkeypatch.setattr(cache, "_split_annotations", lambda split: sample_frame)
    report = cache.verify("test", "raw_gray", n_samples=0)
    assert report["identical"] is True
    assert report["mismatched_images"] == 0
    assert report["compared"] == len(sample_frame)


def test_row_order_is_preserved(tmp_cache, sample_frame, monkeypatch):
    """Row i of the result must be row i of the frame -- for any subset, in any order."""
    monkeypatch.setattr(cache, "_split_annotations", lambda split: sample_frame)
    subset = sample_frame.iloc[[7, 0, 19, 3]]
    cached = cache.load_images(subset, "raw_gray", use_cache=True)
    reference = cache.load_images(subset, "raw_gray", use_cache=False)
    assert np.array_equal(cached, reference)

    full = cache.load_images(sample_frame, "raw_gray", use_cache=True)
    for position, source_row in enumerate([7, 0, 19, 3]):
        assert np.array_equal(cached[position], full[source_row])


def test_subsets_share_one_cached_array(tmp_cache, sample_frame, monkeypatch):
    """A train/val split must not duplicate storage -- both index the same file."""
    monkeypatch.setattr(cache, "_split_annotations", lambda split: sample_frame)
    cache.load_images(sample_frame.iloc[:10], "raw_gray")
    cache.load_images(sample_frame.iloc[10:], "raw_gray")
    assert len(list(tmp_cache.glob("*.npy"))) == 1


# --- staleness -------------------------------------------------------------------------


def test_cache_is_invalidated_when_preprocessing_changes(tmp_cache, sample_frame, monkeypatch):
    """Serving pixels built with different parameters is worse than having no cache."""
    monkeypatch.setattr(cache, "_split_annotations", lambda split: sample_frame)
    first = cache.load_images(sample_frame, "clahe_gray", use_cache=True)
    assert cache.is_valid("test", "clahe_gray")

    monkeypatch.setattr(preprocessing, "CLAHE_CLIP_LIMIT", 40.0)
    assert not cache.is_valid("test", "clahe_gray"), "stale cache reported as valid"

    second = cache.load_images(sample_frame, "clahe_gray", use_cache=True)
    assert not np.array_equal(first, second), "cache was not rebuilt after a param change"
    reference = cache.load_images(sample_frame, "clahe_gray", use_cache=False)
    assert np.array_equal(second, reference)


def test_cache_is_invalidated_when_the_row_set_changes(tmp_cache, sample_frame, monkeypatch):
    monkeypatch.setattr(cache, "_split_annotations", lambda split: sample_frame)
    cache.build("test", "raw_gray", progress=False)
    assert cache.is_valid("test", "raw_gray")

    monkeypatch.setattr(cache, "_split_annotations", lambda split: sample_frame.iloc[:12])
    assert not cache.is_valid("test", "raw_gray")


def test_missing_cache_is_not_valid(tmp_cache, sample_frame, monkeypatch):
    monkeypatch.setattr(cache, "_split_annotations", lambda split: sample_frame)
    assert not cache.is_valid("test", "raw_gray")


def test_build_is_idempotent(tmp_cache, sample_frame, monkeypatch):
    monkeypatch.setattr(cache, "_split_annotations", lambda split: sample_frame)
    path = cache.build("test", "raw_gray", progress=False)
    first_mtime = path.stat().st_mtime_ns
    cache.build("test", "raw_gray", progress=False)
    assert path.stat().st_mtime_ns == first_mtime, "valid cache was rebuilt needlessly"


# --- shapes, dtypes and guards ---------------------------------------------------------


@pytest.mark.parametrize("preproc", ["raw_gray", "clahe_gray", "clahe_hsv"])
def test_shapes_match_the_declared_config(tmp_cache, sample_frame, preproc, monkeypatch):
    monkeypatch.setattr(cache, "_split_annotations", lambda split: sample_frame)
    cfg = preprocessing.get_config(preproc)
    images = cache.load_images(sample_frame, preproc)
    assert images.shape == (len(sample_frame), *cfg.shape)


def test_mmap_returns_the_same_pixels(tmp_cache, sample_frame, monkeypatch):
    monkeypatch.setattr(cache, "_split_annotations", lambda split: sample_frame)
    in_ram = cache.load_images(sample_frame, "raw_gray", mmap=False)
    mapped = cache.load_images(sample_frame, "raw_gray", mmap=True)
    assert np.array_equal(in_ram, mapped)


def test_empty_frame_returns_empty_array_of_the_right_shape():
    empty = data.load_annotations("test").iloc[:0]
    out = cache.load_images(empty, "clahe_hsv", use_cache=False)
    assert out.shape == (0, 48, 48, 3)


def test_rejects_a_frame_mixing_splits(tmp_cache):
    mixed = __import__("pandas").concat(
        [data.load_annotations("train").iloc[:2], data.load_annotations("test").iloc[:2]]
    )
    with pytest.raises(ValueError, match="mixes splits"):
        cache.load_images(mixed, "raw_gray", use_cache=True)


def test_rejects_paths_absent_from_the_split(tmp_cache, sample_frame, monkeypatch):
    """Guards against a frame that did not come from load_annotations."""
    monkeypatch.setattr(cache, "_split_annotations", lambda split: sample_frame.iloc[:12])
    with pytest.raises(KeyError, match="not in the cached"):
        cache.load_images(sample_frame, "raw_gray", use_cache=True)


def test_flatten_produces_unit_range_float32(tmp_cache, sample_frame):
    images = cache.load_images(sample_frame, "raw_gray", use_cache=False)
    flat = cache.flatten(images)
    assert flat.shape == (len(sample_frame), 48 * 48)
    assert flat.dtype == np.float32
    assert 0.0 <= flat.min() and flat.max() <= 1.0
