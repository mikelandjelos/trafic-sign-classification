"""Tests for the timing harness (task 1.5).

Deliberately avoids asserting on wall-clock durations -- those are flaky on a loaded
machine and would make the suite unreliable. What is tested is the *protocol*: that the
warmup call is taken and excluded, that the right number of measured runs happen, and that
the derived statistics are computed correctly from known inputs.
"""

from __future__ import annotations

import pytest

from gtsrb import timing


def test_warmup_call_is_taken_and_excluded():
    calls = []
    result = timing.measure(lambda: calls.append(1), repeats=3)
    assert len(calls) == 4, "expected 1 warmup + 3 measured runs"
    assert len(result.runs_s) == 3
    assert result.discarded_first_s is not None


def test_warmup_can_be_disabled():
    calls = []
    result = timing.measure(lambda: calls.append(1), repeats=3, warmup=False)
    assert len(calls) == 3
    assert result.discarded_first_s is None
    assert result.warmup_ratio is None


def test_rejects_too_few_repeats():
    """The plan requires a median of >= 3 runs; 1 or 2 cannot give a meaningful one."""
    for bad in (1, 2, 0):
        with pytest.raises(ValueError):
            timing.measure(lambda: None, repeats=bad)


def test_ms_per_item():
    result = timing.TimingResult(
        label="x", median_s=2.0, min_s=2.0, iqr_s=0.0, n_items=1000,
        repeats=3, discarded_first_s=None,
    )
    assert result.ms_per_item == pytest.approx(2.0)


def test_ms_per_item_is_none_without_item_count():
    result = timing.TimingResult(
        label="x", median_s=2.0, min_s=2.0, iqr_s=0.0, n_items=None,
        repeats=3, discarded_first_s=None,
    )
    assert result.ms_per_item is None


def test_warmup_ratio():
    result = timing.TimingResult(
        label="x", median_s=0.1, min_s=0.1, iqr_s=0.0, n_items=None,
        repeats=3, discarded_first_s=0.45,
    )
    assert result.warmup_ratio == pytest.approx(4.5)


def test_percentile_matches_known_values():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert timing._percentile(values, 0.5) == pytest.approx(3.0)
    assert timing._percentile(values, 0.25) == pytest.approx(2.0)
    assert timing._percentile(values, 0.75) == pytest.approx(4.0)
    assert timing._percentile([7.0], 0.5) == pytest.approx(7.0)


def test_training_is_measured_once_without_warmup():
    """Refitting a warm process would not measure the cost of training."""
    calls = []
    result = timing.time_training(lambda: calls.append(1))
    assert len(calls) == 1
    assert result.repeats == 1
    assert result.discarded_first_s is None


def test_platform_info_records_what_is_needed_to_interpret_a_timing():
    info = timing.platform_info()
    for key in (
        "timestamp",
        "cpu",
        "cpu_count_logical",
        "os",
        "threads_configured",
        "blas",
        "load_average",
        "versions",
        "gpu",
        "git_commit",
    ):
        assert key in info, f"platform_info() missing {key}"
    assert "numpy" in info["versions"]
    assert "sklearn" in info["versions"]


def test_platform_info_is_json_serialisable(tmp_path):
    """It is written beside results.csv, so it must round-trip."""
    import json

    path = timing.save_platform_info(tmp_path / "platform.json")
    assert json.loads(path.read_text())["cpu"]
