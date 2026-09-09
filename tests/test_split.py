"""Task 1.3: the train/val split must never leak a track across the boundary.

The project plan lists track-disjoint splitting as one of the things that must never be
cut. A leak does not raise anything -- it just inflates validation accuracy and quietly
misdirects model selection -- so it has to be enforced by a test rather than by care.

These tests also pin the properties the split's *correctness argument* depends on:
that every track belongs to exactly one class (which is what lets disjointness and
stratification hold simultaneously), and that the track key is the composite
`CCCCC_TTTTT` rather than the raw filename prefix.
"""

from __future__ import annotations

import pytest

from gtsrb import config, data


@pytest.fixture(scope="module")
def annotations():
    """Training annotations, loaded once for the whole module (reads 43 CSVs)."""
    return data.load_annotations("train")


@pytest.fixture(scope="module")
def split(annotations):
    return data.train_val_split(annotations)


# --- the assertion task 1.3 exists for -------------------------------------------------


def test_no_track_appears_in_both_splits(split):
    train, val = split
    overlap = set(train["track_id"]) & set(val["track_id"])
    assert overlap == set(), f"{len(overlap)} tracks leaked across the split: {sorted(overlap)[:5]}"


def test_no_image_appears_in_both_splits(split):
    train, val = split
    assert set(train["path"]).isdisjoint(set(val["path"]))


def test_split_partitions_the_data(annotations, split):
    """Every row lands on exactly one side -- nothing dropped, nothing duplicated."""
    train, val = split
    assert len(train) + len(val) == len(annotations)
    assert set(train.index).isdisjoint(val.index)
    assert set(train.index) | set(val.index) == set(annotations.index)


# --- properties the correctness argument rests on --------------------------------------


def test_every_track_belongs_to_exactly_one_class(annotations):
    """Splitting per class only yields track-disjointness if tracks are single-class."""
    assert annotations.groupby("track_id")["class_id"].nunique().max() == 1


def test_track_id_is_the_composite_key(annotations):
    """Guards the finding in docs/report-material/02: TTTTT restarts per class directory.

    Using the raw prefix would collapse ~1300 real tracks into ~75 groups, which still
    "works" but destroys stratification.
    """
    raw_prefixes = annotations["path"].str.rsplit("/", n=1).str[-1].str.split("_").str[0]
    assert raw_prefixes.nunique() < annotations["track_id"].nunique()
    assert annotations["track_id"].nunique() == config.N_TRACKS


def test_no_track_exceeds_thirty_frames(annotations):
    assert annotations.groupby("track_id").size().max() <= data.TRACK_FRAMES


# --- stratification --------------------------------------------------------------------


def test_all_classes_present_on_both_sides(split):
    """An empty class in val would make macro-F1 undefined for that class."""
    train, val = split
    assert set(train["class_id"]) == set(range(config.N_CLASSES))
    assert set(val["class_id"]) == set(range(config.N_CLASSES))


def test_every_class_keeps_at_least_one_track_each_side(split):
    train, val = split
    assert train.groupby("class_id")["track_id"].nunique().min() >= 1
    assert val.groupby("class_id")["track_id"].nunique().min() >= 1


def test_validation_fraction_is_close_to_target(split):
    train, val = split
    achieved = len(val) / (len(train) + len(val))
    assert achieved == pytest.approx(config.VAL_FRACTION, abs=0.02)


def test_per_class_fraction_stays_within_quantisation_bounds(split):
    """Tracks are atomic, so per-class shares cannot all be exactly 20 %.

    Class 0 has 7 tracks, so its finest step is 1/7. The bounds here are loose on purpose:
    the test guards against a *broken* stratification, not against the known quantisation.
    """
    train, val = split
    train_counts = train.groupby("class_id").size()
    val_counts = val.groupby("class_id").size()
    per_class = val_counts / (train_counts + val_counts)
    assert per_class.min() >= 0.10
    assert per_class.max() <= 0.35


# --- determinism -----------------------------------------------------------------------


def test_split_is_deterministic(annotations):
    assert data.assign_split(annotations).equals(data.assign_split(annotations))


def test_split_is_independent_of_row_order(annotations):
    """Otherwise the split would silently change when the loader's ordering changes."""
    shuffled = annotations.sample(frac=1, random_state=7)
    reordered = data.assign_split(shuffled).reindex(annotations.index)
    assert data.assign_split(annotations).equals(reordered)


def test_rejects_invalid_fraction(annotations):
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            data.assign_split(annotations, val_fraction=bad)
