"""Task 8.1: the grid runner, and the Q5 pairing guard it exists to enforce.

The failure this file is about is silent. A representation fitted on `clahe_gray` handed
`raw_gray` images produces plausible-looking numbers and raises nothing -- both are 2304
dimensions, so no shape check fires. These tests pin the two properties that make it
impossible: `preproc` comes out of the saved artifact, and an ambiguous set of artifacts is
refused rather than guessed at.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_grid

from gtsrb import degradations, results


class _Dummy:
    """Stands in for a fitted representation; records what it was asked to transform."""

    def __init__(self) -> None:
        self.seen: list[int] = []

    def transform(self, images: np.ndarray) -> np.ndarray:
        self.seen.append(len(images))
        return np.asarray(images, dtype=np.float32).reshape(len(images), -1)


class _DummyClassifier:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.zeros(len(features), dtype=np.int64)


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_grid.config, "MODELS_DIR", tmp_path)
    return tmp_path


def _save(path: Path, preproc: str) -> None:
    joblib.dump({"representation": _Dummy(), "classifier": _DummyClassifier(),
                 "preproc": preproc}, path)


# --- the Q5 guard ------------------------------------------------------------------------


def test_preproc_is_read_from_the_artifact_not_a_default(model_dir) -> None:
    """The whole point: the model dictates its own preprocessing."""
    _save(model_dir / "hog_svm_clahe_hsv.joblib", "clahe_hsv")
    loaded = evaluate_grid.load_method("hog_svm")
    assert loaded.preproc == "clahe_hsv"


def test_a_mislabelled_artifact_still_pairs_with_ITS_OWN_preproc(model_dir) -> None:
    """The filename is not the source of truth -- the stored field is.

    A file named `..._raw_gray.joblib` whose payload says `clahe_gray` must follow the
    payload, because that is what the representation was actually fitted on.
    """
    _save(model_dir / "hog_svm_raw_gray.joblib", "clahe_gray")
    assert evaluate_grid.load_method("hog_svm").preproc == "clahe_gray"


def test_two_saved_models_are_refused_rather_than_guessed(model_dir) -> None:
    """Superseded artifacts must not be silently picked by sort order."""
    _save(model_dir / "pca_svm_raw_gray.joblib", "raw_gray")
    _save(model_dir / "pca_svm_clahe_gray.joblib", "clahe_gray")
    with pytest.raises(SystemExit, match="2 saved models"):
        evaluate_grid.load_method("pca_svm")


def test_a_missing_model_names_the_script_that_makes_it(model_dir) -> None:
    with pytest.raises(SystemExit, match="train_"):
        evaluate_grid.load_method("bovw_svm")


# --- the grid itself ---------------------------------------------------------------------


def test_the_grid_is_eighty_cells() -> None:
    """Five configurations, as the proposal specifies -- bovw_spm was reverted (index Q6)."""
    assert len(evaluate_grid.METHODS) == 5
    assert "bovw_spm_svm" not in evaluate_grid.METHODS
    assert len(evaluate_grid.conditions()) == 16
    assert len(evaluate_grid.METHODS) * len(evaluate_grid.conditions()) == 80


def test_clean_is_present_exactly_once() -> None:
    names = [degradation for degradation, _ in evaluate_grid.conditions()]
    assert names.count("clean") == 1


def test_every_degradation_contributes_all_of_its_levels() -> None:
    grid = evaluate_grid.conditions()
    for degradation in evaluate_grid.GRID_DEGRADATIONS:
        levels = {level for name, level in grid if name == degradation}
        assert levels == set(degradations.levels_for(degradation))


def test_each_degradations_identity_level_is_in_the_grid() -> None:
    """The baseline every robustness curve is normalised against must be a measured cell.

    Gamma's identity is 1.0, in the MIDDLE of its range -- an assumption that `levels[0]` is
    the baseline would silently normalise every gamma curve against gamma=0.4.
    """
    grid = set(evaluate_grid.conditions())
    for degradation in evaluate_grid.GRID_DEGRADATIONS:
        assert (degradation, degradations.identity_for(degradation)) in grid


def test_clean_uses_the_no_level_sentinel() -> None:
    assert ("clean", results.NO_LEVEL) in evaluate_grid.conditions()
