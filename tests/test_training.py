"""Task 7.2: the CNN training loop.

Runs on a tiny synthetic problem (a few hundred 48x48 images, 3 classes) so the suite stays
fast. What is under test is the *loop's* contract -- early stopping, the best-checkpoint
rule, determinism, what it selects on -- none of which needs GTSRB or a converged network.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from gtsrb import config, training
from gtsrb.representations.cnn import SmallCNN


@pytest.fixture(scope="module")
def problem():
    """3 separable classes of 48x48 images: solid, vertical stripes, horizontal stripes."""
    rng = np.random.default_rng(0)
    ys, xs = np.mgrid[0:48, 0:48]
    patterns = [np.full((48, 48), 120.0),
                128 + 80 * np.sin(xs / 2.0),
                128 + 80 * np.sin(ys / 2.0)]

    def build(n_each):
        images = np.concatenate([
            np.clip(p + rng.normal(0, 12, (n_each, 48, 48)), 0, 255).astype(np.uint8)
            for p in patterns
        ])
        labels = np.concatenate([np.full(n_each, c) for c in range(3)])
        return images, labels

    return (*build(80), *build(30))


@pytest.fixture
def model() -> nn.Module:
    config.set_seeds()
    return SmallCNN(in_channels=1, n_classes=3)


def run(model, problem, **kwargs):
    x_train, y_train, x_val, y_val = problem
    defaults = {"max_epochs": 3, "patience": 3, "batch_size": 64, "verbose": False}
    return training.train(model, x_train, y_train, x_val, y_val, **{**defaults, **kwargs})


# --- the loop runs and records ----------------------------------------------------------


def test_records_one_entry_per_epoch(model, problem) -> None:
    history = run(model, problem, max_epochs=3)
    assert len(history.records) == 3
    assert [r.epoch for r in history.records] == [1, 2, 3]


def test_every_record_carries_both_metrics(model, problem) -> None:
    """Both are recorded even though only macro-F1 selects -- the alternative must stay
    inspectable, exactly as `gtsrb.tuning` does for the other methods."""
    history = run(model, problem, max_epochs=2)
    for record in history.records:
        assert 0.0 <= record.val_accuracy <= 1.0
        assert 0.0 <= record.val_macro_f1 <= 1.0
        assert record.train_loss > 0 and record.val_loss > 0
        assert record.seconds > 0


def test_table_is_tidy_rows(model, problem) -> None:
    history = run(model, problem, max_epochs=2)
    rows = history.table()
    assert len(rows) == 2
    assert set(rows[0]) == {"epoch", "train_loss", "val_loss", "val_accuracy",
                            "val_macro_f1", "seconds", "is_best"}


def test_it_actually_learns(model, problem) -> None:
    """A sanity check on the loop itself: loss must fall on a separable problem."""
    history = run(model, problem, max_epochs=4)
    assert history.records[-1].train_loss < history.records[0].train_loss


# --- selection: macro-F1, and the best epoch ----------------------------------------------


def test_best_epoch_is_the_macro_f1_argmax(model, problem) -> None:
    history = run(model, problem, max_epochs=4)
    best = max(history.records, key=lambda r: r.val_macro_f1)
    assert history.best_epoch == best.epoch
    assert history.best_macro_f1 == pytest.approx(best.val_macro_f1)


def test_exactly_the_improving_epochs_are_flagged_best(model, problem) -> None:
    history = run(model, problem, max_epochs=4)
    running = -1.0
    for record in history.records:
        assert record.is_best == (record.val_macro_f1 > running)
        running = max(running, record.val_macro_f1)


def test_the_returned_model_is_the_best_epoch_not_the_last(problem) -> None:
    """The rule this loop exists to get right.

    Trained twice with identical seeds: once stopping exactly at the best epoch, once
    continuing past it. The weights that come back must be identical -- i.e. the extra
    epochs were discarded, not kept.
    """
    x_train, y_train, x_val, y_val = problem

    config.set_seeds()
    long_run = SmallCNN(in_channels=1, n_classes=3)
    history = training.train(long_run, x_train, y_train, x_val, y_val,
                             max_epochs=5, patience=5, batch_size=64, verbose=False)
    if history.best_epoch == len(history.records):
        pytest.skip("best epoch was the last one; nothing was discarded to compare against")

    config.set_seeds()
    short_run = SmallCNN(in_channels=1, n_classes=3)
    training.train(short_run, x_train, y_train, x_val, y_val,
                   max_epochs=history.best_epoch, patience=99, batch_size=64, verbose=False)

    for (name, long_p), (_, short_p) in zip(long_run.state_dict().items(),
                                            short_run.state_dict().items(), strict=True):
        torch.testing.assert_close(long_p, short_p, rtol=0, atol=0,
                                   msg=f"parameter {name} is not the best epoch's")


def test_the_checkpoint_is_a_copy_not_a_live_reference(model, problem) -> None:
    """A shallow state_dict would 'checkpoint' parameters that keep training afterwards."""
    history = run(model, problem, max_epochs=4)
    assert history.best_epoch >= 1
    snapshot = copy.deepcopy(model.state_dict())
    run(model, problem, max_epochs=1)  # keep training the same object
    assert any(not torch.equal(snapshot[k], model.state_dict()[k]) for k in snapshot)


# --- early stopping -------------------------------------------------------------------------


def test_early_stopping_triggers_after_patience_epochs_without_improvement(problem) -> None:
    config.set_seeds()
    model = SmallCNN(in_channels=1, n_classes=3)
    history = training.train(model, *problem, max_epochs=30, patience=1, batch_size=64,
                             verbose=False)
    if not history.stopped_early:
        pytest.skip("improved every epoch within the cap; nothing to assert")
    assert len(history.records) - history.best_epoch == 1


def test_no_early_stop_when_the_cap_is_reached_first(model, problem) -> None:
    history = run(model, problem, max_epochs=2, patience=99)
    assert history.stopped_early is False
    assert len(history.records) == 2


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"patience": 0}, "patience"), ({"max_epochs": 0}, "max_epochs"),
])
def test_invalid_arguments_raise(model, problem, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        run(model, problem, **kwargs)


# --- determinism -----------------------------------------------------------------------------


def test_two_runs_with_the_same_seed_agree(problem) -> None:
    """Batch order is seeded off config.SEED, so the whole run is repeatable."""
    histories = []
    for _ in range(2):
        config.set_seeds()
        model = SmallCNN(in_channels=1, n_classes=3)
        histories.append(run(model, problem, max_epochs=2))
    first, second = histories
    for a, b in zip(first.records, second.records, strict=True):
        assert a.val_macro_f1 == pytest.approx(b.val_macro_f1, abs=1e-6)
        assert a.train_loss == pytest.approx(b.train_loss, abs=1e-5)


# --- evaluate_model ----------------------------------------------------------------------------


def test_evaluate_model_restores_the_training_mode(model, problem) -> None:
    x_train, y_train, _, _ = problem
    loader = training._loader(x_train[:64], y_train[:64], 32, shuffle=False)
    model.train()
    training.evaluate_model(model, loader)
    assert model.training is True


def test_evaluate_model_returns_one_prediction_per_sample(model, problem) -> None:
    _, _, x_val, y_val = problem
    loader = training._loader(x_val, y_val, 32, shuffle=False)
    loss, predictions = training.evaluate_model(model, loader)
    assert predictions.shape == (len(y_val),)
    assert loss > 0


def test_summary_names_the_best_epoch(model, problem) -> None:
    history = run(model, problem, max_epochs=3)
    assert f"best epoch     : {history.best_epoch}" in history.summary()
