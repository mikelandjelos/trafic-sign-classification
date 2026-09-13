"""Task 7.1: the small CNN.

All tests run on tiny random batches -- what is under test is the architecture's contract
(shapes, parameter budget, the eval/no_grad guarantee), none of which needs real data or a
trained model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gtsrb import config
from gtsrb.representations.cnn import EMBEDDING_DIM, SmallCNN, as_batch, build


@pytest.fixture
def model() -> SmallCNN:
    config.set_seeds()
    return SmallCNN(in_channels=1)


@pytest.fixture
def batch() -> torch.Tensor:
    return torch.randn(4, 1, 48, 48)


# --- shapes and budget ------------------------------------------------------------------


def test_forward_returns_class_logits(model: SmallCNN, batch: torch.Tensor) -> None:
    assert model(batch).shape == (4, config.N_CLASSES)


def test_embed_returns_the_penultimate_features(model: SmallCNN, batch) -> None:
    assert model.embed(batch).shape == (4, EMBEDDING_DIM)


def test_parameter_budget_is_under_one_million(model: SmallCNN) -> None:
    """The plan's budget. 4608 -> 256 -> 43 missed it at 1.48 M; this head is 0.88 M.

    Pinned because the budget is easy to blow by widening the head, and the first linear
    layer is ~2/3 of the model -- a change there moves the total far more than a change to
    the convolutions.
    """
    assert model.n_parameters() < 1_000_000
    assert model.n_parameters() > 500_000  # and it has not been gutted either


def test_the_conv_trunk_is_the_small_part(model: SmallCNN) -> None:
    """Records where the parameters actually are, which is what made GAP tempting."""
    trunk = sum(p.numel() for p in model.trunk.parameters())
    assert trunk < 300_000
    assert trunk < model.n_parameters() / 2


def test_trunk_output_is_six_by_six(model: SmallCNN, batch: torch.Tensor) -> None:
    """Three 2x2 pools on 48x48: 48 -> 24 -> 12 -> 6. The head's input width depends on it."""
    assert model.trunk(batch).shape == (4, 128, 6, 6)


def test_colour_input_is_supported() -> None:
    """`clahe_hsv` is 3-channel; only the first conv differs."""
    model = SmallCNN(in_channels=3)
    assert model(torch.randn(2, 3, 48, 48)).shape == (2, config.N_CLASSES)


def test_build_matches_the_preproc_channel_count() -> None:
    assert build("clahe_gray").in_channels == 1
    assert build("clahe_hsv").in_channels == 3


# --- the eval/no_grad guarantee (PROJECT_TASKS section 10) --------------------------------


def test_embed_is_deterministic_even_in_train_mode(model: SmallCNN, batch) -> None:
    """The documented gotcha: dropout left active makes features noisy, not wrong-looking.

    `embed` forces eval() internally, so two calls on the same input must agree exactly even
    when the model is in training mode. Without that, `cnn_feat_svm` would be trained on
    features that differ every time they are extracted.
    """
    model.train()
    first, second = model.embed(batch), model.embed(batch)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_embed_restores_the_previous_mode(model: SmallCNN, batch: torch.Tensor) -> None:
    """It must not silently leave the model in eval() midway through training."""
    model.train()
    model.embed(batch)
    assert model.training is True

    model.eval()
    model.embed(batch)
    assert model.training is False


def test_embed_does_not_build_a_graph(model: SmallCNN, batch: torch.Tensor) -> None:
    assert model.embed(batch).requires_grad is False


def test_forward_keeps_dropout_active_in_train_mode(model: SmallCNN, batch) -> None:
    """The counterpart: `forward` must NOT suppress dropout, or training is silently changed."""
    model.train()
    torch.manual_seed(0)
    first = model(batch)
    torch.manual_seed(1)
    second = model(batch)
    assert not torch.allclose(first, second)


def test_forward_is_deterministic_in_eval_mode(model: SmallCNN, batch) -> None:
    model.eval()
    torch.testing.assert_close(model(batch), model(batch), rtol=0, atol=0)


# --- input conversion ---------------------------------------------------------------------


def test_as_batch_converts_grayscale_images() -> None:
    images = np.random.default_rng(0).integers(0, 256, (5, 48, 48), dtype=np.uint8)
    tensor = as_batch(images)
    assert tensor.shape == (5, 1, 48, 48)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


def test_as_batch_converts_colour_images() -> None:
    images = np.random.default_rng(0).integers(0, 256, (5, 48, 48, 3), dtype=np.uint8)
    assert as_batch(images).shape == (5, 3, 48, 48)


def test_as_batch_scaling_matches_the_other_representations() -> None:
    """[0, 1], the same convention as `cache.flatten` -- all five methods see one scaling."""
    images = np.full((1, 48, 48), 255, dtype=np.uint8)
    torch.testing.assert_close(as_batch(images).max(), torch.tensor(1.0))


def test_as_batch_rejects_a_single_image() -> None:
    with pytest.raises(ValueError, match="expected"):
        as_batch(np.zeros((48, 48, 1, 1, 1), dtype=np.uint8))


def test_as_batch_output_feeds_the_model(model: SmallCNN) -> None:
    images = np.random.default_rng(0).integers(0, 256, (3, 48, 48), dtype=np.uint8)
    assert model(as_batch(images)).shape == (3, config.N_CLASSES)
