"""Task 7.2: the CNN training loop -- per-epoch validation, early stopping, best checkpoint.

    history = training.train(model, train_images, y_train, val_images, y_val)

Deliberately small: one function, no Lightning, no callbacks framework. The project trains
exactly one network once (see 13-cnn.md section 1.1), so the loop only needs to do four
things well.

What it selects on
------------------
**Validation macro-F1**, the same criterion `gtsrb.tuning` uses for every other method. The
CNN has no `C` to tune, but "select on the metric the report leads with" applies to early
stopping just as much as to a hyperparameter grid -- stopping on val *accuracy* while the
report leads with macro-F1 would optimise the network for a criterion the write-up does not
use. With 10.7x imbalance the two do not agree.

The best-checkpoint rule
------------------------
The returned model is the one from the **best epoch**, not the last. These differ whenever
training continues past the optimum, which is exactly what early stopping with patience is
designed to allow -- `patience` epochs of no improvement are *by construction* epochs where
the model got no better and may have got worse. Returning the last would quietly report a
model `patience` epochs past its own peak.

Determinism
-----------
Batch order comes from a generator seeded off `config.SEED`, so two runs see identical
batches. Note what this does and does not buy: torch's CPU kernels are deterministic here,
but the same float32 caveats as everywhere else apply (note 03) -- reproducible to reported
precision, not bit-identical across machines.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gtsrb import config, evaluation
from gtsrb.representations.cnn import as_batch

DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MAX_EPOCHS = 40
DEFAULT_PATIENCE = 6


@dataclass
class EpochRecord:
    """One epoch's numbers -- the raw material for the training-curve figure (task 7.6)."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    seconds: float
    is_best: bool


@dataclass
class TrainingHistory:
    """Everything the run produced, plus which epoch won."""

    records: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = -1
    best_macro_f1: float = -1.0
    stopped_early: bool = False
    total_seconds: float = 0.0

    def table(self) -> list[dict]:
        return [vars(record) for record in self.records]

    def summary(self) -> str:
        lines = [
            f"epochs run     : {len(self.records)}"
            + (" (stopped early)" if self.stopped_early else ""),
            f"best epoch     : {self.best_epoch}  val macro-F1 {self.best_macro_f1:.4f}",
            (
                f"total time     : {self.total_seconds / 60:.1f} min "
                f"({self.total_seconds / max(len(self.records), 1):.0f} s/epoch)"
            ),
        ]
        if self.records:
            last = self.records[-1]
            lines.append(f"final epoch    : {last.epoch}  val macro-F1 {last.val_macro_f1:.4f}")
            if last.epoch != self.best_epoch:
                lines.append(
                    f"  -> the returned model is epoch {self.best_epoch}, not the last; "
                    f"macro-F1 differs by {self.best_macro_f1 - last.val_macro_f1:+.4f}"
                )
        return "\n".join(lines)


def _loader(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool):
    dataset = TensorDataset(as_batch(images), torch.from_numpy(np.asarray(labels)).long())
    generator = torch.Generator().manual_seed(config.SEED) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader) -> tuple[float, np.ndarray]:
    """Mean loss and predicted labels over a loader, in eval mode."""
    was_training = model.training
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, predictions = 0.0, []
    try:
        for batch, targets in loader:
            logits = model(batch)
            total_loss += float(loss_fn(logits, targets))
            predictions.append(logits.argmax(dim=1).numpy())
    finally:
        model.train(was_training)
    return total_loss / len(loader.dataset), np.concatenate(predictions)


def train(
    model: nn.Module,
    train_images: np.ndarray,
    y_train: np.ndarray,
    val_images: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    patience: int = DEFAULT_PATIENCE,
    verbose: bool = True,
) -> TrainingHistory:
    """Train `model` in place, restoring the best epoch's weights before returning.

    `model` is left holding the **best** epoch's parameters, so the caller never has to
    remember to reload a checkpoint -- the object that comes back is the one to evaluate.
    """
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")
    if max_epochs < 1:
        raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")

    train_loader = _loader(train_images, y_train, batch_size, shuffle=True)
    val_loader = _loader(val_images, y_val, batch_size, shuffle=False)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    history = TrainingHistory()
    best_state: dict | None = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        running = 0.0
        for batch, targets in train_loader:
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), targets)
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(batch)
        train_loss = running / len(train_loader.dataset)

        val_loss, predictions = evaluate_model(model, val_loader)
        result = evaluation.evaluate(y_val, predictions)

        is_best = result.macro_f1 > history.best_macro_f1
        if is_best:
            history.best_macro_f1 = result.macro_f1
            history.best_epoch = epoch
            # A deep copy on CPU: the state dict's tensors are the live parameters, so
            # keeping a reference would "checkpoint" a model that keeps changing.
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        record = EpochRecord(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            val_accuracy=result.accuracy, val_macro_f1=result.macro_f1,
            seconds=time.perf_counter() - epoch_started, is_best=is_best,
        )
        history.records.append(record)

        if verbose:
            print(f"  epoch {epoch:>3}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={result.accuracy:.4f}  val_macro_f1={result.macro_f1:.4f}  "
                  f"{record.seconds:.0f}s{'  *' if is_best else ''}", flush=True)

        if epochs_without_improvement >= patience:
            history.stopped_early = True
            if verbose:
                print(f"  early stop: {patience} epochs without improvement", flush=True)
            break

    history.total_seconds = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
