"""Visual explanation of what the CNN learned, and the bug that hides in it (task 7.6).

    poetry run python scripts/demo/cnn_mechanics.py

Four figures into figures/demo/cnn/:

  cnn_training_curves.png   -- what "stopped early" actually means, and why best-epoch matters
  cnn_dropout_check.png     -- THE failure test: the section 10 gotcha, made visible
  cnn_first_layer_filters.png -- what the first 3x3 kernels became
  cnn_activations.png       -- feature maps per block, one sign per super-category

Everything reads the **trained** network from results/models/ and the recorded history; no
model is built or fitted here, so the figures describe what was actually reported.

Why the dropout figure is the failure test
------------------------------------------
PROJECT_TASKS section 10: penultimate features extracted with dropout still active carry
multiplicative noise. **Nothing errors.** `cnn_feat_svm` is merely, inexplicably, a bit worse
-- which is indistinguishable from "the representation is just weaker", the exact conclusion
this project exists to draw carefully.

`SmallCNN.embed()` forces `eval()`/`no_grad()` so the bug cannot happen. This figure shows
both paths side by side on the same image: `embed()` twice (identical), and the unprotected
`embedding(trunk(x))` twice while the model is in train mode (different). It is the one place
the guarantee is *shown* rather than asserted, and the contrast is the whole point -- a figure
of the protected path alone would prove nothing, because a broken implementation would look
exactly the same.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from gtsrb import cache, config, data, preprocessing
from gtsrb.representations import cnn as cnn_module

#: One class per shape family, matching the HOG and BoVW demos so the three are comparable.
SAMPLES: tuple[tuple[int, str], ...] = (
    (14, "octagonal"),
    (25, "triangular warning"),
    (38, "round mandatory"),
    (5, "round prohibitory"),
)


def load_trained(preproc: str) -> cnn_module.SmallCNN:
    path = config.MODELS_DIR / f"cnn_e2e_{preproc}.pt"
    if not path.exists():
        raise SystemExit(f"no trained network at {path}; run scripts/train_cnn.py")
    blob = torch.load(path, weights_only=True)
    model = cnn_module.SmallCNN(in_channels=int(blob["in_channels"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def pick(frame, class_id: int) -> int:
    rows = frame[frame["class_id"] == class_id].nlargest(1, "roi_h")
    return list(frame.index).index(rows.index[0])


def figure_training_curves(history: dict, out_dir: Path) -> Path:
    """What "stopped early" means -- and what returning the last epoch would have cost."""
    epochs = history["epochs"]
    n = [e["epoch"] for e in epochs]
    macro = [e["val_macro_f1"] for e in epochs]
    train_loss = [e["train_loss"] for e in epochs]
    val_loss = [e["val_loss"] for e in epochs]
    best = int(history["best_epoch"])
    best_score = macro[best - 1]
    last_score = macro[-1]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 4.6))

    left.plot(n, macro, "o-", color="#2b6cb0", lw=1.8, ms=5, label="validation macro-F1")
    left.axvspan(best, n[-1], color="#c05621", alpha=0.10)
    left.axvline(best, color="#2f855a", lw=1.6, ls="--")
    left.annotate(f"best: epoch {best}\n{best_score:.4f}", xy=(best, best_score),
                  xytext=(best - 9.5, best_score - 0.028), fontsize=9, color="#2f855a",
                  arrowprops={"arrowstyle": "->", "color": "#2f855a", "lw": 1.2})
    left.annotate(f"stopped: epoch {n[-1]}\n{last_score:.4f}", xy=(n[-1], last_score),
                  xytext=(n[-1] - 7.0, last_score - 0.045), fontsize=9, color="#c05621",
                  arrowprops={"arrowstyle": "->", "color": "#c05621", "lw": 1.2})
    left.set_xlabel("epoch")
    left.set_ylabel("validation macro-F1")
    left.set_title(f"The shaded band is the patience window ({n[-1] - best} epochs)",
                   fontsize=10)
    left.grid(alpha=0.3)
    left.legend(fontsize=9, loc="lower right")

    right.plot(n, train_loss, "o-", color="#c05621", lw=1.8, ms=5, label="train loss")
    right.plot(n, val_loss, "o-", color="#2b6cb0", lw=1.8, ms=5, label="validation loss")
    right.axvline(best, color="#2f855a", lw=1.6, ls="--")
    right.set_yscale("log")
    right.set_xlabel("epoch")
    right.set_ylabel("loss (log scale)")
    right.set_title(f"Train loss reaches {train_loss[-1]:.4f} — the network has memorised",
                    fontsize=10)
    right.grid(alpha=0.3)
    right.legend(fontsize=9)

    fig.suptitle(
        f"'{history['preproc']}' — early stopping is not convergence, and the report says so\n"
        f"Training STOPPED at epoch {n[-1]} having last improved at epoch {best}: that is the "
        f"patience rule firing, not an observed plateau (note 13, addendum 2).\n"
        f"Returning the LAST epoch instead of the best would have cost "
        f"{(best_score - last_score) * 100:.2f} pp macro-F1 — which is why `train()` restores "
        f"the best weights.\n"
        f"Train loss falls ~1000x while validation macro-F1 moves {macro[0]:.3f} → "
        f"{best_score:.3f}: past ~epoch {best} the network is memorising, not generalising.",
        fontsize=10, y=1.03,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    path = out_dir / "cnn_training_curves.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    best epoch {best} ({best_score:.4f}) vs last {n[-1]} ({last_score:.4f}): "
          f"returning the last would cost {(best_score - last_score) * 100:.2f} pp")
    return path


def figure_dropout_check(model: cnn_module.SmallCNN, image: np.ndarray,
                         out_dir: Path) -> Path:
    """The section 10 gotcha, made visible. Both paths, on the same image."""
    batch = cnn_module.as_batch(image[None])

    # The protected path: embed() forces eval()/no_grad() whatever mode the model is in.
    model.train()
    safe_a = model.embed(batch).numpy()[0]
    safe_b = model.embed(batch).numpy()[0]

    # The unprotected path, called deliberately: this is the bug embed() exists to prevent.
    # Dropout is active in train mode, so two calls on the SAME image disagree.
    model.train()
    with torch.no_grad():
        unsafe_a = model.embedding(model.trunk(batch)).numpy()[0]
        unsafe_b = model.embedding(model.trunk(batch)).numpy()[0]
    model.eval()

    safe_delta = float(np.abs(safe_a - safe_b).max())
    unsafe_delta = float(np.abs(unsafe_a - unsafe_b).max())
    # NOT "how many more zeros": ReLU already zeroes most dimensions, and dropout rescales
    # the survivors, so a zero-count difference came out NEGATIVE and meant nothing. The
    # quantity that actually describes the damage is how much of the vector changes, and by
    # how much relative to its own scale.
    changed = float((unsafe_a != unsafe_b).mean())
    relative = unsafe_delta / float(np.abs(unsafe_a).max())

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 6.4))
    dims = np.arange(len(safe_a))

    for ax, (a, b, title, colour, delta) in zip(
        axes[:, 0],
        [(safe_a, safe_b, "embed() — the protected path", "#2f855a", safe_delta),
         (unsafe_a, unsafe_b, "embedding(trunk(x)) in train mode — the bug", "#c05621",
          unsafe_delta)],
        strict=True,
    ):
        ax.plot(dims, a, lw=1.1, color=colour, label="call 1")
        ax.plot(dims, b, lw=1.0, color="#1a202c", ls="--", alpha=0.75, label="call 2")
        ax.set_title(f"{title}\nmax |Δ| between two calls = {delta:.4f}", fontsize=9.5)
        ax.set_xlabel("penultimate feature dimension", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax, (a, b, title, colour) in zip(
        axes[:, 1],
        [(safe_a, safe_b, "protected: the difference is exactly zero", "#2f855a"),
         (unsafe_a, unsafe_b, "unprotected: every call is a different vector", "#c05621")],
        strict=True,
    ):
        ax.bar(dims, a - b, width=1.0, color=colour)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("penultimate feature dimension", fontsize=8)
        ax.set_ylabel("call 1 − call 2", fontsize=8)
        ax.grid(alpha=0.3)
    # Same y-scale on both difference panels, or "exactly zero" is invisible as a choice
    # of limits rather than as a fact.
    span = max(1e-6, float(np.abs(unsafe_a - unsafe_b).max()) * 1.15)
    for ax in axes[:, 1]:
        ax.set_ylim(-span, span)

    fig.suptitle(
        "The gotcha that does not raise: dropout-contaminated features (PROJECT_TASKS §10)\n"
        f"With the model in TRAIN mode, `embed()` returns the identical vector twice "
        f"(max |Δ| = {safe_delta:.4f}); calling the layers directly does not "
        f"(max |Δ| = {unsafe_delta:.4f}).\n"
        f"{changed * 100:.0f} % of the 128 dimensions change between two calls on the SAME "
        f"image, by up to {relative * 100:.0f} % of the vector's own peak — so `cnn_feat_svm` "
        f"would train on noise and simply score lower, with nothing to indicate a bug.\n"
        "The unprotected path is called deliberately here: a figure of the safe path alone "
        "would look the same whether or not the guarantee held.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path = out_dir / "cnn_dropout_check.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    embed() max|Δ| = {safe_delta:.6f} (must be 0); "
          f"unprotected max|Δ| = {unsafe_delta:.4f}")
    return path


def figure_first_layer_filters(model: cnn_module.SmallCNN, out_dir: Path) -> Path:
    """What the first 3x3 kernels became -- 32 of them, on a shared diverging scale."""
    weight = model.trunk[0][0].weight.detach().numpy()  # (32, in_channels, 3, 3)
    kernels = weight[:, 0]
    limit = float(np.abs(kernels).max())

    fig, axes = plt.subplots(4, 8, figsize=(11.0, 6.0))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(kernels[i], cmap="RdBu_r", vmin=-limit, vmax=limit,
                  interpolation="nearest")
        ax.set_title(f"‖w‖={np.linalg.norm(kernels[i]):.2f}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "First convolution: 32 learned 3×3 kernels (red positive, blue negative, shared scale)\n"
        "At 3×3 on 48×48 input these are oriented difference operators — the learned analogue "
        "of the fixed gradient filter HOG and SIFT apply by construction.\n"
        "The norm above each is its weight magnitude: the spread shows the layer does not use "
        "its capacity evenly, and the near-flat kernels contribute little.",
        fontsize=10.5, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    path = out_dir / "cnn_first_layer_filters.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_activations(model: cnn_module.SmallCNN, frame, images: np.ndarray,
                       out_dir: Path) -> Path:
    """Feature maps after each block -- where the spatial resolution goes."""
    fig, axes = plt.subplots(len(SAMPLES), 4, figsize=(11.5, 2.75 * len(SAMPLES)))

    for row, (class_id, shape) in enumerate(SAMPLES):
        image = images[pick(frame, class_id)]
        batch = cnn_module.as_batch(image[None])

        activations = []
        with torch.no_grad():
            x = batch
            for block in model.trunk:
                x = block(x)
                activations.append(x[0].numpy())

        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[row, 0].set_ylabel(f"{config.CLASS_NAMES[class_id][:18]}\n({shape})", fontsize=8)
        if row == 0:
            axes[row, 0].set_title("input 48×48", fontsize=9.5)

        for col, maps in enumerate(activations, start=1):
            # Mean over channels: one panel per block rather than 128 panels, and it answers
            # the question the figure is asking -- WHERE the network is responding.
            axes[row, col].imshow(maps.mean(axis=0), cmap="magma", interpolation="nearest")
            if row == 0:
                axes[row, col].set_title(
                    f"block {col} → {maps.shape[0]}×{maps.shape[1]}×{maps.shape[2]}",
                    fontsize=9.5)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Mean activation after each conv block — 48×48 → 24×24 → 12×12 → 6×6\n"
        "Channels widen 32 → 64 → 128 as resolution halves: the network trades *where* for "
        "*what*, which is the hierarchical property the structural table names.\n"
        "By block 3 a whole sign is described by a 6×6 grid — coarser than HOG's 8×8 cells, "
        "but with 128 learned channels per position instead of 9 fixed orientations.",
        fontsize=10.5, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = out_dir / "cnn_activations.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="CNN mechanics demo (task 7.6).")
    parser.add_argument("--preproc", default=preprocessing.DEFAULT_PREPROC,
                        choices=list(preprocessing.PREPROC_CONFIGS))
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR / "demo" / "cnn")
    args = parser.parse_args()

    config.set_seeds()
    args.out.mkdir(parents=True, exist_ok=True)

    model = load_trained(args.preproc)
    history_path = config.RESULTS_DIR / "histories" / f"cnn_e2e_{args.preproc}.json"
    if not history_path.exists():
        raise SystemExit(f"no recorded history at {history_path}")
    history = json.loads(history_path.read_text())
    print(f"{args.preproc}: {model.n_parameters():,} parameters, "
          f"best epoch {history['best_epoch']} of {len(history['epochs'])}")

    train, _ = data.train_val_split()
    images = cache.load_images(train, args.preproc)

    for path in (
        figure_training_curves(history, args.out),
        figure_dropout_check(model, images[pick(train, SAMPLES[3][0])], args.out),
        figure_first_layer_filters(model, args.out),
        figure_activations(model, train, images, args.out),
    ):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
