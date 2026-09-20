"""Visual explanation of the track-disjoint split (tasks 1.1-1.3).

The project's central methodological claim is that GTSRB must be split by *track*, never by
image. `tests/test_split.py` proves it numerically -- a naive per-image split puts 1,305 of
1,307 tracks on both sides, and 100 % of val images keep a sibling frame in train. Those
numbers are conclusive but abstract. These figures make the hazard visible, which is what a
reader needs to accept the protocol without taking it on trust.

Generates three figures into figures/demo/split/:

  split_leakage.png      -- one track's frames, coloured by where each split sends them
  split_similarity.png   -- same-track vs different-track image correlation, as distributions
  split_balance.png      -- per-class val share, class imbalance, and the ROI size histogram

Run:
    poetry run python scripts/demo/split_mechanics.py

Everything uses `gtsrb.data` itself, so the figures describe the split the experiments run
on rather than a re-derivation of it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gtsrb import cache, config, data
from gtsrb.figures import save_demo_and_report

TRAIN_COLOUR = "#2b6cb0"
VAL_COLOUR = "#c05621"


def naive_image_split(frame, val_fraction: float = config.VAL_FRACTION) -> np.ndarray:
    """The wrong split, for comparison only: assign each *image* independently.

    This is the thing the protocol exists to avoid. It is reimplemented here (rather than
    imported) precisely because it is not part of the library -- `gtsrb.data` offers no way
    to do it, which is the point.
    """
    rng = config.rng_for("demo_naive_split")
    return rng.random(len(frame)) < val_fraction


def figure_leakage(out_dir: Path, frame) -> Path:
    """One track's 30 frames, and where each splitting rule sends them."""
    # A track from a class with plenty of them, so the example is typical rather than a
    # cornered case. Sorted by frame_id so the strip reads as the sequence it is.
    track = frame[frame["track_id"] == "00012_00003"].sort_values("frame_id")
    if track.empty:  # defensive: fall back to the first track with a full 30 frames
        counts = frame.groupby("track_id").size()
        track = frame[frame["track_id"] == counts[counts == 30].index[0]]
        track = track.sort_values("frame_id")

    images = cache.load_images(track, "raw_gray")
    naive = naive_image_split(track)
    disjoint = data.assign_split(track) == "val"

    n = len(track)
    fig, axes = plt.subplots(3, n, figsize=(0.46 * n, 1.9),
                             gridspec_kw={"height_ratios": [1.0, 0.30, 0.30], "hspace": 0.15,
                                          "wspace": 0.06})

    for i in range(n):
        axes[0, i].imshow(images[i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        for row, mask in ((1, naive), (2, np.asarray(disjoint))):
            axes[row, i].set_facecolor(VAL_COLOUR if mask[i] else TRAIN_COLOUR)
        for row in range(3):
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])

    for row, text in ((0, "kadrovi"), (1, "nasumično po slikama"), (2, "po sekvencama  (naše)")):
        axes[row, 0].set_ylabel(text, fontsize=9, rotation=0, ha="right", va="center",
                                labelpad=8, fontweight="bold" if row else "normal")

    n_val_naive = int(naive.sum())
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=TRAIN_COLOUR, label="trening"),
                 plt.Rectangle((0, 0), 1, 1, color=VAL_COLOUR, label="validacija")],
        loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06),
    )
    title = fig.suptitle(
        f"One track = {n} photographs of the SAME physical sign, taken seconds apart as the "
        f"car approaches\n"
        f"A random per-image split sends {n_val_naive} of these frames to validation and "
        f"{n - n_val_naive} to training —\nso the model is validated on signs it has already "
        f"memorised. Splitting by track keeps the whole track on one side.",
        fontsize=10,
        y=1.30,
    )
    fig.tight_layout()
    path = out_dir / "split_leakage.png"
    save_demo_and_report(fig, path, title, dpi=160, bare_rect=None)
    plt.close(fig)
    return path


def figure_similarity(out_dir: Path, frame) -> Path:
    """Quantify 'near-duplicate': correlation within a track vs across tracks."""
    rng = config.rng_for("demo_similarity")
    tracks = frame.groupby("track_id")
    usable = [t for t, g in tracks if len(g) >= 10]
    sample = rng.permutation(usable)[:150]

    within: list[float] = []
    between: list[float] = []
    for track_id in sample:
        group = frame[frame["track_id"] == track_id]
        images = cache.flatten(cache.load_images(group, "raw_gray"))
        for _ in range(4):
            i, j = rng.choice(len(images), 2, replace=False)
            within.append(float(np.corrcoef(images[i], images[j])[0, 1]))

        # A different track of the SAME class -- the hard comparison. If within-track pairs
        # are no more alike than these, there is no leakage to worry about.
        same_class = frame[(frame["class_id"] == group["class_id"].iloc[0])
                           & (frame["track_id"] != track_id)]
        if same_class.empty:
            continue
        others = cache.flatten(cache.load_images(same_class.iloc[
            rng.choice(len(same_class), min(4, len(same_class)), replace=False)
        ], "raw_gray"))
        for k in range(len(others)):
            between.append(float(np.corrcoef(images[0], others[k])[0, 1]))

    fig, ax = plt.subplots(figsize=(9, 4.4))
    bins = np.linspace(-0.4, 1.0, 60)
    ax.hist(between, bins=bins, alpha=0.75, label=f"different tracks, same class "
            f"(median {np.median(between):.2f})", color="#718096", density=True)
    ax.hist(within, bins=bins, alpha=0.8, label=f"same track "
            f"(median {np.median(within):.2f})", color=VAL_COLOUR, density=True)
    ax.axvline(float(np.median(within)), color=VAL_COLOUR, ls="--", lw=1.5)
    ax.axvline(float(np.median(between)), color="#718096", ls="--", lw=1.5)
    ax.set_xlabel("pixel correlation between two images")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    title = fig.suptitle(
        "Why frames of one track are 'near-duplicates', measured rather than asserted\n"
        "Two frames of the same physical sign are far more alike than two different signs "
        "of the same class.\nThat gap is what a per-image split leaks across train and "
        "validation.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = out_dir / "split_similarity.png"
    save_demo_and_report(fig, path, title, dpi=160, bare_rect=None)
    plt.close(fig)
    return path


def figure_balance(out_dir: Path, frame) -> Path:
    """What the track-atomic split costs: quantised per-class validation shares."""
    train, val = data.train_val_split(frame)
    classes = np.arange(config.N_CLASSES)
    n_train = np.array([(train["class_id"] == c).sum() for c in classes])
    n_val = np.array([(val["class_id"] == c).sum() for c in classes])
    share = n_val / (n_train + n_val)
    tracks = np.array([frame[frame["class_id"] == c]["track_id"].nunique() for c in classes])

    fig, (top, mid, bottom) = plt.subplots(3, 1, figsize=(11, 8.4))

    top.bar(classes, n_train, color=TRAIN_COLOUR, label="train")
    top.bar(classes, n_val, bottom=n_train, color=VAL_COLOUR, label="val")
    top.set_ylabel("images")
    total = n_train + n_val
    top.set_title(f"Class imbalance: {total.max()} vs {total.min()} images — "
                  f"{total.max() / total.min():.1f}× "
                  f"(report macro-F1, not just accuracy)", fontsize=10)
    top.legend(fontsize=9)
    top.grid(alpha=0.3, axis="y")

    colours = [VAL_COLOUR if abs(s - 0.2) > 0.04 else TRAIN_COLOUR for s in share]
    mid.bar(classes, share, color=colours)
    mid.axhline(0.2, color="0.3", ls="--", lw=1.2)
    mid.set_ylabel("val share")
    mid.set_title(f"Per-class validation share spans [{share.min():.3f}, {share.max():.3f}] — "
                  f"tracks are atomic, so the split is quantised in whole tracks", fontsize=10)
    mid.grid(alpha=0.3, axis="y")

    bottom.bar(classes, tracks, color="#718096")
    bottom.set_xlabel("class id")
    bottom.set_ylabel("tracks")
    bottom.set_title(f"Tracks per class: {tracks.min()} to {tracks.max()}. A class with "
                     f"{tracks.min()} tracks cannot land on exactly 20 % — its finest step "
                     f"is 1/{tracks.min()} = {1 / tracks.min():.0%}", fontsize=10)
    bottom.grid(alpha=0.3, axis="y")

    title = fig.suptitle(
        "The cost of an atomic-track split, stated rather than hidden\n"
        "Every class keeps at least one track on each side, so macro-F1 is never undefined",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out_dir / "split_balance.png"
    save_demo_and_report(fig, path, title, dpi=160, bare_rect=None)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Track-disjoint split figures (task 1.2/1.3).")
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR / "demo" / "split")
    args = parser.parse_args()

    config.set_seeds()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = data.load_annotations("train")

    for path in (
        figure_leakage(args.out, frame),
        figure_similarity(args.out, frame),
        figure_balance(args.out, frame),
    ):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
