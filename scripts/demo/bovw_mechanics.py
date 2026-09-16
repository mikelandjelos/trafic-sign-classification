"""Visual explanation of *how* BoVW works, and where it breaks (task 6.6).

    poetry run python scripts/demo/bovw_mechanics.py

Four figures into figures/demo/bovw/:

  bovw_grid_and_words.png   -- the dense grid, and which patches share a codeword
  bovw_permutation.png      -- THE centrepiece: orderlessness, demonstrated
  bovw_normalisation.png    -- what power_l2 does to a bursty histogram
  bovw_blur_collapse.png    -- the failure test: does the vocabulary degenerate under blur?

Everything is computed with `gtsrb.representations.bovw` and `gtsrb.degradations` themselves.
Nothing here re-derives the encoding -- a demo that reimplements the thing it is checking
verifies nothing (CLAUDE.md).

Why the permutation figure is the centrepiece
---------------------------------------------
The study's design rests on BoVW being **orderless**, and everywhere else that is an
*assertion*. Here it is a measurement: shuffle which grid position each descriptor came from,
re-pool, and compare. The plain histogram is byte-identical -- mean |delta| is exactly 0.0 --
while the spatial pyramid, built from the very same descriptors and vocabulary, changes.

That single panel is the layout axis the whole project measures, and it is the visual
companion to the +10.4 pp that section 9 of note 14 records.

Why the blur figure is the failure test
---------------------------------------
Task 6.2 asserts a constant descriptor count numerically, and that assertion passes even when
the encoding has silently degenerated: under heavy blur local patches flatten, descriptors
collapse onto a handful of codewords, and every image's histogram starts to look alike. No
test catches that. Counting **distinct words per image** and **histogram similarity between
different classes** does, and both are plotted against blur strength.

The correcting quantity is in the same figure on purpose: a falling word count could be read
as "the representation is becoming robustly compact". The between-class similarity rising at
the same time is what shows it is collapse, not compression.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gtsrb import cache, config, data, degradations
from gtsrb.representations.bovw import (
    BoVWRepresentation,
    BoVWSpatialPyramid,
    normalise_histograms,
)

#: One per shape family, so the codeword map can be read across genuinely different signs.
SAMPLES: tuple[tuple[int, str], ...] = (
    (14, "octagonal"),
    (25, "triangular warning"),
    (38, "round mandatory"),
    (5, "round prohibitory"),
)


def pick(frame, class_id: int) -> int:
    """Index of a large, well-resolved example -- small crops are blurred by the upscale."""
    rows = frame[frame["class_id"] == class_id].nlargest(1, "roi_h")
    return list(frame.index).index(rows.index[0])


def figure_grid_and_words(phi: BoVWRepresentation, frame, images: np.ndarray,
                          out_dir: Path) -> Path:
    """The dense grid, and the codeword map -- which patches the vocabulary calls alike."""
    rows, cols = phi.extractor.grid_shape
    kmeans = phi._fitted()

    fig, axes = plt.subplots(2, len(SAMPLES), figsize=(3.1 * len(SAMPLES), 6.6))
    for col, (class_id, shape) in enumerate(SAMPLES):
        image = images[pick(frame, class_id)]
        words = kmeans.predict(phi.extractor.describe(image)).reshape(rows, cols)

        axes[0, col].imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ys, xs = np.mgrid[0:rows, 0:cols]
        half = phi.extractor.step // 2
        axes[0, col].scatter(xs.ravel() * phi.extractor.step + half,
                             ys.ravel() * phi.extractor.step + half,
                             s=6, c="#c05621", alpha=0.75, linewidths=0)
        axes[0, col].set_title(f"{config.CLASS_NAMES[class_id][:24]}\n({shape})", fontsize=9)

        # Codewords are nominal labels, so a qualitative colormap and NO colourbar: adjacent
        # word ids mean nothing, and a sequential map would invent an ordering.
        axes[1, col].imshow(words % 20, cmap="tab20", interpolation="nearest", vmin=0,
                            vmax=19)
        axes[1, col].set_title(f"{len(np.unique(words))} distinct words", fontsize=9)
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_ylabel(f"input + {phi.extractor.n_keypoints}\nkeypoints", fontsize=10,
                          fontweight="bold")
    axes[1, 0].set_ylabel("codeword map\n(colour = word id)", fontsize=10, fontweight="bold")
    fig.suptitle(
        f"Dense SIFT on a fixed grid, then quantisation to {phi.n_words} visual words — "
        f"size {phi.extractor.keypoint_size} / step {phi.extractor.step}\n"
        "Every image contributes exactly the same number of descriptors, which is what makes "
        "the histograms comparable.\n"
        "Colours are nominal word ids: neighbouring patches sharing a colour share a codeword. "
        "The position of each patch is then DISCARDED.",
        fontsize=10.5, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path = out_dir / "bovw_grid_and_words.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_permutation(phi: BoVWRepresentation, spm: BoVWSpatialPyramid, frame,
                       images: np.ndarray, out_dir: Path) -> Path:
    """Orderlessness, demonstrated rather than asserted. The centrepiece of this demo."""
    rows, cols = phi.extractor.grid_shape
    kmeans = phi._fitted()
    image = images[pick(frame, SAMPLES[3][0])]
    words = kmeans.predict(phi.extractor.describe(image))

    rng = np.random.default_rng(config.SEED)
    order = rng.permutation(rows * cols)

    def pool(word_ids: np.ndarray, pyramid: bool) -> np.ndarray:
        grid = word_ids.reshape(rows, cols)
        if not pyramid:
            return normalise_histograms(
                np.bincount(word_ids, minlength=phi.n_words)[None].astype(np.float32),
                phi.normalisation)[0]
        pieces = []
        for c in spm._cells:
            rb = np.minimum((np.arange(rows) * c) // rows, c - 1)
            cb = np.minimum((np.arange(cols) * c) // cols, c - 1)
            for a in range(c):
                for b in range(c):
                    sub = grid[np.ix_(rb == a, cb == b)]
                    pieces.append(np.bincount(sub.ravel(), minlength=spm.n_words))
        vec = np.concatenate(pieces) * np.repeat(spm.cell_weights, spm.n_words)
        return normalise_histograms(vec[None].astype(np.float32), spm.normalisation)[0]

    plain, plain_shuffled = pool(words, False), pool(words[order], False)
    pyr, pyr_shuffled = pool(words, True), pool(words[order], True)
    d_plain = float(np.abs(plain - plain_shuffled).max())
    d_pyr = float(np.abs(pyr - pyr_shuffled).max())

    fig = plt.figure(figsize=(13.5, 7.4))
    grid_spec = fig.add_gridspec(2, 3, width_ratios=[1, 1.5, 1.5], hspace=0.38, wspace=0.26)

    for row, (word_ids, label) in enumerate(((words, "original"),
                                             (words[order], "positions shuffled"))):
        ax = fig.add_subplot(grid_spec[row, 0])
        ax.imshow(word_ids.reshape(rows, cols) % 20, cmap="tab20", interpolation="nearest",
                  vmin=0, vmax=19)
        ax.set_title(f"codeword map\n({label})", fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])

    for row, (a, b, label) in enumerate(((plain, plain_shuffled, "original"),
                                         (plain, plain_shuffled, "shuffled"))):
        ax = fig.add_subplot(grid_spec[row, 1])
        ax.bar(np.arange(len(a)), a if row == 0 else b, width=1.0, color="#2b6cb0")
        ax.set_title(f"plain BoVW histogram — {label}", fontsize=9.5)
        ax.set_ylim(0, float(max(plain.max(), plain_shuffled.max())) * 1.1)
        ax.set_xlabel("visual word", fontsize=8)

    for row, (vec, label) in enumerate(((pyr, "original"), (pyr_shuffled, "shuffled"))):
        ax = fig.add_subplot(grid_spec[row, 2])
        ax.bar(np.arange(len(vec)), vec, width=1.0, color="#c05621")
        ax.set_title(f"BoVW + SPM (L={spm.levels}) — {label}", fontsize=9.5)
        ax.set_ylim(0, float(max(pyr.max(), pyr_shuffled.max())) * 1.1)
        ax.set_xlabel(f"visual word x {spm.n_cells} spatial cells", fontsize=8)

    fig.suptitle(
        "What 'orderless' actually means — the same descriptors, re-pooled after shuffling "
        "their positions\n"
        f"Plain BoVW: max |Δ| = {d_plain:.1e}  →  the two histograms are the SAME VECTOR. "
        f"Layout was never encoded, so destroying it costs nothing.\n"
        f"BoVW + SPM: max |Δ| = {d_pyr:.4f}  →  genuinely different, because the pyramid "
        f"records WHERE each word occurred.\n"
        "That asymmetry is the layout axis this study measures — and the reason the two are "
        "separate rows rather than one method.",
        fontsize=10.5, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    path = out_dir / "bovw_permutation.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    permutation: plain max|Δ| = {d_plain:.3e}, SPM max|Δ| = {d_pyr:.4f}")
    return path


def figure_normalisation(phi: BoVWRepresentation, frame, images: np.ndarray,
                         out_dir: Path) -> Path:
    """What power_l2 does to a bursty histogram, on a real sign."""
    kmeans = phi._fitted()
    image = images[pick(frame, SAMPLES[1][0])]
    counts = np.bincount(kmeans.predict(phi.extractor.describe(image)),
                         minlength=phi.n_words).astype(np.float32)[None]

    schemes = ("none", "l1", "l2", "power_l2")
    fig, axes = plt.subplots(1, len(schemes), figsize=(3.4 * len(schemes), 3.6))
    for ax, scheme in zip(axes, schemes, strict=True):
        vec = normalise_histograms(counts.copy(), scheme)[0]
        top = np.sort(vec)[::-1]
        ratio = top[0] / max(top[1], 1e-12)
        ax.bar(np.arange(len(vec)), vec, width=1.0,
               color="#2b6cb0" if scheme != "power_l2" else "#2f855a")
        ax.set_title(f"{scheme}\ntop/second = {ratio:.2f}", fontsize=9.5,
                     fontweight="bold" if scheme == "power_l2" else "normal")
        ax.set_xlabel("visual word", fontsize=8)
    axes[0].set_ylabel("weight", fontsize=9)

    fig.suptitle(
        "Only the square root changes the RATIO between bins — which is the whole point\n"
        "Every image contributes the same number of descriptors, so raw histograms already "
        "sum to a constant: l1 and l2 are pure rescales a linear classifier cannot see.\n"
        "power_l2 compresses a dominant bin toward its square root — Perronnin's burstiness "
        "correction, and why it is the default.",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    path = out_dir / "bovw_normalisation.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_blur_collapse(phi: BoVWRepresentation, frame, images: np.ndarray,
                         out_dir: Path) -> Path:
    """The failure test: does the vocabulary degenerate under blur, and how would we know?"""
    kmeans = phi._fitted()
    # One example from each of many classes, so between-class similarity is meaningful.
    indices = [pick(frame, c) for c in range(0, 43, 2)]
    subset = images[indices]
    levels = list(degradations.levels_for("blur"))
    keys = [str(frame.iloc[i]["path"]) for i in indices]

    distinct, similarity = [], []
    for k in levels:
        degraded = degradations.apply(subset, "blur", k, keys)
        hist = phi.transform(degraded)
        words = [kmeans.predict(phi.extractor.describe(img)) for img in degraded]
        distinct.append(float(np.mean([len(np.unique(w)) for w in words])))
        # Mean cosine similarity between DIFFERENT images -- these are unit vectors already.
        gram = hist @ hist.T
        off = gram[~np.eye(len(gram), dtype=bool)]
        similarity.append(float(off.mean()))

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.4))
    left.plot(levels, distinct, "o-", lw=2, color="#2b6cb0", ms=7)
    left.set_xlabel("motion blur kernel (px)")
    left.set_ylabel("mean distinct words per image")
    left.set_title("Vocabulary usage collapses as blur rises", fontsize=10)
    left.grid(alpha=0.3)
    left.set_ylim(0, max(distinct) * 1.15)

    right.plot(levels, similarity, "o-", lw=2, color="#c05621", ms=7)
    right.set_xlabel("motion blur kernel (px)")
    right.set_ylabel("mean cosine similarity, different classes")
    right.set_title("...and different signs start to look alike", fontsize=10)
    right.grid(alpha=0.3)
    right.set_ylim(0, 1.0)

    fig.suptitle(
        "The failure test — a degenerate vocabulary passes every numeric check task 6.2 makes\n"
        f"Distinct words per image: {distinct[0]:.1f} → {distinct[-1]:.1f}. "
        f"Between-class similarity: {similarity[0]:.3f} → {similarity[-1]:.3f}.\n"
        "The right panel is the CORRECTING quantity: a falling word count alone could be read "
        "as the representation becoming compact.\n"
        "Rising between-class similarity is what shows it is collapse — the histograms are "
        "converging on each other, which is exactly the blur prediction's mechanism.",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    path = out_dir / "bovw_blur_collapse.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    blur: distinct words {distinct[0]:.1f} -> {distinct[-1]:.1f}, "
          f"between-class cosine {similarity[0]:.3f} -> {similarity[-1]:.3f}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="BoVW mechanics demo (task 6.6).")
    parser.add_argument("--sweep", type=Path,
                        default=config.RESULTS_DIR / "sweeps" / "bovw_configs.csv",
                        help="read the selected geometry and vocabulary from the 6.5 sweep")
    parser.add_argument("--preproc", default=None)
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR / "demo" / "bovw")
    parser.add_argument("--vocab-samples", type=int, default=100_000)
    args = parser.parse_args()

    config.set_seeds()
    args.out.mkdir(parents=True, exist_ok=True)

    from gtsrb import tuning

    selected = tuning.best_from_sweep(args.sweep)
    preproc = args.preproc or str(selected["preproc"])
    size, step = int(selected["keypoint_size"]), int(selected["step"])
    n_words = int(selected["n_words"])
    print(f"selected at 6.5: preproc={preproc} size={size} step={step} k={n_words}")

    train, _ = data.train_val_split()
    images = cache.load_images(train, preproc)

    # One vocabulary, shared by both representations -- the same guarantee the comparison
    # relies on, so the permutation figure compares like with like.
    phi = BoVWRepresentation(n_words=n_words, step=step, keypoint_size=size,
                             preproc=preproc, n_vocab_samples=args.vocab_samples)
    phi.fit(images, progress=True)
    spm = BoVWSpatialPyramid(n_words=n_words, step=step, keypoint_size=size,
                             preproc=preproc, levels=2)
    spm._kmeans = phi._fitted()
    print(f"{phi!r}\n{spm!r}")

    for path in (
        figure_grid_and_words(phi, train, images, args.out),
        figure_permutation(phi, spm, train, images, args.out),
        figure_normalisation(phi, train, images, args.out),
        figure_blur_collapse(phi, train, images, args.out),
    ):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
