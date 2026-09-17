"""Task 9.5: the robustness curves -- the study's headline figure.

    poetry run python scripts/figure_robustness.py

Three panels (noise, blur, gamma), one line per method, **plotted as retention against each
method's own clean baseline**.

Why retention and not absolute score
------------------------------------
Absolute curves are dominated by where each method *starts*, and the question here is not who
is best -- Table 1 answers that -- but **how fast each one falls**. Normalising to each
method's own clean score makes the panels comparable with one another and keeps them
commensurable with the section 11 jitter panel, which lives on a different test set entirely.

It also matters for reading the gamma panel correctly: HOG's predicted gamma win holds **on
retention** (89.2 % vs the CNNs' 86.1 %) and *not* on absolute macro-F1, where the CNNs score
higher (0.8338 vs 0.8084). A reader checking a different column would conclude the prediction
failed. The axis is therefore labelled as retention, explicitly, and the absolute clean scores
are printed in the legend so the starting points are never hidden.

macro-F1, not accuracy
----------------------
`PROJECT_TASKS` 9.5 originally said "accuracy". This plots **macro-F1**, because that is what
every method was *selected* on (10.7x train imbalance, note 05) and what the report leads
with; plotting a metric the study does not use would invite exactly the mismatch the
selection protocol exists to avoid. `--metric accuracy` produces the alternative, and the two
agree on every qualitative claim -- checked, not assumed.

Gamma's x-axis
--------------
Gamma's identity is **1.0, in the middle of its range**, not at an end. The panel is drawn on
a log-x scale so 0.4 and 2.5 sit symmetrically about it, and the baseline is marked -- an
assumption that `levels[0]` is the baseline would have normalised every gamma curve against
gamma=0.4 (the trap `degradations.identity_for` exists to prevent, task 3.3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gtsrb import config, degradations, results

#: Report order, with the colour each method carries in every figure.
METHODS: tuple[tuple[str, str, str], ...] = (
    ("pca_svm", "PCA", "#2b6cb0"),
    ("hog_svm", "HOG", "#c05621"),
    ("bovw_svm", "BoVW", "#2f855a"),
    ("cnn_feat_svm", "CNN feat + SVM", "#805ad5"),
    ("cnn_e2e", "CNN end-to-end", "#1a202c"),
)

PANELS: tuple[tuple[str, str, bool], ...] = (
    ("noise", "Gaussian noise  σ", False),
    ("blur", "motion blur  kernel (px)", False),
    ("gamma", "gamma  γ", True),
)


def grid(metric: str) -> dict:
    """{method: {degradation: (levels, retention%)}} plus each method's clean baseline."""
    frame = results.latest_per_cell(results.load())
    frame = frame[frame.metric == metric]

    out: dict = {}
    for method, _, _ in METHODS:
        rows = frame[frame.method == method]
        clean = rows[rows.degradation == "clean"]
        if clean.empty:
            raise SystemExit(f"no clean {metric} for {method}; run scripts/evaluate_grid.py")
        baseline = float(clean.value.iloc[0])

        per_degradation = {}
        for degradation, _, _ in PANELS:
            block = rows[rows.degradation == degradation].sort_values("level")
            per_degradation[degradation] = (
                block.level.to_numpy(), block.value.to_numpy() / baseline * 100.0
            )
        out[method] = {"baseline": baseline, "curves": per_degradation}
    return out


def figure(data: dict, metric: str, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9), sharey=True)

    for ax, (degradation, xlabel, log_x) in zip(axes, PANELS, strict=True):
        identity = degradations.identity_for(degradation)
        for method, label, colour in METHODS:
            levels, retained = data[method]["curves"][degradation]
            ax.plot(levels, retained, "o-", color=colour, lw=2.0, ms=5.5, label=label)

        ax.axhline(100, color="#a0aec0", lw=1.0, ls=":")
        ax.axvline(identity, color="#a0aec0", lw=1.0, ls="--")
        if log_x:
            # Log-x so 0.4 and 2.5 sit symmetrically about the identity at 1.0. The MINOR
            # ticks must be silenced explicitly: left on, matplotlib draws "6x10^-0.7" style
            # labels straight through the real ones.
            ax.set_xscale("log")
            ax.set_xticks(list(degradations.levels_for(degradation)))
            ax.set_xticks([], minor=True)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(f"{degradation}   (identity at {identity:g})", fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 108)

    axes[0].set_ylabel(f"{metric.replace('_', '-')} retained\n(% of that method's clean score)",
                       fontsize=10)

    # Annotate what is actually VISIBLE in this panel. An earlier version pointed at "PCA and
    # HOG cross between sigma=0 and sigma=5" -- but these are RETENTION curves, so every line
    # starts at 100 % by construction and nothing can cross there. The crossing is in absolute
    # macro-F1 and belongs in Table 1 and the text, not here. What this panel does show is how
    # early HOG departs, which is the more striking fact anyway.
    axes[0].annotate("HOG loses a fifth of its\nperformance at the MILDEST\nnoise level",
                     xy=(5, 80.6), xytext=(13, 88), fontsize=9, color="#c05621",
                     arrowprops={"arrowstyle": "->", "color": "#c05621", "lw": 1.2})

    handles, _ = axes[0].get_legend_handles_labels()
    legend_labels = [
        f"{label}  (clean {data[method]['baseline']:.3f})"
        for (method, label, _) in METHODS
    ]
    fig.legend(handles, legend_labels, loc="lower center", ncol=5, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.06))

    worst = data["hog_svm"]["curves"]["noise"][1][-1]
    best = data["pca_svm"]["curves"]["noise"][1][-1]
    fig.suptitle(
        "Robustness: how fast each representation degrades, relative to its own clean score\n"
        f"THE RANKING INVERTS UNDER NOISE — PCA is LAST on clean data yet retains "
        f"{best:.1f} % at σ=40, where HOG (third on clean data) retains {worst:.1f} %: "
        f"a {best / worst:.1f}× gap, with both CNNs in between.\n"
        "Blur and gamma do NOT invert — there the ordering roughly follows clean performance. "
        "The interaction is real but stressor-specific, not general.",
        fontsize=10.5, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("robustness_curves.png" if metric == "macro_f1"
                      else f"robustness_curves_{metric}.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def report_crossings(data: dict, metric: str) -> None:
    """Print the numbers the caption claims, so the figure cannot drift from the prose."""
    print(f"\n{metric} retention at the strongest level of each stressor:")
    for degradation, _, _ in PANELS:
        line = []
        for method, label, _ in METHODS:
            levels, retained = data[method]["curves"][degradation]
            strongest = int(np.argmax(np.abs(levels - degradations.identity_for(degradation))))
            line.append((retained[strongest], label))
        line.sort(reverse=True)
        pretty = ", ".join(f"{label} {value:.1f}%" for value, label in line)
        print(f"  {degradation:<6}: {pretty}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Robustness curves (task 9.5).")
    parser.add_argument("--metric", default="macro_f1", choices=["macro_f1", "accuracy"])
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR / "results")
    args = parser.parse_args()

    config.set_seeds()
    data = grid(args.metric)
    report_crossings(data, args.metric)
    print(f"\nwrote {figure(data, args.metric, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
