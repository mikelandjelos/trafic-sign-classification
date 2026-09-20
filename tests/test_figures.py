"""Tests for `gtsrb.figures` -- the demo/report figure pair.

The property that matters is narrow but easy to break: the report copy must actually lose
the title (otherwise the Serbian LaTeX caption is duplicated by an English one inside the
image), and the demo copy must actually keep it (otherwise every demo in the project quietly
loses the diagnostic that CLAUDE.md requires it to carry).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from gtsrb.figures import BARE_SUFFIX, bare_path, save_demo_and_report


@pytest.fixture
def fig():
    figure, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1, 2], [0, 1, 0])
    yield figure
    plt.close(figure)


def test_bare_path_inserts_suffix_before_extension():
    assert bare_path("figures/demo/x/y.png").name == f"y{BARE_SUFFIX}.png"


def test_writes_both_files(tmp_path, fig):
    title = fig.suptitle("a title that the report must not show")
    demo, bare = save_demo_and_report(fig, tmp_path / "f.png", title, dpi=60)

    assert demo.exists() and bare.exists()
    assert bare == tmp_path / f"f{BARE_SUFFIX}.png"


def test_bare_copy_is_shorter_because_the_title_is_gone(tmp_path, fig):
    # The title occupies vertical space, so dropping it and re-tightening must reclaim it.
    # Comparing rendered height is the check that would actually fail if `set_visible(False)`
    # stopped working -- file size would not, reliably.
    title = fig.suptitle("one\ntwo\nthree\nfour", fontsize=14)
    demo, bare = save_demo_and_report(fig, tmp_path / "f.png", title, dpi=60)

    from PIL import Image

    with Image.open(demo) as d, Image.open(bare) as b:
        assert b.height < d.height


def test_figure_is_left_untouched(tmp_path, fig):
    title = fig.suptitle("restored afterwards")
    save_demo_and_report(fig, tmp_path / "f.png", title, dpi=60)

    assert title.get_visible() is True


def test_visibility_restored_even_when_saving_fails(tmp_path, fig):
    title = fig.suptitle("restored even on failure")
    # A directory where the bare file wants to be makes the second savefig raise.
    (tmp_path / f"f{BARE_SUFFIX}.png").mkdir()

    with pytest.raises(OSError):
        save_demo_and_report(fig, tmp_path / "f.png", title, dpi=60)

    assert title.get_visible() is True


def test_bare_rect_none_leaves_a_hand_built_layout_alone(tmp_path):
    # A figure positioned by an explicit gridspec must not be handed to the tight_layout
    # solver, which would discard those coordinates. Cropping alone still removes the strip
    # the hidden title occupied.
    figure = plt.figure(figsize=(4, 3))
    spec = figure.add_gridspec(1, 1, top=0.70, bottom=0.10, left=0.10, right=0.95)
    figure.add_subplot(spec[0, 0]).plot([0, 1], [0, 1])
    before = figure.axes[0].get_position().bounds

    title = figure.suptitle("stripped from the bare copy", fontsize=14)
    demo, bare = save_demo_and_report(figure, tmp_path / "f.png", title, dpi=60,
                                      bare_rect=None)

    assert figure.axes[0].get_position().bounds == before
    from PIL import Image

    with Image.open(demo) as d, Image.open(bare) as b:
        assert b.height < d.height
    plt.close(figure)


def test_without_a_title_the_copy_is_byte_identical(tmp_path, fig):
    # Nothing to strip must mean nothing to re-lay-out: a re-render could shift the layout
    # for no reason, so this path copies instead.
    demo, bare = save_demo_and_report(fig, tmp_path / "f.png", None, dpi=60)

    assert demo.read_bytes() == bare.read_bytes()
