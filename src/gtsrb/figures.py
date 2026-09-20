"""Saving figures that serve two audiences at once.

Every figure in this project is read twice. Once by us, as a demo whose whole job is to
expose a bug that the numeric tests pass over -- and CLAUDE.md is explicit that when a demo
reports a diagnostic which could be misread, the correcting quantity belongs *in the same
figure*. That correcting quantity is almost always the ``suptitle``: "max |delta| = 0.0e+00,
the two histograms are the SAME VECTOR", or "training STOPPED at epoch 24 having last
improved at 18 -- that is the patience rule firing, not an observed plateau".

And once by a reader of the Serbian report, where that same block of English prose is
noise. There the caption is LaTeX's job (``\\caption{}``), the figure should carry no title
of its own, and a second English restatement above the axes only competes with it.

Those two requirements are not in tension as long as the figure is written *once* and saved
*twice*. `save_demo_and_report` does exactly that: the file at ``path`` keeps its title and
stays the demo artifact, and a ``_bare`` sibling is re-laid-out without it for the report.
`scripts/collect_report_figures.py` copies the ``_bare`` variant into ``figures/report/``
under the plain name the report's ``\\includegraphics`` expects.

Why not simply delete the titles
--------------------------------
Because it would silently downgrade every demo in the project to a picture. The suptitles
are where the *measured* reading lives -- several of them record a number that contradicts a
prediction (BoVW's vocabulary does not collapse under blur; the ranking inverts under noise),
and that is precisely the material the report's discussion is built from.

Why not re-run the generators with a flag
-----------------------------------------
Because every one of these scripts needs the 1 GB gitignored dataset, the fitted models, or
both. Producing the pair in one pass means the report copy and the demo copy can never
disagree about the numbers they display, which a second run at a later commit could not
guarantee.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps matplotlib out of import time
    from matplotlib.figure import Figure
    from matplotlib.text import Text

#: Appended to the stem of the title-less copy, e.g. ``x.png`` -> ``x_bare.png``.
BARE_SUFFIX = "_bare"


def bare_path(path: Path | str) -> Path:
    """The title-less sibling of ``path``. Single definition, so the writer and the
    MANIFEST that reads the file cannot drift apart."""
    path = Path(path)
    return path.with_name(f"{path.stem}{BARE_SUFFIX}{path.suffix}")


def save_demo_and_report(
    fig: Figure,
    path: Path | str,
    suptitle: Text | None = None,
    *,
    dpi: int = 160,
    bare_rect: tuple[float, float, float, float] | None = (0.0, 0.0, 1.0, 1.0),
) -> tuple[Path, Path]:
    """Save ``fig`` twice: as-is at ``path``, and without ``suptitle`` at its ``_bare`` sibling.

    Parameters
    ----------
    fig:
        The figure, already fully drawn and laid out for the demo.
    path:
        Where the demo copy goes. Parent directories are created.
    suptitle:
        The artist returned by ``fig.suptitle(...)``. Pass ``None`` when the figure has no
        title, in which case the bare copy is a byte-for-byte duplicate rather than a
        re-render -- a figure with nothing to strip must not be silently re-laid-out.
    dpi:
        Resolution for both files.
    bare_rect:
        ``rect`` handed to ``tight_layout`` for the bare copy. The default reclaims the whole
        canvas, since the space the demo reserved for the title is now free. Pass ``None``
        for a figure laid out by hand -- one built with an explicit ``add_gridspec(top=...)``
        rather than ``tight_layout`` -- where re-running the solver would discard that
        layout. ``bbox_inches="tight"`` still crops away the freed strip, so the bare copy
        comes out correct either way.

    Returns
    -------
    ``(demo_path, bare_path)``.

    The figure is left exactly as it was found -- the title's visibility is restored even if
    saving raises, so a caller that goes on to reuse or re-save ``fig`` is unaffected.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")

    target = bare_path(path)
    if suptitle is None:
        shutil.copyfile(path, target)
        return path, target

    was_visible = suptitle.get_visible()
    suptitle.set_visible(False)
    try:
        if bare_rect is not None:
            fig.tight_layout(rect=bare_rect)
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
    finally:
        suptitle.set_visible(was_visible)
    return path, target
