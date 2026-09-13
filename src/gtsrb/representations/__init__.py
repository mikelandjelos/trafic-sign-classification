"""Tasks 4-7: the representations under comparison.

Each representation is a map phi: image -> fixed-length vector. The classifier downstream
is held fixed (`LinearSVC`), so any difference in results is attributable to phi and not to
the classifier -- that is the whole experimental design, and it only holds if every
representation exposes the same interface and is trained on the same rows.

The shared contract
-------------------
`fit(X)` learns whatever phi needs from **training images only**, `transform(X)` maps any
batch to `(n_samples, n_features)` float32. Some representations learn nothing (HOG), some
learn unsupervised (PCA, BoVW vocabulary) and one learns supervised (CNN); the interface
does not distinguish them, because the grid at task 8.1 must be able to loop over all five.

Two rules that apply to every implementation
--------------------------------------------
1. **Fit on the train split only** -- never train+val, never the test set. This is easy to
   get wrong for the unsupervised ones: PCA and the BoVW vocabulary use no labels, which
   makes fitting them on everything *feel* harmless. It is not. The basis and the mean
   would carry information about the rows they are later scored on.
2. **Fit on clean images only.** Degradations are a test-time stressor (task 8.1); the
   representation never gets to adapt to them. Re-fitting per condition would measure how
   well a method can be *retrained* for a stressor, which is a different question.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Representation(Protocol):
    """The interface every representation implements. See the module docstring."""

    #: Short identifier used in the `method` column of results.csv.
    name: str

    def fit(self, images: np.ndarray) -> Representation:
        """Learn from clean training images, `(n, H, W[, C])` uint8."""
        ...

    def transform(self, images: np.ndarray) -> np.ndarray:
        """Map images to `(n, n_features)` float32."""
        ...

    @property
    def n_features(self) -> int:
        """Output dimensionality -- the feature-dim column of Table 1 (task 9.1)."""
        ...
