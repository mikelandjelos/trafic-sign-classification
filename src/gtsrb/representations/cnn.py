"""Task 7.1: the small CNN -- the learned, hierarchical representation.

    model = cnn.SmallCNN(in_channels=1)
    logits = model(batch)          # (n, 43)   end-to-end head      -> method "cnn_e2e"
    z      = model.embed(batch)    # (n, 128)  penultimate features -> method "cnn_feat_svm"

The CNN appears **twice** in the comparison, which is why both paths live on one model. As
`cnn_e2e` it classifies with its own softmax head; as `cnn_feat_svm` its penultimate layer
feeds the same `LinearSVC` every other representation uses, which is the only way to put a
learned representation on the same footing as PCA, HOG and BoVW.

Architecture
------------
Three blocks of (conv -> BN -> ReLU) x2 -> maxpool, widening 32 -> 64 -> 128, then a dropout
+ fully-connected head. At 48x48 the three pools give 48 -> 24 -> 12 -> 6, so the trunk ends
at 128 x 6 x 6.

Why the head is `Flatten -> 4608 -> 128 -> 43`
----------------------------------------------
The plan sets a budget of **under 1 M parameters**. The obvious head (4608 -> 256 -> 43)
misses it at **1.48 M**, and not because of the convolutions: the conv trunk is only 287 k,
so the first linear layer alone was 80 % of the model. Three ways out were considered:

| head | params | |
|---|---|---|
| `Flatten -> 4608 -> 256 -> 43` | 1.48 M | over budget |
| **`Flatten -> 4608 -> 128 -> 43`** | **0.88 M** | **chosen** |
| global average pool -> 128 -> 43 | 0.29 M | rejected -- see below |

Global average pooling is the cheapest fix by far, and it is rejected for a **structural**
reason rather than a performance one. GAP averages each feature map over all positions, which
discards spatial layout entirely -- it would make the CNN *orderless*, which is precisely the
property that distinguishes BoVW in the comparison table. Two of the four representations
would then sit on the same point of the layout axis and the axis would stop being measurable.
Halving the hidden layer instead keeps the CNN where the study's premise puts it: hierarchical
and pooling-*invariant*, but layout-preserving.

The 128-unit penultimate layer is also a reasonable feature width for task 7.5 -- comparable
to PCA's 256 and far below HOG's 900-2352, which is worth noting when reading the feature-dim
column of Table 1.

Cost, measured before the model was written
-------------------------------------------
On this CPU (8 threads, no CUDA): **116 s/epoch** at batch 128 over 31,379 images, and
**1.2 ms/img** inference. A 30-epoch run is about an hour. The CNN is systematically penalised
by the absence of a GPU and its timings must be read as relative costs in one environment --
see note 06.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from gtsrb import config

#: Penultimate width -- the dimensionality `cnn_feat_svm` hands to `LinearSVC`.
EMBEDDING_DIM = 128

#: Spatial size the conv trunk ends at, for 48x48 input with three 2x2 pools.
TRUNK_SPATIAL = 6


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """(conv -> BN -> ReLU) x2 -> maxpool.

    Two convolutions per block before pooling: it buys a 5x5 effective receptive field at
    the cost of a 3x3 one, which matters on 48x48 inputs where there is little room to go
    deep. BatchNorm sits before the ReLU (the original placement) and also removes any need
    to standardise the input beyond the [0, 1] scaling every representation shares.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Three conv blocks (32 -> 64 -> 128) and a 128-unit penultimate layer."""

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = config.N_CLASSES,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.embedding_dim = int(embedding_dim)

        self.trunk = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * TRUNK_SPATIAL * TRUNK_SPATIAL, self.embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(self.embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits, `(n, 43)`. Dropout is active in train mode -- as it should be."""
        return self.head(self.embedding(self.trunk(x)))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate features, `(n, embedding_dim)`, for task 7.5.

        **Forces `eval()` and `no_grad()` and restores the previous mode.** This is the
        documented gotcha in PROJECT_TASKS section 10: extracted with dropout still active,
        the features carry multiplicative noise and differ between two calls on the same
        image. The result is not an error but a silently worse `cnn_feat_svm`, so the
        guarantee is enforced here rather than left to every caller to remember.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self.embedding(self.trunk(x))
        finally:
            self.train(was_training)

    def n_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not trainable_only)


def as_batch(images: np.ndarray) -> torch.Tensor:
    """`(n, H, W[, C])` uint8 -> `(n, C, H, W)` float32 in [0, 1].

    The same [0, 1] scaling `cache.flatten` and the other representations use, so all five
    methods see identically scaled pixels. Channels-last to channels-first is done here
    rather than in the cache, because torch is the only consumer that wants that layout.
    """
    images = np.asarray(images)
    if images.ndim == 3:  # (n, H, W) grayscale
        images = images[:, :, :, None]
    if images.ndim != 4:
        raise ValueError(f"expected (n, H, W) or (n, H, W, C), got shape {images.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2)))
    return tensor.float().div_(255.0) if images.dtype == np.uint8 else tensor.float()


def build(preproc: str, **kwargs) -> SmallCNN:
    """A model matched to a preprocessing config's channel count."""
    from gtsrb import preprocessing

    return SmallCNN(in_channels=preprocessing.get_config(preproc).channels, **kwargs)
