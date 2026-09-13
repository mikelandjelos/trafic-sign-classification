# 13 — The small CNN: the learned representation (task 7.1)

*Feeds: Methodology → representations; Table 1 (9.1); **Limitations** (no GPU); the
`cnn_e2e` vs `cnn_feat_svm` distinction throughout.*

Implementation: `src/gtsrb/representations/cnn.py`. Tests: `tests/test_cnn.py` (17).
Figure: `scripts/figure_cnn_architecture.py` → `figures/report/cnn_architecture.png` — shapes
and parameter counts are read off the real model with forward hooks, so the diagram cannot
drift from the code.

---

## 1. Why one model appears twice

The CNN is the only representation that is *learned end to end*, which makes it both the
most interesting entry and the hardest to compare fairly. It therefore appears **twice**:

| method | what it is | comparable to |
|---|---|---|
| `cnn_e2e` | the whole network, its own softmax head | nothing else — it is the ceiling |
| `cnn_feat_svm` | penultimate layer → the same `LinearSVC` | PCA, HOG, BoVW directly |

Only the second is a like-for-like comparison. Without it, a CNN win would be ambiguous
between "the representation is better" and "the classifier is better", and the study's whole
design is built to keep those apart. Both paths live on one model: `forward()` and `embed()`.

---

## 2. Architecture

Three blocks of (conv → BN → ReLU) × 2 → maxpool, widening **32 → 64 → 128**, then dropout
and a fully-connected head. At 48×48 the three pools give 48 → 24 → 12 → 6, so the trunk
ends at 128 × 6 × 6.

Two convolutions per block before pooling: a 5×5 effective receptive field for the cost of a
3×3 one, which matters on 48×48 inputs where there is little room to go deep. BatchNorm sits
before the ReLU, and also removes any need to standardise the input beyond the [0, 1] scaling
all five methods share.

### 2.1 The parameter budget, and the decision it forced

The plan sets a budget of **under 1 M parameters**. The obvious head misses it — and
benchmarking *before* writing the model is what caught it:

| head | total params | |
|---|---|---|
| `Flatten → 4608 → 256 → 43` | **1.48 M** | over budget |
| **`Flatten → 4608 → 128 → 43`** | **0.88 M** | **chosen** |
| global average pool → 128 → 43 | 0.29 M | rejected — see below |

Where the parameters actually are, which is what made the third option tempting:

| component | params |
|---|---|
| conv trunk (3 blocks) | 286,880 |
| **first linear layer** | **589,952** |
| classifier head | 5,547 |
| **total** | **882,379** |

The convolutions are a third of the model; **the first linear layer was 80 % of the original
1.48 M version.**

### 2.2 Why global average pooling was rejected — a structural reason, not a performance one

GAP is by far the cheapest fix (0.29 M, a 5× reduction). It is rejected because of what it
would do to the *comparison*, not to the accuracy.

Global average pooling averages each feature map over all spatial positions. That makes the
representation **orderless** — which is precisely the property that defines BoVW in the
structural table:

| Representation | Spatial structure | Locality | Learned? |
|---|---|---|---|
| PCA | holistic, alignment-critical | global | no |
| HOG | rigid grid, layout preserved | local | no |
| BoVW | **orderless**, layout discarded | local | vocabulary only |
| CNN | hierarchical, pooling-invariant | local | fully |

With GAP, the CNN row's "spatial structure" would become *orderless* too. Two of the four
representations would sit on the same point of the layout axis, and **that axis would stop
being measurable** — including the §11 jitter extension, whose whole premise is that
orderless encodings tolerate a shifted box and layout-preserving ones do not.

Halving the hidden layer costs 0.59 M parameters of budget and keeps the CNN where the
study's premise puts it. **Recorded as a limitation:** the network is smaller in its head
than a free choice would make it, for comparability rather than for accuracy.

### 2.3 The 128-unit penultimate layer

Chosen as the budget-compliant width, but worth noting for Table 1's feature-dim column: it
is comparable to PCA's 256 and far below HOG's 900–2352. When the cost table is read, the
CNN's feature vector is among the *smallest*, not the largest.

---

## 3. The `eval()` / `no_grad()` gotcha, enforced rather than documented

PROJECT_TASKS §10 warns that penultimate features extracted with dropout still active carry
multiplicative noise. The failure mode is the dangerous kind: **nothing errors**, and
`cnn_feat_svm` is merely, inexplicably worse than it should be.

`embed()` therefore forces `eval()` and `no_grad()` internally and restores the previous mode
on the way out. Three tests pin the contract:

- `embed` is **deterministic even when the model is in train mode** — two calls on the same
  input agree exactly;
- it **restores** the mode it found, so it cannot silently leave a model in `eval()` midway
  through a training loop;
- the counterpart: `forward` still *does* keep dropout active in train mode, so enforcing the
  first property has not accidentally changed how the network trains.

---

## 4. Cost, measured before the model was written

On this CPU (8 threads, **no CUDA** — the AMD Vega iGPU has no ROCm support):

| | measured |
|---|---|
| training | **116 s/epoch** at batch 128 over 31,379 images |
| inference | **1.2 ms/img** |
| 30 epochs | ~58 min |
| 50 epochs | ~97 min |

Two things to carry into the report:

1. **The CNN is systematically penalised by the absence of a GPU**, and its timings are
   relative costs in one recorded environment — not deployment latency (note 06). On a GPU
   the training column would change by one to two orders of magnitude while the accuracy
   column would not move at all.
2. **Inference is 1.2 ms/img against PCA's 0.33 ms** single-image — so the CNN is ~3.6×
   slower at inference *even batched*, which is the honest version of the cost comparison and
   not something a GPU number would show.

---

## 5. Open for tasks 7.2 / 7.3

- Training loop with per-epoch validation, early stopping and best-checkpoint saving.
- The run is launched in the background (7.3) because at ~2 min/epoch it is the longest single
  compute item in the project.
- Hyperparameters (LR, epochs, batch size) are **not** chosen yet; whatever is swept must go
  through the same validation-macro-F1 protocol as every other method (§7.4 of `11-pca.md`),
  even though the CNN's own head means `C` does not apply to `cnn_e2e`.
