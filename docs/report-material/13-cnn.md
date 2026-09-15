# 13 — The small CNN: the learned representation (task 7.1)

*Feeds: Methodology → representations; Table 1 (9.1); **Limitations** (no GPU); the
`cnn_e2e` vs `cnn_feat_svm` distinction throughout.*

Implementation: `src/gtsrb/representations/cnn.py`. Tests: `tests/test_cnn.py` (17).
Figure: `scripts/figure_cnn_architecture.py` → `figures/report/cnn_architecture.png` — shapes
and parameter counts are read off the real model with forward hooks, so the diagram cannot
drift from the code.
**Status:** complete (tasks 7.1, 7.2); 7.3 run 1 of 3 done (`clahe_gray`)

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

### 1.1 There is exactly one training run

**`cnn_feat_svm` is not trained.** The network is trained end to end through its softmax head
(tasks 7.2–7.4), and `cnn_feat_svm` then reuses *that* network: penultimate layer out,
`LinearSVC` on top, no retraining (task 7.5, ~20 min).

That is deliberate. Training a second network specifically for feature extraction would make
the two CNN rows differ in more than their classifier, which is exactly the confound this
design exists to avoid — and it would also double the project's largest compute item.

**But it carries a caveat that must be stated when the two rows are compared.** The features
were optimised for a softmax head; the network never "knows" it will be read by a linear SVM.
The penultimate layer is shaped by cross-entropy against 43 output units, not by whatever
would make a maximum-margin linear classifier happiest. So:

> If `cnn_feat_svm` underperforms `cnn_e2e`, **part of that gap is the objective mismatch,
> not a property of the representation.** The honest reading of a gap is "these features,
> trained for a softmax, transfer this well to an SVM" — not "the CNN's representation is
> worth less than its classifier".

This is the standard transfer-learning setup and the right choice here, but it is an
asymmetry the other three methods do not have: PCA, HOG and BoVW were never optimised for
*any* classifier, so their features are objective-neutral in a way the CNN's are not.
Recorded as a limitation, and it belongs in the discussion next to whatever gap appears.

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

## 5. A guideline for "is the CNN actually working"

Not a pass/fail gate and not a target — a sanity band, written down before the first run so
it cannot be adjusted to whatever arrives.

> **A small CNN on GTSRB should land comfortably above ~95 % validation macro-F1.**
> Materially below that suggests a *training* problem — learning rate, schedule, epochs —
> rather than a finding about convolutional representations.

Why ~95 % and not the 98–99 % the literature reports: published results use augmentation,
the full 39,209 training images, and often larger or ensembled networks. This project has
none of those. It trains on 31,379 images (train split only, for comparability — §1.1 of
`11-pca.md` §8.1), 0.88 M parameters, no augmentation, on CPU.

**Two things this guideline is deliberately not:**

1. **Not a licence to retrain until the CNN wins.** That would be tailoring the project to a
   desired outcome, which `CLAUDE.md` forbids for predictions and which applies just as much
   to a method. Task 7.4 is the CNN's tuning budget; the other methods each got one sweep.
   **If more than 7.4's budget is spent, the asymmetry gets recorded**, because ten retraining
   attempts against PCA's single sweep would tilt the comparison.
2. **Not about `cnn_feat_svm`.** That method trailing `cnn_e2e` is *expected* — the objective
   mismatch of §1.1 — and is a finding, not a defect. The guideline concerns `cnn_e2e` alone.

---

## 6. Preprocessing: three training runs, not one

Every other method sweeps all three preprocessing configs cheaply, because the representation
is a fixed function and only the classifier is refit. **For the CNN the representation *is*
the trained weights**, so each config costs a full training run: ~1 h each, ~3 h total.

It is done anyway, for two reasons:

- **Protocol symmetry.** Task 4.2 established that each method selects its own preprocessing,
  after measuring that borrowing another config's hyperparameters cost ~1 pp — about 39 % of
  the preprocessing effect itself. Exempting the CNN would reintroduce exactly that confound.
- **Task 8.2 requires it regardless.** The ablation is "best 2 methods × 3 configs", and the
  CNN is almost certainly one of the best two. The three runs are needed either way; doing
  them at 7.3 costs the same compute as discovering the need at 8.2.

Worth noting for the discussion: the headline config `clahe_gray` is **grayscale**, and
traffic signs are colour-coded. A CNN can exploit colour in ways PCA cannot, so `clahe_hsv`
is a live contender here in a way it was not for PCA (where it lost on macro-F1, note 11
§7.1b). This is the method where the colour question may actually change the answer.

---

## 7. Task 7.3, run 1 of 3 — `clahe_gray`

Run `20260913T...`, 6 threads (shared with the HOG sweep), `results/models/cnn_e2e_clahe_gray.pt`,
epoch history in `results/histories/`.

| | value |
|---|---|
| val accuracy | **0.9902** |
| val macro-F1 | **0.9872** |
| val weighted-F1 | 0.9901 |
| best epoch | **18** (of 24 run; early-stopped, patience 6) |
| training time | 3,350 s = **55.8 min**, 140 s/epoch |
| inference, batched | 1.703 ms/img |
| inference, single image | **1.477 ms/img** |
| model size | 3.38 MB |
| feature dim | 128 |

Comfortably above the ~95 % guideline of §5, so no training problem to chase.

### 7.1 The best-epoch rule earned its keep

The run continued 6 epochs past its optimum (patience), and the final epoch was **0.0052
macro-F1 worse** than epoch 18. Returning the last epoch — the easy implementation — would
have reported a model measurably past its own peak, with nothing to indicate it. Small, but
it is exactly the silent-degradation case `training.train` was written to prevent, and it
happened on the very first real run.

### 7.2 Finding — batching barely helps the CNN, unlike PCA

| method | batched | single image | ratio |
|---|---|---|---|
| PCA + LinearSVC | 0.0066 ms | 0.3254 ms | **49×** |
| **CNN** | **1.703 ms** | **1.477 ms** | **0.87×** |

The CNN's single-image figure is *lower* than its batched one — within measurement noise of
each other, and the opposite of PCA's 49× gap. There is so much per-image work in the
convolutions that there is almost nothing left to amortise over a batch.

**This inverts the cost comparison at batch size 1**, which is the realistic setting for a
camera. Batched, PCA looks ~260× cheaper than the CNN; per frame it is only ~4.5×. Any
real-time claim in the report must use the single-image column, and this is the clearest
demonstration of why both are recorded.

### 7.3 Finding — the CNN's hardest classes are PCA's hardest classes

Worst classes by F1, and the top confusions, compared across two methods that share nothing
but the data and the evaluation harness:

| confusion | PCA (macro-F1 0.798) | CNN (macro-F1 0.987) |
|---|---|---|
| End of no passing → End of all speed and passing limits | 60.0 % | **15.0 %** |
| Roundabout mandatory → Priority road | 58.3 % | 8.3 % |

Worst-class overlap is just as strong: classes **41** (*End of no passing*), **32** (*End of
all speed and passing limits*), **40** (*Roundabout mandatory*) and **29** (*Bicycles
crossing*) are in the bottom five for both.

**The same pairs, attenuated roughly 4×.** Two representations with nothing structurally in
common — one holistic and linear, one hierarchical and learned — fail on the same classes.
That says the difficulty is **intrinsic to those classes**, not an artifact of PCA's
holistic encoding, which is what a single method's confusion table could never establish.

The end-of-restriction signs are near-identical grey circles with diagonal strikethroughs,
differing only in fine internal detail; *Roundabout* and *Priority road* are both
diamond/circular high-contrast shapes. Strong material for task 9.3, and a reason to report
confusions for **more than just the best and worst method**.

---

## 8. Open for the remaining 7.3 runs

- `raw_gray` and `clahe_hsv`, ~1 h each. `clahe_hsv` is the live contender: signs are
  colour-coded and a CNN can exploit that in a way PCA could not (note 11 §7.1b).
- Thread count differs across runs (6 for run 1, which shared the machine with the HOG
  sweep; 7–8 for the rest), so **training times are not comparable between runs** — only
  accuracy is. Recorded per run as `torch_threads` in `results.csv`.

- The run is launched in the background (7.3) because at ~2 min/epoch it is the longest single
  compute item in the project.
- Hyperparameters (LR, epochs, batch size) are **not** chosen yet; whatever is swept must go
  through the same validation-macro-F1 protocol as every other method (§7.4 of `11-pca.md`),
  even though the CNN's own head means `C` does not apply to `cnn_e2e`.
