# 13 — The small CNN: the learned representation (tasks 7.1–7.3, 7.5)

*Feeds: Methodology → representations; Table 1 (9.1); **Limitations** (no GPU); the
`cnn_e2e` vs `cnn_feat_svm` distinction throughout.*

Implementation: `src/gtsrb/representations/cnn.py`. Tests: `tests/test_cnn.py` (17).
Figure: `scripts/figure_cnn_architecture.py` → `figures/report/cnn_architecture.png` — shapes
and parameter counts are read off the real model with forward hooks, so the diagram cannot
drift from the code.
**Status:** complete (tasks 7.1–7.3, 7.5)

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


---

## 9. Task 7.5 — `cnn_feat_svm`, the CNN on the same footing as the others

`scripts/train_cnn_features.py`. **No retraining**: the 128-d penultimate layer of the
network task 7.3 already trained, fed to the same `LinearSVC`, with `C` and `class_weight`
tuned on validation exactly as for every other method.

| | `cnn_feat_svm` | `cnn_e2e` | difference |
|---|---|---|---|
| val accuracy | 0.9889 | 0.9902 | **−0.13 pp** |
| val macro-F1 | **0.9797** | **0.9872** | **−0.75 pp** |
| feature dim | 128 | — | |
| inference, batched | 1.1895 ms | 1.7031 ms | |
| inference, single | 1.6346 ms | 1.4772 ms | |

Selected `C = 0.01`, `class_weight="balanced"`.

### 9.1 The gap is small, and it lands where the theory says it should

The objective mismatch of §1.1 costs **0.75 pp macro-F1** — the features were shaped by
cross-entropy against 43 softmax units, not by what would please a maximum-margin linear
classifier. That the penalty is under a point says the CNN's representation transfers well;
the SVM is not the bottleneck.

**The more interesting part is that macro-F1 falls nearly 6× more than accuracy** (0.75 pp vs
0.13 pp). The transfer does not cost uniformly — it costs on the **hard, rare classes**:

| confusion | `cnn_e2e` | `cnn_feat_svm` |
|---|---|---|
| End of no passing → End of all speed limits | 15.0 % | **31.7 %** |
| Roundabout mandatory → Priority road | 8.3 % | 16.7 % |

Both roughly **double**. The softmax head, trained jointly with the features, resolves the
hardest pairs better than a linear classifier bolted on afterwards can — and those pairs are
in the small classes that macro-F1 weights equally. So the right reading is:

> The CNN's features are nearly sufficient on their own, but the last increment on the hardest
> classes comes from the jointly-trained head, not from the representation.

### 9.2 `C` barely matters — informative in itself

All eight grid points landed within **0.001 macro-F1** of each other (0.9788–0.9797). The
features are so nearly linearly separable that regularisation has almost nothing to do.

Contrast PCA, where `C` moved macro-F1 by 2.6 pp and the best value shifted by three orders
of magnitude with dimensionality (note 11 §7.1). **The flatness of this grid is a property of
the representation**, and worth one line in the report: a good representation makes the
classifier's settings stop mattering.

### 9.3 Table 1 caveat — the 0.04 MB model size is not the cost

The saved artifact is **0.04 MB**, because it is only the SVM head (128 × 43). But it cannot
run without the **3.38 MB network** that produced the features, which is shared with
`cnn_e2e`.

So the honest figure for `cnn_feat_svm` is **3.42 MB**, not 0.04 MB. This is the third
distinct thing the `model_size_mb` column measures (note 06): PCA is 96 % learned basis, HOG
is purely SVM coefficients with nothing learned, and this row is an SVM head plus a shared
network. **Table 1 must state what each number contains**, or this row will look like the
cheapest method in the study when it is among the most expensive.


---

## ADDENDUM (2026-09-16) — the reported CNN rows move to `raw_gray`

Preprocessing is now fixed across the comparison (`PROJECT_TASKS.md` §1, index Q7).

- **`cnn_e2e`** reports the `raw_gray` run: val macro-F1 **0.9856**, accuracy **0.9865**
  (checkpoint `results/models/cnn_e2e_raw_gray.pt`, best epoch 18 of 24). The `clahe_gray`
  run (0.9872 / 0.9902) stays in §5 as the measured preprocessing effect. **The cost is
  0.16 pp** — negligible, and the reason is BatchNorm: the CNN normalises internally, so it
  barely cares what CLAHE did.
- **`cnn_feat_svm` re-run, done** (run `20260916T092421-2dbf5d1`): C=0.01, `balanced`, val
  macro-F1 **0.9774**, accuracy **0.9860**, against 0.9797 / 0.9889 on the `clahe_gray`
  network. **−0.23 pp**, i.e. slightly more than `cnn_e2e` gave up (−0.16 pp) — the SVM head
  is marginally more sensitive to the input contrast than the jointly-trained softmax head,
  which is consistent with §1.1's objective-mismatch reading. No retraining: the network
  exists, only the `LinearSVC` on its penultimate features was refitted.
  `C` again barely matters — all 8 grid points fall within 0.003 macro-F1.
  **The objective-mismatch gap is stable across preprocessing**: 0.82 pp here
  (0.9856 → 0.9774) against 0.75 pp on `clahe_gray`. It is a property of the two heads, not
  an artifact of one input config.

**This is the cleanest illustration of the point Q7 closes on.** The preprocessing cost ranges
from 2.64 pp (PCA, no internal normalisation) through 0.4 pp (HOG, block normalisation) to
0.16 pp (CNN, BatchNorm) — i.e. *how much preprocessing matters is itself a property of the
representation*. Under the old per-method protocol that variation was silently absorbed into
the method-vs-method gaps; under the fixed protocol it is visible and reportable.

§1.1's objective-mismatch caveat and §7.3's shared-confusions finding are unaffected.

---

## ADDENDUM 2 (2026-09-17) — task 7.4 closed, and what it does *not* claim

Task 7.4 read "check background run, **tune LR/epochs**, finalize end-to-end CNN". Two parts
of that were not done, and the report says so rather than letting the task title imply
otherwise.

### 7.4a The learning rate was never tuned

**All three runs used Adam at `lr = 1e-3`, the library default. No learning-rate sweep was
ever run.** It was chosen because it is the standard Adam starting point and the first run
worked; nothing about it is measured.

This is a genuine gap, and it is *asymmetric* in a way that matters for the comparison: PCA,
HOG and BoVW each had their representation hyperparameters swept (k, cell size, geometry,
vocabulary), and every method's `C` was tuned. **The CNN is the only method whose principal
optimisation hyperparameter was taken on faith.** The direction of the bias is knowable —
tuning could only help it — so the CNN's reported numbers are, if anything, a *lower* bound,
and the method that already wins is the one that was under-tuned. That is the benign
direction, but it should be stated, not left for a reader to notice.

### 7.4b The selected run is patience-limited, not verified-converged

**Decision (2026-09-17): accept the existing `raw_gray` run; do not retrain.** The reasoning,
in full, because the claim needs to be exact:

`train()` stops after `patience` epochs with no improvement in validation macro-F1, and
returns the **best** epoch's weights. The `raw_gray` run selected **epoch 18** and stopped at
**24** — i.e. exactly `patience = 6` epochs of no improvement.

| preproc | best epoch | epochs run | threads | train s | val macro-F1 |
|---|---|---|---|---|---|
| `raw_gray` (**reported**) | 18 | 24 | 7 | 2,596 | **0.9856** |
| `clahe_gray` | 18 | 24 | 6 | 3,350 | 0.9872 |
| `clahe_hsv` | 5 | 11 | 8 | 1,133 | 0.9642 |

**What this licenses us to say:** the early-stopping criterion fired as designed, and the
returned weights are the best seen. Both grayscale runs stopped at *identical* epoch indices,
which says the behaviour is reproducible and schedule-determined rather than noise.

**What it does not license:** the word *converged*. We did not observe a long-horizon plateau,
and a larger `patience` might have found further improvement. The honest phrasing is
**"training was stopped by the early-stopping rule at epoch 24, having last improved at epoch
18"** — not "training converged".

Not retrained because the cost (~45 min) buys a plausible ~0.1 pp against a **6.8 pp** lead
over the next method: no ranking, no robustness curve and no conclusion in this report turns
on it. Recorded here so the decision is visible rather than silent.

### 7.4c Training times are not comparable across the three runs

`torch_threads` was 6 / 7 / 8 across the three runs, so the 3,350 / 2,596 / 1,133 s figures
measure three different machines' worth of parallelism. **They must not be compared with each
other**, and only the `raw_gray` row (7 threads) is a candidate for Table 1.

This compounds the §10 gotcha already on record — the CPU power profile moved epoch time by
~20 % mid-run, and `scaling_governor` reads `powersave` in both profiles on amd-pstate, so the
change is invisible in the logs. **Table 1's timing column must be re-measured for all six
methods in one pass on an idle machine at a fixed `energy_performance_preference`** (task
9.1). Until then every duration in `results.csv` is indicative only, per note 06.

---

## Task 7.6 — the demo

`scripts/demo/cnn_mechanics.py` → `figures/demo/cnn/`, four figures, all four in the report
manifest. Everything reads the **trained** network and its recorded history; nothing is built
or fitted here, so the figures describe what was actually reported.

### 7.6a The failure test: the §10 gotcha, made visible

`cnn_dropout_check.png`. With the model deliberately left in **train mode**:

| path | max abs difference between two calls on the same image |
|---|---|
| **`embed()`** — forces `eval()`/`no_grad()` | **0.0000** |
| `embedding(trunk(x))` — called directly | **14.3046** |

37 % of the 128 dimensions change between two calls on the *same image*, by up to a third of
the vector's own peak.

**Both paths are shown, and that is the point.** A figure of the protected path alone would
look identical whether or not the guarantee held, so it would prove nothing — the unprotected
call is made deliberately to produce the contrast. This is the one place in the project where
the `embed()` contract is *demonstrated* rather than asserted by a test name.

Why it matters more than its size suggests: **this bug never raises.** It would surface only
as `cnn_feat_svm` scoring a little lower — which is indistinguishable from "the CNN's
representation transfers less well to an SVM", one of the actual conclusions this project
draws. A silent bug that imitates a finding is the worst kind here.

### 7.6b Early stopping is not convergence — the 7.4b disclosure, visually

`cnn_training_curves.png`. The left panel shades the patience window; the right plots both
losses on a log scale.

- Best epoch **18** (0.9856), stopped at **24** (0.9793). **Returning the last epoch instead
  of the best would have cost 0.63 pp macro-F1** — the concrete value of `train()`'s
  best-weights restore, on the run actually reported. (Note 13 §5 records 0.52 pp for the
  `clahe_gray` run; this is the `raw_gray` equivalent.)
- **Train loss falls to 0.0013** — roughly 1000× — while validation macro-F1 is flat after
  epoch 18. Past that point the network is memorising, not generalising, which is the honest
  visual accompaniment to "we did not tune the learning rate and did not verify convergence".

### 7.6c What the network learned

`cnn_first_layer_filters.png` — the 32 learned 3×3 kernels, on a shared diverging scale. They
are **oriented difference operators**: the learned analogue of the fixed gradient filter HOG
and SIFT apply by construction. That is the most compact answer the report can give to "what
does *learned* actually buy here" — not a different kind of operator, but one fitted to the
data, stacked, and followed by two more blocks. The weight-norm spread also shows the layer
does not use its capacity evenly.

`cnn_activations.png` — mean activation after each block for four sign shapes:
48×48 → 24×24 → 12×12 → 6×6 as channels widen 32 → 64 → 128. **The network trades *where* for
*what*.** Useful directly against HOG: by block 3 a whole sign is described on a 6×6 grid,
*coarser* than HOG's 8×8 cells — but with 128 learned channels per position instead of 9 fixed
orientation bins.

### 7.6d A caption statistic that was wrong, and how it was caught

The dropout figure first reported "dropout zeroes **−1 %** more of the vector per call" — a
negative percentage, i.e. meaningless. The quantity was a difference in zero *counts*, but
ReLU already zeroes most dimensions and dropout rescales the survivors, so the count barely
moves and its sign is arbitrary. Replaced with what actually describes the damage: the
**fraction of dimensions that change** and the change **relative to the vector's own peak**.

Caught by reading the rendered figure, not by any test — the same way the motion-blur kernel
defect (§3.2) and the BoVW blur captions (note 14 §13.3) were caught.
