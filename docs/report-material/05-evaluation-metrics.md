# 05 — Evaluation metrics and the measurement harness

**Feeds:** Methodology → evaluation; Results → Table 1; supports figures 9.2–9.4.
**Status:** complete (tasks 1.4, 4.2 — macro-F1 also became the *selection* criterion)

Implemented as `gtsrb.evaluation`; tested in `tests/test_evaluation.py`.

---

## Why the harness is built before any method exists

Tasks 4–7 are the four experiments; task 1 calibrates the instrument they are all measured
on. Building it first is deliberate: if the evaluation code grew alongside the first method
written, it would end up — without anyone intending it — shaped to suit that method. Built
blind to the methods, it cannot flatter any of them.

Practically, it is also what makes the 5 × 16 = 80-run grid tractable. Evaluation is one
function call returning one object, so no run can record its metrics slightly differently
from another.

## Metrics, and why each is present

| Metric | Why |
|---|---|
| **Accuracy** | the headline number, and what GTSRB results are conventionally quoted as |
| **Macro-F1** | the honest number under 10.7× imbalance — see below |
| **Weighted-F1** | included for completeness; tracks accuracy closely and is not load-bearing |
| **Per-class F1** (43,) | locates *which* classes a representation fails on, feeding the discussion |
| **Confusion matrix** (43×43) | figure 9.2, and the source of the confused-pairs table 9.3 |

### Accuracy alone would be misleading here

Class frequencies differ by **10.7×** (note 02). A classifier that neglects class 0
entirely — 210 of 39,209 training images — sacrifices about 0.5 % accuracy and still looks
strong. Macro-F1 weights all 43 classes equally and exposes exactly that failure.

The harness test suite pins this with a deliberately extreme case: a classifier that never
predicts a rare class scores **0.98 accuracy but under 0.03 macro-F1**. Both numbers appear
in Table 1 for every method, and where they diverge, that divergence is itself a finding.

## The fixed-label decision

Every metric is computed with `labels=range(43)` pinned explicitly, rather than letting
scikit-learn infer the label set from the data.

This is not defensive boilerplate — it is required by the experimental design. The grid
includes heavily degraded conditions (σ = 40 noise, 15 px blur, 40 % bbox jitter) where a
method may never predict some class at all. With inferred labels, such a run silently
produces a **42×42** confusion matrix and a 42-element per-class F1 vector. Demonstrated:

```
ours          : (43, 43)   per-class F1: (43,)
sklearn naive : (3, 3)     <- would break the grid
```

Nothing raises. The per-class arrays simply stop being comparable across rows of
`results.csv`, and every per-class number after the missing class shifts by one index. The
test suite pins both the shape and the index meaning (`per_class_f1[i]` is class `i`, not
the i-th class that happened to appear).

## Size-stratified results come for free

`accuracy_by_group(y_true, y_pred, size_buckets(df["roi_h"]))` produces figure 9.4 from the
*existing* clean-test predictions, grouped differently — no extra inference, no extra
model. Buckets are half-open `[0,32) [32,48) [48,72) [72,∞)` over ROI height, and the test
split populates them 47.7 / 28.1 / 16.3 / 7.9 % (note 02), so every bucket has at least
1,004 test images.

## Confused pairs and class names

`config.CLASS_NAMES` carries the 43 official GTSRB names, so task 9.3's table reads
"Speed limit (30km/h) → Speed limit (50km/h)" rather than "class 1 → class 2". The
project plan predicts speed limits will confuse predictably; that prediction is only
checkable if the table is legible.

`most_confused_pairs(k)` returns `(true_id, pred_id, count, fraction_of_true_class)`,
excluding the diagonal. The fraction matters as much as the count: 60 errors out of a
2,250-image class is a very different phenomenon from 60 out of 210.

---

## Caution when reading the smoke-test output

The harness was exercised on **synthetic predictions** with a speed-limit confusion bias
injected deliberately, purely to check plumbing. The resulting numbers (0.92 accuracy,
0.90 macro-F1, speed limits topping the confusion table) are **not results** and must never
be quoted as such. Real numbers arrive with task 4.3 onward.


---

## Addendum (task 4.2) — macro-F1 is also the *selection* criterion

This note argues that macro-F1 must be **reported** alongside accuracy, because GTSRB is
imbalanced 10.7x and accuracy is dominated by the large classes. Task 4.2 extends that from
reporting to **model selection**: every hyperparameter choice in the project — `k`, `C`,
`class_weight`, and later HOG's cell size and BoVW's vocabulary size — is made by maximising
**validation macro-F1**, in `gtsrb.tuning.select_best`.

The reason is consistency rather than a new argument. Selecting on accuracy and then leading
the report with macro-F1 would mean the models were optimised for a criterion the report does
not headline — indefensible if a reader asks. Both metrics are recorded at every grid point
regardless, so the alternative selection can always be inspected.

**It changes the answer, so it is not a formality.** In the PCA sweep the two criteria pick
different cells at *k* = 128: macro-F1 selects `class_weight=None` (0.7610) while accuracy
would have preferred a different point, and at *k* = 256 `balanced` wins on macro-F1 by
0.55 pp. The preprocessing configs disagree more sharply — `clahe_hsv` beats `raw_gray` on
**accuracy** (0.8257 vs 0.8140) but loses on **macro-F1** (0.7674 vs 0.7713), i.e. colour
helps the common classes and not the rare ones. Any table comparing those two configs must
therefore say which metric it is ranking by.

Ties break toward the **smaller `C`**, so selection is a deterministic function of the
recorded grid rather than of the order rows arrive in.
