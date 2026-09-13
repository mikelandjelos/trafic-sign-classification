# 09 — Bounding-box jitter: why it moved off GTSRB

**Feeds:** Methodology → scope; **Limitations**; Future work. The measurement below is a
result in its own right and should appear in the report, not only as a justification.
**Status:** decided (planning); execution scheduled as the extension in `PROJECT_TASKS.md` §11.

---

## The finding

**GTSRB cannot support honest bounding-box jitter.** This is not a flaw in the dataset — it
is a consequence of what GTSRB is. It is a *classification* benchmark, and its images were
pre-cropped by the dataset authors in 2011 to the sign plus a small margin. Everything
outside that margin — road, sky, pole — no longer exists in any file.

Measured over all 39,209 training images:

| Margin around the ROI | left | top | right | bottom |
|---|---|---|---|---|
| min (px) | 0 | 5 | 1 | 5 |
| median (px) | 6 | 6 | 5 | 5 |
| max (px) | 20 | 20 | 20 | 20 |

Median margin ≈ **16.7 % of the ROI size**. The maximum outward scaling before a box runs
off the edge of the file is **1.30× median** (min 1.00×, max 1.80×). So:

| Expand ROI by | Images that fit | Images that would clip |
|---|---|---|
| +10 % | 100.0 % | 12 |
| +20 % | 86.2 % | 5 410 |
| +30 % | 50.7 % | 19 330 |
| **+40 %** | **28.2 %** | **28 166** |

The project plan specified ±40 %. **That is impossible for 72 % of the dataset.**

The only ways to produce it anyway are to pad with invented pixels or to clamp the box —
both of which fabricate or distort exactly the background content the experiment is
supposed to measure the effect of. A jitter curve built that way measures the padding
strategy, not the representation.

**This is worth stating plainly in the report.** Bbox jitter on GTSRB appears in published
work, usually implemented with padding. Measuring the headroom and declining to fake it is
a stronger methodological position than producing a jitter panel that looks complete.

## What was decided

Jitter is **not in the MVP**: it appears nowhere in training, nowhere in the core evaluation
grid. It becomes a *test-time stressor applied only to uncropped datasets*, where the
pixels genuinely exist.

Consequences already applied:

- Core grid: **5 methods × 16 conditions = 80 runs** (clean + noise/blur/gamma × 5 levels),
  down from the planned 105. No cell is compromised.
- GTSRB preprocessing uses the dataset's own framing (`roi=False`) — see note 08, Decision 3.
- The predictions table **keeps its jitter row**, marked as tested in the extension. A
  prediction we could not test on the planned data is honest; a silently deleted one is not.

## Why this is better science, not just cheaper

Excluding jitter from *training* was the decisive simplification, and it improves the design:

- **The control stays pure.** Classifier fixed, training pipeline fixed, jitter purely a
  test-time perturbation. A trained-with/trained-without axis would double the grid and
  weaken the claim that differences are attributable to the representation.
- **It asks the more interesting question.** "Which representation is *intrinsically* robust
  to box error?" is worth answering precisely *because* the model has never seen jitter.
  "Does augmentation help?" is already known to be yes.
- **It removes an asymmetry we could not have escaped.** Training-time jitter on GTSRB would
  also be limited by the same margin, so a model could only learn *inward* perturbation
  while being tested on both directions. GTSDB cannot fill that gap either — 1,213 signs is
  far too few to train on.

## The staged plan

| Stage | What | Status |
|---|---|---|
| 1 | Core pipeline, no jitter anywhere. Establishes the baseline results. | MVP |
| 2 | Evaluate the **unmodified** models on GTSDB across jitter levels. Measures intrinsic robustness to box error. | extension |
| 3 | If stage 2 shows the models are *not* robust, propose GTSDB as a training supplement and build a jitter-aware model. | conditional on stage 2 |
| 4 | A further uncropped dataset (BelgiumTS the only realistic candidate) for cross-country generalisation. | future work |

Stage 3 is explicitly **conditional**: it is only worth doing if stage 2 shows a robustness
gap. Reporting "the representations were already robust to ±X %, so no augmentation was
warranted" is a legitimate and cheaper outcome.

## GTSDB — verified before committing

Checked by reading the archive's central directory and `gt.txt` over HTTP range requests
(237 KB transferred, no full download):

| | |
|---|---|
| Annotated signs | **1 213** across 741 frames (1.64 per frame) |
| Frame size | 1360×800 — full road scenes, so jitter has real pixels in every direction |
| Class scheme | `ClassId` **0–42, identical to GTSRB, all 43 classes present** |
| `gt.txt` format | `file;x1;y1;x2;y2;ClassId` — same semantics as GTSRB's ROI columns |
| Sign height | 16 / **37** / 128 px (min/median/max) — GTSRB is 15/32/193, so comparable |
| Per-class support | min **2**, median 16, max 88 |
| Archive | `FullIJCNN2013.zip`, 1 585 MB, contains full frames *and* per-class crops |

### Three conditions the stage-2 evaluation must satisfy

1. **Jitter level 0 must replicate GTSRB's framing** — the annotated box *plus an equivalent
   ~17 % margin*, not the tight box. Otherwise a framing mismatch is measured and reported
   as domain shift.
2. **The panel must be read relative to GTSDB-at-jitter-0**, not to GTSRB-clean. The gap
   between those two baselines is a separate and valuable quantity: a direct **cross-dataset
   generalisation** measurement the original plan would not have produced.
3. **Accuracy only, not macro-F1.** 14 of 43 classes have fewer than 10 instances and the
   smallest has 2; macro-F1 over such classes is quantisation noise. Whole-set accuracy at
   n = 1 213 carries a ±1.7 pp 95 % CI — ample for the multi-point drops jitter causes,
   but not enough for per-class claims.

### Framing for the report

The four stressors are **not four instances of one experiment**, and saying so is better
than implying otherwise:

- noise, blur, gamma → *controlled degradations*: same signs, degraded pixels, clean causal
  attribution.
- jitter on GTSDB → *deployment probe*: different dataset, imperfect boxes, external
  validity.

To keep the panels commensurable, plot every robustness curve as accuracy **relative to its
own baseline**, so what is compared is the rate of degradation rather than the starting
point.

### BelgiumTS caveat

BelgiumTS has 62 classes against GTSRB's 43 — overlapping but not identical, so it needs an
explicit class mapping and would cover only a subset. It is scoped as future work rather
than a deliverable, since it is the one item here that could expand without bound.
