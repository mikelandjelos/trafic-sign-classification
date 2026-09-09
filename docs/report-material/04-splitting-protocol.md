# 04 — The train/validation splitting protocol

**Feeds:** Methodology → experimental protocol. This is one of the two things the project
plan marks as "never cut", so it deserves explicit treatment in the report rather than a
passing mention.
**Status:** complete (task 1.2)

Implemented as `gtsrb.data.assign_split()` / `train_val_split()`.

---

## The hazard

GTSRB's training images are not independent samples. They come in **tracks**: 30
consecutive frames of the *same physical sign*, captured as the vehicle approaches it.
Consecutive frames differ mainly in scale and a little in blur and viewpoint — they are
near-duplicates.

A random split by image therefore puts near-duplicates of the same sign on **both** sides
of the train/validation boundary. The model is then validated on signs it has effectively
already seen, and validation accuracy measures memorisation rather than generalisation.
The number it reports is inflated and the model selection it drives is unreliable.

This is not a hypothetical concern with GTSRB — it is the standard mistake, and it is
invisible: nothing errors, and the inflated accuracy looks like success.

## The apparent conflict, and why it dissolves

Two requirements pull in different directions at first glance:

1. **Track-disjointness** — no track may appear on both sides.
2. **Class stratification** — every class needs a proportional share on both sides,
   because macro-F1 averages over all 43 classes and imbalance is 10.7× (see note 02).

Classes hold very different numbers of tracks (7 for class 0, 75 for class 2), so a global
80/20 partition of the 1,307 tracks would give no control over per-class balance.

The conflict is only apparent. **Each track belongs to exactly one class** — asserted in
`_validate`, not assumed — so partitioning *each class's own tracks* 80/20 satisfies both
constraints simultaneously. No track is split, and every class keeps its share.

## Achieved split

```
train         : 31379 images, 1046 tracks
val           :  7830 images,  261 tracks
val fraction  : 0.200 by image, 0.200 by track
classes       : train 43, val 43
track overlap : 0
per-class val : min 0.143 (class 0), max 0.250 (class 27)
```

The image-level and track-level fractions both land on exactly 0.200, which is worth
noting: because tracks are near-uniformly 30 frames, splitting by track does not distort
the image-level proportions.

## Quantisation, stated rather than hidden

Tracks are atomic, so the split is quantised in units of ~30 images. Class 0 has only
**7 tracks**, so its finest achievable step is 1/7 ≈ 14.3 % and it cannot land on 20 %; it
receives 1 validation track (14.3 %). Class 27, with 8 tracks, receives 2 (25.0 %).

The per-class validation share therefore ranges over **[0.143, 0.250]** rather than being
exactly 0.2 everywhere. This is an unavoidable consequence of refusing to split tracks,
and the right trade: a slightly uneven share costs a little precision in per-class metrics,
whereas splitting tracks would corrupt the metrics outright.

The implementation guarantees **at least one track per class on each side**. Without that
guarantee a small class could land with zero validation images, which would make macro-F1
undefined for that class and silently change what the averaged metric means.

## Determinism

The split is a **pure function** of (annotations, `SEED`, `val_fraction`) and needs no
on-disk manifest:

- The seed is derived per class via `config.rng_for("train_val_split", class_id)` rather
  than drawn from one shared stream, so a class's assignment does not depend on how many
  classes were processed before it.
- Track lists are sorted before shuffling, so the result does not depend on the row order
  of the input dataframe.

Both properties are verified, not assumed: repeated calls return identical assignments, and
shuffling the input dataframe (`sample(frac=1)`) produces the same assignment after
reindexing. Task 1.3 makes the disjointness property a permanent test rather than a
one-off check.

---

## Opportunity: measure the leakage rather than only asserting it

The report currently argues that a random split *would* inflate validation accuracy. That
argument is much stronger as a measured number than as a claim.

Once any single method exists (PCA + `LinearSVC` trains in about a minute), the experiment
is nearly free: train the same model twice, once with the track-disjoint split and once
with a random per-image split, and compare validation accuracy. The gap is a direct
measurement of the leakage this protocol prevents.

**Estimated cost: ~15 minutes after task 4.3.** Logged as an opportunity in `00-INDEX.md`.
