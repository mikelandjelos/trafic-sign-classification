# 02 — Dataset structure and the track identifier

**Feeds:** Data section; Methodology → train/val protocol. The track finding below is
report-worthy in its own right — it is a concrete instance of the leakage hazard the
proposal already commits to addressing.
**Status:** complete (task 0.2)

---

## What was downloaded

| Archive | SHA-256 (first 16) | Size | Contents |
|---|---|---|---|
| `GTSRB_Final_Training_Images.zip` | `d32ac4b5fa9a1cbd` | 263 MB | 39 209 PPM + 43 × `GT-*.csv` |
| `GTSRB_Final_Test_Images.zip` | `48ba6fab7e877eb6` | 85 MB | 12 630 PPM |
| `GTSRB_Final_Test_GT.zip` | `f94e5a7614d75845` | <1 MB | `GT-final_test.csv` |

Full digests in `data/CHECKSUMS.txt`. On disk: 349 MB archives + 561 MB extracted.

Verified counts (task 0.2 acceptance criteria, all passing):

```
train images     39209    class directories   43
test images      12630    per-class GT csv    43
                          test GT csv          1
```

Annotation header, identical for train and test:

```
Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
```

The ROI columns are present as required — bounding-box jitter (task 3.4) is therefore
feasible on real coordinates rather than simulated by padding.

---

## The track identifier is not globally unique

**This is the single most important structural fact about GTSRB for this project, and it
is easy to get wrong in a way that produces no error.**

Training images are named `TTTTT_FFFFF.ppm`, where `TTTTT` is the track (30 consecutive
frames of the same physical sign as the vehicle approaches it) and `FFFFF` the frame.
The obvious reading — that `TTTTT` identifies a track — is **wrong**. The numbering
restarts at `00000` inside every class directory:

| Quantity | Value |
|---|---|
| Training images | 39 209 |
| Distinct raw `TTTTT` prefixes | **75** |
| Distinct `(class_id, track_id)` pairs | **1 307** |
| Frames per track | 30 for 1 306 tracks; 29 for one (`class 33, track 19`) |
| Tracks per class | min 7 (class 0), max 75 (class 2) |

1 307 × 30 = 39 210, one more than the 39 209 images present — accounted for exactly by
the single 29-frame track. The track structure is otherwise perfectly regular.

**The correct track key is therefore `(class_id, track_id)`, never `track_id` alone.**

### Why this matters, and why it is worth a paragraph in the report

Keying on the raw prefix does not produce an obvious crash. It produces 75 groups instead
of 1 307, and every downstream step still runs:

- An 80/20 split over 75 groups is far coarser than over 1 307; with 43 classes to
  stratify, class balance across the split becomes impossible to control.
- Track `00000` of *every* class is forced onto the same side of the split, so the
  grouping no longer corresponds to any physical property of the data.
- Validation accuracy would still look plausible, and nothing would flag the error.

This is the same *class* of hazard as the random-split leakage the project already guards
against — a splitting bug that inflates or distorts results silently rather than loudly.
Worth stating in the report that the invariant is **asserted in code**, not merely
observed once: `scripts/download_data.py` fails if scoping by class does not increase the
track count, and if any track exceeds 30 frames. Task 1.3's test enforces the disjointness
property itself.

### Consequence for task 1.1

The parsed dataframe must carry a composite track key. Storing the raw prefix as
`track_id` invites exactly the bug above; the column should be built as
`f"{class_id:05d}_{track:05d}"` (or an equivalent tuple) so that it is globally unique by
construction and any later `groupby` is correct by default.

---

## Test labels ship separately

`GTSRB_Final_Test_Images.zip` contains images only; the labels are in a separate archive.
This is a leftover from the original 2011 IJCNN blind competition format, and is worth one
sentence in the report as a reminder that GTSRB was designed as a held-out challenge
rather than an ordinary supervised split. Practically: the test set is used **only** for
final reporting, and all model selection happens on the track-disjoint validation split
carved out of the training data.

---

## Incidental: an extraction bug worth remembering

The first run of `download_data.py` reported 0 test images while exiting successfully. The
cause was an idempotence check that skipped an archive when its *last* ZIP member already
existed on disk — and the test archive's last member is a bare `GTSRB/` directory entry
that the training extraction had already created. All 12 630 test images were skipped.

The check now compares the full list of non-directory members against disk. The general
lesson, and the reason it is recorded here: **the verification step is what caught it.**
Had the script merely downloaded and extracted without asserting counts, the missing test
set would have surfaced days later, during evaluation.
