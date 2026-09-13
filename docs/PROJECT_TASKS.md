# GTSRB — Poređenje reprezentacija za prepoznavanje saobraćajnih znakova

Task tracker for the Computer Vision (Računarski vid) project.
Working doc in English; the final report is in Serbian.

---

## 1. Scope

**What this project is:** a controlled comparison of five ways to represent a cropped
traffic sign image, evaluated on GTSRB.

**Input:** one RGB crop containing exactly one traffic sign (15×15 to 250×250 px).
**Output:** one of 43 class labels.

**Research question:** the four representations differ along several structural axes at
once. Which one wins is expected to *depend on the stressor* — the contribution is the
interaction between representation and degradation type, not a leaderboard.

| Representation | Spatial structure | Locality | Learned? |
|---|---|---|---|
| PCA | holistic, alignment-critical | global | no |
| HOG | rigid grid, layout preserved | local | no |
| BoVW | **orderless**, layout discarded | local | vocabulary only |
| CNN | hierarchical, pooling-invariant | local | fully |

**Experimental design.** The classifier is held fixed (`LinearSVC`) across all four
representations, so any difference is attributable to the representation itself and not
confounded with the classifier. The CNN therefore appears **twice**: once as a feature
extractor (penultimate layer → `LinearSVC`, comparable to the others) and once
end-to-end with its own softmax head. Five rows total.

### Explicitly out of scope

Not because they are uninteresting, but because they belong to the full pipeline of
which this project is one module:

- Detection / localization on full frames (GTSDB)
- Video, tracking, optical flow
- Camera calibration, camera model, distance estimation
- Any real-time end-to-end system

The report describes the full system architecture and then marks, without ambiguity,
which single module this project delivers.

---

## 2. Predictions (fill in BEFORE running anything)

Write these down first. Being wrong about one is more interesting than being right
about all of them, and a prediction made after seeing results is worthless.

> Rewritten after the scope change: the original table predated the jitter rescope and had
> no row for gamma, leaving only 2 of 4 rows describing MVP conditions. Entries revised from
> that original are marked ⚠ with the reason. Recorded in `predictions.md` on 2026-09-13,
> before any model was trained.

### MVP stressors

| Stressor | Most robust | Least robust | Reasoning |
|---|---|---|---|
| **Gaussian noise** σ 0→40 | PCA | HOG | Noise is high-frequency and isotropic; PCA keeps only k ≈ 128 of 2304 dimensions, so most noise energy falls outside the retained subspace and is averaged away. HOG differentiates — a high-pass operation that amplifies noise directly and randomises orientation bins. BoVW is also gradient-based but SIFT's 4×4 spatial pooling averages somewhat, so it should land between. |
| **Motion blur** k 0→15 px | PCA | ⚠ BoVW | Blur is low-pass and preserves exactly the low-frequency structure the leading eigenvectors encode. BoVW suffers twice: local 12 px patches become near-uniform, so descriptors collapse onto a few codewords, *and* the orderless histogram has no spatial layout left to fall back on. ⚠ Revised from "HOG / BoVW": HOG keeps the rigid grid, so even weakened gradients in the right cells still carry signal — it should beat BoVW here. |
| **Gamma** γ 0.4 ↔ 2.5 | HOG | PCA | *New row.* Gamma is a monotone point transform: edge locations are untouched, only contrast changes. HOG's block-wise L2 normalisation cancels contrast scaling almost entirely, and SIFT's descriptor normalisation gives BoVW similar (slightly weaker) protection. PCA operates on raw intensities, so a global remap shifts every projection coefficient and moves the test point away from the training distribution. At γ = 2.5, 3.71 % of pixels crush to zero — irreversible loss that hurts the intensity-based method most. |
| **Small signs** <32 px ROI height | CNN | BoVW | Upsampled small signs carry no genuine high-frequency detail, so dense SIFT has too little local support to form meaningful descriptors. The CNN is trained on the same size distribution (47.6 % of training images are <32 px — this is the modal case, not a tail), so it can learn whatever coarse cues exist. Note this measures scale-dependence, *not* distribution shift: train and test size distributions agree to within 0.1 pp (note 02). |

### Extension stressors (§11 — GTSDB, not part of the MVP)

| Stressor | Most robust | Least robust | Reasoning |
|---|---|---|---|
| **Bbox jitter** ±40 % | BoVW | PCA | Orderless encoding discards layout, so a shifted or rescaled box perturbs the histogram far less than it perturbs an alignment-critical subspace. HOG's rigid grid should also suffer, as content crosses cell boundaries; CNN pooling gives partial tolerance. |
| **Cross-dataset** GTSRB → GTSDB @ jitter 0 | HOG | CNN (e2e) | *New row.* This measures domain shift alone. The CNN has the most capacity to exploit GTSRB-specific statistics — camera, compression, track structure — so it has the most to lose. Contrast-normalised hand-designed features encode structure that transfers; only the linear classifier on top is fitted to GTSRB. |

### Headline prediction

**No single representation wins everywhere, and the ranking inverts between stressors.** The
sharpest falsifiable case: **PCA is predicted best under noise and worst under gamma, while
HOG is predicted worst under noise and best under gamma.** If that crossing appears, the
project's central claim is demonstrated in one figure. If the ranking is stable across all
stressors, the premise is wrong — which is a more interesting result than confirming it.

- [x] **P.1** Record the approved table in `predictions.md` with a timestamp.
      — Recorded **2026-09-13**, before task 4.1, i.e. before any model existed. (Moved
      earlier than the original "before task 8", which would have meant predicting after
      seeing PCA, HOG, BoVW and CNN results.)

---

## 3. Hardware

```
CPU : AMD Ryzen 7 5800H, 8C/16T
RAM : 13 GB
GPU : AMD Vega iGPU (Cezanne) — NO CUDA; ROCm does not support this chip
```

RAM is not a binding constraint at this data scale. The one real constraint is the
absence of a usable GPU.

| Constraint | Decision |
|---|---|
| No usable GPU | CPU-only PyTorch. Small CNN at 48×48 → a few min/epoch on 16 threads. Fine. |
| CPU-only | `LinearSVC` (liblinear), not `SVC(kernel='rbf')` — kernel SVM is O(n²)–O(n³) and will take hours on 39k samples. This is a *compute* limit, not a memory one. |
| CPU-only | `MiniBatchKMeans` for the BoVW vocabulary — full `KMeans` on ~600k descriptors would fit in RAM but is needlessly slow. |
| CPU-only | Randomized SVD for PCA (`svd_solver='randomized'`). |

Pin thread counts explicitly — sklearn and torch each grabbing 16 threads will thrash.

> **Verified at task 0.3:** `OMP_NUM_THREADS` alone is **not sufficient**. BLAS backends
> read it only at their own import time, and an import sorter places `import numpy` above
> a first-party `from gtsrb import config` — so the pinning is silently inert in exactly
> the scripts that matter. `gtsrb.config.set_seeds()` clamps the already-loaded pools at
> runtime via `threadpoolctl`; env vars are kept only for subprocesses. Pinned to 8
> (physical cores), not 16 (SMT).

---

## 4. Stack

```
Python 3.11
opencv-python  >= 4.5     # SIFT (patent expired, now in main pkg), CLAHE, color, filters
scikit-image              # HOG — more configurable than cv2.HOGDescriptor for 48x48 crops
scikit-learn              # PCA, MiniBatchKMeans, LinearSVC, metrics
torch (CPU build)         # CNN
numpy, pandas, matplotlib
```

Install torch CPU explicitly:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`

`cv2.HOGDescriptor` defaults are tuned for 64×128 pedestrian windows and are awkward to
reconfigure for small square crops — use `skimage.feature.hog`.

---

## 5. Data

- **GTSRB** — 43 classes, 39,209 train / 12,630 test crops.
- Kaggle: `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`, or
  `torchvision.datasets.GTSRB(download=True)`.
- Original site `benchmark.ini.rub.de` is live but historically flaky.

### Critical: track-disjoint splitting

Training images are grouped into **tracks of 30 frames of the same physical sign**
(filenames `TTTTT_FFFFF.ppm`). A random train/val split puts near-duplicate frames of the
same sign on both sides and inflates validation accuracy.

**Split by track, never by image.** State this explicitly in the report.

> **Verified at task 0.2 — `TTTTT` is NOT a global track ID.** The numbering restarts at
> `00000` inside every class directory: 39,209 images yield only **75** distinct raw
> prefixes but **1,307** distinct `(class_id, track_id)` pairs (30 frames each, except one
> 29-frame track in class 33). **The track key must be `(class_id, track_id)`.** Using the
> raw prefix collapses 1,307 tracks into 75 groups, wrecks per-class stratification, and
> fails silently. See `docs/report-material/02-dataset-structure.md`.

Annotation CSVs give per-image `Width`, `Height`, `Roi.X1/Y1/X2/Y2`, `ClassId`.

> **Measured at task 2.2 — GTSRB cannot support bbox jitter.** Its images are *already
> cropped* to the sign plus a median ~16.7 % margin; everything beyond that was discarded
> when the dataset was built. Max outward scaling before running off the file is 1.30×
> (median), so **+40 % is impossible for 72 % of images** (28,166 would clip). Producing it
> anyway requires padding with invented pixels — which measures the padding strategy, not
> the representation. Jitter is therefore **not in the MVP**; it moves to full-frame
> datasets in §11. See `docs/report-material/09-jitter-and-datasets.md`.

The ROI columns are still used: `roi_h` is the size measure for the accuracy-vs-size
buckets (task 9.4). Images themselves are used with the framing GTSRB provides.

---

## 6. Tasks

### Day 1 — Infrastructure (~5.5 h)

- [x] **0.1** venv, install stack, verify `cv2.SIFT_create` exists *(task 0 = 1.0 h)*
      — Python 3.11.11 (pyenv) + Poetry, groups `main` / `research` / `dev`; torch from
      the explicit CPU index. Verified by `scripts/verify_env.py` (6/6 checks, incl. a
      real dense `sift.compute()` returning `(9, 128)`). See
      `docs/report-material/01-environment-and-setup.md`.
- [x] **0.2** Download GTSRB, verify counts: 39,209 / 12,630 / 43
      — `scripts/download_data.py` (idempotent; sha256 → `data/CHECKSUMS.txt`). All five
      count checks pass. **Found: `TTTTT` restarts per class dir — 75 raw prefixes vs
      1307 real `(class, track)` pairs.** See `docs/report-material/02-dataset-structure.md`.
- [x] **0.3** `config.py` with one global seed; seed numpy, torch, sklearn
      — `src/gtsrb/config.py`: `SEED=42`, `set_seeds()`, paths, dataset constants, and
      `rng_for(*parts)` for **order-independent** degradation seeding (content-derived, so
      all 5 methods see identical degraded pixels at task 8.1). Threads pinned to 8 at
      runtime via `threadpoolctl` — env vars alone are inert once an import sorter puts
      `import numpy` first. See `docs/report-material/03-reproducibility.md`.
- [x] **1.1** Parse annotation CSVs → dataframe: `path, class_id, track_id, roi_*, w, h` *(task 1 = 2.5 h)*
      — `src/gtsrb/data.py`, `load_annotations("train"|"test")`. `track_id` is the
      composite `CCCCC_TTTTT` (globally unique by construction); `roi_w`/`roi_h` derived
      for task 9.4. Validates row counts, class range, ROI-within-bounds, track count and
      one-class-per-track. **Gotcha: use `data/GT-final_test.csv`, not
      `Final_Test/Images/GT-final_test.test.csv` — the latter has no `ClassId` but the same
      12,630 rows.** Stats in `docs/report-material/02-dataset-structure.md`.
- [x] **1.2** Track-disjoint train/val split (80/20 by track ID, stratified by class)
      — `data.assign_split()` / `train_val_split()`. Splits *each class's own tracks* 80/20,
      so disjointness and stratification hold at once (every track is single-class).
      Achieved: 31,379 / 7,830 images, 1,046 / 261 tracks, **0.200 by both image and
      track**, 0 overlap, 43 classes each side. Per-class val share spans [0.143, 0.250]
      because tracks are atomic (class 0 has only 7). Pure function of (annotations, SEED,
      val_fraction) — no manifest needed. See `docs/report-material/04-splitting-protocol.md`.
- [x] **1.3** Write a test asserting no track ID appears in both splits
      — `tests/test_split.py`, 13 tests, ~1 s (`poetry run pytest`). Covers leakage,
      the correctness premises (single-class tracks, composite key), stratification and
      determinism. **Measured the hazard: a naive random per-image split puts 1,305 of
      1,307 tracks on both sides, and 100% of val images (7,808/7,808) have a sibling
      frame in train.** See `docs/report-material/04-splitting-protocol.md`.
- [x] **1.4** Eval harness → accuracy, macro-F1, per-class F1, confusion matrix
      — `gtsrb.evaluation.evaluate()` returns one `ClassificationResult` with all of them,
      plus `most_confused_pairs()` (9.3), `accuracy_by_group()`/`size_buckets()` (9.4) and
      `config.CLASS_NAMES`. **All metrics pinned to `labels=range(43)`** — inferred labels
      would silently yield a 42×42 matrix on degraded runs where a class is never
      predicted. 10 tests. See `docs/report-material/05-evaluation-metrics.md`.
- [x] **1.5** Timing harness → train wall-clock, inference ms/img (median of ≥3 runs, discard first)
      — `gtsrb.timing`: `time_training()`, `time_inference()`, and `platform_info()` →
      `results/platform.json` (CPU, RAM, OS, pinned threads, **achieved** BLAS/torch thread
      counts, load average, library versions, git commit). Warmup discard justified
      empirically: first call measured **4.5× slower** than the median. 10 tests.
      **Timings are relative costs in one recorded environment, not deployment latency —
      porting to a target platform can reorder the ranking, and the CNN is systematically
      penalised by the absence of a GPU.** See `docs/report-material/06-cost-measurement.md`.
- [x] **1.6** Append all results to one tidy CSV:
      `run_id, method, preproc, degradation, level, metric, value`
      — `gtsrb.results`. **`run_id` added to the planned schema** (`<timestamp>-<commit>`):
      the file is append-only so runs coexist, and a commit alone can't separate two runs
      or survive a dirty tree; it keys into `results/platform_<run_id>.json`, which holds
      the commit. `pivot()` refuses to silently average levels/runs; `append_rows()` refuses
      NaN; `check_complete()` reports missing grid cells for 8.3. 19 tests.
      See `docs/report-material/07-results-table.md`.
- [x] **2.1** Preprocessing: `to_gray`, `to_hsv`, `clahe`, `gaussian_blur`, `resize(48,48)` *(task 2 = 1.5 h)*
      — `gtsrb.preprocessing`, all uint8-in/uint8-out for the 2.3 cache, plus `load_image`
      and `crop_roi`. **`resize` picks interpolation per image** (39.2% of crops shrink,
      60.8% enlarge; a fixed choice injects a ~4 gray-level, size-correlated artifact —
      as large as σ=5 noise — straight into fig. 9.4). **CLAHE runs after resize**, tile
      grid (4,4), V-channel only on colour. 23 tests.
      See `docs/report-material/08-preprocessing.md`.
- [x] **2.2** Three named configs for the ablation: `raw_gray`, `clahe_gray`, `clahe_hsv`
      — `preprocessing.PREPROC_CONFIGS`, each declaring `shape`/`flat_dim` (2304, 2304,
      6912). They form a ladder: each adds exactly one factor to the previous. Plus
      `load_and_preprocess(row, preproc)`. **GTSRB images keep the framing the dataset
      provides** (`roi=False`) — only pixels are processed, the box is untouched. (An earlier
      revision cropped to the tight ROI to align with jitter level 0; reverted when jitter
      moved to §11.) 11 tests. See `docs/report-material/08-preprocessing.md`.
- [x] **2.3** Cached loader — preprocess once, store uint8 `.npy`, reload fast
      — `gtsrb.cache.load_images(frame, preproc)`. Cached by default; **`use_cache=False`
      recomputes from the PPMs and is the reference implementation** the cache is checked
      against bit-for-bit (`--verify`, `np.array_equal`, no tolerance). 6 caches built
      (86/86/258 MB train, 28/28/83 MB test); train/val subsets share one array per split.
      Manifest fingerprints the pipeline + path order, so a stale cache rebuilds instead of
      serving wrong pixels. **83× faster** (1.4345 s → 0.0172 s on 12,630 images).
      Found and fixed: `clahe()` bound its params as default args, so the fingerprint could
      change while pixels did not. 18 tests.
      See `docs/report-material/08-preprocessing.md`.
- [x] **3.1** Degradation: Gaussian noise, σ ∈ {0, 5, 10, 20, 40} *(task 3 = 1.5 h)*
      — `gtsrb.degradations.apply(images, "noise", level, keys)`. **Governing decision:
      degradations apply to the preprocessed 48×48 model input, not the source image** —
      crops span 25–266 px, so degrading at source would make one σ mean different things
      by sign size and confound fig. 9.5 with fig. 9.4. Seeded per image by
      `rng_for("noise", σ, path)` — **keyed on path, not row position**, so a subset matches
      the full batch exactly. Level 0 returns the input unchanged. Found and fixed:
      `astype(uint8)` truncates, darkening every pixel by −0.49 levels at every σ; now
      `np.rint`. 22 tests. See `docs/report-material/10-degradations.md`.
- [x] **3.2** Degradation: motion blur, kernel ∈ {0, 3, 5, 9, 15} px, random angle
      — normalised line kernel (sum 1, so blur doesn't also shift brightness and confound
      3.3); `BORDER_REFLECT_101` **not** zero padding, which would vignette in proportion to
      kernel size; angle ~U[0,180) per image from `rng_for("blur", k, path)` — [0,360) would
      sample each orientation twice. Edge energy 100→80→66→51→**41%**, mean preserved to
      ±0.01. **Kernel built by sub-pixel sampling, not `cv2.line`** — rasterising made the
      blur extent depend on the angle (k=5: 4.00 px at 0° vs 2.83 px at 45°), so a random
      angle changed the strength at a fixed nominal level. Caught by *plotting* the kernels;
      no numeric check saw it. 23 tests. See `docs/report-material/10-degradations.md`.
- [x] **3.3** Degradation: gamma, γ ∈ {0.4, 0.7, 1.0, 1.5, 2.5}
      — 256-entry LUT (exact, memoised), monotone, endpoints fixed. Mean 99.4 → 165.2 /
      125.6 / 99.4 / 72.5 / 47.2, monotone across the grid. **The only non-stochastic
      stressor** (`stochastic=False`; rng ignored). **Its identity is γ=1.0, in the *middle*
      of its range** — the registry now declares `identity` per degradation
      (`identity_for()`), replacing the `levels[0]` assumption that was silently wrong here
      and would have baselined every gamma curve against γ=0.4. At γ=2.5, 3.71% of pixels
      crush to zero (irreversible: round-trip error 1.42 levels). 11 tests.
      See `docs/report-material/10-degradations.md`.
- [x] **3.4** ~~Degradation: bbox jitter~~ — **moved to §11 (extension)**. GTSRB is
      pre-cropped; +40% expansion is impossible for 72% of images. Not an MVP deliverable.
- [x] **3.5** Contact-sheet figure: every degradation × every level on one sample → straight
      into the report. Three rows (noise, blur, gamma) × 5 levels; jitter is not on GTSRB.
      — `scripts/demo/degradation_contact_sheet.py` → `figures/demo/degradation/`. Shows the real 48×48
      model input degraded by `gtsrb.degradations` itself; identity level boxed (gamma's
      sits mid-row). See `docs/report-material/10-degradations.md`.

### Day 2 — PCA, HOG, CNN kickoff (~5.5 h)

- [ ] **4.1** PCA on flattened 48×48 grayscale, `svd_solver='randomized'` *(task 4 = 1.5 h)*
- [ ] **4.2** Sweep n_components ∈ {32, 64, 128, 256} on val, pick one
- [ ] **4.3** `LinearSVC` on PCA features; record cost metrics
- [ ] **4.4** Figure: top-16 eigenvectors as an image grid ("eigensigns") — ties directly to the Eigenfaces lecture
- [ ] **4.5** **Leakage measurement** — train PCA+`LinearSVC` twice, once on the
      track-disjoint split and once on a random per-image split, and report the gap in val
      accuracy. Turns the project's central methodological claim from an argument into a
      measured number. ~15 min; feeds the discussion section. (Q2, approved)
- [ ] **5.1** HOG via `skimage.feature.hog` on 48×48 *(task 5 = 1.5 h)*
- [ ] **5.2** Sweep `pixels_per_cell` ∈ {(6,6),(8,8)}, `orientations` ∈ {9,12}
- [ ] **5.3** `LinearSVC` on HOG features; record cost metrics
- [ ] **5.4** Figure: HOG visualization, one sample per super-category
- [ ] **7.1** Define small CNN: 3 conv blocks (32→64→128), BN, maxpool, dropout, FC head. Target < 1M params. *(task 7 = 3.0 h)*
- [ ] **7.2** Training loop: val each epoch, early stopping, best-checkpoint save
- [ ] **7.3** **Launch baseline training in the background** — it trains while you write BoVW tomorrow

### Day 3 — BoVW, CNN finalize, full grid (~5.5 h)

- [ ] **6.1** **Dense** SIFT: fixed grid (stride ~6 px on 48×48), fixed keypoint size (~12 px), no detector *(task 6 = 2.5 h)*
- [ ] **6.2** Assert every image yields the same nonzero descriptor count
- [ ] **6.3** Subsample ~200k descriptors, `MiniBatchKMeans`, k ∈ {200, 500}
- [ ] **6.4** Encode as k-dim histogram; L2 or power normalization
- [ ] **6.5** `LinearSVC` on BoVW histograms; record cost metrics
- [ ] **7.4** Check background run, tune LR/epochs, finalize end-to-end CNN
- [ ] **7.5** **CNN-as-feature-extractor**: penultimate layer → `LinearSVC`. Puts the CNN on the same footing as the other three. ~20 min, you already have both pieces.
- [x] **P.1** ~~Write `predictions.md` — do this before task 8~~ — **done early**, on
      2026-09-13 before task 4.1. Waiting until Day 3 would have meant predicting after
      seeing PCA, HOG, BoVW and CNN results. See §2 and `predictions.md`.
- [ ] **8.1** Run full evaluation grid (§7) — inference only, no retraining *(task 8 = 1.5 h)*
- [ ] **8.2** Preprocessing ablation: best 2 methods × 3 configs
- [ ] **8.3** Verify results CSV is complete, no NaNs

### Day 4 — Analysis and figures (~5.5 h)

- [ ] **9.1** Table 1: accuracy, macro-F1, train time, inference ms/img, model size MB, feature dim — one row per method (5 rows) *(task 9 = 2.0 h)*
- [ ] **9.2** Figure: confusion matrix, best and worst method
- [ ] **9.3** Table: top-10 most-confused class pairs + commentary (speed limits confuse predictably)
- [ ] **9.4** Figure: accuracy vs. sign size — buckets [0,32), [32,48), [48,72), [72,∞) by ROI height, one line per method
- [ ] **9.5** Figure: robustness curves — 3 panels (noise, blur, gamma), one line per method.
      Plot accuracy **relative to each method's own clean baseline**, so the figure compares
      rate of degradation rather than starting point (and stays commensurable with the §11
      jitter panel, which lives on a different test set).
- [ ] **9.6** **Table: predictions vs. outcomes** — which held, which didn't, and why. This is the core of the discussion section.
- [ ] **9.7** Table: preprocessing ablation
- [ ] **9.8** Write 5 concrete findings as bullets — raw material for the conclusion
- [ ] Buffer

### Day 5 — Report (Serbian)

- [ ] Write up, has same formatting as the proposal, but needs to be created based on this research and the implementation:
  - [ ] consult and brainstorm about sections, before starting to write -- the structure needs to be scaffolded firstly;

---

## 7. Experiment grid

Models are trained once; only inference varies across the grid. Whole grid runs in well
under an hour.

```
methods       = [pca_svm, hog_svm, bovw_svm, cnn_feat_svm, cnn_e2e]   # 5
degradations  = [clean] + [noise, blur, gamma] x 5 levels             # 16
preproc       = [raw_gray, clahe_gray, clahe_hsv]                     # ablation, best 2 methods only
```

Core grid: 5 × 16 = **80** evaluation runs. (Was 105; bbox jitter moved to §11 — no cell in
the remaining grid is compromised.)
Size-stratified results are free — group the existing clean-test predictions by ROI height.

**No jitter enters training or the core grid.** Models are trained on GTSRB with the
framing the dataset provides; jitter is a test-time stressor applied only in §11, on
datasets whose pixels actually exist outside the box.

---

## 8. Deliverables

- [ ] Table 1 — headline comparison (accuracy, macro-F1, cost)
- [ ] Table — **predictions vs. outcomes**
- [ ] Figure — eigensigns
- [ ] Figure — HOG visualization
- [ ] Figure — degradation contact sheet
- [ ] Figure — confusion matrices (best + worst)
- [ ] Figure — accuracy vs. sign size
- [ ] Figure — robustness curves (3 panels: noise, blur, gamma; relative to clean)
- [ ] Table — preprocessing ablation
- [ ] Table — top confused pairs
- [ ] `results.csv` — reproducible from a single script

---

## 9. Cut list

Cut in this order if you fall behind. The comparison survives all three.

1. **BoVW** (task 6, ~2.5 h) — the fiddliest. Leaves PCA, HOG, and both CNN variants.
2. **Preprocessing ablation** (8.2) — report one config, note as a limitation.
3. **Hyperparameter sweeps** (4.2, 5.2, 6.3) — use sensible defaults, note as a limitation.

Never cut: track-disjoint splitting, the cost table, the predictions table, and the
**measurement** of why jitter cannot be done on GTSRB (§11 — the number is the finding, even
if the extension itself is never run).
Those are what make this a study rather than a tutorial.

---

## 10. Known gotchas

- **Track leakage** — split by track, assert it. See §5.
- **Track ID is class-scoped** — `TTTTT` restarts per class directory; the key is
  `(class_id, track_id)`. 75 raw prefixes vs 1,307 real tracks. Confirmed at task 0.2.
- **Detector SIFT returns zero keypoints** on small crops, silently breaking BoVW. Use dense SIFT.
- **`SVC(kernel='rbf')`** on 39k samples will run for hours. `LinearSVC`.
- **Class imbalance** — measured at task 1.1: **10.7×** train (210 for class 0 vs 2,250 for
  class 2), 12.5× test. Report macro-F1, not just accuracy. Consider `class_weight='balanced'`.
- **Two test GT files** — `data/GT-final_test.csv` has `ClassId`;
  `Final_Test/Images/GT-final_test.test.csv` does not. Same row count, so the wrong one
  passes every count check. `config.TEST_GT_CSV` names the right one.
- **BLAS oversubscription** — pin thread counts for sklearn and torch, at *runtime*
  (`threadpoolctl`), not just via env vars. See §3. Confirmed at task 0.3.
- **Degradation seeding** — seeding once at startup makes the degraded test set depend on
  execution order, so methods would be compared on different pixels. Use
  `config.rng_for(degradation, level, index)`, which derives the seed from content.
- **Timing** — discard the first inference call (lazy init / cache warmup); median of ≥3 runs.
  Confirmed at task 1.5: the first call ran **4.5× slower** than the median. Always record
  `platform.json` with the numbers — a duration without its environment is uninterpretable,
  and the timings are relative, not deployment latency.
- **PPM format** — original GTSRB is P6 PPM; `cv2.imread` handles it natively.
- **CNN feature extraction** — put the model in `eval()` and wrap in `torch.no_grad()`, or the penultimate features will carry dropout noise.

---

## 11. Extension — bbox jitter on full-frame datasets

Not part of the MVP. Scheduled after Day 4, once baseline results exist.

**Why it is here and not in §6:** GTSRB is pre-cropped, so outward jitter cannot be
simulated without inventing pixels (§5, measured: +40% impossible for 72% of images). The
correct instrument is a dataset of full road scenes. Full rationale, verification and
statistical caveats: `docs/report-material/09-jitter-and-datasets.md`.

### Stage 1 — MVP (§6)
No jitter anywhere. Establishes the baseline results and Table 1.

### Stage 2 — evaluate the *unmodified* models on GTSDB
- [ ] **11.1** Download `FullIJCNN2013.zip` (1,585 MB); parse `gt.txt`
      (`file;x1;y1;x2;y2;ClassId`) into the standard annotations schema.
      *Verified over HTTP range requests (237 KB): 1,213 signs / 741 frames, 1360×800,
      ClassId 0–42 identical to GTSRB, all 43 classes present.*
- [ ] **11.2** Jitter: perturb the box ±{0,10,20,30,40}% in scale and position, crop from
      the **full frame** — real background, no padding, no clipping.
      **Level 0 must replicate GTSRB's framing** (box + ~17% margin), or a framing mismatch
      gets measured and reported as domain shift.
- [ ] **11.3** Run all 5 methods across the jitter levels. No retraining — inference only;
      the task-1 harness handles it unchanged.
- [ ] **11.4** Report **accuracy only, not macro-F1** — 14 of 43 GTSDB classes have <10
      instances (min 2). n=1,213 gives ±1.7 pp CI: fine for curve shape, not for per-class
      claims.
- [ ] **11.5** Report GTSRB-clean vs GTSDB-jitter-0 separately: that gap is a **cross-dataset
      generalisation** measurement, a bonus the original plan would not have produced.

### Stage 3 — *conditional* on stage 2 showing a robustness gap
- [ ] **11.6** Only if stage 2 shows the representations are *not* robust: propose GTSDB as a
      training supplement and build a jitter-aware model. Note the asymmetry — GTSRB can only
      supply *inward* jitter for training, while stage 2 tests both directions.
      "Already robust, so no augmentation was warranted" is a legitimate, cheaper outcome.

### Stage 4 — future work
- [ ] **11.7** A further uncropped dataset for cross-country generalisation. BelgiumTS is the
      only realistic candidate (62 classes vs 43 — needs an explicit mapping, covers a
      subset). Scoped as future work; it is the one item that could expand without bound.

---

## 12. Optional — composite degradation scenarios

**Not part of the MVP.** Only after the core grid (§7) is complete and working. First thing
to cut if time is short; the one-factor-at-a-time grid stands alone without it.

**Why the core grid is one-factor-at-a-time.** The research question is the interaction
between *representation* and *degradation type*. A combined condition cannot be interpreted
without the marginal effects first — if `blur+noise` hurts BoVW badly, nothing tells you
whether it was the blur, the noise, or the pair. A full factorial is also prohibitive:
5³ = 125 combinations × 5 methods = 625 runs against 80.

**Why a few named scenarios are still worth having.** Real conditions co-occur, and a
deployment chooses a representation for a *condition*, not for a single stressor — which is
the framing of proposal §1. Three named scenarios cost **15 runs** (inference only,
seconds) and need no schema change (`degradation="night"`, `level=0`).

| Scenario | Composition | Represents |
|---|---|---|
| `night` | blur 5 → noise 20 → γ 2.5 | dark, long exposure, sensor noise |
| `dusk_motion` | blur 9 → noise 10 → γ 1.5 | moving vehicle, fading light |
| `glare` | noise 5 → γ 0.4 | low sun / overexposure |

- [ ] **12.1** Implement composite scenarios with a **fixed, documented application order**:
      **blur (optical) → noise (sensor) → gamma (response curve)**, following the physical
      imaging chain. Order is not cosmetic: motion blur is a low-pass filter, so applying it
      *after* noise averages the noise away and materially lowers the effective σ. Any order
      is defensible only if it is stated.
- [ ] **12.2** Evaluate all 5 methods on the 3 scenarios; report alongside the OFAT grid,
      never in place of it.
