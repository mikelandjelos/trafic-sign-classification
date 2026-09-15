# GTSRB — Poređenje reprezentacija za prepoznavanje saobraćajnih znakova

Task tracker for the Computer Vision (Računarski vid) project.
Working doc in English; the final report is in Serbian.

---

## 1. Scope

**What this project is:** a controlled comparison of five ways to represent a cropped
traffic sign image, evaluated on GTSRB.

**Input:** one RGB crop containing exactly one traffic sign (15×15 to 250×250 px).
**Output:** one of 43 class labels.

**Where degradations enter.** After preprocessing, on the model input — so what is injected
is the degradation preprocessing *failed to remove*, the residual the representation must
cope with. Capture-time simulation is a separate question, scoped to §12. Pipeline diagram:
`docs/diagrams/pipeline.puml` → `figures/diagrams/pipeline.png`.

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
      `results/platform_<run_id>.json` (CPU, RAM, OS, pinned threads, **achieved** BLAS/torch thread
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
- [x] **1.7** **Demo** for the splitting protocol — `scripts/demo/split_mechanics.py` →
      `figures/demo/split/`. The 1,305-of-1,307 leakage count is conclusive but abstract;
      these make it visible. **New measurement: "near-duplicate" quantified** — median pixel
      correlation is **0.61** between two frames of the same track vs **0.17** between two
      images from *different tracks of the same class* (the hard comparison). ~3.5×, with
      barely overlapping distributions, and independent of any model — this is the number to
      quote in the report. The frame strip is ordered by `frame_id`, so it reads as a car
      approaching: the frames are *near*-duplicates, not identical, which is why any
      deduplication check would miss the leak. The naive per-image split is reimplemented in
      the demo rather than imported, since `gtsrb.data` deliberately cannot do it.
      See `docs/report-material/04-splitting-protocol.md`.
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
> **Why these three stressors.** Each attacks a *different* structural property, which is
> what makes the representation × stressor interaction measurable: noise **adds**
> high-frequency energy, blur **removes** it (opposite operations on the same axis), and
> gamma is **orthogonal** — geometry untouched, intensity remapped. Three distinct failure
> mechanisms, matching the axes the representations differ on. Occlusion, JPEG artifacts,
> rotation and weather were excluded — reasons in `docs/report-material/10-degradations.md`.

- [x] **3.1** Degradation: Gaussian noise, σ ∈ {0, 5, 10, 20, 40} *(task 3 = 1.5 h)*
      — `gtsrb.degradations.apply(images, "noise", level, keys)`. **Governing decision:
      degradations apply to the preprocessed 48×48 model input.** Injected *after*
      preprocessing, the perturbation is by construction **the degradation preprocessing did
      not remove** — the residual that reaches the representation. That is the quantity the
      study compares methods on, and it needs no claim about cameras. It also keeps one
      level meaning one condition: degrading at source would make σ=20 span a measured 2.4×
      range of effective strengths by sign size, entangling fig. 9.5 with fig. 9.4. Seeded per image by
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

- [x] **4.1** PCA on flattened 48×48 grayscale, `svd_solver='randomized'` *(task 4 = 1.5 h)*
      — `gtsrb.representations.pca.PCARepresentation`, behind the shared `Representation`
      interface tasks 5–7 also implement. Fitted on the **training split only** and on
      **clean images only** (`fit_on_train()` enforces both). **`whiten=False`, deliberately:**
      whitening rescales each retained component to unit variance, which amplifies exactly
      the low-variance directions where noise concentrates — it would suppress the very
      mechanism the locked noise prediction rests on. Kept available as an ablation, and
      reported as a limitation. Measured: 80 % of variance needs **8** components on
      `raw_gray`, **25** on `clahe_gray`, **106** on `clahe_hsv` — *CLAHE makes the data
      harder to compress*, because the direction it normalises away (PC1 = brightness,
      **52.3 %** of variance, r = **+0.9975** with mean intensity) was absorbing half of it.
      Train/val reconstruction error agree to 0.1 gray levels, so the basis generalises
      across held-out tracks. Found and fixed: `fit` leaves `components_` F-contiguous and
      joblib reloads it C-contiguous — identical values, different BLAS kernel, projections
      differing by 1e-6, i.e. *the model changed when reloaded*. 31 tests.
      See `docs/report-material/11-pca.md`.
- [x] **4.1b** Demo figures + risk check for the tasks that build on PCA.
      — `scripts/demo/pca_mechanics.py` → `figures/demo/pca/` (4 figures: reconstruction
      ladder, spectrum, PC1-is-brightness, degraded-through-clean-basis). **At k=2 every
      sign — a Yield triangle included — reconstructs toward a round speed-limit disc**: the
      leading components encode the dataset's modal shape, which makes "holistic, no notion
      of a part" visible. PCA also spends variance budget on *background* (GTSRB's ~17 %
      margin), one for the limitations. **Risk checked and cleared:** unwhitened PCA features
      span 29× in σ (865× in variance), which should distort `LinearSVC`'s single L2
      penalty — measured, it does not: 0.8013 unscaled vs 0.8004 standardised, both
      converged, so no scaling step is needed. **Two risks left open**, Q4 (is one `C` across
      five methods a confound? PCA is the only representation not internally normalised) and
      Q5 (preproc/representation pairing is unenforced and a `raw_gray`/`clahe_gray` mix-up
      is silent). See `docs/report-material/11-pca.md` §5–6.
- [x] **4.2** Sweep n_components ∈ {32, 64, 128, 256} on val, pick one
      — `scripts/sweep_pca.py` + `gtsrb.tuning` (shared by 5.2/6.3). **Selected k = 256,
      C = 0.01, `class_weight="balanced"`** — val macro-F1 **0.7977**, accuracy 0.8442.
      **Swept k × C × class_weight jointly, because they interact:** best C falls
      monotonically as k grows (10 → 0.1 → 0.1 → 0.01), and at a fixed C = 1 the same k=256
      scores 0.7715 — fixing C first would have understated the best config by **2.6 pp**,
      comparable to the between-method gaps this study measures. **Selected on macro-F1**,
      not accuracy (10.7× imbalance). **Caveat: 256 is the top of the specified grid and the
      curve is still rising** — best of the four offered, not a located optimum; widening the
      grid *because* the edge won would void the protocol, so it is reported as a limitation.
      **Variance and accuracy disagree**: variance spans 0.824→0.972 while macro-F1 spans
      0.49→0.80, so the "keep 95 % of variance" heuristic would have picked ~155 components
      and lost accuracy. `balanced` won by only 0.55 pp and loses at k=128 — recorded as
      within noise. All 32 grid points converged. 12 tests.
      **All three preprocessing configs swept independently** (96 grid points): `clahe_gray`
      wins (0.7977) over `raw_gray` (0.7713) and `clahe_hsv` (0.7674), so `DEFAULT_PREPROC`
      is now *measured* rather than the bare unjustified assignment it had been. Each config
      selects **different** hyperparameters (C = 0.01 / 0.10 / 0.01; balanced only for
      `clahe_gray`), and borrowing `clahe_gray`'s costs `raw_gray` **1.03 pp** and
      `clahe_hsv` **0.87 pp** — against a preprocessing effect of 2.64 pp, i.e. **~39 % of
      the signal 8.2 measures**, biased toward the config the hyperparameters came from.
      **8.2 must therefore tune per (method, preproc) cell.** Also: variance ranks the three
      configs *backwards* (`raw_gray` has the best variance and the worse macro-F1), a third
      independent instance of variance ≠ discriminability.
      See `docs/report-material/11-pca.md` §7.
> **Classifier protocol (Q4, decided at 4.1).** `C` is **tuned per method** on validation,
> and the chosen value is recorded per method in `results.csv` and reported alongside Table 1.
> "Fixed classifier" means the same estimator and the same training protocol — *not* the same
> nuisance hyperparameter. PCA is the only representation whose features are not internally
> normalised (HOG block-normalises, BoVW L2-normalises, the CNN has batch norm; PCA's feature
> σ spans 29×), so a frozen `C` would hand each representation a dial calibrated for another's
> feature scale. Applies to **4.3, 5.3, 6.5, 7.5** and the grid at **8.1**.

- [x] **4.3** `LinearSVC` on PCA features, `C` tuned on val; record cost metrics and `C`
      — `scripts/train_pca.py` → **the first rows in `results/results.csv`** (58 rows, run
      `20260913T203406-7374a4f`) + `results/models/pca_svm_clahe_gray.joblib` (2.35 MB).
      Val accuracy **0.8442**, macro-F1 **0.7977** — reproduces the 4.2 selected cell exactly,
      a free end-to-end check since the script reads hyperparameters from the sweep CSV rather
      than from literals. Train 28.7 s (PCA fit **included** — timing only the SVM would
      flatter PCA against HOG, which has no fitted stage). **Trained on the train split only,
      not train+val**, because the CNN needs val for early stopping and giving the other
      methods 25 % more data would confound the comparison — applies to all five methods.
      **Test set untouched**: rows are `val_*`; 8.1 writes the plain test metrics.
      **Found: batched throughput is 49× the single-image latency** (0.0066 vs 0.3254 ms/img),
      so both are recorded — the batched figure is right for grid cost and wrong for any
      real-time claim. Early 9.3 signal: the two worst confusions are *not* speed limits but
      the near-identical **end-of-restriction** signs (60.0 % and 58.3 %).
      See `docs/report-material/11-pca.md` §8.
- [x] **4.4** Figure: top-16 eigenvectors as an image grid ("eigensigns") — ties directly to the Eigenfaces lecture
      — `scripts/figure_eigensigns.py` → `figures/report/pca_eigensigns.png`, rendered from
      the **fitted** basis (not a refit) so it matches the reported results. Signed components
      on a diverging map centred at zero (grayscale would turn a bipolar pattern into a
      brightness gradient); each scaled to its own amplitude with the variance share printed
      above, since PC16 has ~1/50 of PC1's. Mean panel included — it is subtracted at every
      projection, and it renders as a blurred "30" disc, the dataset's modal class.
      **Interpretable finding: PC1 = brightness (39.8 %, a pure nuisance), PC2 = round vs
      triangular (12.7 %, the coarsest class distinction), and the discriminative detail —
      which digit, which pictogram — lives below 1 % per component.** PCA spends capacity in
      order of variance, not usefulness. See `docs/report-material/11-pca.md` §9.
- [x] **4.5** **Leakage measurement** — train PCA+`LinearSVC` twice, once on the
      track-disjoint split and once on a random per-image split, and report the gap in val
      accuracy. Turns the project's central methodological claim from an argument into a
      measured number. (Q2, **closed**)
      — `scripts/measure_leakage.py` → `figures/report/split_leakage_measured.png` +
      `leakage_*` rows in `results.csv`. Identical model and hyperparameters; only the split
      rule differs; random split averaged over 3 seeds (σ ≈ 0.3 pp).
      **A random per-image split inflates val accuracy by +5.08 pp (0.8442 → 0.8950) and
      macro-F1 by +9.58 pp (0.7977 → 0.8936).** Cause measured on the same assignments:
      1,305–1,306 of 1,307 tracks land on both sides and **100 % of val images keep a sibling
      frame in train**. **Second finding: macro-F1 inflates ~2× as much as accuracy** —
      leakage flatters the *rare* classes most (a 7-track class has no diversity to
      generalise across, so its held-out frames become near-lookup), and macro-F1 weights
      those equally. So the metric chosen *because* of imbalance is the one leakage corrupts
      worst. See `docs/report-material/04-splitting-protocol.md`.
- [x] **5.1** HOG via `skimage.feature.hog` on 48×48 *(task 5 = 1.5 h)*
      — `gtsrb.representations.hog.HOGRepresentation`, behind the same `Representation`
      interface. **`fit` is a no-op — HOG learns nothing**, pinned by a test (fitting on two
      disjoint halves must give byte-identical descriptors): its training cost is only the
      SVM fit and its "model" is only the SVM coefficients, both of which Table 1 must
      distinguish from PCA's. `block_norm="L2-Hys"` is the gamma mechanism and is
      test-pinned; `cells_per_block=(2,2)` not (3,3), since a 6×6 cell grid cannot support
      3×3 blocks; `transform_sqrt=False` — named explicitly because it *is* a gamma transform
      (γ=0.5) and would interact with task 3.3, so it is an ablation candidate rather than a
      quiet default. Dims 900 / 1200 / 1764 / 2352 for the 5.2 grid, all test-pinned;
      0.53–0.83 ms/img. **Verified: a linear contrast change is cancelled exactly (shift
      0.000); gamma is only partly cancelled (0.358).** 28 tests.
      See `docs/report-material/12-hog.md`.
- [x] **5.2** Sweep `pixels_per_cell` ∈ {(6,6),(8,8)}, `orientations` ∈ {9,12}
      — `scripts/sweep_hog.py`, 96 points (× 3 preproc × `C` × `class_weight`), ~2 h 20 m,
      **all converged**. **Selected `raw_gray`, 6px/9o, C=0.01, balanced → macro-F1 0.9175,
      accuracy 0.9298.**
      **Finding: HOG selects `raw_gray` where PCA selected `clahe_gray`** — the methods
      genuinely disagree about preprocessing, which is exactly what the 4.2 per-method
      decision was for. Mechanism: HOG block-normalises internally, so CLAHE is redundant for
      it; PCA has no internal normalisation.
      **Finding: cell size dominates orientation count** — 8px→6px is worth ~3.5 pp, 9→12
      orientations is worth *nothing* (slightly negative). Spatial resolution is the binding
      constraint on 48×48, not angular. The selected config is **not** the highest-dimensional.
      `clahe_hsv` costs HOG −9.7 pp; hypothesis recorded (hue wraps at red, so gradients across
      the discontinuity are spurious and win skimage's max-magnitude channel selection) —
      **unverified, worth checking before 9.7**.
      Third confirmation of the dimensionality/`C` coupling (C=0.01 at 1764 dims, 0.10 at 900).
      See `docs/report-material/12-hog.md` §4.
- [x] **5.3** `LinearSVC` on HOG features, `C` tuned on val; record cost metrics and `C`
      — `scripts/train_hog.py`. Val accuracy **0.9298**, macro-F1 **0.9175** (reproduces the
      5.2 cell exactly). Model **0.58 MB** — *entirely* SVM coefficients, since HOG learns
      nothing, against PCA's 2.35 MB that is 96 % basis. Batching ratio **1.3×**, completing
      the spread: PCA 49×, HOG 1.3×, CNN 0.87×.
      **Finding: HOG fails on completely different classes than PCA and the CNN.** Four of its
      five top confusions are *speed limits* (80→50 at 16.9 %), where PCA and the CNN both
      struggle with the *end-of-restriction* signs that barely trouble HOG. Mechanism: the
      strikethrough is a strong oriented gradient HOG encodes and a holistic subspace cannot;
      speed-limit digits differ in fine stroke detail *within* a cell, which HOG pools away.
      **The representation × difficulty interaction, visible on clean data.**
      See `docs/report-material/12-hog.md` §5.2.
- [x] **5.4** Figure: HOG visualization, one sample per super-category
      — `scripts/figure_hog.py` → `figures/report/hog_visualization.png`. Four *shapes*, not
      four arbitrary classes; config read from the sweep CSV. States explicitly that it renders
      the **un-normalised** histograms (the classifier sees the block-normalised vector) and
      that brightness is rescaled for display only. **Corroborates 5.3 visually**: on
      *Speed limit (80)* the circular border gives strong strokes while the digits give only
      short weak ones — the detail separating 80 from 50 is pooled away, which is precisely
      HOG's worst confusion. See `docs/report-material/12-hog.md` §6.
- [x] **5.5** **Demo** (`scripts/demo/hog_mechanics.py`) — three figures into
      `figures/demo/hog/`, all built from the real modules, and pointed at *Speed limit (80)* —
      HOG's **worst** class (5.3) rather than a flattering one.
      **Block normalisation quantified:** linear contrast ×0.5 gives mean |Δdescriptor| =
      **0.0000** (traces exactly superimposed); gamma 2.5 gives **0.0339**. Exact protection
      against contrast, partial against gamma — the 9.5 panel's mechanism on one image.
      **The failure test:** per-cell gradient energy relative to clean as σ rises. Corner cells
      (low-contrast sky, no real gradient) reach **6×+** their clean energy while the sign
      centre stays near 1× — noise swamps exactly the cells that had nothing to report, so
      their orientation votes go random. Mean |Δdescriptor| at σ=40 is 0.1261, ~4× gamma's.
      **Cell size** shown side by side: at 8 px the digits fall inside single cells.
      See `docs/report-material/12-hog.md` §7. **Task 5 complete.**
- [x] **7.1** Define small CNN: 3 conv blocks (32→64→128), BN, maxpool, dropout, FC head. Target < 1M params. *(task 7 = 3.0 h)*
      — `gtsrb.representations.cnn.SmallCNN`. **The specified head misses the budget at
      1.48 M** — the conv trunk is only 287 k, so the first linear layer was 80 % of the
      model. `Flatten → 4608 → 128 → 43` gives **0.88 M**. **Global average pooling (0.29 M)
      was rejected for a structural reason, not a performance one**: GAP makes the
      representation *orderless*, which is BoVW's defining property — two of the four methods
      would then sit on the same point of the layout axis and that axis would stop being
      measurable (including the §11 jitter extension). Recorded as a limitation: the head is
      smaller than a free choice would make it, for comparability.
      `embed()` **forces `eval()`/`no_grad()` and restores the prior mode**, enforcing the
      §10 gotcha rather than documenting it — dropout-contaminated features do not error,
      they just make `cnn_feat_svm` quietly worse. 17 tests.
      Measured cost: **116 s/epoch**, 1.2 ms/img inference (~3.6× PCA, batched).
      Architecture diagram: `figures/report/cnn_architecture.png` (shapes read from the
      model via forward hooks, so it cannot drift). Shows the branch that makes one
      network produce two of the five rows.
      See `docs/report-material/13-cnn.md`.
- [x] **7.2** Training loop: val each epoch, early stopping, best-checkpoint save
      — `gtsrb.training.train()`. **Early stopping on validation macro-F1**, the same
      criterion every other method selects on — stopping on accuracy while the report leads
      with macro-F1 would optimise for a metric the write-up does not use, and under 10.7×
      imbalance the two disagree. **The returned model holds the *best* epoch's weights, not
      the last**: `patience` epochs of no improvement are by construction epochs where the
      model got no better and may have got worse. Pinned by a test that trains twice with
      identical seeds — once stopping at the best epoch, once running past it — and asserts
      the weights come back bit-identical. Batch order seeded off `config.SEED`. 16 tests.
      See `docs/report-material/13-cnn.md`.
- [ ] **7.3** **Launch baseline training in the background** — it trains while you write BoVW tomorrow
      **Run 1 of 3 done (`clahe_gray`): val accuracy 0.9902, macro-F1 0.9872**, best epoch 18
      of 24 (early-stopped), 55.8 min at 6 threads. Well above the ~95 % guideline.
      The best-epoch rule mattered on the very first run: the final epoch was 0.0052 macro-F1
      *worse* than epoch 18, so returning the last would have reported a model past its peak.
      **Finding: the CNN gains nothing from batching** (1.703 ms batched vs 1.477 single) where
      PCA gains 49× — so at batch size 1, the realistic camera setting, PCA's cost advantage
      shrinks from ~260× to ~4.5×. Runs 2-3 (`raw_gray`, `clahe_hsv`) pending.
      **Three runs, one per preprocessing config** (~1 h each). Unlike the other methods, the
      CNN's representation *is* its weights, so each config is a full retrain — but 8.2
      ("best 2 methods × 3 configs") needs them regardless, and exempting the CNN would
      reintroduce the hyperparameter-transfer confound measured at 4.2. `clahe_hsv` is a live
      contender here in a way it was not for PCA: signs are colour-coded and a CNN can exploit
      that. Guideline, not a gate: `cnn_e2e` below ~95 % val macro-F1 indicates a training
      problem, not a finding — see `13-cnn.md` §5.

### Day 3 — BoVW, CNN finalize, full grid (~5.5 h)

- [x] **6.1** **Dense** SIFT: fixed grid (stride ~6 px on 48×48), fixed keypoint size (~12 px), no detector *(task 6 = 2.5 h)*
      — `gtsrb.representations.bovw.DenseSIFT`. step=6, size=12 → **8×8 = 64 keypoints**,
      neighbourhoods overlapping by half; 0.57 ms/img (~18 s per training pass). Descriptors
      arrive from OpenCV **already L2-normalised to ≈512**, so they sit on a sphere — the
      right footing for k-means, and no rescaling is applied. **`upright=True` is a decision,
      not a default**: signs are upright so absolute gradient orientation is signal, and
      SIFT's per-patch orientation estimate is unstable on low-contrast patches. Found and
      fixed: `sample_descriptors` returned 480 for a request of 500 (`round` undershoots),
      which would have silently shrunk the 6.3 vocabulary sample.
      See `docs/report-material/14-bovw.md`.
- [x] **6.2** Assert every image yields the same nonzero descriptor count
      — **0 keypoints pruned** at every (step, size) tried, and the guard raises loudly if
      OpenCV ever prunes one (a dropped keypoint makes that image's histogram incomparable,
      silently). A test also pins *why* a detector is unusable here: it returns a **variable**
      count per image, so histograms would not be comparable even where it fires.
      **Zero descriptors: 0 of 128,000 clean, and 0 of 38,400 under blur 15 / noise 40 /
      gamma 2.5** — checked rather than assumed, since heavy blur flattens local structure.
      But a *perfectly* uniform patch does produce a zero vector, so the code relies on
      "never happens on GTSRB", not on "cannot happen" — both are asserted.
- [x] **6.3** Subsample ~200k descriptors, `MiniBatchKMeans`, k ∈ {200, 500}
      — `BoVWRepresentation.fit()`. Seeded vocabulary (two fits with the same seed agree,
      asserted). `sample_descriptors` streams instead of materialising all ~2M descriptors
      (the full array would be ~1 GB).
- [x] **6.4** Encode as k-dim histogram; L2 or power normalization
      — **`power_l2` by default, and the reason is narrower than it looks**: every image
      contributes exactly 64 descriptors, so raw histograms *already* sum to a constant and
      `l1`/`l2` are pure rescales that cannot change the ratio between two bins. Only the
      square root does — Perronnin's burstiness correction. Measured on a 43-class stratified
      subsample: power_l2 0.2769 > l2 0.2695 > none 0.2426 macro-F1. Fixed with a stated
      reason rather than swept, following the precedent of HOG's `block_norm`.
      Encoding is chunked for memory; `chunk=7` and `chunk=1000` give byte-identical results.
      **Trap recorded:** the first probe reported chance-level macro-F1 (0.023) with plausible
      accuracy (0.446) — it had subsampled with `iloc[:3000]` on class-ordered annotations and
      contained 3 of 43 classes. A large accuracy/macro-F1 gap is a label-space signature.
- [ ] **6.5** `LinearSVC` on BoVW histograms, `C` tuned on val; record cost metrics and `C`
      **OPEN: the plan's sampling density is too coarse.** Measured on the full split, k=500:
      size12/step6 (the value specified in 6.1) gives macro-F1 **0.3402**; size8/step4 gives
      0.4583; size6/step3 gives **0.5248**. +18.5 pp from density alone — more than `k` or `C`
      move anything. Not a bug (vocabulary fully used, ~37 distinct words/img, larger `C`
      monotonically worse). **Decision needed: add `(step, keypoint_size)` to the sweep?**
      Separately, BoVW trailing the others is *expected* — spatial pyramid matching exists to
      fix orderless weakness on aligned objects, and we deliberately omit it because it would
      collapse BoVW onto HOG's position on the layout axis. See `14-bovw.md` §6.3.
- [ ] **6.6** **Demo** (`scripts/demo/bovw_mechanics.py`) — the dense keypoint grid drawn on
      a sign, codeword assignment as a colour map (which patches share a word), and the
      histogram before/after normalisation. **This is the demo most likely to catch a real
      bug**: 6.2 asserts a constant descriptor count numerically, but a silently degenerate
      vocabulary — most patches collapsing onto one codeword under blur — passes that
      assertion and is obvious on sight.
- [ ] **7.4** Check background run, tune LR/epochs, finalize end-to-end CNN
- [ ] **7.5** **CNN-as-feature-extractor**: penultimate layer → `LinearSVC`. Puts the CNN on the same footing as the other three. ~20 min, you already have both pieces.
      **No retraining — it reuses 7.3's network.** There is exactly one training run in the
      whole project. Caveat to report alongside the result: those features were optimised for
      a *softmax head*, so if `cnn_feat_svm` trails `cnn_e2e`, part of the gap is objective
      mismatch rather than the representation. PCA, HOG and BoVW were never optimised for any
      classifier, so this asymmetry is the CNN's alone.
      See `docs/report-material/13-cnn.md` §1.1.
- [ ] **7.6** **Demo** (`scripts/demo/cnn_mechanics.py`) — training curves, first-layer
      filters, and feature-map activations for one sign per super-category. Include the
      `eval()`/`no_grad()` check as a *visual*: penultimate features extracted twice must be
      identical, and are not if dropout is still active (§10) — a bug that otherwise only
      shows up as mysteriously poor `cnn_feat_svm` accuracy.
> **P.1 (see §2)** — ~~write `predictions.md` before task 8~~ was **done early**, on
>   2026-09-13 before task 4.1. Waiting until Day 3 would have meant predicting after
>   seeing PCA, HOG, BoVW and CNN results. Recorded once, in §2.
- [ ] **8.1** Run full evaluation grid (§7) — inference only, no retraining *(task 8 = 1.5 h)*
      **The runner pairs each representation with the cache it was fitted from** (Q5, decided
      at 4.1): a representation is never obtainable without its images, with a test asserting
      it. A `raw_gray`/`clahe_gray` mix-up is otherwise silent — both are 2304 dims, so no
      shape check fires and the results look plausible. This is the one place it can happen,
      because 8.2 loops over all three configs.
- [ ] **8.2** Preprocessing ablation: best 2 methods × 3 configs.
      **Largely answered already** (note 08 addendum): all three completed methods swept all
      three configs, and **the ranking is stable — CNN > HOG > PCA under both grayscale
      configs**. The largest preprocessing effect (+3.0 pp, PCA) is smaller than the smallest
      between-method gap (8.6 pp). CLAHE helps **only** the method with no internal contrast
      normalisation (PCA +2.6 pp), is neutral for the CNN (BatchNorm, +0.2) and *hurts* HOG
      (L2-Hys already does it, −0.4). **Tune per (method,
      preproc) cell**, not per method — measured at 4.2: reusing one config's hyperparameters
      on another costs ~1 pp, about 39 % of the preprocessing effect itself, and biases toward
      whichever config they were selected on. **Its job is a
      ranking-stability check, not an accuracy sweep**: show the *ranking* of methods within
      each stressor is unchanged across `raw_gray` / `clahe_gray` / `clahe_hsv`, so the
      conclusion is not an artifact of one preprocessing choice. Claim ordinally — with one
      seed and no repeats we cannot support "no significant difference". A ranking that does
      flip is a finding and gets reported.
- [ ] **8.3** Verify results CSV is complete, no NaNs

### Day 4 — Analysis and figures (~5.5 h)

- [ ] **9.1** Table 1: accuracy, macro-F1, train time, inference ms/img, model size MB, feature dim — one row per method (5 rows) *(task 9 = 2.0 h)*
      **Two caveats measured at 4.3, both must appear with the table.** (a) Report the
      **single-image** inference figure, or state which is which: batched throughput is 49×
      the per-frame latency for PCA, and the ratio is method-specific. (b) `model_size_mb`
      is not like-for-like — 96 % of PCA's 2.35 MB is the *basis*, while HOG has no fitted
      stage at all (~0.1 MB of SVM coefficients). Say what dominates each row, or the table
      reads as "PCA is 20× bigger than HOG" when the real difference is a learned dictionary
      vs none. See `docs/report-material/06-cost-measurement.md`.
- [ ] **9.2** Figure: confusion matrix, best and worst method
- [ ] **9.3** Table: top-10 most-confused class pairs + commentary (speed limits confuse predictably)
      **Report confusions for more than the best and worst method.** PCA and the CNN — which
      share nothing structurally — fail on the *same* classes, with the same top confusion
      (End of no passing → End of all speed and passing limits: 60.0 % for PCA, 15.0 % for the
      CNN) and four of five worst classes in common (41, 32, 40, 29). Same pairs, attenuated
      ~4×. That establishes the difficulty is **intrinsic to those classes**, which a single
      method's confusion table cannot show. Note also the commentary premise needs revising:
      the hardest pairs are *not* the speed limits but the near-identical end-of-restriction
      signs. See `docs/report-material/13-cnn.md` §7.3.
- [ ] **9.4** Figure: accuracy vs. sign size — buckets [0,32), [32,48), [48,72), [72,∞) by ROI height, one line per method
- [ ] **9.5** Figure: robustness curves — 3 panels (noise, blur, gamma), one line per method.
      Plot accuracy **relative to each method's own clean baseline**, so the figure compares
      rate of degradation rather than starting point (and stays commensurable with the §11
      jitter panel, which lives on a different test set).
- [ ] **9.6** **Table: predictions vs. outcomes** — which held, which didn't, and why. This is the core of the discussion section.
- [ ] **9.7** Table: preprocessing ablation — **must carry the hue-wrap finding.**
      `clahe_hsv` stores OpenCV HSV where hue wraps at 0/179, and red — the commonest sign
      colour — sits on the wrap: *Stop* has 56.5 % of saturated pixels at hue<10 and 4.4 % at
      hue>169. **3.98 % of adjacent-pixel hue gradients exceed 90**, and mean hue gradients
      match intensity gradients, so they win any max-magnitude selection. Costs scale with
      reliance on derivatives: **HOG −9.7 pp** (selects max-magnitude gradient across
      channels), **CNN −2.3 pp and peaks at epoch 5 vs 18**, **PCA −3.0 pp** (never
      differentiates). **Do not report this as "colour is uninformative"** — it is a defective
      *encoding* of colour. Fix is known (hue as cos/sin, or CIELab) and deliberately not
      applied: all three sweeps used the current config. See `08-preprocessing.md` addendum.
      Reported as **ranking stability per
      stressor** across the three configs (see 8.2), not as a bare accuracy comparison.
- [ ] **9.8** Write 5 concrete findings as bullets — raw material for the conclusion
- [ ] Buffer

### Day 5 — Report (Serbian)

- [ ] Write up, has same formatting as the proposal, but needs to be created based on this research and the implementation:

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
- [x] Figure — eigensigns
- [ ] Figure — HOG visualization
- [ ] Figure — degradation contact sheet
- [ ] Figure — confusion matrices (best + worst)
- [ ] Figure — accuracy vs. sign size
- [ ] Figure — robustness curves (3 panels: noise, blur, gamma; relative to clean)
- [ ] Table — preprocessing ablation
- [ ] Table — top confused pairs
- [ ] `results.csv` — reproducible from a single script

> **Report figures are committed.** `figures/` is gitignored except `figures/report/`, which
> **is** in the repository — the report is written on Day 5, possibly elsewhere, and "re-run
> the scripts" is not a plan when they need a 1 GB gitignored dataset. Populate it with
> `poetry run python scripts/collect_report_figures.py`; the MANIFEST in that script is the
> single explicit list of what belongs in the report, and `--check` prints what is still
> pending. Figures named "Figure:" above are `deliverable=True`; ones judged important enough
> to argue for are `deliverable=False` with the argument recorded inline.

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
  class 2), 12.5× test. Report macro-F1, not just accuracy. **`class_weight` is swept,
  not assumed** — it is on the `gtsrb.tuning` grid alongside `C` for every method
  (decided at 4.2). For PCA `balanced` won, but by only 0.55 pp and it *loses* at
  k=128, so no class-weighting effect should be claimed from it.
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
- **A long job redirected to a log looks stalled** — Python block-buffers stdout when it is
  not a terminal, so a sweep writing one line per fit can go 25 minutes without touching the
  file while sitting at 98 % CPU. Check `ps -o stat=,%cpu=` before concluding anything is
  wrong, and pass `flush=True` on progress prints. Hit at task 5.2.
- **The CPU power profile changes timings by ~20 % and is not recorded anywhere** — switching
  from "power saver" to "balanced" cut CNN epochs from 153 s to 120 s mid-run. Note that
  `scaling_governor` reads `powersave` in *both* profiles on amd-pstate; the knob is
  `energy_performance_preference`. Table 1 timings must be re-measured in one pass on an idle
  machine at a fixed profile. See `docs/report-material/06-cost-measurement.md`.
- **`.gitignore` patterns are not recursive** — `results/*.joblib` matches nothing in
  `results/models/`, so a 2.4 MB checkpoint sat untracked-but-unignored and would have been
  committed by the next `git add -A`. Found at task 4.3; fixed with `results/**/*.joblib`.
  Check with `git check-ignore -v <path>`, which prints the matching rule or exits non-zero.
- **CNN feature extraction** — put the model in `eval()` and wrap in `torch.no_grad()`, or the penultimate features will carry dropout noise.
- **Memory layout changes float32 results** — `PCA.fit` leaves `components_` F-contiguous,
  a joblib round-trip restores it C-contiguous, and BLAS picks a different GEMM kernel for
  each. Values bit-identical, projections differing by ~1e-6: *a reloaded model behaves
  differently from the one just fitted*. Normalise with `np.ascontiguousarray` at fit time.
  Confirmed at task 4.1. The residual case — a subset batch blocking differently from the
  full batch — is unfixable and documented instead (same 1e-6, far below any SVM margin).

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

## 12. Optional — capture-time degradation and composite scenarios

**Not part of the MVP.** The MVP injects degradation *after* preprocessing, which models the
residual preprocessing did not remove (§6, task 3.1). The extensions below model degradation
at *capture* instead, which is a different question and out of scope.

- [ ] **12.0** **Capture-time ordering.** Degrade at source resolution, before preprocessing,
      so CLAHE sees the damage. Would measure the preprocessing × degradation interaction the
      MVP cannot see — e.g. CLAHE amplifying sensor noise (×1.84, note 08) or partially
      undoing a gamma shift. Requires blur to be reparameterised as a *fraction* of sign size
      (`k × max(h,w)/48`), since `k` is the only spatial parameter and a fixed 15 px kernel is
      larger than a 25 px crop. Keep the existing input-space path alongside it; having both
      makes the order-sensitivity comparison nearly free.

**Composite scenarios — also not part of the MVP.** Only after the core grid (§7) is complete and working. First thing
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
