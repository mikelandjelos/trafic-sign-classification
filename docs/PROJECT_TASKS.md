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

| Stressor | Predicted most robust | Predicted least robust | Reasoning |
|---|---|---|---|
| Bbox jitter | BoVW | PCA | orderless encoding is translation-invariant; PCA subspace assumes alignment |
| Motion blur | PCA | HOG / BoVW | gradient-based methods lose their signal; PCA works on raw intensity |
| Gaussian noise | PCA | HOG | low-dim projection averages noise away; gradients amplify it |
| Small signs (<32 px) | HOG / CNN | BoVW | too little support for meaningful local descriptors |

- [ ] **P.1** Record predictions in `predictions.md` with a timestamp, before task 8

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

Pin thread counts explicitly (`torch.set_num_threads`, `OMP_NUM_THREADS`) — sklearn and
torch each grabbing 16 threads will thrash.

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
(filenames `TTTTT_FFFFF.ppm`, `TTTTT` = track ID). A random train/val split puts
near-duplicate frames of the same sign on both sides and inflates validation accuracy.

**Split by track ID, never by image.** State this explicitly in the report.

Annotation CSVs give per-image `Width`, `Height`, `Roi.X1/Y1/X2/Y2`, `ClassId`. The ROI
coordinates are what make honest bbox jitter possible — you re-crop from the source
image with perturbed coordinates rather than faking it with padding.

---

## 6. Tasks

### Day 1 — Infrastructure (~5.5 h)

- [ ] **0.1** venv, install stack, verify `cv2.SIFT_create` exists *(task 0 = 1.0 h)*
- [ ] **0.2** Download GTSRB, verify counts: 39,209 / 12,630 / 43
- [ ] **0.3** `config.py` with one global seed; seed numpy, torch, sklearn
- [ ] **1.1** Parse annotation CSVs → dataframe: `path, class_id, track_id, roi_*, w, h` *(task 1 = 2.5 h)*
- [ ] **1.2** Track-disjoint train/val split (80/20 by track ID, stratified by class)
- [ ] **1.3** Write a test asserting no track ID appears in both splits
- [ ] **1.4** Eval harness → accuracy, macro-F1, per-class F1, confusion matrix
- [ ] **1.5** Timing harness → train wall-clock, inference ms/img (median of ≥3 runs, discard first)
- [ ] **1.6** Append all results to one tidy CSV: `method, preproc, degradation, level, metric, value`
- [ ] **2.1** Preprocessing: `to_gray`, `to_hsv`, `clahe`, `gaussian_blur`, `resize(48,48)` *(task 2 = 1.5 h)*
- [ ] **2.2** Three named configs for the ablation: `raw_gray`, `clahe_gray`, `clahe_hsv`
- [ ] **2.3** Cached loader — preprocess once, store uint8 `.npy`, reload fast
- [ ] **3.1** Degradation: Gaussian noise, σ ∈ {0, 5, 10, 20, 40} *(task 3 = 1.5 h)*
- [ ] **3.2** Degradation: motion blur, kernel ∈ {0, 3, 5, 9, 15} px, random angle
- [ ] **3.3** Degradation: gamma, γ ∈ {0.4, 0.7, 1.0, 1.5, 2.5}
- [ ] **3.4** Degradation: **bbox jitter** — perturb ROI ±{0,10,20,30,40}% in scale and position, re-crop from source (no padding; let real background in)
- [ ] **3.5** Contact-sheet figure: every degradation × every level on one sample → straight into the report

### Day 2 — PCA, HOG, CNN kickoff (~5.5 h)

- [ ] **4.1** PCA on flattened 48×48 grayscale, `svd_solver='randomized'` *(task 4 = 1.5 h)*
- [ ] **4.2** Sweep n_components ∈ {32, 64, 128, 256} on val, pick one
- [ ] **4.3** `LinearSVC` on PCA features; record cost metrics
- [ ] **4.4** Figure: top-16 eigenvectors as an image grid ("eigensigns") — ties directly to the Eigenfaces lecture
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
- [ ] **P.1** Write `predictions.md` — do this before task 8
- [ ] **8.1** Run full evaluation grid (§7) — inference only, no retraining *(task 8 = 1.5 h)*
- [ ] **8.2** Preprocessing ablation: best 2 methods × 3 configs
- [ ] **8.3** Verify results CSV is complete, no NaNs

### Day 4 — Analysis and figures (~5.5 h)

- [ ] **9.1** Table 1: accuracy, macro-F1, train time, inference ms/img, model size MB, feature dim — one row per method (5 rows) *(task 9 = 2.0 h)*
- [ ] **9.2** Figure: confusion matrix, best and worst method
- [ ] **9.3** Table: top-10 most-confused class pairs + commentary (speed limits confuse predictably)
- [ ] **9.4** Figure: accuracy vs. sign size — buckets [0,32), [32,48), [48,72), [72,∞) by ROI height, one line per method
- [ ] **9.5** Figure: robustness curves — 4 panels (noise, blur, gamma, jitter), one line per method
- [ ] **9.6** **Table: predictions vs. outcomes** — which held, which didn't, and why. This is the core of the discussion section.
- [ ] **9.7** Table: preprocessing ablation
- [ ] **9.8** Write 5 concrete findings as bullets — raw material for the conclusion
- [ ] Buffer

### Day 5 — Report (Serbian)

- [ ] Write up per the proposal structure

---

## 7. Experiment grid

Models are trained once; only inference varies across the grid. Whole grid runs in well
under an hour.

```
methods       = [pca_svm, hog_svm, bovw_svm, cnn_feat_svm, cnn_e2e]   # 5
degradations  = [clean] + [noise, blur, gamma, jitter] x 5 levels     # 21
preproc       = [raw_gray, clahe_gray, clahe_hsv]                     # ablation, best 2 methods only
```

Core grid: 5 × 21 = 105 evaluation runs.
Size-stratified results are free — group the existing clean-test predictions by ROI height.

---

## 8. Deliverables

- [ ] Table 1 — headline comparison (accuracy, macro-F1, cost)
- [ ] Table — **predictions vs. outcomes**
- [ ] Figure — eigensigns
- [ ] Figure — HOG visualization
- [ ] Figure — degradation contact sheet
- [ ] Figure — confusion matrices (best + worst)
- [ ] Figure — accuracy vs. sign size
- [ ] Figure — robustness curves (4 panels incl. bbox jitter)
- [ ] Table — preprocessing ablation
- [ ] Table — top confused pairs
- [ ] `results.csv` — reproducible from a single script

---

## 9. Cut list

Cut in this order if you fall behind. The comparison survives all three.

1. **BoVW** (task 6, ~2.5 h) — the fiddliest. Leaves PCA, HOG, and both CNN variants.
2. **Preprocessing ablation** (8.2) — report one config, note as a limitation.
3. **Hyperparameter sweeps** (4.2, 5.2, 6.3) — use sensible defaults, note as a limitation.

Never cut: track-disjoint splitting, bbox jitter, the cost table, the predictions table.
Those are what make this a study rather than a tutorial.

---

## 10. Known gotchas

- **Track leakage** — split by track, assert it. See §5.
- **Detector SIFT returns zero keypoints** on small crops, silently breaking BoVW. Use dense SIFT.
- **`SVC(kernel='rbf')`** on 39k samples will run for hours. `LinearSVC`.
- **Class imbalance** — GTSRB classes differ ~10×. Report macro-F1, not just accuracy. Consider `class_weight='balanced'`.
- **BLAS oversubscription** — pin thread counts for sklearn and torch.
- **Timing** — discard the first inference call (lazy init / cache warmup); median of ≥3 runs.
- **PPM format** — original GTSRB is P6 PPM; `cv2.imread` handles it natively.
- **CNN feature extraction** — put the model in `eval()` and wrap in `torch.no_grad()`, or the penultimate features will carry dropout noise.
