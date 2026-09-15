# 06 — Cost measurement: what the timing numbers can and cannot claim

**Feeds:** Methodology → cost measurement; Table 1 (cost columns); **Limitations**.
**Status:** complete (tasks 1.5, 4.3, 5.2, 7.3 — batched-vs-single latency, model-size composition, tuning parallelism, CPU power state)

Implemented as `gtsrb.timing`; tested in `tests/test_timing.py`.

---

## The claim boundary — state this before quoting any number

Table 1 reports training wall-clock and inference ms/img for each representation. Those
numbers are **relative costs measured in one fully documented environment**. They are not
deployment latency, and the report must not let a reader take them as such.

Two limits bound the claim:

### 1. This measures an implementation, not an algorithm

The measurements are Python / NumPy / scikit-learn / PyTorch, on a general-purpose desktop
OS, with other processes running. A production traffic-sign recogniser is not that. It
would be **ported to the target platform** — compiled, very likely fixed-point, on an
automotive SoC, embedded ARM core, DSP or NPU.

The important consequence is not that the numbers would shrink. It is that they would
**not shrink uniformly**. HOG and BoVW are dense, regular, SIMD-friendly workloads; a CNN
maps onto a neural accelerator in a way none of the others do; PCA is a single dense
matrix multiply that a vendor BLAS will service extremely well. So porting can reorder the
ranking, not merely rescale it. Any sentence of the form "HOG is the cheapest
representation" is unsupported by this work; "HOG was cheapest **in this environment**" is
supported.

The honest formulation for the report: *the timing column establishes the order of
magnitude of the cost differences under a fixed, recorded implementation. Where those
differences span orders of magnitude they are informative; where they are 10–20 % they are
within the noise of implementation choices and should not be interpreted.*

### 2. There is no usable GPU on this machine

The CNN is timed on CPU because the available hardware is an AMD Vega iGPU with no CUDA
and no ROCm support. On any device with a neural accelerator, the CNN's relative position
would improve more than any other method's — it is the method most penalised by this
constraint. This should be named explicitly rather than left for a reader to infer, since
it is a systematic disadvantage to exactly one of the five configurations.

**This is a limitation of scope, not a flaw in the comparison.** The classifier and the
environment are held fixed across all five, so the *accuracy* comparison — the project's
actual subject — is unaffected. Only the cost column carries this caveat.

---

## What is recorded alongside every timing

A duration without its environment is uninterpretable, so `timing.platform_info()` captures
the environment and `save_platform_info()` writes it to `results/platform_<run_id>.json` next to
`results.csv`:

```json
{
  "timestamp": "2026-09-09T20:12:15+0200",
  "git_commit": "4f41ee4",
  "cpu": "AMD Ryzen 7 5800H with Radeon Graphics",
  "cpu_count_logical": 16,
  "ram_gb": 13.5,
  "gpu": "none (AMD Vega iGPU, no CUDA/ROCm support) -- all timings are CPU",
  "os": "Linux-7.0.0-30-generic-x86_64-with-glibc2.39",
  "threads_configured": 8,
  "threads_torch": 8,
  "blas": [{"library": "openblas", "threads": 8}, {"library": "openmp", "threads": 8}],
  "load_average": [1.28, 1.16, 1.16],
  "versions": {"python": "3.11.11", "numpy": "2.4.6", "sklearn": "1.9.0",
               "cv2": "4.14.0", "skimage": "0.26.0", "torch": "2.14.0+cpu"}
}
```

Three of these fields earn their place for specific reasons:

- **`blas` and `threads_torch`** *prove* the thread pinning from note 03 was actually in
  effect. An unpinned run oversubscribes 16 SMT threads and produces numbers that cannot
  be compared against pinned ones. Recording the achieved state means a mispinned run is
  detectable after the fact instead of silently polluting Table 1.
- **`load_average`** flags a contaminated measurement. A timing taken while the machine
  was busy is not comparable with one taken while it was idle.
- **`git_commit`** makes any row traceable to the code that produced it.

---

## Aside — real-time feasibility of the preprocessing stage

**Secondary material.** This project delivers module M3 offline; nothing here is a claim
about a deployed system. It is recorded because it bears on the extended framing (a
real-time pipeline) and would be tedious to reconstruct later.

Measured single-threaded (`cv2.setNumThreads(1)`) over 200 real crops, median 43 px:

| Stage | µs/sign | | Full pipeline | µs/sign | signs/s |
|---|---|---|---|---|---|
| `cvtColor` BGR→GRAY (source res) | 2.7 | | `raw_gray` | **10.9** | 91 600 |
| `cvtColor` BGR→HSV (source res) | 6.4 | | `clahe_gray` | **29.2** | 34 200 |
| + `resize` → 48×48 | 10.6 | | `clahe_hsv` | **43.7** | 22 900 |
| + CLAHE on 48×48 | 29.2 | | | | |

**Why it is this cheap:** the pipeline runs on 2 304 pixels. A 720p frame is 921 600 — 400×
more. All of this happens *after* the crop, so the cost is bounded by the sign, not the
frame. The working set is 2.3 KB and fits in L1 on any embedded core.

Scaling to a Cortex-A53 class core (≈10–20× slower than this Zen3 core for integer image
work: lower clock and IPC, NEON rather than AVX2, smaller cache), at the pessimistic end:

```
clahe_gray:  29.2 µs × 20 ≈ 0.6 ms/sign
GTSDB averages 1.64 signs/frame → ~1.0 ms/frame
30 fps budget = 33.3 ms/frame  → ~3 % of the budget
```

All four operations are **fixed-point friendly**, which matters on parts without an FPU:
`cvtColor` is an integer weighted sum, `INTER_AREA`/`INTER_LINEAR` use fixed-point weights,
and CLAHE is integer histograms plus a LUT and bilinear interpolation. OpenCV already
computes these in fixed point for 8-bit input, so there is no float dependency to remove.

Three caveats, consistent with the claim boundary above:

1. **This is an estimate, not a measurement.** Desktop OpenCV scaled by a rule of thumb.
   The only way to know is to run it on the target.
2. **The bottleneck would be detection (M1), which is out of scope** — a full-frame search
   is 400× the pixel count. The classifier is a separate question; PCA/HOG + `LinearSVC`
   are microseconds, but CNN inference on a weak CPU needs the task-7 numbers.
3. **CLAHE is 62 % of the pipeline cost** (18 of 29 µs). That makes the preprocessing
   ablation (task 8.2) a deployment question as well as an accuracy one: if `clahe_gray`
   does not beat `raw_gray` by a meaningful margin, `raw_gray` is 2.7× cheaper. Worth
   framing the ablation that way in the report rather than as a pure accuracy sweep.

## Measurement protocol

**Inference:** median of 5 repeats, first call discarded. **Training:** a single run, no
warmup — repeating it would measure refitting a warm process rather than the cost of
training, and for the CNN it would be prohibitively slow. Training time is therefore quoted
to at most two significant figures.

`time.perf_counter` is used rather than `time.time`: it is monotonic and immune to system
clock adjustments during a long run.

### The warmup discard is justified empirically, not by convention

Measured on `LinearSVC.predict` over 2,000 samples:

```
runs (s): [0.00087, 0.00063, 0.00059, 0.00056, 0.00056]
           ^ discarded -- 4.5x slower than the median
```

The first call was **4.5× slower** than the steady state, carrying lazy imports, BLAS
thread-pool spin-up and cache warmup. Including it would have inflated the reported cost by
roughly 45 % on a 5-run mean. `TimingResult.warmup_ratio` exposes this, so the
justification can be quoted from the actual runs rather than asserted.

**The IQR is reported as a trust signal.** A wide spread means the machine was busy during
measurement and the number should be treated with suspicion — better than presenting a
median that silently averages a contaminated run.


---

## Addendum (task 4.3) — batched throughput vs single-image latency, measured

This note warns that timings are relative costs in one environment, not deployment latency.
Task 4.3 supplies the magnitude, and it is larger than the warning implies:

| PCA + LinearSVC inference | ms/img | implied rate |
|---|---|---|
| batched (7,830 images at once) | 0.0066 | 151,000 img/s |
| **single image** | **0.3254** | 3,070 img/s |

**49×.** Both are now recorded for every method (`inference_ms_per_image` and
`inference_single_ms_per_image`), because they answer different questions: batched is what
the 80-cell grid costs, single-image is what a camera would experience. Quoting the batched
figure as per-frame latency would overstate the system by a factor of 49, and it is the
figure that falls out of a naive `time(predict(X_all)) / len(X_all)`.

Expect the gap to differ by method — BLAS amortises a large matrix multiply well, so PCA
benefits most; the CNN's per-image overhead is dominated by different costs. **The ratio
itself is therefore a per-method property and must not be assumed constant across Table 1.**


---

## Addendum (task 4.3) — `model_size_mb` measures structurally different things per method

Measured for PCA + `LinearSVC` (`results/models/pca_svm_clahe_gray.joblib`, 2.35 MB):

| contents | shape | dtype | size |
|---|---|---|---|
| PCA `components_` | (256, 2304) | float32 | **2.25 MB** |
| PCA `mean_` | (2304,) | float32 | 0.01 MB |
| `LinearSVC.coef_` | (43, 256) | float64 | 0.08 MB |
| `LinearSVC.intercept_` | (43,) | float64 | ~0 |

**96 % of PCA's "model" is the basis, not the classifier.** That is a property of the
representation, and it will not hold for the others: HOG has *no* fitted stage at all, so its
model is only the SVM coefficients (expect ~0.1 MB); BoVW carries a k-means codebook; the CNN
carries its weights. The column is worth reporting — a 2.35 MB dictionary is a real
deployment cost — but the numbers are **not like-for-like**, and Table 1 (9.1) must say what
dominates each, or a reader will conclude PCA is 20× "bigger" than HOG when what differs is
that one has a learned dictionary and the other has none.

Also note the format: `.joblib` is pickle with raw NumPy buffers. It is Python-only and
version-fragile (it stores references to `gtsrb...PCARepresentation` and to sklearn's
internal attribute layout), which is tolerable only because models are regenerable in ~30 s
and therefore gitignored. Any claim about embedded deployment would need the weights exported
as plain arrays or ONNX — the arithmetic is one matrix multiply plus 43 dot products, so that
is trivial, but the `.joblib` is not the artifact one would ship.


---

## Addendum (task 5.2) — parallelising the tuning grid: measured, and mostly not worth it

The HOG sweep is 96 `LinearSVC` fits, ~2 hours. Since the fits are independent, parallelising
them looked like an easy win. Measured on 4 fits over 12,000 × 900 features:

| mode | time | speedup | results identical? |
|---|---|---|---|
| sequential (8 BLAS threads) | 28.3 s | — | — |
| joblib `threading`, n_jobs=4, 1 thread each | **16.3 s** | **1.73×** | **yes** |
| joblib `loky`, n_jobs=4, 1 thread each | 17.5 s | 1.61× | yes |

**The speedup is 1.7×, not the 4× the core count suggests.** The reason is that the
sequential baseline is *already* parallel where it can be: `LinearSVC.predict` is a BLAS GEMM
using all 8 threads. What parallelising adds is only across liblinear's coordinate-descent
`fit`, which is single-threaded either way. Threading and loky land within 7 % of each other,
so memory pressure was not the limiter — this is close to the real ceiling at 4 workers.

**Both backends give bit-identical macro-F1 (agreement to 1e-12).** That answers the question
that actually mattered: parallelising would *not* have required re-running the PCA sweep,
task 4.3 or task 4.5. Worth recording because the opposite was plausible — per-worker thread
limits change BLAS summation order, which is exactly the 1e-6 effect documented in note 03.

Two practical notes for anyone repeating this:

- **Each worker must be pinned to one inner thread.** Otherwise N workers × 8 BLAS threads
  oversubscribes the machine and runs *slower* than sequential — the same class of silent
  failure as the env-var thread pinning at task 0.3.
- `inner_max_num_threads` is **not accepted by joblib's threading backend** (threads share
  one process); it needs `threadpool_limits(1)` around the block instead.

Not adopted: a 45-minute saving did not justify putting untested parallel code into the one
module every remaining method depends on. Recorded so the decision is not re-litigated from
scratch, and so the 1.7× figure is available if a later sweep is large enough to warrant it.


---

## Addendum (task 7.3) — the CPU power profile silently changes every timing

Mid-session the machine was switched from the "power saver" profile to "balanced", and the
effect on the running jobs was immediately visible: CNN epochs dropped from **153 s to 120 s**
(-22 %) with no code change.

What the knob actually is, since the naming misleads:

| | value |
|---|---|
| `scaling_governor` | `powersave` — **on amd-pstate this is the normal name, not a throttle** |
| `energy_performance_preference` (EPP) | changed to `balance_performance` |
| observed clock | ~3456 MHz avg, 3554 max (Ryzen 7 5800H base 3.2 GHz) |

**The governor string is not the setting.** It reads `powersave` in both profiles; the EPP is
what moved. Anyone checking only `scaling_governor` would conclude nothing had changed.

### Consequence for Table 1

Timings taken under different power profiles are **not comparable**, and nothing in
`results.csv` records the profile. This compounds two effects already noted:

1. thread count (recorded as `torch_threads`, and it differs between CNN runs),
2. machine contention (two jobs sharing cores),
3. **CPU power profile** (not recorded).

All three move wall-clock time without touching accuracy. The existing caveat — *timings are
relative costs in one recorded environment, not deployment latency* — therefore has to be
stated more strongly than "one environment": within a single session the environment itself
changed by 22 %.

**Practical rule for the final numbers:** the timings that go into Table 1 should be
re-measured in one pass, on an idle machine, at a fixed power profile and thread count, rather
than taken from whichever run happened to produce them. Accuracy needs no such re-run.

### Addendum — the CNN inverts PCA's batching advantage

Measured at task 7.3 and recorded here because it changes how the cost column reads:

| method | batched | single image | ratio |
|---|---|---|---|
| PCA + LinearSVC | 0.0066 ms | 0.3254 ms | **49×** |
| CNN | 1.703 ms | 1.477 ms | **0.87×** |

The CNN gains essentially nothing from batching — its per-image convolution work leaves
nothing to amortise. So the ranking depends on which column is read: batched, PCA looks ~260×
cheaper; **per frame it is only ~4.5×**. The single-image column is the honest one for any
real-time claim.
