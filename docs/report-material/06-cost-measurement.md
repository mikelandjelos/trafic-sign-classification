# 06 — Cost measurement: what the timing numbers can and cannot claim

**Feeds:** Methodology → cost measurement; Table 1 (cost columns); **Limitations**.
**Status:** complete (task 1.5)

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
the environment and `save_platform_info()` writes it to `results/platform.json` next to
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
