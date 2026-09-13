# 03 — Reproducibility: seeds, determinism, and thread control

**Feeds:** Methodology → experimental protocol; Limitations. The degradation-seeding
argument below is not housekeeping — it is what makes the robustness curves (figure 9.5)
a valid comparison rather than five methods measured on five different datasets.
**Status:** complete (task 0.3)

Implemented in `src/gtsrb/config.py`.

---

## Why seed control is load-bearing for *this* study

The design holds the classifier fixed (`LinearSVC`) so that any difference between
representations is attributable to the representation. That argument only works if the
measured differences exceed run-to-run variance. If HOG scores 96.1 % and BoVW 95.4 %, the
0.7-point gap means something only if re-running neither method moves it by 0.7 points.

Randomness enters this project in more places than is obvious:

| Stage | Random component |
|---|---|
| Train/val split (1.2) | which tracks are assigned to validation |
| PCA (4.1) | `svd_solver='randomized'` uses random projections |
| BoVW (6.3) | subsampling ~200 k of ~600 k descriptors; `MiniBatchKMeans` centroid init |
| `LinearSVC` | the dual coordinate-descent solver shuffles internally |
| CNN (7.x) | weight init, batch shuffling, dropout masks |
| Degradations (3.x) | noise draws, motion-blur angle |

**scikit-learn has no global seed.** Unlike numpy, it cannot be seeded once — every
estimator takes its own `random_state=`. A single `SEED` constant in `config.py` is
therefore not a convenience but the mechanism: one value that every call site passes, so
there is exactly one number to change when checking whether a result is a lucky draw.

---

## The degradation seeding problem

**This is the subtle part, and the part most worth a paragraph in the report.**

At task 8.1, five methods are evaluated against the same 16 conditions. When σ = 20
Gaussian noise is applied, PCA, HOG, BoVW and both CNN variants must see the *same noisy
pixels*. Otherwise the robustness curves compare methods on different data, and the
representation × stressor interaction — the entire contribution of the project — is
contaminated by which method happened to draw unlucky noise.

**Seeding once at startup does not achieve this.** A single global stream means the values
drawn for a given image depend on how many draws happened before it. Evaluating the
methods in a different order, or adding a method, silently changes the degraded test set.

The solution is to derive the seed from *content* rather than from stream position:

```python
def derive_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(f"{SEED}|{key}".encode()).digest()[:4], "big")

rng = config.rng_for("noise", 20, image_index)
```

The degraded test set becomes a pure function of `(degradation, level, image)` and is
independent of execution order, of how many methods are run, and of the order they run in.
Verified: the same key reproduces its stream after a thousand intervening draws from other
keys, and distinct keys give distinct streams.

---

## Thread control, and a design mistake worth recording

GTSRB timing numbers (task 1.5, and the cost column of Table 1) are only comparable if
every method gets the same CPU budget. Both sklearn (via OpenMP/BLAS) and torch default to
grabbing all 16 SMT threads; run together they oversubscribe the pool and thrash. The
budget here is pinned to **8** — the physical core count.

The first implementation set `OMP_NUM_THREADS` and friends at module import, with a
warning if `gtsrb.config` was imported after numpy. **That design was wrong**, and the
linter is what exposed it: BLAS backends read those variables only when *they* are
imported, and an import sorter places a first-party `from gtsrb import config` *below*
`import numpy`. So from task 1.1 onward — every script that touches numpy or pandas — the
pinning would have been silently inert, leaving only a warning nobody reads. Timing
numbers would have been collected under uncontrolled thread counts.

The fix makes the mechanism independent of import order: `set_seeds()` calls
`threadpoolctl.threadpool_limits()`, which reaches into the *already loaded* OpenBLAS /
OpenMP libraries at runtime. The environment variables are retained only as belt-and-braces
for subprocesses. Demonstrated under the worst case — `OMP_NUM_THREADS=16` forced in the
environment and numpy imported first:

```
before set_seeds(): {'blas': 16}
after  set_seeds(): {'blas': 8, 'openmp': 8}     torch threads: 8
```

The general lesson, and the reason it belongs in the report's limitations discussion
rather than being quietly fixed: **a reproducibility control that fails silently is worse
than none**, because it produces the appearance of rigour. The remaining controls in this
project are therefore all asserted or demonstrated, not merely configured.

---

## What is *not* controlled, and should be stated as a limitation

- **`PYTHONHASHSEED`** cannot be set from inside a running interpreter. It affects only
  str/bytes hash randomisation; nothing here depends on it for ordering.
- **BLAS floating-point non-determinism.** Multi-threaded reductions may sum in different
  orders between runs, so results can differ in the last bits. This is far below any
  effect size the study discusses, but it means "bitwise identical" is not claimed —
  **reproducible to reported precision** is.
- **Single seed, not multiple runs.** With more compute, every configuration would be run
  over several seeds and reported as mean ± std. The evaluation grid is 5 × 16 = 80 runs
  at one seed; repeating it across seeds is the obvious extension and should be named as
  such rather than glossed over.
