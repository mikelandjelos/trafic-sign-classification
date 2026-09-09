# 01 — Environment and setup

**Feeds:** Methodology → reproducibility subsection; Appendix (exact versions).
**Status:** in progress — version table filled after `poetry install` completes.

---

## Hardware

| | |
|---|---|
| CPU | AMD Ryzen 7 5800H, 8 cores / 16 threads |
| RAM | 13 GB total |
| GPU | AMD Vega iGPU (Cezanne) — **no CUDA**; ROCm does not support this chip |
| Disk | 530 GB free |
| OS | Ubuntu 24.04.4 LTS, kernel 7.0.0-30-generic |

The single binding constraint is **the absence of a usable GPU**, not memory. Every
methodological decision below follows from that, and the report should present them as
consequences of one constraint rather than as four independent choices:

| Decision | Forced by |
|---|---|
| `LinearSVC` (liblinear), never `SVC(kernel='rbf')` | kernel SVM is O(n²)–O(n³); ~39 k training samples would run for hours on CPU |
| `MiniBatchKMeans` for the BoVW vocabulary | full k-means over ~600 k descriptors fits in RAM but is needlessly slow |
| randomized SVD for PCA (`svd_solver='randomized'`) | exact SVD on a 39 209 × 2304 matrix is wasteful when only ≤256 components are kept |
| small CNN at 48×48, CPU-only PyTorch | a few minutes per epoch on 16 threads is acceptable; anything larger is not |
| explicit thread pinning | sklearn (OpenMP) and torch each grabbing 16 threads causes BLAS oversubscription and thrashing |

**Point worth making in the report:** these are *compute* limits, not accuracy ceilings.
The comparison stays valid because the classifier is held fixed across all
representations — a slower classifier would move every row of Table 1 in the same
direction and would not change the ranking, which is what the study measures.

---

## Software environment

### Decisions and rationale

**Python 3.11.11 via pyenv.** Nothing in the stack strictly requires 3.11 — OpenCV,
scikit-image and PyTorch all ship wheels for 3.12/3.13. 3.11 was chosen because it has
the widest wheel coverage of any currently supported release, removing the risk of a
missing build for any single package mid-project. Pinned in `.python-version` so a
clone reproduces the interpreter, not just the packages.

**Poetry for dependency management.** Constraints in `pyproject.toml` are deliberately
*loose* (`>=` lower bounds only); the exact versions live in the committed
`poetry.lock`. This is the reproducibility model: the lock file is the record, the
constraints only express real incompatibilities.

**Dependency groups.** Three, so that reproducing results does not require the research
tooling:

| Group | Contents | Why separate |
|---|---|---|
| main | numpy, pandas, opencv-python, scikit-image, scikit-learn, matplotlib, tqdm, torch | everything `src/gtsrb/` imports — `poetry install --only main` must suffice to regenerate `results.csv` |
| research | jupyterlab, ipykernel, seaborn | exploratory notebooks; never imported by the package |
| dev | pytest, ruff | tests (incl. the track-disjointness assertion) and linting |

**CPU-only PyTorch via an explicit source.** `download.pytorch.org/whl/cpu` is
registered with `priority = "explicit"`, meaning Poetry consults it *only* for
dependencies that name it. Without this, resolving `torch` from PyPI on Linux pulls the
CUDA build (~2.5 GB of unusable NVIDIA runtime libraries).

**`.python-version` pins two interpreters, not one.** It lists `3.11.11` *then* `3.13.0`.
pyenv resolves shims against the listed versions in order, so `python` is 3.11.11 (the
project interpreter) while `poetry` — which is installed only under 3.13.0 — still
resolves. Pinning 3.11.11 alone makes the `poetry` shim fail with
`pyenv: poetry: command not found` inside the project directory, which is confusing
because Poetry is plainly installed. Worth an appendix footnote: Poetry's own
interpreter is unrelated to the project interpreter it manages, and conflating them is a
common pyenv+Poetry trip-up.

**`cv2.SIFT_create` needs no contrib package.** The SIFT patent expired in March 2020
and the algorithm moved from `opencv-contrib-python` into the main `opencv-python`
distribution. Verified explicitly at task 0.1 rather than assumed, because a missing
SIFT would silently block BoVW (task 6) two days later.

### Exact versions

Verified by `scripts/verify_env.py` (task 0.1), 2026-09-09. Exact pins live in the
committed `poetry.lock`; this table is the appendix-ready subset.

| Package | Version |
|---|---|
| Python | 3.11.11 |
| numpy | 2.4.6 |
| pandas | 2.3.3 |
| opencv-python | 4.14.0 |
| scikit-image | 0.26.0 |
| scikit-learn | 1.9.0 |
| torch | 2.14.0+cpu |
| matplotlib | 3.11.1 |

Platform string: `Linux-7.0.0-30-generic-x86_64-with-glibc2.39`.

**Verification is behavioural, not just import-based.** `verify_env.py` does not merely
check that `cv2.SIFT_create` exists — it runs `sift.compute()` on a synthetic 48×48 crop
with nine manually placed keypoints and asserts a `(9, 128)` descriptor array. That
exercises exactly the *dense* code path task 6 depends on, rather than the detector path,
which is the one that silently returns zero keypoints on small crops. It likewise asserts
`torch.cuda.is_available() is False` and that `"+cu"` is absent from the torch version
string, so a wrong-wheel install fails loudly at setup instead of at training time.

`skimage.feature.hog` on a 48×48 input with `orientations=9`, `pixels_per_cell=(8,8)`,
`cells_per_block=(2,2)` yields **900** dimensions — a useful sanity number to quote when
Table 1 reports feature dimensionality.

### Decision: upper bounds on OpenCV and pandas

Left unconstrained, the resolver initially selected three recent **major** releases:
OpenCV 5.0.0, pandas 3.0.5 and torch 2.14. All task-0.1 checks passed on them. They were
nevertheless pinned back to `opencv-python >=4.10,<5` (→ 4.14.0) and `pandas >=2.2,<3`
(→ 2.3.3), deliberately, because both majors change behaviour in ways that fail *quietly*
rather than loudly:

- **pandas 3.0** makes copy-on-write and the string dtype the defaults. Chained assignment
  that worked in 2.x silently becomes a no-op — precisely the failure mode that would
  corrupt the annotation dataframe (task 1.1) without raising anything.
- **OpenCV 5.0** is a major API break relative to the 4.x that every course reference and
  tutorial assumes. Nothing this project needs was broken, but reading 4.x documentation
  while running 5.x invites errors that cost more than the upgrade is worth.

torch was left at 2.14 — its major-version cadence does not carry comparable semantic
changes for the small CPU CNN used here.

**The general principle, worth one sentence in the report:** for a short project, an
environment that matches the reference material is worth more than an environment that is
current. Reproducibility comes from `poetry.lock`, not from being on the newest release.

---

## Repository layout

```
src/gtsrb/        installable package (`poetry install` puts it on the path editable)
scripts/          entry points — one per pipeline stage; results.csv reproducible from these
notebooks/        exploration only; findings get promoted into docs/report-material/
tests/            pytest; includes the track-disjointness assertion (task 1.3)
data/             GTSRB archives + extracted images (gitignored)
results/          results.csv (committed — it is a deliverable) + model checkpoints (ignored)
figures/          generated figures (gitignored; regenerable from scripts/)
docs/             PROJECT_TASKS.md, proposition/, report-material/, report/
```

An installable package rather than loose scripts, so that notebooks, tests and pipeline
scripts all import the *same* code by the same path — no `sys.path` manipulation, and no
risk that a notebook silently diverges from the code that produced `results.csv`.

---

## Data acquisition

Source: original GTSRB archives from the `sid.erda.dk` mirror (the RUB benchmark site is
live but historically flaky).

| Archive | Size | Contents |
|---|---|---|
| `GTSRB_Final_Training_Images.zip` | 263 MB | 39 209 PPM crops, per-class `GT-*.csv` |
| `GTSRB_Final_Test_Images.zip` | 85 MB | 12 630 PPM crops |
| `GTSRB_Final_Test_GT.zip` | <1 MB | `GT-final_test.csv` — test labels, withheld from the images archive |

Chosen over `torchvision.datasets.GTSRB` and the Kaggle mirror because the original
archives are the only source guaranteed to preserve **both** things this study depends on:

1. **ROI coordinates** (`Roi.X1/Y1/X2/Y2` per image) — without them, bounding-box jitter
   (task 3.4) cannot be done honestly; it would have to be faked with padding.
2. **Track IDs**, encoded in the filename `TTTTT_FFFFF.ppm` — without them, the
   track-disjoint split (task 1.2) is impossible and validation accuracy is inflated by
   near-duplicate frames of the same physical sign appearing on both sides of the split.

Test labels ship separately from test images, which is a leftover from the original 2011
IJCNN competition format — worth one sentence in the report, as it is a small reminder
that GTSRB was designed as a blind challenge.
