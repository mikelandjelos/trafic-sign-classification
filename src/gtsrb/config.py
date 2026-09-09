"""Task 0.3: global configuration -- seeds, thread pinning, paths, shared constants.

Call this at the top of every script and notebook:

    from gtsrb import config
    config.set_seeds()

Thread pinning is applied two ways, deliberately. Environment variables are set at import
time, but BLAS backends read those only when *they* are imported -- and an import sorter
will happily place `import numpy` above a first-party `from gtsrb import config`, which
would silently defeat the env vars in every script from task 1.1 onward. So `set_seeds()`
additionally clamps the already-loaded thread pools at runtime via `threadpoolctl`, which
works regardless of import order. The env vars remain as belt-and-braces for subprocesses.

Why this file exists at all (see docs/report-material/03-reproducibility.md):

The study's claim is that differences between representations are attributable to the
representation, because the classifier is held fixed. A gap of 0.7 percentage points only
supports that claim if re-running would not move it by 0.7 points on its own. Seed control
is what separates a real effect from run-to-run noise.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# --- thread pinning ----------------------------------------------------------------
# These env vars only bite if set before the BLAS libraries are imported, which is not
# guaranteed; `limit_threads()` is the authoritative mechanism and works either way.
# Defaults to the 8 physical cores rather than all 16 SMT threads: sklearn (OpenMP) and
# torch each grabbing 16 oversubscribes the BLAS pool and thrashes. Override with
# GTSRB_NUM_THREADS=N if benchmarking.
NUM_THREADS: int = int(os.environ.get("GTSRB_NUM_THREADS", "8"))

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

for _var in _THREAD_ENV_VARS:
    # setdefault, not assignment: an explicit value from the shell always wins.
    os.environ.setdefault(_var, str(NUM_THREADS))

# Holds the threadpoolctl limiter so it is not garbage-collected -- dropping the
# reference can restore the original (unlimited) limits.
_LIMITER = None


def limit_threads(n_threads: int = NUM_THREADS) -> None:
    """Clamp BLAS/OpenMP thread pools at runtime, regardless of import order.

    `threadpoolctl` reaches into the already-loaded OpenBLAS/MKL/OpenMP libraries, so this
    works even when numpy was imported before this module -- which is the normal case once
    an import sorter has had its way with a script.
    """
    global _LIMITER
    from threadpoolctl import threadpool_limits

    _LIMITER = threadpool_limits(limits=n_threads)

# --- the seed ----------------------------------------------------------------------
# One arbitrary but fixed value. Every random_state= in the project derives from it, so
# there is exactly one number to change to check that a result is not a lucky draw.
SEED: int = 42

# --- paths -------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
GTSRB_DIR: Path = DATA_DIR / "GTSRB"
TRAIN_IMAGES_DIR: Path = GTSRB_DIR / "Final_Training" / "Images"
TEST_IMAGES_DIR: Path = GTSRB_DIR / "Final_Test" / "Images"
CACHE_DIR: Path = DATA_DIR / "cache"  # preprocessed uint8 .npy arrays (task 2.3)

RESULTS_DIR: Path = PROJECT_ROOT / "results"
RESULTS_CSV: Path = RESULTS_DIR / "results.csv"  # the tidy results table (task 1.6)
MODELS_DIR: Path = RESULTS_DIR / "models"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"

# --- dataset constants -------------------------------------------------------------
N_CLASSES: int = 43
N_TRAIN_IMAGES: int = 39_209
N_TEST_IMAGES: int = 12_630
N_TRACKS: int = 1_307  # (class_id, track_id) pairs -- see docs/report-material/02

IMAGE_SIZE: tuple[int, int] = (48, 48)  # (H, W) every representation operates on
VAL_FRACTION: float = 0.2  # track-disjoint, stratified by class (task 1.2)

# --- seeding -----------------------------------------------------------------------


def set_seeds(seed: int = SEED) -> None:
    """Seed `random`, numpy and torch, and pin every thread pool.

    Call once at the top of every script. torch is imported lazily -- the PCA, HOG and
    BoVW pipelines never need it, and importing it costs a couple of seconds.

    Note: `PYTHONHASHSEED` cannot be set from inside a running interpreter. It affects
    only str/bytes hash randomisation, which nothing here depends on for ordering.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    limit_threads()

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        return

    torch.manual_seed(seed)
    torch.set_num_threads(NUM_THREADS)


def derive_seed(*parts: object) -> int:
    """Derive a stable 32-bit seed from SEED plus arbitrary labels.

    This is what keeps the robustness grid honest. At task 8.1 five methods are evaluated
    against the same 21 conditions, and every method must see *identical* degraded pixels
    -- otherwise the robustness curves compare methods on different data and the
    representation x stressor interaction, which is the whole point of the study, gets
    contaminated by which method drew unlucky noise.

    Seeding once at startup is NOT sufficient: the stream position would then depend on
    how many draws happened earlier, so simply running the methods in a different order
    would change the noise. Deriving from content instead makes the degraded test set a
    pure function of (degradation, level, image) and independent of execution order.

        rng = rng_for("noise", 20, image_index)
    """
    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(f"{SEED}|{key}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def rng_for(*parts: object):
    """A `numpy.random.Generator` seeded deterministically by `parts`. See `derive_seed`."""
    import numpy as np

    return np.random.default_rng(derive_seed(*parts))


def ensure_dirs() -> None:
    """Create the output directories that scripts write into."""
    for directory in (CACHE_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """One-block summary of the active configuration, for logs and the report appendix."""
    return "\n".join(
        [
            f"seed          : {SEED}",
            f"threads       : {NUM_THREADS} (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})",
            f"project root  : {PROJECT_ROOT}",
            f"image size    : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
            f"classes       : {N_CLASSES}",
            f"val fraction  : {VAL_FRACTION} (track-disjoint, stratified)",
        ]
    )


if __name__ == "__main__":
    set_seeds()
    ensure_dirs()
    print(describe())
