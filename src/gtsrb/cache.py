"""Task 2.3: preprocess once, store as uint8 `.npy`, reload fast.

    X = cache.load_images(train_df, "clahe_gray")                 # cached (default)
    X = cache.load_images(train_df, "clahe_gray", use_cache=False) # recomputed from PPM

Why this exists
---------------
The evaluation grid is 80 runs, and every one of them would otherwise re-decode 12 630 PPM
files and re-run CLAHE over them. That work is identical every time. Caching turns it into
one pass per (split, preproc) and is most of what makes "the whole grid runs in under an
hour" true.

The cache must be provably invisible
------------------------------------
A cache that silently changes the pixels would corrupt every result downstream while
looking like a speedup. So `use_cache=False` is not a fallback -- it is the **reference
implementation**, and `verify()` checks the cached array against it bit-for-bit:

    poetry run python -m gtsrb.cache --verify

Both paths run the same `preprocessing.load_and_preprocess` over the same rows in the same
order, so agreement is expected to be exact, not approximate. `verify()` asserts equality
with `np.array_equal`; there is no tolerance parameter, because any difference at all means
a bug rather than rounding.

Staleness
---------
A cache built with different preprocessing parameters is worse than no cache. Every file
carries a manifest recording the pipeline fingerprint (preproc name, image size, CLAHE
clip limit and tile grid, plus a CACHE_VERSION to bump on any change) and a hash of the
ordered path list. A mismatch rebuilds rather than silently serving stale pixels.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from gtsrb import config, data, preprocessing

#: Bump when anything about the preprocessing pipeline changes shape or semantics in a way
#: the parameter fingerprint below would not capture.
CACHE_VERSION = 1


def _fingerprint(preproc: str) -> str:
    """Hash of everything that affects the cached pixels."""
    payload = {
        "version": CACHE_VERSION,
        "preproc": preproc,
        "image_size": list(config.IMAGE_SIZE),
        "clahe_clip": preprocessing.CLAHE_CLIP_LIMIT,
        "clahe_tile": list(preprocessing.CLAHE_TILE_GRID),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _paths_hash(paths) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode())
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def array_path(split: str, preproc: str) -> Path:
    return config.CACHE_DIR / f"{split}_{preproc}.npy"


def manifest_path(split: str, preproc: str) -> Path:
    return config.CACHE_DIR / f"{split}_{preproc}.json"


@lru_cache(maxsize=4)
def _split_annotations(split: str) -> pd.DataFrame:
    """Full annotations for a split, memoised -- the cache is built in this exact order."""
    return data.load_annotations(split)


def _read_manifest(split: str, preproc: str) -> dict | None:
    path = manifest_path(split, preproc)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def is_valid(split: str, preproc: str) -> bool:
    """True when a cache exists and matches the current pipeline and annotation order."""
    manifest = _read_manifest(split, preproc)
    if manifest is None or not array_path(split, preproc).exists():
        return False
    frame = _split_annotations(split)
    return (
        manifest.get("fingerprint") == _fingerprint(preproc)
        and manifest.get("paths_hash") == _paths_hash(frame["path"])
        and manifest.get("n") == len(frame)
    )


def build(split: str, preproc: str, force: bool = False, progress: bool = True) -> Path:
    """Preprocess a whole split once and store it as uint8 `.npy`."""
    if not force and is_valid(split, preproc):
        return array_path(split, preproc)

    frame = _split_annotations(split)
    cfg = preprocessing.get_config(preproc)
    out = np.empty((len(frame), *cfg.shape), dtype=np.uint8)

    rows = frame.itertuples(index=False)
    if progress:
        rows = tqdm(rows, total=len(frame), desc=f"  {split}/{preproc}", unit="img")
    for i, row in enumerate(rows):
        out[i] = preprocessing.load_and_preprocess(row._asdict(), preproc)

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(array_path(split, preproc), out)
    manifest_path(split, preproc).write_text(
        json.dumps(
            {
                "split": split,
                "preproc": preproc,
                "n": len(frame),
                "shape": list(out.shape),
                "dtype": str(out.dtype),
                "fingerprint": _fingerprint(preproc),
                "paths_hash": _paths_hash(frame["path"]),
                "megabytes": round(out.nbytes / 1024**2, 1),
            },
            indent=2,
        )
        + "\n"
    )
    return array_path(split, preproc)


def _compute(frame: pd.DataFrame, preproc: str, progress: bool = False) -> np.ndarray:
    """The reference implementation: preprocess every row from source, no cache."""
    cfg = preprocessing.get_config(preproc)
    out = np.empty((len(frame), *cfg.shape), dtype=np.uint8)
    rows = frame.itertuples(index=False)
    if progress:
        rows = tqdm(rows, total=len(frame), desc=f"  {preproc} (uncached)", unit="img")
    for i, row in enumerate(rows):
        out[i] = preprocessing.load_and_preprocess(row._asdict(), preproc)
    return out


def load_images(
    frame: pd.DataFrame,
    preproc: str = preprocessing.DEFAULT_PREPROC,
    use_cache: bool = True,
    mmap: bool = False,
    progress: bool = False,
) -> np.ndarray:
    """Preprocessed images for the rows of `frame`, as a uint8 array.

    Row *i* of the result corresponds to row *i* of `frame`, cached or not.

    `use_cache=True` (default) builds the split's cache if needed, then selects the rows
    `frame` asks for -- so a train/val subset costs no extra storage, since both index into
    the one cached array for the split.

    `use_cache=False` recomputes everything from the PPM files. Identical output by
    construction; use it to verify the cache, or when preprocessing is being changed.

    `mmap=True` memory-maps the cached array instead of reading it into RAM. Useful for
    `clahe_hsv`, which is 3x the size of the grayscale configs.
    """
    if frame.empty:
        cfg = preprocessing.get_config(preproc)
        return np.empty((0, *cfg.shape), dtype=np.uint8)

    if not use_cache:
        return _compute(frame, preproc, progress=progress)

    splits = frame["split"].unique()
    if len(splits) != 1:
        raise ValueError(f"frame mixes splits {list(splits)}; load them separately")
    split = str(splits[0])

    build(split, preproc, progress=progress)

    full = _split_annotations(split)
    index_of = {path: i for i, path in enumerate(full["path"])}
    try:
        indices = np.fromiter((index_of[p] for p in frame["path"]), dtype=np.int64,
                              count=len(frame))
    except KeyError as exc:
        raise KeyError(
            f"path {exc} is not in the cached {split} split -- the frame does not come "
            f"from data.load_annotations({split!r})"
        ) from None

    stored = np.load(array_path(split, preproc), mmap_mode="r" if mmap else None)
    return np.asarray(stored[indices])


def flatten(images: np.ndarray) -> np.ndarray:
    """(N, H, W[, C]) uint8 -> (N, D) float32 in [0, 1], the form the estimators consume."""
    return images.reshape(len(images), -1).astype(np.float32) / 255.0


def verify(split: str, preproc: str, n_samples: int = 200, seed: int = 0) -> dict:
    """Check the cache against the uncached reference, bit-for-bit.

    Compares a random sample rather than the whole split so the check is cheap enough to
    run routinely; pass `n_samples=0` to compare everything.
    """
    frame = _split_annotations(split)
    if n_samples and n_samples < len(frame):
        rng = np.random.default_rng(seed)
        frame = frame.iloc[np.sort(rng.choice(len(frame), n_samples, replace=False))]

    cached = load_images(frame, preproc, use_cache=True)
    reference = load_images(frame, preproc, use_cache=False)

    identical = np.array_equal(cached, reference)
    mismatches = 0 if identical else int((cached != reference).any(axis=tuple(range(1, cached.ndim))).sum())
    return {
        "split": split,
        "preproc": preproc,
        "compared": len(frame),
        "identical": identical,
        "mismatched_images": mismatches,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build and verify the preprocessing cache.")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--preproc", nargs="+", default=list(preprocessing.PREPROC_CONFIGS))
    parser.add_argument("--force", action="store_true", help="rebuild even if valid")
    parser.add_argument("--verify", action="store_true", help="check cached == uncached")
    parser.add_argument("--samples", type=int, default=200, help="0 = compare everything")
    args = parser.parse_args()

    config.set_seeds()
    failures = 0

    for split in args.splits:
        for preproc in args.preproc:
            path = build(split, preproc, force=args.force)
            manifest = _read_manifest(split, preproc) or {}
            print(f"{path.name:<28} {manifest.get('megabytes', '?')} MB  "
                  f"shape={tuple(manifest.get('shape', ()))}")

    if args.verify:
        print("\nVerifying cached == uncached (bit-for-bit)")
        for split in args.splits:
            for preproc in args.preproc:
                report = verify(split, preproc, n_samples=args.samples)
                status = "OK  " if report["identical"] else "FAIL"
                if not report["identical"]:
                    failures += 1
                print(f"  {status} {split}/{preproc:<12} {report['compared']} images compared, "
                      f"{report['mismatched_images']} mismatched")

    if failures:
        print(f"\n{failures} cache(s) do not match the reference implementation.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
