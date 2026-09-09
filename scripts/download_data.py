"""Task 0.2: download and extract GTSRB, then verify the counts that matter.

Source: the sid.erda.dk mirror of the original 2011 IJCNN competition archives. Chosen
over torchvision/Kaggle because it is the only source guaranteed to preserve BOTH
things this study depends on (see docs/report-material/01-environment-and-setup.md):

  1. ROI coordinates per image      -> honest bbox jitter (task 3.4)
  2. track IDs in the filenames     -> track-disjoint splitting (task 1.2)

Test labels ship in a separate archive from the test images -- a leftover from the
original blind-challenge format.

Run: poetry run python scripts/download_data.py
Idempotent: re-running skips archives already present at the right size and skips
extraction if the expected file counts are already on disk.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from gtsrb import config

DATA_DIR = config.DATA_DIR
RAW_DIR = config.RAW_DIR

BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370"
ARCHIVES = {
    "GTSRB_Final_Training_Images.zip": "training images + per-class GT-*.csv (ROI, ClassId)",
    "GTSRB_Final_Test_Images.zip": "test images (labels withheld)",
    "GTSRB_Final_Test_GT.zip": "test labels: GT-final_test.csv",
}

# Acceptance criteria for task 0.2, from the GTSRB paper. Owned by gtsrb.config.
EXPECTED_TRAIN_IMAGES = config.N_TRAIN_IMAGES
EXPECTED_TEST_IMAGES = config.N_TEST_IMAGES
EXPECTED_CLASSES = config.N_CLASSES


class _DownloadProgress(tqdm):
    def update_to(self, blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            self.total = total
        self.update(blocks * block_size - self.n)


def download(name: str, dest: Path) -> Path:
    """Download `name` unless a complete copy is already on disk."""
    url = f"{BASE_URL}/{name}"
    target = dest / name

    with urllib.request.urlopen(url) as response:
        remote_size = int(response.headers.get("Content-Length", 0))

    if target.exists() and remote_size and target.stat().st_size == remote_size:
        print(f"  skip     {name} (already complete, {remote_size / 1e6:.0f} MB)")
        return target

    if target.exists():
        print(f"  refetch  {name} (size {target.stat().st_size} != remote {remote_size})")

    with _DownloadProgress(unit="B", unit_scale=True, unit_divisor=1024, desc=f"  {name}") as bar:
        urllib.request.urlretrieve(url, target, reporthook=bar.update_to)
    return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        # Idempotence must be decided on *file* members only. The test archive ends with
        # a bare "GTSRB/" directory entry, which the training extraction already created
        # -- checking the last member alone silently skipped all 12 630 test images.
        files = [m for m in zf.namelist() if not m.endswith("/")]
        missing = [m for m in files if not (dest / m).exists()]
        if not missing:
            print(f"  skip     {archive.name} (all {len(files)} files already extracted)")
            return
        for member in tqdm(missing, desc=f"  {archive.name}", unit="file"):
            zf.extract(member, dest)


def verify(root: Path) -> bool:
    """Assert the three counts that define a correct GTSRB download."""
    train_root = root / "GTSRB" / "Final_Training" / "Images"
    test_root = root / "GTSRB" / "Final_Test" / "Images"

    train_images = sorted(train_root.rglob("*.ppm"))
    test_images = sorted(test_root.rglob("*.ppm"))
    class_dirs = sorted(d for d in train_root.iterdir() if d.is_dir())
    train_csvs = sorted(train_root.rglob("GT-*.csv"))
    test_gt = list(root.rglob("GT-final_test.csv"))

    checks = [
        ("train images", len(train_images), EXPECTED_TRAIN_IMAGES),
        ("test images", len(test_images), EXPECTED_TEST_IMAGES),
        ("class directories", len(class_dirs), EXPECTED_CLASSES),
        ("per-class GT csv", len(train_csvs), EXPECTED_CLASSES),
        ("test GT csv", len(test_gt), 1),
    ]

    ok = True
    print("\nVerification")
    for label, got, want in checks:
        status = "OK  " if got == want else "FAIL"
        ok &= got == want
        print(f"  {status} {label:<20} {got:>6} (expected {want})")

    # The two structural properties the rest of the project rests on.
    if train_images:
        sample = train_images[0]
        track_id, frame_id = sample.stem.split("_")
        print(f"\n  filename pattern: {sample.name} -> track={track_id} frame={frame_id}")

        # CRITICAL: TTTTT restarts at 00000 inside every class directory, so the raw
        # prefix is NOT a global track identifier -- it collapses ~1300 real tracks into
        # ~75 groups. The track key must be (class_id, track_id). Verified here so the
        # invariant is enforced rather than remembered.
        raw = {p.stem.split("_")[0] for p in train_images}
        scoped = {(p.parent.name, p.stem.split("_")[0]) for p in train_images}
        print(f"  raw TTTTT prefixes:        {len(raw):>5}  <- NOT globally unique")
        print(f"  (class, track) pairs:      {len(scoped):>5}  <- the real track count")
        if len(scoped) <= len(raw):
            print("  FAIL scoping by class did not increase the track count -- check layout")
            ok = False

        sizes: dict[tuple[str, str], int] = {}
        for p in train_images:
            sizes[(p.parent.name, p.stem.split("_")[0])] = (
                sizes.get((p.parent.name, p.stem.split("_")[0]), 0) + 1
            )
        counts = sorted(set(sizes.values()))
        print(f"  frames per track:          {counts} (GTSRB tracks are 30 frames)")
        if max(sizes.values()) > 30:
            print("  FAIL a track has >30 frames -- track key is wrong")
            ok = False

    if train_csvs:
        header = train_csvs[0].read_text().splitlines()[0]
        print(f"  GT csv header: {header}")
        for column in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"):
            if column not in header:
                print(f"  FAIL missing column {column} -- bbox jitter impossible")
                ok = False

    return ok


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading to {RAW_DIR}")
    archives = []
    for name, description in ARCHIVES.items():
        print(f"\n{name}  -- {description}")
        archives.append(download(name, RAW_DIR))

    print(f"\nExtracting to {DATA_DIR}")
    for archive in archives:
        extract(archive, DATA_DIR)

    checksums = DATA_DIR / "CHECKSUMS.txt"
    print("\nChecksums (recorded for reproducibility)")
    lines = []
    for archive in archives:
        digest = sha256(archive)
        print(f"  {digest}  {archive.name}")
        lines.append(f"{digest}  {archive.name}")
    checksums.write_text("\n".join(lines) + "\n")

    if not verify(DATA_DIR):
        print("\nDownload verification FAILED -- do not proceed to task 1.1.")
        return 1
    print("\nAll counts match. Task 0.2 verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
