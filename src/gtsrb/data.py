"""Task 1.1: parse the GTSRB annotation CSVs into one tidy dataframe.

    from gtsrb import data
    train = data.load_annotations("train")   # 39 209 rows
    test  = data.load_annotations("test")    # 12 630 rows

Columns
-------
path        str   image location, relative to the project root (POSIX separators)
split       str   "train" or "test"
class_id    int   0..42
track_id    str   "CCCCC_TTTTT" for train, None for test -- see the warning below
frame_id    int   frame within the track, train only (-1 for test)
width       int   source image width
height      int   source image height
roi_x1/y1/x2/y2  int   annotated sign bounding box within the image
roi_w       int   roi_x2 - roi_x1
roi_h       int   roi_y2 - roi_y1   <- the size bucket key for task 9.4

The track_id trap
-----------------
GTSRB filenames are `TTTTT_FFFFF.ppm`, but `TTTTT` restarts at 00000 inside every class
directory -- 39 209 images give only 75 distinct raw prefixes against 1 307 real tracks.
`track_id` here is therefore the *composite* key `f"{class_id:05d}_{track:05d}"`, so it is
globally unique by construction and any later groupby is correct by default. See
docs/report-material/02-dataset-structure.md.

Test images carry no track information -- they are single frames, not tracks -- so
`track_id` is None for the test split. That is correct rather than missing data: the test
set is used only for final reporting, and all model selection happens on the
track-disjoint validation split carved out of the training data.
"""

from __future__ import annotations

import pandas as pd

from gtsrb import config

#: Column order of the returned dataframe.
COLUMNS = [
    "path",
    "split",
    "class_id",
    "track_id",
    "frame_id",
    "width",
    "height",
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "roi_w",
    "roi_h",
]

_CSV_RENAME = {
    "Width": "width",
    "Height": "height",
    "Roi.X1": "roi_x1",
    "Roi.Y1": "roi_y1",
    "Roi.X2": "roi_x2",
    "Roi.Y2": "roi_y2",
    "ClassId": "class_id",
}

TRACK_FRAMES = 30  # every track is 30 frames, except one 29-frame track in class 33


def _read_gt_csv(path):
    """Read one GTSRB annotation CSV.

    Semicolon-separated with CRLF line endings (the files are from 2011); pandas handles
    both, but the separator is not the default and silently yields a single column if
    omitted.
    """
    frame = pd.read_csv(path, sep=";")
    missing = set(_CSV_RENAME) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing)}")
    return frame.rename(columns=_CSV_RENAME)


def _load_train() -> pd.DataFrame:
    class_dirs = sorted(d for d in config.TRAIN_IMAGES_DIR.iterdir() if d.is_dir())
    if len(class_dirs) != config.N_CLASSES:
        raise ValueError(f"expected {config.N_CLASSES} class dirs, found {len(class_dirs)}")

    parts = []
    for class_dir in class_dirs:
        csv_path = class_dir / f"GT-{class_dir.name}.csv"
        frame = _read_gt_csv(csv_path)

        # The class is encoded twice -- in the directory name and in ClassId. Cross-check
        # them rather than trusting either.
        dir_class = int(class_dir.name)
        if not (frame["class_id"] == dir_class).all():
            raise ValueError(f"{csv_path}: ClassId disagrees with directory {dir_class}")

        stems = frame["Filename"].str.removesuffix(".ppm").str.split("_", expand=True)
        frame["track_id"] = f"{dir_class:05d}_" + stems[0]
        frame["frame_id"] = stems[1].astype(int)
        frame["path"] = frame["Filename"].map(
            lambda name, d=class_dir: (d / name).relative_to(config.PROJECT_ROOT).as_posix()
        )
        parts.append(frame)

    return pd.concat(parts, ignore_index=True)


def _load_test() -> pd.DataFrame:
    if not config.TEST_GT_CSV.exists():
        raise FileNotFoundError(
            f"{config.TEST_GT_CSV} not found. Run scripts/download_data.py -- note the "
            f"labels ship in a separate archive from the test images."
        )

    frame = _read_gt_csv(config.TEST_GT_CSV)
    frame["track_id"] = None  # test images are single frames, not tracks
    frame["frame_id"] = -1
    frame["path"] = frame["Filename"].map(
        lambda name: (config.TEST_IMAGES_DIR / name).relative_to(config.PROJECT_ROOT).as_posix()
    )
    return frame


def _validate(frame: pd.DataFrame, split: str) -> None:
    """Fail loudly on anything that would corrupt a downstream stage silently."""
    expected_rows = config.N_TRAIN_IMAGES if split == "train" else config.N_TEST_IMAGES
    if len(frame) != expected_rows:
        raise ValueError(f"{split}: expected {expected_rows} rows, got {len(frame)}")

    if not frame["class_id"].between(0, config.N_CLASSES - 1).all():
        raise ValueError(f"{split}: class_id outside 0..{config.N_CLASSES - 1}")

    if frame["path"].duplicated().any():
        raise ValueError(f"{split}: duplicate image paths")

    # A ROI that escapes its image would break bbox jitter (task 3.4) by producing crops
    # that are partly out of bounds.
    out_of_bounds = (
        (frame["roi_x1"] < 0)
        | (frame["roi_y1"] < 0)
        | (frame["roi_x2"] > frame["width"])
        | (frame["roi_y2"] > frame["height"])
        | (frame["roi_x2"] <= frame["roi_x1"])
        | (frame["roi_y2"] <= frame["roi_y1"])
    )
    if out_of_bounds.any():
        raise ValueError(f"{split}: {int(out_of_bounds.sum())} ROIs outside image bounds")

    if split == "train":
        if frame["track_id"].isna().any():
            raise ValueError("train: missing track_id")
        n_tracks = frame["track_id"].nunique()
        if n_tracks != config.N_TRACKS:
            raise ValueError(f"train: expected {config.N_TRACKS} tracks, got {n_tracks}")

        # Guards against the composite key being built wrong: if track_id were the raw
        # prefix, tracks would contain up to 43x30 frames.
        largest = frame.groupby("track_id").size().max()
        if largest > TRACK_FRAMES:
            raise ValueError(f"train: a track has {largest} frames (> {TRACK_FRAMES})")

        # Each track must belong to exactly one class, or the split cannot be stratified.
        per_track_classes = frame.groupby("track_id")["class_id"].nunique().max()
        if per_track_classes != 1:
            raise ValueError("train: a track spans multiple classes")


def load_annotations(split: str = "train", validate: bool = True) -> pd.DataFrame:
    """Load the annotations for `split` ("train" or "test") as a tidy dataframe."""
    if split == "train":
        frame = _load_train()
    elif split == "test":
        frame = _load_test()
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    frame["split"] = split
    frame["roi_w"] = frame["roi_x2"] - frame["roi_x1"]
    frame["roi_h"] = frame["roi_y2"] - frame["roi_y1"]
    frame = frame[COLUMNS].sort_values("path", ignore_index=True)

    if validate:
        _validate(frame, split)
    return frame


def summarise(frame: pd.DataFrame) -> str:
    """Human-readable summary, for logs and the report's data section."""
    split = frame["split"].iloc[0]
    lines = [
        f"split         : {split}",
        f"images        : {len(frame)}",
        f"classes       : {frame['class_id'].nunique()}",
    ]
    if frame["track_id"].notna().any():
        sizes = frame.groupby("track_id").size()
        distinct = sorted(int(v) for v in sizes.unique())
        lines.append(f"tracks        : {len(sizes)} (frames/track: {distinct})")
    counts = frame["class_id"].value_counts()
    lines += [
        (
            f"class balance : min {counts.min()} (class {counts.idxmin()}), "
            f"max {counts.max()} (class {counts.idxmax()}), "
            f"ratio {counts.max() / counts.min():.1f}x"
        ),
        (
            f"image size    : {frame['width'].min()}-{frame['width'].max()} w, "
            f"{frame['height'].min()}-{frame['height'].max()} h"
        ),
        (
            f"roi height    : min {frame['roi_h'].min()}, "
            f"median {int(frame['roi_h'].median())}, max {frame['roi_h'].max()}"
        ),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    for name in ("train", "test"):
        print(summarise(load_annotations(name)))
        print()
