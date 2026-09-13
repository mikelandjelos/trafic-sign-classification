"""Task 2.1: preprocessing primitives shared by every representation.

    bgr  = preprocessing.load_image(path)
    gray = preprocessing.resize(preprocessing.to_gray(bgr))

All functions take and return **uint8** arrays, so intermediate results can be cached
directly as uint8 `.npy` (task 2.3) without a lossy round-trip through float.

Colour convention: `cv2.imread` returns **BGR**, not RGB. Every conversion here starts
from BGR. Matplotlib expects RGB, so figures must convert before displaying -- otherwise
red prohibition signs render blue and nobody notices until the report is printed.

Two decisions worth stating
---------------------------
1. *Interpolation is chosen per image, not fixed.* GTSRB crops range from 25 px to 266 px
   against a 48x48 target, so the same call both upscales and downscales. `INTER_AREA` is
   correct for shrinking (it averages over the source footprint and avoids aliasing);
   `INTER_LINEAR` is correct for enlarging, where `INTER_AREA` degenerates to something
   close to nearest-neighbour. Using one setting for both would systematically damage one
   end of the size range -- and since accuracy-vs-size (task 9.4) is a headline figure,
   that damage would be misread as a property of the representations.

2. *CLAHE runs after the resize, not before.* Applied at native resolution, a fixed tile
   grid covers 3 px tiles on a 25 px crop and 33 px tiles on a 266 px one -- the operation
   would mean something different for every image, confounding the preprocessing ablation
   (task 8.2) with sign size. Applied at 48x48 it is identical for every sample.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gtsrb import config

#: CLAHE tile grid for 48x48 inputs. The OpenCV default of (8, 8) gives 6x6 px tiles,
#: small enough to amplify sensor noise into structure; (4, 4) gives 12x12 px tiles.
CLAHE_TILE_GRID = (4, 4)
CLAHE_CLIP_LIMIT = 2.0


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR uint8. Handles GTSRB's P6 PPM natively."""
    path = Path(path)
    if not path.is_absolute():
        path = config.PROJECT_ROOT / path
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    """BGR -> single-channel grayscale uint8."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_hsv(image: np.ndarray) -> np.ndarray:
    """BGR -> HSV uint8.

    Motivated by the data: GTSRB classes are strongly colour-coded (red prohibition, blue
    mandatory, yellow priority), and HSV separates that hue information from the
    illumination changes that dominate the raw intensities.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected a 3-channel BGR image, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: tuple[int, int] = CLAHE_TILE_GRID,
) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation.

    Grayscale input is equalised directly. **Three-channel input is treated as HSV** and
    equalised on V only -- equalising H would rotate hues and destroy exactly the colour
    information that makes traffic signs separable.

    GTSRB contains heavily under- and over-exposed frames (the tracks are captured while
    driving, through changing light), so local contrast normalisation is the one
    preprocessing step with an obvious physical motivation here.
    """
    operator = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    if image.ndim == 2:
        return operator.apply(image)
    if image.ndim == 3 and image.shape[2] == 3:
        out = image.copy()
        out[:, :, 2] = operator.apply(out[:, :, 2])
        return out
    raise ValueError(f"unsupported image shape for CLAHE: {image.shape}")


def gaussian_blur(image: np.ndarray, ksize: int = 3, sigma: float = 0.0) -> np.ndarray:
    """Gaussian smoothing. `sigma=0` lets OpenCV derive sigma from the kernel size.

    This is the *preprocessing* blur (mild denoising). The motion blur used as a stressor
    in task 3.2 is a different operation with a different purpose -- do not conflate them.
    """
    if ksize <= 0:
        return image
    if ksize % 2 == 0:
        raise ValueError(f"gaussian kernel size must be odd, got {ksize}")
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def resize(
    image: np.ndarray,
    size: tuple[int, int] = config.IMAGE_SIZE,
    interpolation: int | None = None,
) -> np.ndarray:
    """Resize to (height, width), picking the interpolation appropriate to the direction.

    Pass `interpolation` explicitly to override. See the module docstring for why the
    adaptive default matters for the accuracy-vs-size analysis.
    """
    target_h, target_w = size
    source_h, source_w = image.shape[:2]
    if interpolation is None:
        shrinking = target_h * target_w < source_h * source_w
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    # cv2.resize takes (width, height), the opposite of numpy's (rows, cols).
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def _pipeline_raw_gray(image: np.ndarray) -> np.ndarray:
    return resize(to_gray(image))


def _pipeline_clahe_gray(image: np.ndarray) -> np.ndarray:
    return clahe(resize(to_gray(image)))


def _pipeline_clahe_hsv(image: np.ndarray) -> np.ndarray:
    return clahe(resize(to_hsv(image)))


@dataclass(frozen=True)
class PreprocConfig:
    """One named preprocessing pipeline for the ablation (task 8.2)."""

    name: str
    description: str
    channels: int
    _apply: Callable[[np.ndarray], np.ndarray]

    def apply(self, image: np.ndarray) -> np.ndarray:
        """BGR crop -> preprocessed 48x48 uint8 array."""
        return self._apply(image)

    @property
    def shape(self) -> tuple[int, ...]:
        h, w = config.IMAGE_SIZE
        return (h, w) if self.channels == 1 else (h, w, self.channels)

    @property
    def flat_dim(self) -> int:
        """Dimensionality after flattening -- the input width PCA sees."""
        h, w = config.IMAGE_SIZE
        return h * w * self.channels


#: The three configs compared in the preprocessing ablation (task 2.2 / 8.2).
PREPROC_CONFIGS: dict[str, PreprocConfig] = {
    "raw_gray": PreprocConfig(
        name="raw_gray",
        description="grayscale, resized only -- the baseline, no contrast normalisation",
        channels=1,
        _apply=_pipeline_raw_gray,
    ),
    "clahe_gray": PreprocConfig(
        name="clahe_gray",
        description="grayscale + CLAHE -- isolates the effect of contrast normalisation",
        channels=1,
        _apply=_pipeline_clahe_gray,
    ),
    "clahe_hsv": PreprocConfig(
        name="clahe_hsv",
        description="HSV + CLAHE on V -- adds colour, which GTSRB classes are coded by",
        channels=3,
        _apply=_pipeline_clahe_hsv,
    ),
}

DEFAULT_PREPROC = "clahe_gray"


def get_config(name: str) -> PreprocConfig:
    if name not in PREPROC_CONFIGS:
        raise KeyError(f"unknown preproc config {name!r}; available: {sorted(PREPROC_CONFIGS)}")
    return PREPROC_CONFIGS[name]


def preprocess(image: np.ndarray, preproc: str = DEFAULT_PREPROC) -> np.ndarray:
    """Apply a named pipeline to a BGR crop."""
    return get_config(preproc).apply(image)


def load_and_preprocess(row, preproc: str = DEFAULT_PREPROC, roi: bool = False) -> np.ndarray:
    """Load one annotated image and run it through a named pipeline.

    `row` is a row of the annotations dataframe (needs `path` and, when `roi=True`, the
    `roi_*` columns).

    **GTSRB images are used with the framing the dataset provides** -- the sign plus its
    ~17 % margin -- so `roi=False` is the default. Only the pixels are processed
    (colour conversion, CLAHE, resize); the box is left alone.

    An earlier revision cropped to the tight ROI so that the clean condition would coincide
    with bbox jitter at level 0. That reason no longer exists: jitter was moved out of the
    core grid onto full-frame datasets (see docs/report-material/09-jitter-and-datasets.md),
    because GTSRB's pre-cropped images cannot support outward jitter without inventing
    pixels. Using the provided framing is also how GTSRB is conventionally used, which keeps
    the headline numbers comparable with published results.

    `roi=True` remains available: it is how the GTSDB evaluation reproduces GTSRB's framing
    (annotated box plus an equivalent margin) so that jitter level 0 is a like-for-like
    baseline rather than a framing mismatch.
    """
    image = load_image(row["path"])
    if roi:
        image = crop_roi(image, row["roi_x1"], row["roi_y1"], row["roi_x2"], row["roi_y2"])
    return preprocess(image, preproc)


def crop_roi(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop to the annotated ROI, clamped to the image bounds.

    Clamping is silent here by design; task 3.4 measures and reports how often bbox jitter
    actually pushes a box out of bounds, which is a separate question.
    """
    height, width = image.shape[:2]
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    return image[y1:y2, x1:x2]
