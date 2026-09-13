"""Task 3.x: controlled degradations applied at evaluation time.

    noisy = degradations.apply(images, "noise", sigma=20, keys=frame["path"])

Every degradation takes uint8 and returns uint8 of the same shape, so a degraded batch
drops straight into the same estimator as a clean one.

Where in the pipeline the degradation is applied
------------------------------------------------
**On the preprocessed 48x48 model input, not on the source image.** This is a deliberate
choice of *control over physical realism*, and it needs stating in the report because the
opposite choice is equally defensible.

Real sensor noise enters at capture, before any resizing -- so a physically faithful
simulation would add noise to the source crop and then downsample. The problem is that
GTSRB crops span 25-266 px against a 48x48 target: downsampling a 200 px crop averages
noise away, while a 25 px crop is upsampled and keeps all of it. The *same* sigma would
then mean a different effective noise level depending on sign size, silently coupling the
noise axis to the size axis (task 9.4) and making "sigma = 20" an ambiguous label.

Applying the degradation to the model input instead gives:

- one well-defined meaning of each level across the whole dataset;
- independence from sign size, so the robustness curves and the size analysis stay
  separable rather than confounded;
- independence from the preprocessing config, so the ablation (task 8.2) does not
  interact with the degradation axis;
- identical pixels for all five methods, which is the point of the whole design.

The cost is that these are *controlled perturbations of the classifier's input*, not
simulations of a camera. The report should say so plainly rather than implying the latter.

Determinism
-----------
Each image's noise is drawn from a stream keyed by `(degradation, level, image path)` --
the **path**, not the row position. Subsetting, reordering or evaluating a different number
of methods therefore cannot change the pixels any given image receives. See
`config.rng_for` and docs/report-material/03-reproducibility.md.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np

from gtsrb import config

#: Levels for each degradation. Index 0 is always the identity (the clean condition).
NOISE_LEVELS: tuple[float, ...] = (0, 5, 10, 20, 40)
BLUR_LEVELS: tuple[float, ...] = (0, 3, 5, 9, 15)


def gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise of standard deviation `sigma` (in 0-255 units).

    `sigma = 0` returns the input unchanged, so level 0 is exactly the clean condition
    rather than merely close to it.

    Values are clipped to [0, 255] before casting back to uint8. The clipping is a real
    effect, not an implementation detail: at sigma = 40 a bright sign face genuinely
    saturates, which is what an over-exposed sensor does.

    The result is **rounded** rather than truncated. `astype(np.uint8)` truncates toward
    zero, which would bias every perturbed pixel down by ~0.5 levels and make the
    degradation darken the image slightly at every sigma -- a systematic shift on top of
    the zero-mean noise it is supposed to apply.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if sigma == 0:
        return image
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, size=image.shape)
    return np.rint(np.clip(noisy, 0, 255)).astype(np.uint8)


def motion_blur_kernel(size: int, angle_deg: float) -> np.ndarray:
    """A normalised line kernel of `size` px at `angle_deg`, for linear motion blur.

    The kernel sums to 1, so the operation preserves mean intensity: the blur must not
    also darken or brighten the image, or the degradation would be confounded with a
    brightness shift (which is what task 3.3 measures separately).
    """
    if size < 1 or size % 2 == 0:
        raise ValueError(f"kernel size must be odd and >= 1, got {size}")
    kernel = np.zeros((size, size), dtype=np.float32)
    centre = (size - 1) / 2.0
    radians = np.deg2rad(angle_deg)
    dx, dy = np.cos(radians) * centre, np.sin(radians) * centre
    start = (round(float(centre - dx)), round(float(centre - dy)))
    end = (round(float(centre + dx)), round(float(centre + dy)))
    cv2.line(kernel, start, end, color=1.0, thickness=1)
    return kernel / kernel.sum()


def motion_blur(image: np.ndarray, ksize: int, rng: np.random.Generator) -> np.ndarray:
    """Linear motion blur with a `ksize`-px kernel at a uniformly random angle.

    `ksize = 0` returns the input unchanged.

    The angle is drawn per image from the supplied stream, so it is reproducible from
    `(degradation, level, path)` like everything else. It is sampled over **[0, 180)**
    rather than [0, 360): a line kernel at theta and at theta + 180 is the same kernel, so
    the wider range would merely sample each orientation twice.

    Borders use `BORDER_REFLECT_101`. The default alternative, zero padding, would darken
    every edge in proportion to the kernel size -- a systematic vignette that grows with
    the degradation level and would be measured as part of the blur's effect.
    """
    if ksize == 0:
        return image
    angle = float(rng.uniform(0.0, 180.0))
    kernel = motion_blur_kernel(int(ksize), angle)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


#: Registry of degradation name -> (per-image function, levels).
_DEGRADATIONS: dict[str, tuple[Callable[..., np.ndarray], tuple[float, ...]]] = {
    "noise": (gaussian_noise, NOISE_LEVELS),
    "blur": (motion_blur, BLUR_LEVELS),
}


def levels_for(degradation: str) -> tuple[float, ...]:
    if degradation not in _DEGRADATIONS:
        raise KeyError(
            f"unknown degradation {degradation!r}; available: {sorted(_DEGRADATIONS)}"
        )
    return _DEGRADATIONS[degradation][1]


def apply(
    images: np.ndarray,
    degradation: str,
    level: float,
    keys: Sequence[str],
) -> np.ndarray:
    """Apply `degradation` at `level` to a batch of preprocessed images.

    `keys` must be stable per-image identifiers -- the annotation `path` column. They seed
    each image's noise independently of its position in the batch, so every method in the
    grid sees byte-identical degraded pixels regardless of evaluation order.
    """
    if degradation == "clean":
        return images
    if degradation not in _DEGRADATIONS:
        raise KeyError(
            f"unknown degradation {degradation!r}; available: "
            f"{['clean', *sorted(_DEGRADATIONS)]}"
        )
    if len(keys) != len(images):
        raise ValueError(f"keys has {len(keys)} entries for {len(images)} images")

    function, valid_levels = _DEGRADATIONS[degradation]
    if level not in valid_levels:
        raise ValueError(
            f"{degradation} level {level} is not one of {valid_levels}; the grid is "
            f"defined over these levels and off-grid values would not be comparable"
        )
    if level == valid_levels[0]:
        return images  # identity level -- returned unchanged, not merely recomputed

    out = np.empty_like(images)
    for i, key in enumerate(keys):
        rng = config.rng_for(degradation, level, key)
        out[i] = function(images[i], level, rng)
    return out
