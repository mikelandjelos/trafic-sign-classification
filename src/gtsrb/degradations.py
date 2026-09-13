"""Task 3.x: controlled degradations applied at evaluation time.

    noisy = degradations.apply(images, "noise", sigma=20, keys=frame["path"])

Every degradation takes uint8 and returns uint8 of the same shape, so a degraded batch
drops straight into the same estimator as a clean one.

What the injected degradation represents
---------------------------------------
**It is applied to the preprocessed 48x48 model input, not to the source image.** Injected
*after* preprocessing, the perturbation is by construction **the degradation preprocessing
did not remove** -- the residual that actually reaches the representation.

That is the quantity this study compares methods on. Preprocessing is never perfect: some
noise survives denoising, some exposure error survives contrast normalisation. Whatever
gets through is what the representation must cope with, so the question answered here is
"given a degradation your pipeline failed to remove, which representation copes best?" --
which requires no claim about cameras at all.

Four properties follow, and under this framing they are requirements rather than
trade-offs:

- one well-defined meaning of each level across the whole dataset -- a residual of a stated
  magnitude is one condition, not an average over sign sizes;
- independence from sign size (degrading at source would make sigma = 20 span a measured
  2.4x range of effective strengths, entangling task 9.5 with task 9.4);
- independence from the preprocessing config, so the residual is specified directly;
- identical pixels for all five methods, which is the point of the whole design.

Out of scope, deliberately: this does **not** simulate capture-time degradation, so it
cannot measure how preprocessing *interacts* with a stressor -- CLAHE amplifying sensor
noise, or partially undoing a gamma shift applied before it. Those are real effects, named
as an extension in PROJECT_TASKS.md section 12, not a limitation of this measurement.

Determinism
-----------
Each image's noise is drawn from a stream keyed by `(degradation, level, image path)` --
the **path**, not the row position. Subsetting, reordering or evaluating a different number
of methods therefore cannot change the pixels any given image receives. See
`config.rng_for` and docs/report-material/03-reproducibility.md.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from gtsrb import config

#: Grid levels per degradation. Note gamma's identity (1.0) sits in the *middle* of its
#: range, not at the start -- it is the one stressor with two directions.
NOISE_LEVELS: tuple[float, ...] = (0, 5, 10, 20, 40)
BLUR_LEVELS: tuple[float, ...] = (0, 3, 5, 9, 15)
GAMMA_LEVELS: tuple[float, ...] = (0.4, 0.7, 1.0, 1.5, 2.5)


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

    Built by **sub-pixel sampling**, not by rasterising a line with `cv2.line`. A rasterised
    line rounds its endpoints to integer pixels, which makes the blur's physical extent
    depend on the angle -- and not by a constant factor:

        size  0 deg extent   45 deg extent
          3       2.00           2.83   (diagonal 41 % LONGER)
          5       4.00           2.83   (diagonal 29 % SHORTER)
         15      14.00          14.14   (near-equal by luck)

    Since the angle is drawn at random per image, that would give images at the same
    nominal level materially different blur strengths, breaking the "one level means one
    condition" property the whole degradation design rests on.

    Sampling `size` points evenly along a line of length `size - 1` at the requested angle,
    and splatting each bilinearly, keeps the extent exactly `size - 1` at every angle. It
    also anti-aliases the kernel, which is closer to real motion blur than a hard-rounded
    line anyway.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError(f"kernel size must be odd and >= 1, got {size}")
    kernel = np.zeros((size, size), dtype=np.float32)
    if size == 1:
        kernel[0, 0] = 1.0
        return kernel

    centre = (size - 1) / 2.0
    radians = np.deg2rad(angle_deg)
    offsets = np.linspace(-centre, centre, size)
    xs = centre + offsets * np.cos(radians)
    ys = centre + offsets * np.sin(radians)

    for x, y in zip(xs, ys, strict=True):
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for dy in (0, 1):
            for dx in (0, 1):
                xi, yi = x0 + dx, y0 + dy
                if 0 <= xi < size and 0 <= yi < size:
                    weight = (fx if dx else 1.0 - fx) * (fy if dy else 1.0 - fy)
                    kernel[yi, xi] += weight

    # Drop numerical dust: cos(90 deg) is 6.1e-17 rather than 0, which otherwise splats a
    # vanishing weight into a neighbouring column and leaves an axis-aligned kernel
    # looking two pixels wide.
    kernel[kernel < 1e-6] = 0.0
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


@lru_cache(maxsize=16)
def _gamma_lut(gamma: float) -> np.ndarray:
    """256-entry lookup table for a gamma curve.

    Memoised because the table depends only on gamma, not on the image -- rebuilding it
    for each of 12 630 images would dominate the cost of an otherwise trivial operation.
    """
    scaled = (np.arange(256, dtype=np.float64) / 255.0) ** gamma
    return np.rint(np.clip(scaled * 255.0, 0, 255)).astype(np.uint8)


def gamma_correction(
    image: np.ndarray, gamma: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Apply `out = 255 * (in/255) ** gamma`.

    gamma < 1 brightens and lifts shadows; gamma > 1 darkens. gamma = 1 is the identity.

    **This is the one degradation with no randomness at all** -- the mapping is a fixed
    function of the pixel value, so `rng` is accepted for interface uniformity and ignored.
    Nothing about it depends on the seed, which also means it is the one stressor where
    the "all methods see identical pixels" guarantee is trivially satisfied.

    Implemented as a lookup table rather than per-pixel arithmetic: the transform has only
    256 possible inputs, so the table is exact and the operation is a memory lookup.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if gamma == 1.0:
        return image
    return cv2.LUT(image, _gamma_lut(float(gamma)))


@dataclass(frozen=True)
class Degradation:
    """One stressor: how to apply it, its grid levels, and which level is the identity."""

    name: str
    function: Callable[..., np.ndarray]
    levels: tuple[float, ...]
    identity: float
    stochastic: bool
    description: str


#: Registry of the controlled degradations (task 3.x).
_DEGRADATIONS: dict[str, Degradation] = {
    "noise": Degradation(
        name="noise",
        function=gaussian_noise,
        levels=NOISE_LEVELS,
        identity=0,
        stochastic=True,
        description="additive zero-mean Gaussian noise, sigma in 0-255 units",
    ),
    "blur": Degradation(
        name="blur",
        function=motion_blur,
        levels=BLUR_LEVELS,
        identity=0,
        stochastic=True,
        description="linear motion blur, kernel length in px at a random angle",
    ),
    "gamma": Degradation(
        name="gamma",
        function=gamma_correction,
        levels=GAMMA_LEVELS,
        identity=1.0,
        stochastic=False,
        description="gamma correction; <1 brightens, >1 darkens, 1.0 is the identity",
    ),
}


def get_degradation(degradation: str) -> Degradation:
    if degradation not in _DEGRADATIONS:
        raise KeyError(
            f"unknown degradation {degradation!r}; available: {sorted(_DEGRADATIONS)}"
        )
    return _DEGRADATIONS[degradation]


def levels_for(degradation: str) -> tuple[float, ...]:
    return get_degradation(degradation).levels


def identity_for(degradation: str) -> float:
    """The level at which the degradation is a no-op.

    Not simply `levels[0]`: gamma's identity is 1.0, in the middle of its range, because
    it perturbs in two directions. Anything iterating the grid must ask rather than assume.
    """
    return get_degradation(degradation).identity


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
    spec = get_degradation(degradation)
    if len(keys) != len(images):
        raise ValueError(f"keys has {len(keys)} entries for {len(images)} images")

    if level not in spec.levels:
        raise ValueError(
            f"{degradation} level {level} is not one of {spec.levels}; the grid is "
            f"defined over these levels and off-grid values would not be comparable"
        )
    if level == spec.identity:
        # Returned unchanged rather than recomputed, so the identity level and the clean
        # condition are the same array. Note this is `spec.identity`, not `levels[0]` --
        # gamma's identity is 1.0, in the middle of its range.
        return images

    out = np.empty_like(images)
    for i, key in enumerate(keys):
        rng = config.rng_for(degradation, level, key) if spec.stochastic else None
        out[i] = spec.function(images[i], level, rng)
    return out
