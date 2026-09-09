"""Task 0.1 verification: prove the environment can run every stage of the project.

Each check corresponds to something that would otherwise fail *silently and late*:

- SIFT missing        -> BoVW (task 6) dies on day 3
- torch with CUDA     -> 2.5 GB of unusable NVIDIA libs, wrong wheel
- skimage.hog missing -> HOG (task 5) dies on day 2
- LinearSVC missing   -> the fixed classifier the whole design rests on

Run: poetry run python scripts/verify_env.py
Emits a markdown version table for docs/report-material/01-environment-and-setup.md.
"""

from __future__ import annotations

import platform
import sys

FAILURES: list[str] = []


def check(label: str, fn) -> str:
    """Run a check, returning a short status string; never raises."""
    try:
        return f"OK    {label}: {fn()}"
    except Exception as exc:  # noqa: BLE001 - we want every failure, not the first
        FAILURES.append(f"{label}: {exc!r}")
        return f"FAIL  {label}: {exc!r}"


def main() -> int:
    versions: dict[str, str] = {"Python": platform.python_version()}
    lines: list[str] = []

    def _sift() -> str:
        import cv2

        versions["opencv-python"] = cv2.__version__
        sift = cv2.SIFT_create()
        # Constructing is not enough -- confirm it actually descriptors something.
        import numpy as np

        img = np.zeros((48, 48), dtype=np.uint8)
        img[12:36, 12:36] = 255
        kps = [cv2.KeyPoint(x=float(x), y=float(y), size=12.0) for x in (12, 24, 36) for y in (12, 24, 36)]
        _, desc = sift.compute(img, kps)
        assert desc.shape == (9, 128), f"unexpected descriptor shape {desc.shape}"
        return f"cv2 {cv2.__version__}, dense compute -> {desc.shape}"

    def _torch() -> str:
        import torch

        versions["torch"] = torch.__version__
        assert not torch.cuda.is_available(), "CUDA reported available -- wrong wheel?"
        assert "+cu" not in torch.__version__, f"CUDA build installed: {torch.__version__}"
        x = torch.randn(2, 3) @ torch.randn(3, 2)
        return f"{torch.__version__}, cuda={torch.cuda.is_available()}, threads={torch.get_num_threads()}, matmul {tuple(x.shape)}"

    def _skimage() -> str:
        import numpy as np
        import skimage
        from skimage.feature import hog

        versions["scikit-image"] = skimage.__version__
        feat = hog(
            np.zeros((48, 48), dtype=np.float32),
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
        )
        return f"{skimage.__version__}, hog(48x48) -> dim {feat.shape[0]}"

    def _sklearn() -> str:
        import sklearn
        from sklearn.cluster import MiniBatchKMeans  # noqa: F401 - BoVW vocabulary
        from sklearn.decomposition import PCA  # noqa: F401 - task 4
        from sklearn.svm import LinearSVC  # noqa: F401 - the fixed classifier

        versions["scikit-learn"] = sklearn.__version__
        return f"{sklearn.__version__}, LinearSVC / PCA / MiniBatchKMeans importable"

    def _numpy_pandas() -> str:
        import numpy as np
        import pandas as pd

        versions["numpy"] = np.__version__
        versions["pandas"] = pd.__version__
        return f"numpy {np.__version__}, pandas {pd.__version__}"

    def _matplotlib() -> str:
        import matplotlib

        matplotlib.use("Agg")  # headless: figures are written to files, never shown
        import matplotlib.pyplot as plt  # noqa: F401

        versions["matplotlib"] = matplotlib.__version__
        return f"{matplotlib.__version__}, Agg backend"

    lines.append(check("SIFT (BoVW, task 6)", _sift))
    lines.append(check("torch CPU (task 7)", _torch))
    lines.append(check("skimage.hog (task 5)", _skimage))
    lines.append(check("sklearn (tasks 4/6, classifier)", _sklearn))
    lines.append(check("numpy/pandas", _numpy_pandas))
    lines.append(check("matplotlib (figures)", _matplotlib))

    print(f"interpreter: {sys.executable}")
    print(f"platform:    {platform.platform()}")
    print()
    for line in lines:
        print(line)

    print("\n--- markdown table for docs/report-material/01-environment-and-setup.md ---\n")
    print("| Package | Version |")
    print("|---|---|")
    for name, ver in versions.items():
        print(f"| {name} | {ver} |")

    if FAILURES:
        print(f"\n{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\nAll checks passed. Task 0.1 verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
