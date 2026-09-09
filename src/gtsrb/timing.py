"""Task 1.5: the timing harness -- train wall-clock and inference ms/img.

    env = timing.platform_info()          # what the numbers were measured on
    fit = timing.time_training(lambda: model.fit(X, y))
    inf = timing.time_inference(lambda: model.predict(X), n_items=len(X))

What these numbers mean, and what they do not
---------------------------------------------
They are **relative costs measured in one documented environment**, not deployment
latency. Two limits bound the claim, and both belong in the report rather than in a
footnote:

1. *Implementation, not algorithm.* This measures Python/NumPy/scikit-learn/PyTorch on a
   general-purpose desktop OS, with other processes running. A production traffic-sign
   recogniser would be ported to the target platform -- compiled, probably fixed-point, on
   an automotive SoC, embedded ARM, DSP or NPU. Timings there can differ by orders of
   magnitude, and not by a constant factor: HOG and BoVW vectorise very differently from a
   CNN on hardware with SIMD or a neural accelerator, so the *ranking* itself may change.
   Nothing here should be read as evidence about what is fast on an ECU.

2. *No usable GPU here.* The CNN is timed on CPU. On any device with a neural accelerator
   its relative position would improve substantially -- arguably more than any other
   method's.

What the numbers do support is the comparison the project is actually making: with the
environment held fixed and recorded, the cost differences between representations are
real, and orders-of-magnitude gaps (say PCA vs. dense-SIFT BoVW) are robust to the
implementation caveat above in a way that 10-20 % differences are not.

Measurement protocol
--------------------
Median of >= 3 repeats, discarding the first. The first call carries lazy imports, BLAS
thread-pool spin-up, memory allocation and cache warmup, and is routinely several times
slower -- including it would measure startup rather than steady-state cost. The spread
(IQR) is reported too: a wide spread means the machine was busy and the number should not
be trusted.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from gtsrb import config

# --- what the measurement ran on -------------------------------------------------------


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
        match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _total_ram_gb() -> float | None:
    try:
        text = Path("/proc/meminfo").read_text()
        match = re.search(r"^MemTotal:\s*(\d+) kB$", text, re.MULTILINE)
        if match:
            return round(int(match.group(1)) / 1024**2, 1)
    except OSError:
        pass
    return None


def git_commit() -> str | None:
    """The commit the measurement was taken at -- makes a timing row traceable to code."""
    try:
        result = subprocess.run(
            ["git", "-C", str(config.PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "sklearn", "cv2", "skimage", "torch"):
        try:
            versions[name] = __import__(name).__version__
        except (ImportError, AttributeError):
            versions[name] = "not installed"
    return versions


def platform_info() -> dict:
    """Everything needed to interpret a timing number. Written beside results.csv.

    `blas` and `threads_pinned` are captured to prove the thread pinning from
    `config.set_seeds()` was actually in effect -- an unpinned run oversubscribes 16 SMT
    threads and produces numbers that cannot be compared with pinned ones.
    """
    try:
        from threadpoolctl import threadpool_info

        blas = [
            {"library": p.get("internal_api"), "threads": p.get("num_threads")}
            for p in threadpool_info()
        ]
    except Exception:  # noqa: BLE001
        blas = []

    torch_threads = None
    if "torch" in sys.modules:
        torch_threads = sys.modules["torch"].get_num_threads()

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(),
        "cpu": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "ram_gb": _total_ram_gb(),
        "gpu": "none (AMD Vega iGPU, no CUDA/ROCm support) -- all timings are CPU",
        "os": platform.platform(),
        "threads_configured": config.NUM_THREADS,
        "threads_torch": torch_threads,
        "blas": blas,
        "load_average": list(os.getloadavg()),
        "versions": _library_versions(),
    }


def save_platform_info(path: Path | None = None) -> Path:
    """Write platform_info() as JSON beside results.csv."""
    path = path or config.RESULTS_DIR / "platform.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(platform_info(), indent=2) + "\n")
    return path


# --- measurement -----------------------------------------------------------------------


@dataclass(frozen=True)
class TimingResult:
    """Timing for one operation. `ms_per_item` is None when n_items was not given."""

    label: str
    median_s: float
    min_s: float
    iqr_s: float
    n_items: int | None
    repeats: int
    discarded_first_s: float | None
    runs_s: list[float] = field(default_factory=list)

    @property
    def ms_per_item(self) -> float | None:
        if not self.n_items:
            return None
        return self.median_s * 1000.0 / self.n_items

    @property
    def warmup_ratio(self) -> float | None:
        """How much slower the discarded first call was. >2 justifies discarding it."""
        if self.discarded_first_s is None or self.median_s == 0:
            return None
        return self.discarded_first_s / self.median_s

    def summary(self) -> str:
        parts = [f"{self.label}: median {self.median_s:.4f}s"]
        if self.ms_per_item is not None:
            parts.append(f"{self.ms_per_item:.4f} ms/img")
        parts.append(f"IQR {self.iqr_s:.4f}s over {self.repeats} runs")
        if self.warmup_ratio is not None:
            parts.append(f"first call {self.warmup_ratio:.1f}x slower (discarded)")
        return ", ".join(parts)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["ms_per_item"] = self.ms_per_item
        data["warmup_ratio"] = self.warmup_ratio
        return data


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def measure(
    fn: Callable[[], object],
    label: str = "operation",
    repeats: int = 5,
    n_items: int | None = None,
    warmup: bool = True,
) -> TimingResult:
    """Time `fn`, discarding a warmup call, and report the median of `repeats` runs.

    `time.perf_counter` is used rather than `time.time`: it is monotonic and unaffected by
    system clock adjustments during a long run.
    """
    if repeats < 3:
        raise ValueError(f"repeats must be >= 3 for a meaningful median, got {repeats}")

    discarded = None
    if warmup:
        start = time.perf_counter()
        fn()
        discarded = time.perf_counter() - start

    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - start)

    return TimingResult(
        label=label,
        median_s=_percentile(runs, 0.5),
        min_s=min(runs),
        iqr_s=_percentile(runs, 0.75) - _percentile(runs, 0.25),
        n_items=n_items,
        repeats=repeats,
        discarded_first_s=discarded,
        runs_s=runs,
    )


def time_training(fn: Callable[[], object], label: str = "train") -> TimingResult:
    """Wall-clock for a single training run.

    Training is measured **once**, with no warmup: repeating it would measure refitting a
    warm process rather than the cost of training the model, and for the CNN it would be
    prohibitively slow. The single-run caveat is why training time is reported to two
    significant figures at most.
    """
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    return TimingResult(
        label=label,
        median_s=elapsed,
        min_s=elapsed,
        iqr_s=0.0,
        n_items=None,
        repeats=1,
        discarded_first_s=None,
        runs_s=[elapsed],
    )


def time_inference(
    fn: Callable[[], object], n_items: int, label: str = "inference", repeats: int = 5
) -> TimingResult:
    """Inference cost, reported as ms/img. Median of `repeats`, first call discarded."""
    return measure(fn, label=label, repeats=repeats, n_items=n_items, warmup=True)


if __name__ == "__main__":
    config.set_seeds()
    print(json.dumps(platform_info(), indent=2))
