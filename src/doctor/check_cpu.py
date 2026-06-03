"""Check CPU core count and available RAM."""
from __future__ import annotations

import os

from src.doctor.runner import CheckResult

MIN_CORES = 4
MIN_RAM_GB = 16.0
WARN_RAM_GB = 32.0


def check_cpu(_scenario) -> CheckResult:
    """Report physical CPU cores and logical threads. Warn if below minimum for vLLM."""
    try:
        physical = _physical_cores()
        logical  = os.cpu_count() or 1

        if physical < MIN_CORES:
            return CheckResult(
                name="CPU cores",
                passed=False,
                message=f"{physical} physical cores, {logical} threads — need ≥{MIN_CORES} cores",
                detail="vLLM tokenisation and scheduling are CPU-bound. Pick a host with more cores.",
            )
        return CheckResult(
            name="CPU cores",
            passed=True,
            message=f"{physical} physical cores, {logical} threads",
        )
    except Exception as exc:
        return CheckResult(name="CPU cores", passed=False, message=str(exc))


def check_ram(_scenario) -> CheckResult:
    """Report total and available RAM. Warn if below recommended for vLLM."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb     = mem.total     / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        used_pct     = mem.percent

        msg = f"{available_gb:.1f} GB free / {total_gb:.1f} GB total ({used_pct:.0f}% used)"

        if total_gb < MIN_RAM_GB:
            return CheckResult(
                name="RAM",
                passed=False,
                message=msg,
                detail=f"Need ≥{MIN_RAM_GB:.0f} GB RAM. vLLM uses system RAM for CPU tensors and KV-cache management.",
            )
        if total_gb < WARN_RAM_GB:
            return CheckResult(
                name="RAM",
                passed=False,
                message=msg,
                severity="warning",
                detail=f"≥{WARN_RAM_GB:.0f} GB RAM recommended for stable performance under load.",
            )
        return CheckResult(name="RAM", passed=True, message=msg)

    except ImportError:
        return CheckResult(
            name="RAM",
            passed=False,
            message="psutil not installed",
            detail="Run: pip install psutil",
        )
    except Exception as exc:
        return CheckResult(name="RAM", passed=False, message=str(exc))


def _physical_cores() -> int:
    """Return physical core count (not hyperthreaded threads)."""
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
        return cores if cores else os.cpu_count() or 1
    except ImportError:
        pass
    # Fallback: parse /proc/cpuinfo on Linux
    try:
        ids: set[str] = set()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("core id"):
                    ids.add(line.split(":")[1].strip())
        return len(ids) or (os.cpu_count() or 1)
    except OSError:
        return os.cpu_count() or 1
