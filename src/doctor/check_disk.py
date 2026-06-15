"""Check available disk space and per-GPU VRAM."""
from __future__ import annotations

import shutil

from src.doctor.runner import CheckResult

_DISK_HEADROOM_GB = 10  # overhead beyond model size for KV cache swap, etc.


def check_disk(scenario) -> CheckResult:
    """Verify enough disk is free: model size + headroom."""
    try:
        import os
        min_disk_gb = scenario.estimated_vram_gb() + _DISK_HEADROOM_GB
        cache_dir = os.getenv("HF_HOME", "/root/.cache/huggingface")
        # Walk up to find a mounted path
        check_path = cache_dir
        import pathlib
        while check_path and not pathlib.Path(check_path).exists():
            check_path = str(pathlib.Path(check_path).parent)
        check_path = check_path or "/"

        usage = shutil.disk_usage(check_path)
        free_gb = usage.free / (1024 ** 3)

        if free_gb < min_disk_gb:
            return CheckResult(
                name="Disk space",
                passed=False,
                message=f"{free_gb:.0f} GB free — need ≥{min_disk_gb:.0f} GB",
                detail=f"Path: {check_path}. Free up space or attach a larger volume.",
            )
        return CheckResult(
            name="Disk space",
            passed=True,
            message=f"{free_gb:.0f} GB free on {check_path}",
        )
    except Exception as exc:
        return CheckResult(name="Disk space", passed=False, message=str(exc))


def check_vram(scenario) -> CheckResult:
    """Verify there is enough VRAM to load the scenario's model."""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        total_free_mb = 0
        gpu_names = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            total_free_mb += mem.free // (1024 * 1024)
            total_gb = mem.total // (1024 ** 3)
            gpu_names.append(f"{name} ({total_gb} GB)")
        pynvml.nvmlShutdown()

        needed_gb = scenario.estimated_vram_gb()
        free_gb = total_free_mb / 1024

        if free_gb < needed_gb * 1.1:
            return CheckResult(
                name="VRAM",
                passed=False,
                message=f"{free_gb:.1f} GB free, need ~{needed_gb:.1f} GB",
                detail=f"GPUs: {', '.join(gpu_names)}. Use a smaller model or quantized scenario.",
            )
        return CheckResult(
            name="VRAM",
            passed=True,
            message=f"{device_count}× {' | '.join(gpu_names)} — {free_gb:.0f} GB free",
        )
    except ImportError:
        return CheckResult(
            name="VRAM",
            passed=False,
            message="pynvml not installed",
            detail="Run: pip install pynvml",
        )
    except Exception as exc:
        return CheckResult(name="VRAM", passed=False, message=str(exc))
