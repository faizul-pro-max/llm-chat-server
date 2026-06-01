"""Check CUDA driver, Python version, and PyTorch CUDA availability."""
from __future__ import annotations

import subprocess
import sys

from src.doctor.runner import CheckResult

MIN_DRIVER = 525
MIN_CUDA_MAJOR = 12
MIN_CUDA_MINOR = 1


def check_cuda(_scenario) -> CheckResult:
    """Verify nvidia-smi, driver version, CUDA, Python, and PyTorch."""
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()[0]
        driver_str, gpu_name = [x.strip() for x in smi.split(",", 1)]
        driver_major = int(driver_str.split(".")[0])
    except FileNotFoundError:
        return CheckResult(
            name="CUDA + Driver",
            passed=False,
            message="nvidia-smi not found — no GPU or driver not installed",
            detail="This orchestrator requires an NVIDIA GPU with CUDA drivers.",
        )
    except Exception as exc:
        return CheckResult(name="CUDA + Driver", passed=False, message=str(exc))

    if driver_major < MIN_DRIVER:
        return CheckResult(
            name="CUDA + Driver",
            passed=False,
            message=f"Driver {driver_str} too old (need ≥{MIN_DRIVER})",
            detail=f"GPU: {gpu_name}. Update the NVIDIA driver.",
        )

    # Check CUDA version from nvcc or nvidia-smi
    cuda_version = _get_cuda_version()
    if cuda_version:
        major, minor = cuda_version
        if (major, minor) < (MIN_CUDA_MAJOR, MIN_CUDA_MINOR):
            return CheckResult(
                name="CUDA + Driver",
                passed=False,
                message=f"CUDA {major}.{minor} too old (need ≥{MIN_CUDA_MAJOR}.{MIN_CUDA_MINOR})",
            )
        cuda_str = f"CUDA {major}.{minor}"
    else:
        cuda_str = "CUDA unknown"

    py = f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Check PyTorch
    try:
        import torch
        if not torch.cuda.is_available():
            return CheckResult(
                name="CUDA + Driver",
                passed=False,
                message="torch.cuda.is_available() = False",
                detail="PyTorch installed but cannot see GPU. Check CUDA install.",
            )
        torch_str = f"torch {torch.__version__}"
    except ImportError:
        torch_str = "torch not installed (ok — vllm installs it)"

    return CheckResult(
        name="CUDA + Driver",
        passed=True,
        message=f"{driver_str} / {cuda_str} / {py} / {torch_str} / {gpu_name}",
    )


def _get_cuda_version() -> tuple[int, int] | None:
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], stderr=subprocess.DEVNULL
        ).decode()
        for part in out.split():
            if part.startswith("V"):
                nums = part[1:].split(".")
                return int(nums[0]), int(nums[1])
    except Exception:
        pass
    # Fallback: parse nvidia-smi output
    try:
        out = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.DEVNULL
        ).decode()
        import re
        m = re.search(r"CUDA Version:\s+(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None
