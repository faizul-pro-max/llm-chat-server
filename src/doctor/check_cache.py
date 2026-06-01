"""Report which models are already cached on disk."""
from __future__ import annotations

import os
from pathlib import Path

from src.doctor.runner import CheckResult


def check_cache(_scenario) -> CheckResult:
    """List models already downloaded into HF_HOME. Informational only."""
    try:
        hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        hub_path = Path(hf_home) / "hub"

        if not hub_path.exists():
            return CheckResult(
                name="Cached models",
                passed=True,
                message="No cache directory yet",
                severity="info",
            )

        models = []
        for model_dir in hub_path.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("models--"):
                name = model_dir.name.replace("models--", "").replace("--", "/")
                size_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                size_gb = size_bytes / (1024 ** 3)
                models.append(f"{name} ({size_gb:.1f} GB)")

        if models:
            return CheckResult(
                name="Cached models",
                passed=True,
                message=f"{len(models)} model(s) cached",
                severity="info",
                detail="  " + "\n  ".join(models),
            )
        return CheckResult(
            name="Cached models",
            passed=True,
            message="No models cached yet",
            severity="info",
        )
    except Exception as exc:
        return CheckResult(
            name="Cached models",
            passed=True,
            message=f"Could not read cache: {exc}",
            severity="info",
        )
