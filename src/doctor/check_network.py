"""Measure download throughput from HuggingFace to estimate model download cost."""
from __future__ import annotations

import time

from src.doctor.runner import CheckResult

WARN_MB_S = 10.0
FAIL_MB_S = 5.0
TEST_URL = "https://huggingface.co/datasets/openai/openai_humaneval/resolve/main/openai_humaneval.tar.gz"
TARGET_BYTES = 50_000_000  # 50 MB sample is enough for a reliable estimate
GPU_HOURLY_RATE = 1.07


def check_network(scenario) -> CheckResult:
    """Download a sample from HF Hub to measure throughput and estimate model cost."""
    try:
        import httpx
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn

        start = time.monotonic()
        bytes_downloaded = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Testing network speed...", total=TARGET_BYTES)
            with httpx.stream("GET", TEST_URL, follow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes(chunk_size=131_072):
                    bytes_downloaded += len(chunk)
                    progress.update(task, advance=len(chunk))
                    if bytes_downloaded >= TARGET_BYTES:
                        break

        elapsed = time.monotonic() - start
        speed_mb_s = bytes_downloaded / elapsed / 1_000_000

        model_size_gb = scenario.estimated_vram_gb() * 1.1  # rough proxy
        est_seconds = (model_size_gb * 1000) / speed_mb_s
        est_cost = (est_seconds / 3600) * GPU_HOURLY_RATE

        msg = f"{speed_mb_s:.1f} MB/s  (download ~{est_seconds / 60:.1f} min ≈ ${est_cost:.2f})"

        if speed_mb_s < FAIL_MB_S:
            return CheckResult(
                name="Network speed",
                passed=False,
                message=msg,
                detail=(
                    f"Network is too slow. Downloading at {speed_mb_s:.1f} MB/s will cost ~${est_cost:.2f} in GPU time.\n"
                    "    Options: pick a host with faster network, use --skip-network to bypass."
                ),
            )
        if speed_mb_s < WARN_MB_S:
            return CheckResult(
                name="Network speed",
                passed=False,
                message=msg,
                severity="warning",
                detail="Network is slow — download will take a while. Use --skip-network to bypass.",
            )
        return CheckResult(name="Network speed", passed=True, message=msg)

    except Exception as exc:
        return CheckResult(
            name="Network speed",
            passed=False,
            message=f"Speed test failed: {exc}",
            severity="warning",
            detail="Could not measure speed. Use --skip-network to skip this check.",
        )
