"""Measure download and upload throughput using speedtest CLI (falls back to HTTP)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

from src.doctor.runner import CheckResult

WARN_MB_S = 10.0
FAIL_MB_S = 5.0
GPU_HOURLY_RATE = 1.07

# HTTP fallback — tried in order if speedtest CLI is not installed
_FALLBACK_DOWNLOAD_URLS = [
    "http://speedtest.tele2.net/100MB.zip",
    "https://proof.ovh.net/files/100Mb.dat",
    "https://speed.hetzner.de/100MB.test",
]
_FALLBACK_UPLOAD_URLS = [
    "https://httpbin.org/post",
    "https://postman-echo.com/post",
]
_TARGET_BYTES  = 50_000_000
_UPLOAD_BYTES  = 5_000_000


def check_network(scenario) -> CheckResult:
    """Measure download + upload speed; estimate model download cost."""
    try:
        if shutil.which("speedtest"):
            down_mb_s, up_mb_s = _run_speedtest_cli()
            method = "speedtest"
        else:
            down_mb_s = _http_download()
            up_mb_s   = _http_upload()
            method = "http"

        model_size_gb = scenario.estimated_vram_gb() * 1.1
        est_seconds   = (model_size_gb * 1000) / down_mb_s if down_mb_s else float("inf")
        est_cost      = (est_seconds / 3600) * GPU_HOURLY_RATE

        up_str = f"{up_mb_s:.1f} MB/s" if up_mb_s is not None else "n/a"
        msg = (
            f"↓ {down_mb_s:.1f} MB/s  ↑ {up_str}"
            f"  (model dl ~{est_seconds / 60:.1f} min ≈ ${est_cost:.2f})"
            + (f"  [via {method}]" if method == "http" else "")
        )

        if down_mb_s < FAIL_MB_S:
            return CheckResult(
                name="Network speed",
                passed=False,
                message=msg,
                severity="warning",
                detail=(
                    f"Download very slow ({down_mb_s:.1f} MB/s). "
                    f"Fetching the model will cost ~${est_cost:.2f} in GPU time.\n"
                    "    `make start` will ask before continuing. To skip this check\n"
                    "    entirely, use --skip-network, or pick a host with faster network."
                ),
            )
        if down_mb_s < WARN_MB_S:
            return CheckResult(
                name="Network speed",
                passed=False,
                message=msg,
                severity="warning",
                detail="Download is slow — model fetch will take a while. Use --skip-network to bypass.",
            )
        return CheckResult(name="Network speed", passed=True, message=msg)

    except Exception as exc:
        return CheckResult(
            name="Network speed",
            passed=False,
            message=f"Speed test failed: {exc}",
            severity="warning",
            detail=(
                "Could not measure speed.\n"
                "    Run `make install` to install the speedtest CLI, or use --skip-network to skip."
            ),
        )


# ── speedtest CLI path ────────────────────────────────────────────────────────

def _run_speedtest_cli() -> tuple[float, float]:
    """Run `speedtest --accept-license --accept-gdpr --format=json` and return (down, up) MB/s."""
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as p:
        p.add_task("Running speedtest (Ookla)…")
        result = subprocess.run(
            ["speedtest", "--accept-license", "--accept-gdpr", "--format=json"],
            capture_output=True,
            text=True,
            timeout=120,
        )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "speedtest exited non-zero")

    data = json.loads(result.stdout)
    # bandwidth is in bytes/second
    down_mb_s = data["download"]["bandwidth"] / 1_000_000
    up_mb_s   = data["upload"]["bandwidth"]   / 1_000_000
    return down_mb_s, up_mb_s


# ── HTTP fallback path ────────────────────────────────────────────────────────

def _http_download() -> float:
    import httpx
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn

    url = _first_reachable(_FALLBACK_DOWNLOAD_URLS)
    if url is None:
        raise RuntimeError("No download test server reachable — check internet connection")

    received = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("↓ [progress.description]{task.description}"),
        BarColumn(), DownloadColumn(), TransferSpeedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Testing download…", total=_TARGET_BYTES)
        start = time.monotonic()
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes(chunk_size=131_072):
                received += len(chunk)
                progress.update(task, advance=len(chunk))
                if received >= _TARGET_BYTES:
                    break

    return received / (time.monotonic() - start) / 1_000_000


def _http_upload() -> float | None:
    try:
        import httpx
        url = _first_reachable(_FALLBACK_UPLOAD_URLS)
        if url is None:
            return None
        payload = os.urandom(_UPLOAD_BYTES)
        start = time.monotonic()
        with httpx.stream(
            "POST", url,
            content=payload,
            headers={"Content-Length": str(_UPLOAD_BYTES)},
            timeout=60,
        ) as r:
            for _ in r.iter_bytes():
                pass
        return _UPLOAD_BYTES / (time.monotonic() - start) / 1_000_000
    except Exception:
        return None


def _first_reachable(urls: list[str]) -> str | None:
    import httpx
    for url in urls:
        try:
            httpx.head(url, follow_redirects=True, timeout=5).raise_for_status()
            return url
        except Exception:
            continue
    return None
