"""Send warmup requests to vLLM to stabilise KV cache and TTFT."""
from __future__ import annotations

import itertools
import os
import sys
import time
from typing import List

import httpx
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from src.utils import logging as log

VLLM_URL = "http://localhost:8000/v1/completions"


def run(prompts: List[str], count: int = 20) -> None:
    """Send `count` completion requests cycling through `prompts`."""
    token = os.getenv("VLLM_API_KEY", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    prompt_cycle = itertools.cycle(prompts)
    ttfts: List[float] = []

    log.info(f"Sending {count} warmup requests to {VLLM_URL}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Warming up...", total=count)

        for i in range(count):
            prompt = next(prompt_cycle)
            payload = {
                "model": _detect_model(headers),
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0.0,
            }
            t0 = time.monotonic()
            try:
                r = httpx.post(VLLM_URL, json=payload, headers=headers, timeout=60)
                r.raise_for_status()
                ttft = time.monotonic() - t0
                ttfts.append(ttft)
            except Exception as exc:
                log.warning(f"Warmup request {i + 1} failed: {exc}")
            progress.advance(task)

    if ttfts:
        avg_ms = (sum(ttfts) / len(ttfts)) * 1000
        log.success(f"{len(ttfts)}/{count} warmup requests complete — avg TTFT {avg_ms:.0f} ms")
    else:
        log.warning("All warmup requests failed — check vLLM logs")


def _detect_model(headers: dict = None) -> str:
    """Ask vLLM which model is loaded."""
    try:
        r = httpx.get("http://localhost:8000/v1/models", headers=headers or {}, timeout=5)
        data = r.json()
        return data["data"][0]["id"]
    except Exception:
        return "default"
