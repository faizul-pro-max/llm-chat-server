"""FastAPI observer agent — /gpu, /health, /system, /stream endpoints on :9100."""
from __future__ import annotations

import asyncio
import json
import os
import platform
import time
from typing import Any, Dict

import psutil
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse

from src.observer_agent.auth import require_api_key
from src.observer_agent.nvml_reader import read_gpu, read_all_gpus

app = FastAPI(title="GPU Observer Agent", version="1.0.0")


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Unauthenticated health check."""
    return {"status": "ok", "ts": time.time()}


@app.get("/gpu", dependencies=[Depends(require_api_key)])
async def gpu(index: int = 0) -> Dict[str, Any]:
    """Return current metrics for GPU at the given index."""
    return read_gpu(index)


@app.get("/gpus", dependencies=[Depends(require_api_key)])
async def gpus() -> list:
    """Return metrics for all GPUs."""
    return read_all_gpus()


@app.get("/system", dependencies=[Depends(require_api_key)])
async def system_info() -> Dict[str, Any]:
    """Return host-level CPU, RAM, and platform info."""
    vm = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=0.1)
    return {
        "ts": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_percent": cpu_pct,
        "ram_total_gb": round(vm.total / (1024 ** 3), 1),
        "ram_used_gb": round(vm.used / (1024 ** 3), 1),
        "ram_percent": vm.percent,
    }


@app.get("/stream")
async def stream(x_api_key: str = "", index: int = 0):
    """Server-Sent Events stream of GPU metrics at ~500 ms intervals.

    Pass x-api-key as a query param for browser/SSE clients that can't set headers.
    """
    secret = os.getenv("AGENT_SECRET", "")
    if secret and x_api_key != secret:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    async def event_generator():
        while True:
            try:
                data = read_gpu(index)
                yield f"data: {json.dumps(data)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def main() -> None:
    port = int(os.getenv("AGENT_PORT", "9100"))
    uvicorn.run("src.observer_agent.server:app", host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
