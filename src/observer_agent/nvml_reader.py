"""pynvml wrapper — reads GPU metrics for single and multi-GPU hosts."""
from __future__ import annotations

import socket
import time
from typing import Any, Dict, List

import pynvml


def _init() -> None:
    pynvml.nvmlInit()


def _shutdown() -> None:
    pynvml.nvmlShutdown()


def read_gpu(gpu_index: int = 0) -> Dict[str, Any]:
    """Return metrics for a single GPU as a JSON-serialisable dict."""
    _init()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_w = power_mw / 1000.0
        except pynvml.NVMLError:
            power_w = None

        return {
            "ts": time.time(),
            "gpu_index": gpu_index,
            "gpu_name": name,
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "vram_used_mb": mem.used // (1024 * 1024),
            "vram_free_mb": mem.free // (1024 * 1024),
            "vram_total_mb": mem.total // (1024 * 1024),
            "power_w": round(power_w, 1) if power_w is not None else None,
            "temp_c": temp,
            "hostname": socket.gethostname(),
        }
    finally:
        _shutdown()


def read_all_gpus() -> List[Dict[str, Any]]:
    """Return metrics for every GPU on this host."""
    _init()
    try:
        count = pynvml.nvmlDeviceGetCount()
    finally:
        _shutdown()
    return [read_gpu(i) for i in range(count)]
