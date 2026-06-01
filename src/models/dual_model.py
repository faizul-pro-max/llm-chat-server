"""Scenario 7 — two vLLM instances on ports 8000 and 8001."""
from __future__ import annotations

from typing import List

from src.models._base import BaseScenario


class DualModelScenario(BaseScenario):
    name: str = "dual_model"
    description: str = "2x vLLM instances: port 8000 (FP16) + port 8001 (AWQ)"

    # Primary instance — FP16 on :8000
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    dtype: str = "float16"
    port: int = 8000

    # Secondary instance config — AWQ on :8001
    secondary_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    secondary_quantization: str = "awq"
    secondary_port: int = 8001

    def estimated_vram_gb(self) -> float:
        # FP16 14 GB + AWQ 7 GB
        return 21.0

    def build_secondary_command(self) -> List[str]:
        """vLLM command for the second instance."""
        return [
            "vllm", "serve", self.secondary_model,
            "--host", self.host,
            "--port", str(self.secondary_port),
            "--dtype", self.dtype,
            "--quantization", self.secondary_quantization,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--disable-log-requests",
        ]


scenario = DualModelScenario()
