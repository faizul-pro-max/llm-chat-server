"""Scenario 4 — AWQ INT4 quantized model."""
from src.models._base import BaseScenario


class AwqQuantScenario(BaseScenario):
    name: str = "awq_quant"
    description: str = "AWQ INT4 quantized Qwen2.5-7B (half the VRAM of FP16)"

    model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    quantization: str = "awq"
    dtype: str = "float16"
    gpu_memory_utilization: float = 0.90

    def estimated_vram_gb(self) -> float:
        return 7.0


scenario = AwqQuantScenario()
