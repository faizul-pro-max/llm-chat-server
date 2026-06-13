"""Mistral-7B-Instruct-v0.3 — no HF token required."""
from src.models._base import BaseScenario


class MistralScenario(BaseScenario):
    name: str = "mistral"
    description: str = "Mistral-7B-Instruct-v0.3, FP16, no auth required"

    model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    dtype: str = "float16"
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    max_num_seqs: int = 256

    enable_prefix_caching: bool = True

    def estimated_vram_gb(self) -> float:
        return 14.0


scenario = MistralScenario()
