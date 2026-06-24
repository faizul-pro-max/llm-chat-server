"""Baseline scenario — no optimizations. The zero line.

Runs the model on plain HuggingFace Transformers (naive `model.generate()`, one
request at a time — no continuous batching, no paged attention). This is the
un-optimized reference point: benchmark it against any vLLM scenario to measure
how much throughput/latency vLLM actually buys you.
"""
from src.models._base import BaseScenario


class BaselineScenario(BaseScenario):
    name: str = "baseline"
    description: str = "Naive HuggingFace Transformers FP16, no optimizations"

    backend: str = "transformers"

    model: str = "Qwen/Qwen2.5-7B-Instruct"
    dtype: str = "float16"
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    max_num_seqs: int = 256

    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    speculative_model: None = None
    num_speculative_tokens: int = 0

    def estimated_vram_gb(self) -> float:
        return 14.0


scenario = BaselineScenario()
