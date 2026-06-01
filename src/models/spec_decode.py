"""Scenario 5 — speculative decoding with a small draft model."""
from src.models._base import BaseScenario


class SpecDecodeScenario(BaseScenario):
    name: str = "spec_decode"
    description: str = "Speculative decoding: Qwen2.5-7B target + 0.5B draft"

    model: str = "Qwen/Qwen2.5-7B-Instruct"
    speculative_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    num_speculative_tokens: int = 5
    dtype: str = "float16"

    def estimated_vram_gb(self) -> float:
        # 7B target + 0.5B draft
        return 16.0


scenario = SpecDecodeScenario()
