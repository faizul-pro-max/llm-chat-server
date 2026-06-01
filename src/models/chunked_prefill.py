"""Scenario 3 — chunked prefill enabled."""
from src.models._base import BaseScenario


class ChunkedPrefillScenario(BaseScenario):
    name: str = "chunked_prefill"
    description: str = "vLLM FP16 with --enable-chunked-prefill"

    enable_chunked_prefill: bool = True

    def estimated_vram_gb(self) -> float:
        return 14.0


scenario = ChunkedPrefillScenario()
