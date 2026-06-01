"""Scenario 2 — prefix caching enabled."""
from src.models._base import BaseScenario


class PrefixCacheScenario(BaseScenario):
    name: str = "prefix_cache"
    description: str = "vLLM FP16 with --enable-prefix-caching"

    enable_prefix_caching: bool = True

    def estimated_vram_gb(self) -> float:
        return 14.0


scenario = PrefixCacheScenario()
