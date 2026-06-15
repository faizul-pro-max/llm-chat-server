"""BaseScenario — common fields and interface shared by all scenarios."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BaseScenario(BaseModel):
    # Identity
    name: str = "base"
    description: str = ""

    # Model
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    dtype: str = "float16"
    quantization: Optional[str] = None

    # vLLM server
    host: str = "0.0.0.0"
    port: int = 8000
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    max_num_seqs: int = 256

    # Optimisation flags
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 0

    # Warmup
    warmup_requests: int = 20
    warmup_prompts: List[str] = Field(default_factory=lambda: [
        "Short prompt.",
        "Explain how transformers work in detail.",
        "Write a long essay on attention mechanisms in neural networks. " * 5,
    ])

    class Config:
        # Allow subclasses to set class-level attributes as defaults
        arbitrary_types_allowed = True

    def build_vllm_command(self) -> List[str]:
        """Build the full `vllm serve` argument list for this scenario."""
        cmd = [
            "vllm", "serve", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--max-num-seqs", str(self.max_num_seqs),
        ]
        if self.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        if self.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")
        if self.speculative_model:
            cmd += ["--speculative-model", self.speculative_model,
                    "--num-speculative-tokens", str(self.num_speculative_tokens)]
        if self.quantization:
            cmd += ["--quantization", self.quantization]
        return cmd

    def estimated_vram_gb(self) -> float:
        """Minimum VRAM required. Override in subclass for accuracy."""
        return 14.0

    def summary(self) -> str:
        flags = []
        if self.enable_prefix_caching:
            flags.append("prefix-caching")
        if self.enable_chunked_prefill:
            flags.append("chunked-prefill")
        if self.speculative_model:
            flags.append(f"spec-decode({self.num_speculative_tokens}tok)")
        if self.quantization:
            flags.append(f"quant={self.quantization}")
        return f"{self.dtype}" + (f" [{', '.join(flags)}]" if flags else "")
