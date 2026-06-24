"""Launch vLLM server in a detached tmux session."""
from __future__ import annotations

import os
import sys

from src.utils import logging as log
from src.utils import tmux

SESSION = "vllm"


def start_in_tmux(scenario, log_file: str = "logs/vllm.log") -> None:
    """Launch the inference server in tmux session 'vllm'.

    The session name stays 'vllm' regardless of backend so stop/status/logs keep
    working; the actual command depends on `scenario.backend` (vLLM, or the naive
    HuggingFace Transformers baseline server). Returns immediately — does NOT wait
    for ready.
    """
    if tmux.is_running(SESSION):
        log.error(f"tmux session '{SESSION}' already exists. Run `make stop` first.")
        sys.exit(2)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    parts = scenario.build_command()
    if scenario.backend == "vllm":
        # Resolve the vllm entrypoint to the venv so tmux's login shell finds it.
        venv_bin = os.path.dirname(sys.executable)
        parts[0] = os.path.join(venv_bin, "vllm")
    command = " ".join(parts)
    log.info(f"Backend:   {scenario.backend}")
    log.info(f"Command:   {command}")
    log.info(f"Log:       {log_file}")

    tmux.create_session(SESSION, command, log_file=log_file)
    log.info(f"Launched in tmux session '{SESSION}'.")
    log.info(f"  Attach:  tmux attach -t {SESSION}")
