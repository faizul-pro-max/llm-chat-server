"""Launch vLLM server in a detached tmux session."""
from __future__ import annotations

import os
import sys

from src.utils import logging as log
from src.utils import tmux

SESSION = "vllm"


def start_in_tmux(scenario, log_file: str = "logs/vllm.log") -> None:
    """Launch vLLM in tmux session 'vllm'. Returns immediately — does NOT wait for ready."""
    if tmux.is_running(SESSION):
        log.error(f"tmux session '{SESSION}' already exists. Run `make stop` first.")
        sys.exit(2)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    command = " ".join(scenario.build_vllm_command())
    log.info(f"Command:   {command}")
    log.info(f"Log:       {log_file}")

    tmux.create_session(SESSION, command, log_file=log_file)
    log.info(f"Launched in tmux session '{SESSION}'.")
    log.info(f"  Attach:  tmux attach -t {SESSION}")
