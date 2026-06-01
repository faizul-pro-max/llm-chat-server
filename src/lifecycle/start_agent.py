"""Launch the observer agent in a detached tmux session."""
from __future__ import annotations

import os
import sys

from src.utils import logging as log
from src.utils import tmux

SESSION = "agent"


def start_in_tmux(port: int = 9100, log_file: str = "logs/agent.log") -> None:
    """Launch the FastAPI observer agent in tmux session 'agent'."""
    if tmux.is_running(SESSION):
        log.error(f"tmux session '{SESSION}' already exists. Run `make stop` first.")
        sys.exit(2)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    command = f"AGENT_PORT={port} python -m src.observer_agent.server"
    log.info(f"Command:   {command}")
    log.info(f"Log:       {log_file}")

    tmux.create_session(SESSION, command, log_file=log_file)
    log.info(f"Launched in tmux session '{SESSION}'.")
    log.info(f"  Attach:  tmux attach -t {SESSION}")
