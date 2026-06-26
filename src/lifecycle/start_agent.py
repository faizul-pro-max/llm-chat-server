"""Launch the observer agent in a detached tmux session."""
from __future__ import annotations

import os
import shlex
import sys

from src.utils import logging as log
from src.utils import tmux

SESSION = "agent"


def start_in_tmux(port: int = 9100, log_file: str = "logs/agent.log",
                  api_key: str = None) -> None:
    """Launch the FastAPI observer agent in tmux session 'agent'.

    When `api_key` is given it is injected as AGENT_SECRET so the agent's
    x-api-key auth is enforced with this run's generated key (overriding any
    placeholder in .env, since the inline env var wins over the .env load).
    """
    if tmux.is_running(SESSION):
        log.error(f"tmux session '{SESSION}' already exists. Run `make stop` first.")
        sys.exit(2)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    env_prefix = f"AGENT_PORT={port}"
    if api_key:
        env_prefix += f" AGENT_SECRET={shlex.quote(api_key)}"
    command = f"{env_prefix} {sys.executable} -m src.observer_agent.server"
    log.info(f"Command:   {command}")
    log.info(f"Log:       {log_file}")

    tmux.create_session(SESSION, command, log_file=log_file)
    log.info(f"Launched in tmux session '{SESSION}'.")
    log.info(f"  Attach:  tmux attach -t {SESSION}")
