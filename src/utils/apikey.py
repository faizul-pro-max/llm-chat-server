"""Per-run API key — one UUID v4 shared by the vLLM server and the observer agent.

`make start` generates a fresh key, injects it into each server's launch command
before starting them (vLLM `--api-key`, agent `AGENT_SECRET`), and prints it in
the ready banner. It is persisted to a gitignored runtime file so `make info` can
reprint it without restarting anything.

Path defaults to `$LOG_DIR/api_key` (LOG_DIR=/app/logs in Docker → ./logs on the
host), overridable via API_KEY_FILE.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional


def key_file() -> str:
    explicit = os.getenv("API_KEY_FILE")
    if explicit:
        return explicit
    return os.path.join(os.getenv("LOG_DIR", "logs"), "api_key")


def generate() -> str:
    """Return a new UUID v4 string."""
    return str(uuid.uuid4())


def save(key: str) -> str:
    """Persist the key (mode 0600) so reconnect flows can read it back."""
    path = key_file()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(key + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def load() -> Optional[str]:
    """Return the saved key, or None if no run has generated one yet."""
    try:
        with open(key_file()) as fh:
            return fh.read().strip() or None
    except FileNotFoundError:
        return None
