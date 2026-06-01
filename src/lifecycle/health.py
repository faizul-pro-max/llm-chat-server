"""Poll health endpoints until services are ready or timeout expires."""
from __future__ import annotations

import sys
import time

import httpx
from rich.live import Live
from rich.text import Text

from src.utils import logging as log
from src.utils import tmux


def wait_for_vllm(timeout: int = 180, show_log_tail: bool = True) -> None:
    """Block until vLLM /health returns 200 or timeout is reached."""
    _wait(
        url="http://localhost:8000/health",
        service_name="vLLM",
        tmux_session="vllm",
        timeout=timeout,
        show_log_tail=show_log_tail,
    )


def wait_for_agent(timeout: int = 15) -> None:
    """Block until the observer agent /health returns 200 or timeout is reached."""
    _wait(
        url="http://localhost:9100/health",
        service_name="observer agent",
        tmux_session="agent",
        timeout=timeout,
        show_log_tail=False,
    )


def _wait(url: str, service_name: str, tmux_session: str, timeout: int, show_log_tail: bool) -> None:
    deadline = time.monotonic() + timeout
    interval = 2.0
    attempt = 0

    with Live(refresh_per_second=2) as live:
        while time.monotonic() < deadline:
            attempt += 1
            elapsed = int(time.monotonic() - (deadline - timeout))
            status_line = Text(f"  Waiting for {service_name}... ({elapsed}s / {timeout}s)")

            if show_log_tail and tmux.is_running(tmux_session):
                tail = tmux.capture_pane(tmux_session, lines=3)
                status_line.append(f"\n  [log] {tail.splitlines()[-1] if tail.strip() else '…'}")

            live.update(status_line)

            try:
                r = httpx.get(url, timeout=3.0)
                if r.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass

            time.sleep(interval)

    log.error(f"{service_name} did not become ready within {timeout}s.")
    log.info(f"  Check logs: tmux attach -t {tmux_session}")
    sys.exit(2)
