"""Check that vLLM port :8000 and agent port :9100 are free."""
from __future__ import annotations

import socket

from src.doctor.runner import CheckResult

PORTS = [8000, 9100]


def check_ports(_scenario) -> CheckResult:
    """Verify ports 8000 and 9100 are not already bound."""
    occupied = []
    for port in PORTS:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", port))
        except OSError:
            occupied.append(port)

    if occupied:
        return CheckResult(
            name="Ports 8000 + 9100 free",
            passed=False,
            message=f"Port(s) in use: {', '.join(':' + str(p) for p in occupied)}",
            detail="Run `make stop` to kill existing sessions, or check with `lsof -i :<port>`.",
        )
    return CheckResult(
        name="Ports 8000 + 9100 free",
        passed=True,
        message="Both available",
    )
