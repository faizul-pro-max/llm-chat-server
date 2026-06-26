"""Active-experiment state — capture the running scenario's config so the
observer agent can report *what* is being benchmarked.

`make start` loads a scenario (a Pydantic class in src/models/) and launches the
inference server from it. That config (model, dtype, optimisation flags, the
exact launch command, …) is otherwise invisible to API clients. We snapshot it
to a small JSON file at start time; the agent serves it from `/experiment`.

Path resolution keeps the host (tmux) and container (Docker) flows in agreement:
both default to `$LOG_DIR/active_experiment.json` (LOG_DIR=/app/logs in Docker,
mapped to ./logs on the host), overridable via EXPERIMENT_STATE_FILE.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


def state_file() -> str:
    explicit = os.getenv("EXPERIMENT_STATE_FILE")
    if explicit:
        return explicit
    return os.path.join(os.getenv("LOG_DIR", "logs"), "active_experiment.json")


def build_payload(scenario) -> Dict[str, Any]:
    """Flatten a scenario into a JSON-serialisable experiment description."""
    return {
        "name": scenario.name,
        "description": scenario.description,
        "backend": scenario.backend,
        "model": scenario.model,
        "summary": scenario.summary(),
        "launch_command": scenario.build_command(),
        "config": scenario.model_dump(),
    }


def write(scenario) -> str:
    """Snapshot the active scenario to the state file. Returns the path written."""
    payload = build_payload(scenario)
    payload["started_at"] = time.time()
    path = state_file()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


def read() -> Optional[Dict[str, Any]]:
    """Return the snapshot written at start, or None if absent/unreadable."""
    path = state_file()
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def current() -> Optional[Dict[str, Any]]:
    """Best-effort active experiment for API responses.

    Prefers the snapshot file (reflects the actual `make start`); falls back to
    loading the scenario named by the SCENARIO env var (the Docker path, where
    the agent and inference server are launched straight from that var).
    """
    snapshot = read()
    if snapshot is not None:
        return snapshot

    name = os.getenv("SCENARIO")
    if not name:
        return None
    try:
        from src.orchestrator import load_scenario
        return build_payload(load_scenario(name))
    except SystemExit:
        # load_scenario exits on an unknown name; don't take the agent down.
        return None
    except Exception:
        return None
