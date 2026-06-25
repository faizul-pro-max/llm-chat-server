"""Main lifecycle — called by `make start SCENARIO=name`."""
from __future__ import annotations

import importlib
import os
import socket
import sys

from src.models._base import BaseScenario
from src.utils import logging as log


def load_scenario(name: str) -> BaseScenario:
    """Load src/models/{name}.py and return its `scenario` singleton."""
    try:
        module = importlib.import_module(f"src.models.{name}")
    except ModuleNotFoundError:
        log.error(f"Unknown scenario: '{name}'")
        log.info("Run `python -m src.cli scenarios list` to see available scenarios.")
        sys.exit(3)
    if not hasattr(module, "scenario"):
        log.error(f"src/models/{name}.py does not expose a `scenario` object.")
        sys.exit(3)
    return module.scenario


def start(
    scenario_name: str,
    skip_warmup: bool = False,
    skip_download: bool = False,
) -> None:
    """Full lifecycle: doctor → download → start agent → start vLLM → warmup → ready."""
    from src.doctor import runner as doctor_runner
    from src.lifecycle import download, start_vllm, start_agent, warmup, health

    # ── Step 1: Load scenario ────────────────────────────────────────────────
    log.section("Loading scenario")
    scenario = load_scenario(scenario_name)
    log.info(f"Scenario:  {scenario.name}")
    log.info(f"Model:     {scenario.model}")
    log.info(f"Config:    {scenario.summary()}")

    # ── Step 2: Doctor (mandatory) ───────────────────────────────────────────
    log.section("Pre-flight checks")
    passed = doctor_runner.run_all(scenario)
    if not passed:
        log.error("Doctor failed — aborting. Fix the issues above before retrying.")
        sys.exit(1)

    # ── Step 3: Download model ───────────────────────────────────────────────
    if not skip_download:
        log.section("Model download")
        download.download_model(scenario.model, show_progress=True)

    # ── Step 4: Start observer agent (fast) ─────────────────────────────────
    log.section("Observer agent")
    start_agent.start_in_tmux(port=9100)
    health.wait_for_agent(timeout=15)
    log.success("Agent ready at :9100")

    # ── Step 5: Start inference server (slow — 60-180s) ─────────────────────
    backend_label = "HF Transformers" if scenario.backend == "transformers" else "vLLM"
    log.section(f"{backend_label} server")
    start_vllm.start_in_tmux(scenario)
    health.wait_for_vllm(timeout=600, show_log_tail=True)
    log.success(f"{backend_label} ready at :{scenario.port}")

    # ── Step 6: Warmup ───────────────────────────────────────────────────────
    if not skip_warmup:
        log.section("Warmup")
        warmup.run(scenario.warmup_prompts, count=scenario.warmup_requests)

    # ── Step 7: Print connection info ────────────────────────────────────────
    log.section("Ready")
    _print_ready(scenario)


def start_docker(scenario_name: str, skip_warmup: bool = False) -> None:
    """Host-side orchestration for the Docker stack.

    The compose stack (inference-server: vLLM + agent) is already started by
    `docker compose up`. This does the part Docker can't: wait for both /health
    endpoints, send warmup requests, and print the connection banner. tmux is not
    involved — process supervision is handled by supervisord inside the container.
    """
    from src.lifecycle import warmup, health

    log.section("Loading scenario")
    scenario = load_scenario(scenario_name)
    log.info(f"Scenario:  {scenario.name}")
    log.info(f"Model:     {scenario.model}")
    log.info(f"Config:    {scenario.summary()}")

    hint = "Check logs:  make docker-logs   (or: docker compose logs inference-server)"

    log.section("Waiting for services")
    health.wait_for_agent(timeout=60, fail_hint=hint)
    log.success("Agent ready at :9100")

    backend_label = "HF Transformers" if scenario.backend == "transformers" else "vLLM"
    # First run pulls the model from HuggingFace, so allow a long startup window.
    health.wait_for_vllm(timeout=1800, show_log_tail=False, fail_hint=hint)
    log.success(f"{backend_label} ready at :{scenario.port}")

    if not skip_warmup:
        log.section("Warmup")
        warmup.run(scenario.warmup_prompts, count=scenario.warmup_requests)

    log.section("Ready")
    _print_ready(scenario)


_AGENT_INTERNAL_PORT = 9100


def _print_ready(scenario: BaseScenario) -> None:
    public_ip = _get_public_ip()
    vllm_ext = _get_external_port(scenario.port)
    agent_ext = _get_external_port(_AGENT_INTERNAL_PORT)

    from src.utils.logging import console
    from src.utils import logging as log

    console.print(f"\n[bold white]{'═' * 63}[/bold white]")
    console.print("[bold green]✓  READY FOR BENCHMARKS[/bold green]")
    console.print(f"[bold white]{'═' * 63}[/bold white]\n")

    log.print_kv("Scenario:", scenario.name)
    log.print_kv("Model:", scenario.model)
    log.print_kv("Public IP:", public_ip)
    console.print()

    log.print_kv("Endpoints:", "")
    _print_endpoint(console, "vLLM", public_ip, scenario.port, vllm_ext)
    _print_endpoint(console, "Agent", public_ip, _AGENT_INTERNAL_PORT, agent_ext)
    console.print()

    log.print_kv("Paste into Node.js .env:", "")
    console.print(f"  [bold cyan]GPU_SERVER_IP={public_ip}[/bold cyan]")
    console.print(f"  [bold cyan]VLLM_PORT={vllm_ext}[/bold cyan]")
    console.print(f"  [bold cyan]GPU_AGENT_PORT={agent_ext}[/bold cyan]")
    console.print()

    log.info("Useful commands:")
    log.info("  make logs               — tail combined logs")
    log.info("  make attach SVC=vllm    — see vLLM live output")
    log.info("  make attach SVC=agent   — see agent live output")
    log.info("  make stop               — stop all services")
    console.print()


def _print_endpoint(console, label: str, ip: str, internal: int, external: int) -> None:
    """Print one endpoint line, noting port mapping when external != internal."""
    url = f"http://{ip}:{external}"
    if external != internal:
        console.print(f"  [dim]{label + ':':<18}[/dim] [white]{url}[/white]  [dim](internal :{internal})[/dim]")
    else:
        console.print(f"  [dim]{label + ':':<18}[/dim] [white]{url}[/white]")


def _get_external_port(internal: int) -> int:
    """Return the provider-mapped external port for an internal port, or the port itself."""
    val = os.environ.get(f"VAST_TCP_PORT_{internal}")
    if val and val.isdigit():
        return int(val)
    return internal


def _get_public_ip() -> str:
    try:
        import httpx
        r = httpx.get("https://api.ipify.org", timeout=5)
        return r.text.strip()
    except Exception:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "unknown"
