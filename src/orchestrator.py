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

    # ── Step 5: Start vLLM (slow — 60-180s) ─────────────────────────────────
    log.section("vLLM server")
    start_vllm.start_in_tmux(scenario)
    health.wait_for_vllm(timeout=180, show_log_tail=True)
    log.success("vLLM ready at :8000")

    # ── Step 6: Warmup ───────────────────────────────────────────────────────
    if not skip_warmup:
        log.section("Warmup")
        warmup.run(scenario.warmup_prompts, count=scenario.warmup_requests)

    # ── Step 7: Print connection info ────────────────────────────────────────
    log.section("Ready")
    _print_ready(scenario)


def _print_ready(scenario: BaseScenario) -> None:
    public_ip = _get_public_ip()
    from src.utils.logging import console
    console.print(f"\n[bold white]{'═' * 63}[/bold white]")
    console.print("[bold green]✓  READY FOR BENCHMARKS[/bold green]")
    console.print(f"[bold white]{'═' * 63}[/bold white]\n")

    from src.utils import logging as log
    log.print_kv("Scenario:", scenario.name)
    log.print_kv("Model:", scenario.model)
    log.print_kv("Public IP:", public_ip)
    console.print()
    log.print_kv("Endpoints:", "")
    log.print_kv("  vLLM:", f"http://{public_ip}:8000")
    log.print_kv("  Agent:", f"http://{public_ip}:9100")
    console.print()
    log.print_kv("For .env:", "")
    log.print_kv("  GPU_SERVER_IP=", public_ip)
    log.print_kv("  VLLM_PORT=", "8000")
    log.print_kv("  GPU_AGENT_PORT=", "9100")
    console.print()
    log.info("Useful commands:")
    log.info("  make logs               — tail combined logs")
    log.info("  make attach SVC=vllm    — see vLLM live output")
    log.info("  make attach SVC=agent   — see agent live output")
    log.info("  make stop               — stop all services")
    console.print()


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
