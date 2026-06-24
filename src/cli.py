"""Click CLI entry point — `start`, `stop`, `doctor`, `status`, `logs`, `scenarios`."""
import sys

import click

from src.utils import env as envutil

envutil.load()


@click.group()
def cli() -> None:
    """GPU Server Orchestrator — manage vLLM + metrics agent lifecycle."""


@cli.command()
@click.option("--skip-network", is_flag=True, help="Skip the network speed test (ignored when --check=network)")
@click.option("--scenario",     default="baseline", show_default=True, help="Scenario to validate VRAM/config against")
@click.option("--check",        default=None, metavar="NAME",
              help=f"Run a single check only. One of: cuda, disk, vram, cpu, ram, network, hf, ports, cache, tmux, sessions")
@click.option("--simple",       is_flag=True, help="Use plain dot-format output instead of the table")
def doctor(skip_network: bool, scenario: str, check: str, simple: bool) -> None:
    """Run pre-flight checks. NO GPU cost — run this first.

    \b
    Examples:
      make doctor                       run all checks (table view)
      make doctor CHECK=network         run network check only
      make doctor SIMPLE=1              plain dot-format output
      python -m src.cli doctor --check cuda
      python -m src.cli doctor --simple
    """
    from src.doctor import runner as doctor_runner
    from src.doctor.runner import CHECK_KEYS
    from src.orchestrator import load_scenario

    if check and check not in CHECK_KEYS:
        raise click.BadParameter(
            f"'{check}' is not a valid check name.\nChoose from: {', '.join(CHECK_KEYS)}",
            param_hint="--check",
        )

    s = load_scenario(scenario)
    passed = doctor_runner.run_all(s, skip_network=skip_network, only=check, simple=simple)
    sys.exit(0 if passed else 1)


@cli.command()
@click.option("--scenario", required=True, help="Scenario name (e.g. baseline, prefix_cache)")
@click.option("--no-warmup", is_flag=True, help="Skip warmup requests")
@click.option("--no-download", is_flag=True, help="Skip model download (model must already be cached)")
def start(scenario: str, no_warmup: bool, no_download: bool) -> None:
    """Full lifecycle: doctor → download → start vLLM + agent → warmup → ready."""
    from src import orchestrator
    orchestrator.start(scenario, skip_warmup=no_warmup, skip_download=no_download)


@cli.command()
@click.option("--service", type=click.Choice(["vllm", "agent", "all"]), default="all", show_default=True)
def stop(service: str) -> None:
    """Stop tmux sessions for vLLM / agent."""
    from src.utils import tmux

    targets = ["vllm", "agent"] if service == "all" else [service]
    for name in targets:
        if tmux.is_running(name):
            tmux.kill_session(name)
            click.echo(f"  Stopped: {name}")
        else:
            click.echo(f"  Not running: {name}")


@cli.command()
def status() -> None:
    """Show service status — tmux sessions + health endpoints."""
    from src.utils import tmux
    from src.utils import http as httputil

    client = httputil.get_client(timeout=3.0, retries=1)

    for name in ("vllm", "agent"):
        running = tmux.is_running(name)
        state = "[green]running[/green]" if running else "[red]stopped[/red]"
        click.echo(f"  {name:<10} tmux={state}")

    # Check HTTP health endpoints
    for label, url in [("vLLM :8000", "http://localhost:8000/health"), ("agent :9100", "http://localhost:9100/health")]:
        try:
            r = client.get(url)
            http_state = "OK" if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception:
            http_state = "unreachable"
        click.echo(f"  {label:<16} http={http_state}")


@cli.command()
@click.option("--scenario", default="baseline", show_default=True, help="Scenario whose port the server was started with")
def info(scenario: str) -> None:
    """Reprint the connection banner (public IP, mapped ports, .env snippet).

    Read-only — does not restart or disturb running services. Useful when you
    cleared the terminal and lost the output from `make start`.
    """
    from src.orchestrator import _print_ready, load_scenario

    _print_ready(load_scenario(scenario))


@cli.command()
@click.option("--service", type=click.Choice(["vllm", "agent"]), default=None, help="Which service logs to tail")
@click.option("--tail", default=50, show_default=True, help="Number of lines to show")
def logs(service: str, tail: int) -> None:
    """Tail log output from tmux sessions."""
    from src.utils import tmux

    targets = [service] if service else ["vllm", "agent"]
    for name in targets:
        click.echo(f"\n{'─' * 60}")
        click.echo(f"  {name} (last {tail} lines)")
        click.echo(f"{'─' * 60}")
        output = tmux.capture_pane(name, lines=tail)
        if output:
            click.echo(output)
        else:
            click.echo("  (no output — session may not be running)")


@cli.group()
def scenarios() -> None:
    """Scenario management commands."""


@scenarios.command(name="list")
def list_scenarios() -> None:
    """List all available scenarios."""
    import importlib
    import pkgutil
    import src.models as models_pkg

    click.echo("\nAvailable scenarios:\n")
    for info in pkgutil.iter_modules(models_pkg.__path__):
        if info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"src.models.{info.name}")
            s = mod.scenario
            click.echo(f"  {s.name:<20} {s.description}")
        except Exception as exc:
            click.echo(f"  {info.name:<20} (error loading: {exc})")
    click.echo()


@scenarios.command()
@click.argument("name")
def show(name: str) -> None:
    """Show full config of a named scenario."""
    from src.orchestrator import load_scenario

    s = load_scenario(name)
    click.echo(f"\nScenario: {s.name}")
    click.echo(f"Description: {s.description}\n")
    for key, value in s.model_dump().items():
        click.echo(f"  {key:<35} {value}")
    click.echo()
    click.echo(f"Launch command (backend={s.backend}):")
    click.echo("  " + " ".join(s.build_command()))
    click.echo()


@cli.command()
@click.option("--scenario", default="baseline", show_default=True)
@click.option("--show-progress", is_flag=True, default=True)
def download(scenario: str, show_progress: bool) -> None:
    """Download the model for a scenario (standalone, without full start)."""
    from src.orchestrator import load_scenario
    from src.lifecycle import download as dl

    s = load_scenario(scenario)
    dl.download_model(s.model, show_progress=show_progress)


@cli.command()
@click.option("--count", default=20, show_default=True, help="Number of warmup requests")
def warmup(count: int) -> None:
    """Send warmup requests to vLLM (must already be running)."""
    from src.lifecycle import warmup as wm
    from src.orchestrator import load_scenario

    s = load_scenario("baseline")
    wm.run(s.warmup_prompts, count=count)


if __name__ == "__main__":
    cli()
