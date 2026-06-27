"""Doctor runner — executes all pre-flight checks and formats a rich table."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from rich import box
from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


@dataclass
class CheckResult:
    """Returned by every check function."""
    name:     str
    passed:   bool
    message:  str
    severity: str = "error"       # error | warning | info
    detail:   Optional[str] = None


# ── Check registry ────────────────────────────────────────────────────────────
# Each entry: (key, display_name, factory)
# key is what the user passes to --check / make doctor CHECK=<key>

def _registry(scenario) -> List[Tuple[str, str, Callable]]:
    from src.doctor.check_cuda    import check_cuda
    from src.doctor.check_disk    import check_disk, check_vram
    from src.doctor.check_cpu     import check_cpu, check_ram
    from src.doctor.check_network import check_network
    from src.doctor.check_ports   import check_ports
    from src.doctor.check_hf      import check_hf_access
    from src.doctor.check_cache   import check_cache

    return [
        ("cuda",     "CUDA + Driver",       lambda: check_cuda(scenario)),
        ("vllm",     "vLLM installed",       lambda: _check_vllm(scenario)),
        ("disk",     "Disk space",           lambda: check_disk(scenario)),
        ("vram",     "VRAM",                 lambda: check_vram(scenario)),
        ("cpu",      "CPU cores",            lambda: check_cpu(scenario)),
        ("ram",      "RAM",                  lambda: check_ram(scenario)),
        ("network",  "Network speed",        lambda: check_network(scenario)),
        ("hf",       "HuggingFace Hub",      lambda: check_hf_access(scenario)),
        ("ports",    "Ports 8000 + 9100",    lambda: check_ports(scenario)),
        ("cache",    "Cached models",        lambda: check_cache(scenario)),
        ("tmux",     "tmux",                 lambda: _check_tmux(scenario)),
        ("sessions", "Existing sessions",    lambda: _check_existing_sessions(scenario)),
    ]


# Exported so cli.py can build the Click Choice list without importing checks
CHECK_KEYS = [
    "cuda", "vllm", "disk", "vram", "cpu", "ram",
    "network", "hf", "ports", "cache", "tmux", "sessions",
]


# ── Public entry point ────────────────────────────────────────────────────────

def run_all(
    scenario,
    skip_network: bool = False,
    only: Optional[str] = None,   # single check key, or None for all
    simple: bool = False,          # True → old dot-format output
) -> List[CheckResult]:
    """Run checks, render output, and return the list of results.

    Use `has_blocking_errors(results)` to decide whether to abort — warnings
    (e.g. a slow network) do not block on their own.
    """
    registry = _registry(scenario)

    if only:
        checks = [(k, n, fn) for k, n, fn in registry if k == only]
    else:
        checks = [(k, n, fn) for k, n, fn in registry
                  if not (k == "network" and skip_network)]

    total   = len(checks)
    results: List[CheckResult] = []

    if simple:
        _run_simple(checks, results, total)
    else:
        _run_table(checks, results, total)

    _print_details(results)
    _print_summary(results, simple=simple)
    return results


def has_blocking_errors(results: List[CheckResult]) -> bool:
    """True when any check failed with error severity (warnings don't block)."""
    return any(not r.passed and r.severity == "error" for r in results)


# ── Table output (default) ────────────────────────────────────────────────────

def _run_table(
    checks:  List[Tuple[str, str, Callable]],
    results: List[CheckResult],
    total:   int,
) -> None:
    console.print()
    console.print(
        Panel(
            "[bold cyan]🩺  GPU Server Orchestrator — Pre-flight Doctor[/bold cyan]",
            border_style="cyan",
            expand=False,
            padding=(0, 4),
        )
    )
    console.print()

    with Live(console=console, refresh_per_second=12, vertical_overflow="visible") as live:
        for i, (key, name, check_fn) in enumerate(checks, start=1):
            live.update(_build_table(results, total, running=(i, name)))
            try:
                result = check_fn()
            except Exception as exc:
                result = CheckResult(name=name, passed=False, message=f"Unexpected error: {exc}")
            results.append(result)
            live.update(_build_table(results, total))


def _build_table(
    results:  List[CheckResult],
    total:    int,
    running:  Optional[Tuple[int, str]] = None,
) -> Table:
    table = Table(
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold white on grey23",
        row_styles=["", "on grey7"],
        show_lines=False,
        expand=False,
        padding=(0, 1),
        min_width=72,
    )
    table.add_column(" # ",    justify="right",  style="dim",  no_wrap=True, width=4)
    table.add_column("Check",                    style="bold", no_wrap=True, min_width=22)
    table.add_column("Status", justify="center",               no_wrap=True, width=8)
    table.add_column("Result",                                  min_width=36)

    for i, result in enumerate(results, start=1):
        icon, icon_style, msg_style = _styles(result)
        table.add_row(
            str(i),
            result.name,
            Text(icon, style=icon_style, justify="center"),
            Text(result.message, style=msg_style),
        )

    if running:
        idx, name = running
        table.add_row(
            str(idx),
            Text(name, style="dim"),
            Text("…", style="dim", justify="center"),
            Text("running…", style="dim italic"),
        )

    filled = len(results) + (1 if running else 0)
    for j in range(filled + 1, total + 1):
        table.add_row(str(j), Text("", style="dim"), Text("·", style="dim", justify="center"), "")

    return table


# ── Simple output (--simple flag) ─────────────────────────────────────────────

def _run_simple(
    checks:  List[Tuple[str, str, Callable]],
    results: List[CheckResult],
    total:   int,
) -> None:
    console.print()
    console.print("[bold cyan]🩺  LLM Bench Doctor — Pre-flight Checks[/bold cyan]")
    console.print("[bold white]" + "═" * 63 + "[/bold white]")
    console.print()

    for i, (key, name, check_fn) in enumerate(checks, start=1):
        try:
            result = check_fn()
        except Exception as exc:
            result = CheckResult(name=name, passed=False, message=f"Unexpected error: {exc}")
        results.append(result)
        _print_result(i, total, result)

    console.print()


def _print_result(index: int, total: int, result: CheckResult) -> None:
    label = f"[{index}/{total}] {result.name}"
    dots  = "." * max(1, 55 - len(label))
    if result.passed:
        status = f"[bold green]✓[/bold green] [green]{result.message}[/green]"
    elif result.severity == "warning":
        status = f"[bold yellow]⚠[/bold yellow] [yellow]{result.message}[/yellow]"
    elif result.severity == "info":
        status = f"[bold blue]ⓘ[/bold blue] [blue]{result.message}[/blue]"
    else:
        status = f"[bold red]✗[/bold red] [red]{result.message}[/red]"
    console.print(f"{label} {dots} {status}")
    if result.detail:
        console.print(f"    [dim]{result.detail}[/dim]")


# ── Shared helpers ────────────────────────────────────────────────────────────

def _styles(result: CheckResult) -> Tuple[str, str, str]:
    """Return (icon, icon_style, message_style)."""
    if result.passed:
        return "✓", "bold green", "green"
    if result.severity == "warning":
        return "⚠", "bold yellow", "yellow"
    if result.severity == "info":
        return "ⓘ", "bold blue", "blue"
    return "✗", "bold red", "red"


def _print_details(results: List[CheckResult]) -> None:
    has_details = [r for r in results if not r.passed and r.detail]
    if not has_details:
        return
    console.print()
    for r in has_details:
        _, icon_style, _ = _styles(r)
        console.print(f"  [{icon_style}]{r.name}[/{icon_style}]")
        for line in r.detail.strip().splitlines():
            console.print(f"    [dim]{line.strip()}[/dim]")


def _print_summary(results: List[CheckResult], simple: bool = False) -> None:
    errors   = [r for r in results if not r.passed and r.severity == "error"]
    warnings = [r for r in results if not r.passed and r.severity == "warning"]

    if errors:
        parts = [f"[bold red]{len(errors)} error{'s' if len(errors) > 1 else ''}[/bold red]"]
        if warnings:
            parts.append(f"[yellow]{len(warnings)} warning{'s' if len(warnings) > 1 else ''}[/yellow]")
        body  = f"[bold red]✗  CHECKS FAILED[/bold red]  —  {',  '.join(parts)}"
        style = "red"
    elif warnings:
        body  = f"[bold yellow]⚠  PASSED WITH {len(warnings)} WARNING{'S' if len(warnings) > 1 else ''}[/bold yellow]"
        style = "yellow"
    else:
        body  = "[bold green]✓  ALL CHECKS PASSED — ready to start scenarios[/bold green]"
        style = "green"

    if simple:
        console.print("[bold white]" + "═" * 63 + "[/bold white]")
        console.print(body)
        console.print("[bold white]" + "═" * 63 + "[/bold white]")
    else:
        console.print()
        console.print(Panel(Padding(body, (0, 2)), border_style=style, expand=False, padding=(0, 2)))
    console.print()


# ── Inline checks ─────────────────────────────────────────────────────────────

def _check_vllm(_scenario) -> CheckResult:
    """Verify vLLM is importable in the active interpreter (the .venv used by doctor)."""
    import importlib.metadata as md
    try:
        version = md.version("vllm")
        return CheckResult(name="vLLM installed", passed=True, message=f"v{version}")
    except md.PackageNotFoundError:
        return CheckResult(
            name="vLLM installed",
            passed=False,
            message="not installed",
            severity="warning",
            detail="Run `make install` (installs requirements-gpu.txt), or `pip install -r requirements-gpu.txt`.",
        )
    except Exception as exc:
        return CheckResult(name="vLLM installed", passed=False, message=str(exc))


def _check_tmux(_scenario) -> CheckResult:
    import subprocess
    try:
        out = subprocess.check_output(["tmux", "-V"], stderr=subprocess.DEVNULL).decode().strip()
        return CheckResult(name="tmux", passed=True, message=out)
    except FileNotFoundError:
        return CheckResult(name="tmux", passed=False, message="not found", detail="Run: apt-get install tmux")
    except Exception as exc:
        return CheckResult(name="tmux", passed=False, message=str(exc))


def _check_existing_sessions(_scenario) -> CheckResult:
    try:
        import libtmux
        server = libtmux.Server()
        found  = [n for n in ("vllm", "agent") if server.has_session(n)]
        if found:
            return CheckResult(
                name="Existing sessions",
                passed=False,
                message=f"Orphaned: {', '.join(found)}",
                severity="warning",
                detail="Run `make stop` to clear them before starting fresh.",
            )
        return CheckResult(name="Existing sessions", passed=True, message="None running")
    except Exception as exc:
        return CheckResult(name="Existing sessions", passed=True, message=f"Could not check: {exc}", severity="info")
