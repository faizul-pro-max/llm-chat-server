"""Doctor runner — executes all pre-flight checks and formats output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from rich.console import Console

console = Console()


@dataclass
class CheckResult:
    """Returned by every check function."""
    name: str
    passed: bool
    message: str
    severity: str = "error"     # error | warning | info
    detail: Optional[str] = None


def _print_result(index: int, total: int, result: CheckResult) -> None:
    label = f"[{index}/{total}] {result.name}"
    dots = "." * max(1, 55 - len(label))
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


def run_all(scenario, skip_network: bool = False) -> bool:
    """Run all checks in order. Returns True if all passed (warnings don't block)."""
    from src.doctor.check_cuda import check_cuda
    from src.doctor.check_disk import check_disk, check_vram
    from src.doctor.check_network import check_network
    from src.doctor.check_ports import check_ports
    from src.doctor.check_hf import check_hf_access
    from src.doctor.check_cache import check_cache

    checks: List[Callable] = [
        lambda: check_cuda(scenario),
        lambda: check_disk(scenario),
        lambda: check_vram(scenario),
    ]
    if not skip_network:
        checks.append(lambda: check_network(scenario))
    checks += [
        lambda: check_hf_access(scenario),
        lambda: check_ports(scenario),
        lambda: check_cache(scenario),
        lambda: _check_tmux(scenario),
        lambda: _check_existing_sessions(scenario),
    ]

    total = len(checks)
    results: List[CheckResult] = []

    console.print()
    console.print("[bold cyan]🩺  LLM Bench Doctor — Pre-flight Checks[/bold cyan]")
    console.print("[bold white]" + "═" * 63 + "[/bold white]")
    console.print()

    for i, check_fn in enumerate(checks, start=1):
        try:
            result = check_fn()
        except Exception as exc:
            result = CheckResult(
                name=f"check_{i}",
                passed=False,
                message=f"Unexpected error: {exc}",
                severity="error",
            )
        results.append(result)
        _print_result(i, total, result)

        if not result.passed and result.severity == "error":
            _print_summary(results, aborted=True)
            return False

    console.print()
    _print_summary(results, aborted=False)
    return all(r.passed or r.severity != "error" for r in results)


def _print_summary(results: List[CheckResult], aborted: bool) -> None:
    console.print()
    console.print("[bold white]" + "═" * 63 + "[/bold white]")
    failed = [r for r in results if not r.passed and r.severity == "error"]
    if failed or aborted:
        console.print(f"[bold red]✗  CHECKS FAILED — {len(failed)} error(s)[/bold red]")
    else:
        console.print("[bold green]✓  ALL CHECKS PASSED — ready to start scenarios[/bold green]")
    console.print("[bold white]" + "═" * 63 + "[/bold white]")
    console.print()


# ── Inline checks that don't need their own file ─────────────────────────────

def _check_tmux(_scenario) -> CheckResult:
    """Check tmux is installed."""
    import subprocess
    try:
        out = subprocess.check_output(["tmux", "-V"], stderr=subprocess.DEVNULL).decode().strip()
        return CheckResult(name="tmux installed", passed=True, message=out)
    except FileNotFoundError:
        return CheckResult(
            name="tmux installed",
            passed=False,
            message="tmux not found",
            detail="Run: apt-get install tmux",
        )
    except Exception as exc:
        return CheckResult(name="tmux installed", passed=False, message=str(exc))


def _check_existing_sessions(_scenario) -> CheckResult:
    """Warn if orphaned vllm/agent tmux sessions exist."""
    try:
        import libtmux
        server = libtmux.Server()
        found = [n for n in ("vllm", "agent") if server.has_session(n)]
        if found:
            return CheckResult(
                name="Existing tmux sessions",
                passed=False,
                message=f"Orphaned sessions: {', '.join(found)}",
                severity="warning",
                detail="Run `make stop` to clear them before starting fresh.",
            )
        return CheckResult(name="Existing tmux sessions", passed=True, message="None running")
    except Exception as exc:
        return CheckResult(name="Existing tmux sessions", passed=True, message=f"Could not check: {exc}", severity="info")
