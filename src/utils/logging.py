"""Rich console wrappers for consistent coloured output."""
from rich.console import Console
from rich.text import Text

console = Console()


def section(title: str) -> None:
    """Print a visual section header."""
    console.print(f"\n[bold white]{'═' * 63}[/bold white]")
    console.print(f"[bold cyan]  {title}[/bold cyan]")
    console.print(f"[bold white]{'═' * 63}[/bold white]")


def info(message: str) -> None:
    console.print(f"  [white]{message}[/white]")


def success(message: str) -> None:
    console.print(f"  [bold green]✓[/bold green] [green]{message}[/green]")


def warning(message: str) -> None:
    console.print(f"  [bold yellow]⚠[/bold yellow] [yellow]{message}[/yellow]")


def error(message: str) -> None:
    console.print(f"  [bold red]✗[/bold red] [red]{message}[/red]")


def print_kv(key: str, value: str) -> None:
    console.print(f"  [dim]{key:<18}[/dim] [white]{value}[/white]")
