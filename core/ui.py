from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def print_user(text: str) -> None:
    console.print(Panel(Text(text, style="bold cyan"), title="[cyan]You", border_style="cyan"))

def print_jarvis(text: str) -> None:
    console.print(Panel(Text(text, style="bold green"), title="[green]J.A.R.V.I.S", border_style="green"))

def print_status(msg: str) -> None:
    console.print(f"[dim]{msg}[/dim]")

def print_error(msg: str) -> None:
    console.print(f"[red]ERROR: {msg}[/red]")

def print_banner() -> None:
    console.print(Panel(
        "[bold green]J.A.R.V.I.S AGI[/bold green]\n[dim]Local-first AI Voice Assistant[/dim]",
        border_style="green",
    ))
