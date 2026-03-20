"""
CLI display layer — purely presentational, no effect on internal state or parsing.

All user-facing output goes through this module. Raw LLM text and parsed objects
remain unchanged upstream; this module only controls what the user sees on screen.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

from models import Opportunity, Proposal, ProjectGoal

console = Console()


# ── Startup / errors ──────────────────────────────────────────────────────────

def print_usage() -> None:
    console.print("[bold red]Usage:[/bold red] python main.py [cyan]<project_path>[/cyan]")


def print_error(msg: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {msg}")


def print_scanning(path: str) -> None:
    console.print(f"\n[bold cyan]Scanning[/bold cyan] [dim]{path}[/dim] ...\n")


def print_done() -> None:
    console.print("\n[bold green]✓[/bold green] Proposals confirmed. Ready to proceed.\n")


# ── Spinner while LLM is working ──────────────────────────────────────────────

@contextmanager
def thinking(message: str = "Thinking...") -> Generator[None, None, None]:
    """Context manager that shows a spinner while the LLM is running."""
    with console.status(f"[dim]{message}[/dim]", spinner="dots"):
        yield


# ── Confirm node ──────────────────────────────────────────────────────────────

def print_summary(summary: str) -> None:
    console.print(
        Panel(
            summary,
            title="[bold blue]Project Understanding[/bold blue]",
            border_style="blue",
            box=box.ROUNDED,
        )
    )


def print_confirm_prompt() -> None:
    console.print("[dim]Does this look right? (yes to continue, or tell me what I got wrong)[/dim]")


def ask_user() -> str:
    return console.input("[cyan]>[/cyan] ").strip()


# ── Goal ──────────────────────────────────────────────────────────────────────

def print_goal(goal: ProjectGoal) -> None:
    console.print(
        Panel(
            f"[white]{goal.project_goal}[/white]",
            title="[bold blue]Project Goal[/bold blue]",
            border_style="blue",
            box=box.ROUNDED,
        )
    )


# ── Opportunities ─────────────────────────────────────────────────────────────

def print_opportunities(opportunities: list[Opportunity]) -> None:
    lines = Text()
    for i, opp in enumerate(opportunities, start=1):
        if i > 1:
            lines.append("\n")
        lines.append(f"  {i}.  ", style="bold cyan")
        lines.append(opp.title)
    console.print(
        Panel(
            lines,
            title="[bold blue]Opportunities[/bold blue]",
            border_style="blue",
            box=box.ROUNDED,
        )
    )


# ── Proposals ─────────────────────────────────────────────────────────────────

def print_proposals(proposals: list[Proposal]) -> None:
    _EFFORT_COLOR = {"quick win": "green", "medium": "yellow", "larger task": "red"}
    lines = Text()
    for i, p in enumerate(proposals, start=1):
        if i > 1:
            lines.append("\n\n")
        lines.append(f"  {i}.  {p.title}\n", style="bold white")
        lines.append("      What:    ", style="dim")
        lines.append(f"{p.what}\n")
        lines.append("      Effort:  ", style="dim")
        lines.append(p.effort, style=_EFFORT_COLOR.get(p.effort.lower(), "white"))
    console.print(
        Panel(
            lines,
            title="[bold blue]Proposals[/bold blue]",
            border_style="blue",
            box=box.ROUNDED,
        )
    )


def print_proposal_prompt() -> None:
    console.print("[dim]Happy with these? (yes to proceed, or give feedback / suggest your own idea)[/dim]")
