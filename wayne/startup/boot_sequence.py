from __future__ import annotations

import subprocess
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.text import Text

console = Console()

WAYNE_LOGO = """
 ██╗    ██╗ █████╗ ██╗   ██╗███╗   ██╗███████╗
 ██║    ██║██╔══██╗╚██╗ ██╔╝████╗  ██║██╔════╝
 ██║ █╗ ██║███████║ ╚████╔╝ ██╔██╗ ██║█████╗
 ██║███╗██║██╔══██║  ╚██╔╝  ██║╚██╗██║██╔══╝
 ╚███╔███╔╝██║  ██║   ██║   ██║ ╚████║███████╗
  ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═══╝╚══════╝
"""

BOOT_STEPS = [
    ("Initializing neural core", 0.3),
    ("Loading Gemma local language model", 0.8),
    ("Checking local backend link", 0.4),
    ("Syncing task database", 0.3),
    ("Mounting file system index", 0.3),
    ("Activating device control layer", 0.4),
    ("Establishing laptop agent link", 0.3),
    ("Loading calendar integration", 0.3),
    ("Warming up voice synthesis", 0.4),
    ("All systems nominal", 0.2),
]


class BootSequence:
    def run_cli(self) -> None:
        console.clear()
        self._print_logo()
        self._print_subtitle()
        self._run_boot_steps()
        self._print_online_banner()
        self._speak_online()

    def _print_logo(self) -> None:
        console.print(Text(WAYNE_LOGO, style="bold cyan"))
        time.sleep(0.3)

    def _print_subtitle(self) -> None:
        console.print("[bold amber1]  W.A.Y.N.E - Wireless Artificial Yielding Network Engine[/]")
        console.print("[dim]  Version 1.0  |  Gemma Local  |  Supabase/SQLite  |  Zero Cloud AI[/]\n")
        time.sleep(0.2)

    def _run_boot_steps(self) -> None:
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[cyan]{task.description}"),
            BarColumn(bar_width=30, style="cyan", complete_style="green"),
            TextColumn("[green]{task.percentage:>3.0f}%"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Booting...", total=len(BOOT_STEPS))
            for step_name, delay in BOOT_STEPS:
                progress.update(task, description=f"[cyan]{step_name}...[/]")
                time.sleep(delay)
                console.print(f"  [green]✓[/] [dim]{step_name}[/]")
                progress.advance(task)

    def _print_online_banner(self) -> None:
        console.print()
        console.print(
            Panel(
                "[bold green]W.A.Y.N.E ONLINE[/]\n"
                "[cyan]Wireless Artificial Yielding Network Engine[/]\n"
                "[dim]Gemma · Supabase/SQLite · Local AI · Device Control[/]",
                border_style="cyan",
                title="[bold cyan]SYSTEM READY[/]",
                subtitle="[dim]Say 'Sleep WAYNE' to return to passive mode[/]",
            )
        )
        console.print()
        console.print("[bold cyan]W.A.Y.N.E ❯[/] [green]Online. How can I assist you, Sir?[/]\n")

    def _speak_online(self) -> None:
        text = "W.A.Y.N.E online. How can I assist you, Sir?"
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["say", "-v", "Samantha", text])
            elif sys.platform == "win32":
                command = (
                    "Add-Type -AssemblyName System.Speech; "
                    f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"
                )
                subprocess.Popen(["powershell", "-NoProfile", "-Command", command])
            else:
                subprocess.Popen(["espeak", "-s", "150", "-p", "30", text])
        except Exception:
            pass
