from __future__ import annotations

from datetime import datetime
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

console = Console()

LOGO = r"""
 ██╗    ██╗ █████╗ ██╗   ██╗███╗   ██╗███████╗
 ██║    ██║██╔══██╗╚██╗ ██╔╝████╗  ██║██╔════╝
 ██║ █╗ ██║███████║ ╚████╔╝ ██╔██╗ ██║█████╗
 ██║███╗██║██╔══██║  ╚██╔╝  ██║╚██╗██║██╔══╝
 ╚███╔███╔╝██║  ██║   ██║   ██║ ╚████║███████╗
  ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═══╝╚══════╝
"""


def show_startup(online: bool, status: dict[str, Any]) -> None:
    console.print(Text(LOGO, style="bold cyan"))
    state = "[green]ONLINE[/green]" if online else "[yellow]OFFLINE[/yellow]"
    console.print("[amber1]W.A.Y.N.E - Wireless Artificial Yielding Network Engine[/amber1]")
    console.print("[amber1]Version 1.0 | Local AI | Gemma via Ollama[/amber1]")
    console.print(
        Panel(
            f"[ W.A.Y.N.E v1.0 ] [ Gemma Local ] [ {datetime.now().strftime('%H:%M:%S')} ] [ {state} ]",
            border_style="cyan",
            box=box.SQUARE,
        )
    )
    console.print("[green]W.A.Y.N.E ONLINE. Awaiting your command, Sir.[/green]")


def extract_tag(text: str) -> str:
    if text.startswith("[") and "]" in text:
        return text[1 : text.index("]")]
    return "AI RESPONSE"


def show_ai_response(reply: str) -> None:
    tag = extract_tag(reply)
    console.print(Panel(reply, title=tag, border_style="cyan", box=box.ROUNDED))


def show_error(message: str) -> None:
    console.print(Panel(message, title="ERROR", border_style="red"))


def show_meta(tokens: int, elapsed: float) -> None:
    console.print(f"[dim]{tokens} tokens approx | {elapsed:.2f}s[/dim]")


def task_table(tasks: list[dict[str, Any]]) -> Table:
    table = Table(title="Tasks", box=box.SIMPLE_HEAVY, border_style="cyan")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Task")
    table.add_column("Priority")
    table.add_column("Status")
    for task in tasks:
        priority = task.get("priority", "medium")
        priority_style = {"high": "red", "medium": "yellow", "low": "green"}.get(priority, "white")
        status = "[green]Complete[/green]" if task.get("completed") else "[yellow]Open[/yellow]"
        table.add_row(str(task.get("id")), task.get("title", ""), f"[{priority_style}]{priority}[/{priority_style}]", status)
    return table


def file_table(files: list[dict[str, Any]]) -> Table:
    table = Table(title="Files", box=box.SIMPLE_HEAVY, border_style="cyan")
    table.add_column("Name")
    table.add_column("Size", justify="right")
    table.add_column("Modified")
    for item in files:
        name = item.get("name", "")
        if item.get("type") == "directory":
            name = f"[cyan]{name}/[/cyan]"
        table.add_row(name, str(item.get("size", 0)), item.get("modified", ""))
    return table


def calendar_table(events: list[dict[str, Any]]) -> Table:
    table = Table(title="Today's Schedule", box=box.SIMPLE_HEAVY, border_style="cyan")
    table.add_column("Time")
    table.add_column("Event")
    table.add_column("Duration")
    for event in events:
        start = str(event.get("start", ""))
        end = str(event.get("end", ""))
        table.add_row(start[11:16] if len(start) > 15 else start, event.get("title", "Untitled"), end[11:16] if len(end) > 15 else end)
    return table


def system_table(status: dict[str, Any]) -> Table:
    table = Table(title="System Status", box=box.SIMPLE_HEAVY, border_style="cyan")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Bar")
    for label, key in [("CPU", "cpu_percent"), ("RAM", "ram_percent"), ("Disk", "disk_percent"), ("Battery", "battery_percent")]:
        raw = status.get(key, 0)
        value = float(raw if raw is not None else 0)
        table.add_row(label, f"{value:.1f}%", ProgressBar(total=100, completed=value, width=24))
    return table


def device_table(devices: list[dict[str, Any]]) -> Table:
    table = Table(title="Devices", box=box.SIMPLE_HEAVY, border_style="cyan")
    table.add_column("Device")
    table.add_column("Type")
    table.add_column("Battery")
    table.add_column("Online")
    table.add_column("Last Seen")
    for device in devices:
        online = "[green]Online[/green]" if device.get("is_online") else "[red]Offline[/red]"
        table.add_row(
            device.get("device_name") or device.get("device_id", ""),
            device.get("device_type", ""),
            f"{device.get('battery_level', 0)}%",
            online,
            str(device.get("last_seen", "")),
        )
    return table


def help_panel() -> None:
    commands = "\n".join(
        [
            "/tasks         show all tasks",
            "/files [path]  list files in directory",
            "/schedule      show today's calendar events",
            "/status        show CPU, RAM, disk",
            "/devices       show registered devices",
            "/shutdown laptop send shutdown command",
            "/restart laptop  send restart command",
            "/sleep laptop    send sleep command",
            "/lock laptop     send lock command",
            "/notify \"msg\"   send push notification",
            "/voice         record voice input",
            "/clear         clear terminal",
            "/history       show last 10 conversation turns",
            "/help          show all commands",
            "/exit          quit WAYNE",
        ]
    )
    console.print(Panel(commands, title="Commands", border_style="cyan"))
