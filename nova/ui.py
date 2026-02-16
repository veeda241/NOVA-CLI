from rich import print
from rich.panel import Panel
from rich.layout import Layout
from rich.align import Align
from rich.text import Text
from rich.style import Style
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.prompt import Prompt
from rich.console import Group
import psutil
import os
import datetime
import time
import threading
import queue

# Global console
console = Console()

# Queue for thread-safe UI updates
ui_queue = queue.Queue()

# Store chat history for rendering
chat_history = []
max_history_lines = 15

# Global layout object
layout = Layout()

def get_header():
    ascii_art = """
    ███    ██  ██████  ██    ██  █████ 
    ████   ██ ██    ██ ██    ██ ██   ██
    ██ ██  ██ ██    ██ ██    ██ ███████
    ██  ██ ██ ██    ██  ██  ██  ██   ██w
    ██   ████  ██████    ████   ██   ██
    """
    return Align.center(Text(ascii_art, style="bold blue on black"), vertical="middle")

def get_system_panel():
    """Generates a system monitoring panel."""
    try:
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
    except:
        cpu_percent = 0
        mem = type('obj', (object,), {'percent': 0})
    
    table = Table(show_header=False, expand=True, box=None)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right", style="bold green")
    
    table.add_row("CPU Usage:", f"{cpu_percent}%")
    table.add_row("Memory:", f"{mem.percent}%")
    table.add_row("OS:", os.name.upper())

    return Panel(
        table,
        title="[bold cyan]System Status[/bold cyan]",
        border_style="cyan"
    )

def get_files_panel():
    """Shows recent files in current directory."""
    try:
        files = os.listdir('.')[:8] # Top 8 files
        files_text = "\n".join([f"📄 {f}" if os.path.isfile(f) else f"📁 {f}" for f in files])
    except:
        files_text = "N/A"
        
    return Panel(
        files_text,
        title=f"[bold magenta]Workspace: {os.path.basename(os.getcwd())}[/bold magenta]",
        border_style="magenta"
    )

def get_chat_panel():
    """Renders the chat history."""
    content = Group(*chat_history)
    return Panel(
        content,
        title="[bold green]Chat Output[/bold green]",
        border_style="green",
        padding=(1, 1)
    )

def get_footer():
    """Render a footer status bar."""
    now = datetime.datetime.now().strftime("%H:%M:%S")
    user = os.getenv('USERNAME') or 'User'
    cwd = os.getcwd()
    # Simple footer string
    return f" USER: {user} | DIR: {cwd} | TIME: {now} | STATUS: [bold green]ONLINE[/bold green]"

def setup_layout():
    """Creates the initial dashboard layout structure."""
    global layout
    layout.split_column(
        Layout(name="header", size=8),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    
    layout["body"].split_row(
        Layout(name="main", ratio=3),
        Layout(name="sidebar", ratio=1)
    )
    
    layout["sidebar"].split_column(
        Layout(name="system", ratio=1),
        Layout(name="files", ratio=1)
    )
    
    layout["header"].update(Panel(get_header(), style="blue"))
    layout["footer"].update(Panel(Align.center(get_footer()), style="white on black"))
    layout["main"].update(get_chat_panel())
    layout["system"].update(get_system_panel())
    layout["files"].update(get_files_panel())

def run_dashboard():
    """Runs the live dashboard with non-blocking input to exit."""
    setup_layout()
    
    # Windows-specific non-blocking input
    import msvcrt
    
    with Live(layout, refresh_per_second=4, screen=True) as live:
        console.print("[bold yellow]Dashboard Active. Press 'q' or 'Esc' to exit.[/bold yellow]")
        try:
            while True:
                # Update data
                layout["system"].update(get_system_panel())
                
                # Update Footer with exit instruction
                footer_text = get_footer() + " | [bold yellow]Press 'q' to exit[/bold yellow]"
                layout["footer"].update(Panel(Align.center(footer_text), style="white on black"))
                
                layout["files"].update(get_files_panel())
                
                # Check for input without blocking
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key.lower() == b'q' or key == b'\x1b': # q or ESC
                        break
                        
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass

def add_chat_message(role, message):
    """Adds a message to the chat history."""
    timestamp = datetime.datetime.now().strftime("%H:%M")
    if role == "user":
        text = Text(f"[{timestamp}] You: {message}", style="bold cyan")
    elif role == "system":
        text = Text(f"[{timestamp}] System: {message}", style="dim white")
    else:
        text = Text(f"[{timestamp}] TinyLlama: {message}", style="green")
    
    chat_history.append(text)
    if len(chat_history) > max_history_lines:
        chat_history.pop(0)

__all__ = ['setup_layout', 'layout', 'run_dashboard', 'add_chat_message', 'chat_history']
