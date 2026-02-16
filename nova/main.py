import argparse
import sys
import os
import io
import time

# Fix Windows console encoding (must be before Rich import)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from nova import __version__
from nova.chat import chat_manager
from nova.hf_chat import hf_chat_manager

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.align import Align
    from rich.table import Table
    from rich.columns import Columns
    from rich.status import Status
    import psutil
    import datetime
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console(file=sys.stdout, force_terminal=True)

# Global state
use_hf = True

def render_header():
    """Renders the full header dashboard once."""
    console.clear()
    
    # ASCII Logo
    ascii_art = """
    ███    ██  ██████  ██    ██  █████ 
    ████   ██ ██    ██ ██    ██ ██   ██
    ██ ██  ██ ██    ██ ██    ██ ███████
    ██  ██ ██ ██    ██  ██  ██  ██   ██
    ██   ████  ██████    ████   ██   ██
    """
    console.print(Panel(
        Align.center(Text(ascii_art, style="bold blue")),
        style="blue",
    ))
    
    # System Info Row
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
    except:
        cpu, mem = 0, 0
    
    import getpass
    now = datetime.datetime.now().strftime("%H:%M:%S")
    user_name = getpass.getuser()
    
    sys_table = Table(show_header=False, box=None, expand=True)
    sys_table.add_column("", justify="center")
    sys_table.add_column("", justify="center")
    sys_table.add_column("", justify="center")
    sys_table.add_column("", justify="center")
    sys_table.add_row(
        f"[bold cyan]User:[/bold cyan] {user_name}",
        f"[bold cyan]CPU:[/bold cyan] {cpu}%",
        f"[bold cyan]RAM:[/bold cyan] {mem}%",
        f"[bold cyan]Time:[/bold cyan] {now}",
    )
    console.print(Panel(sys_table, border_style="cyan", title="[bold]System Status[/bold]"))
    
    # Workspace files (compact)
    try:
        files = os.listdir('.')[:6]
        files_str = "  ".join([f"📁{f}" if os.path.isdir(f) else f"📄{f}" for f in files])
    except:
        files_str = "N/A"
    console.print(Panel(files_str, border_style="magenta", title=f"[bold]Workspace: {os.path.basename(os.getcwd())}[/bold]"))
    
    # Tips
    tips = (
        "[green]hello[/green] · [green]status[/green] · [green]/model[/green] · [green]clear[/green] · [green]exit[/green] · [yellow]Type to chat![/yellow]"
    )
    console.print(Panel(tips, border_style="yellow", title="[bold]Quick Commands[/bold]"))
    console.rule(style="dim")

def check_model_health():
    """Checks if the current HF model works, falls back if not."""
    global use_hf
    
    fallback_models = [
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen 2.5"),
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek R1"),
        ("google/gemma-2-2b-it", "Gemma 2B"),
    ]
    
    with console.status("[bold blue]Performing Smart Health Check...[/bold blue]", spinner="dots") as status:
        # 1. Check HF
        if hf_chat_manager.api_token:
            for model_id, label in fallback_models:
                status.update(f"[bold blue]Testing {label}...[/bold blue]")
                hf_chat_manager.set_model(model_id)
                response = hf_chat_manager.stream_response("ping")
                
                if "Error" not in response and "HF Error" not in response:
                    use_hf = True
                    console.print(f"[dim green]✓ Auto-switched to working model: {label}[/dim green]")
                    return True
        
        # 2. Check Ollama if HF fails
        status.update("[bold blue]Trying Local Ollama...[/bold blue]")
        if chat_manager.check_ollama_ready():
            use_hf = False
            console.print("[dim yellow]⚠ All HF models failed. Switched to Local Ollama.[/dim yellow]")
            return True
            
        console.print("[dim red]✗ No working AI models found (HF or Local).[/dim red]")
        return False

def interactive_mode():
    global use_hf
    
    if not HAS_RICH:
        print("Rich library not found.")
        return

    # Import NIE and consciousness
    from nova.engine import nie_engine
    from nova.consciousness import nova_consciousness
    from nova.memory import nova_memory
    import uuid
    import re

    # Start a new session
    session_id = str(uuid.uuid4())[:8]
    nova_consciousness.start_session(session_id)

    render_header()

    # Show NIE model info
    nie_info = nie_engine.get_model_info()
    if nie_info.get("loaded"):
        console.print(Panel(
            f"[bold cyan]Neural Intent Engine[/bold cyan] loaded\n"
            f"  Vocab: {nie_info['vocab_size']} words | "
            f"Embed: {nie_info['embed_dim']}d | "
            f"Intents: {', '.join(nie_info['intents'])}\n"
            f"  [dim]System commands are handled INSTANTLY (<1ms) without LLM[/dim]",
            border_style="cyan",
            title="[bold]NIE Status[/bold]"
        ))
    else:
        console.print("[dim yellow]NIE not loaded. Run: python -m nova.nie_trainer[/dim yellow]")

    # Consciousness-aware greeting
    greeting = nova_consciousness.get_greeting()
    console.print(f"\n[bold green]{greeting}[/bold green]\n")

    # Start the Smart Health Check for LLM fallback
    check_model_health()
    console.print()
    
    while True:
        try:
            prompt_label = "[bold magenta]HF[/bold magenta]" if use_hf else "[bold cyan]LOCAL[/bold cyan]"
            model_name = hf_chat_manager.model_id if use_hf else "Ollama (TinyLlama)"
            command = console.input(f"[bold blue]NOVA[/bold blue] ({prompt_label}) [dim]({model_name})[/dim] [bold blue]>[/bold blue] ")
            
            if command.strip() == "":
                continue
            
            if command.lower() in ('exit', 'quit'):
                # End session and show stats
                nova_consciousness.end_session()
                stats = nova_memory.get_lifetime_stats()
                console.print(f"\n[yellow]Goodbye! Session complete.[/yellow]")
                console.print(f"[dim]Total lifetime messages: {stats['total_messages']} | "
                             f"Facts learned: {stats['facts_learned']} | "
                             f"Sessions: {stats['total_sessions']}[/dim]")
                break
            
            if command.lower() == 'clear' or command.lower() == 'status':
                render_header()
                if command.lower() == 'status':
                    console.print("[green]Dashboard refreshed with live system stats.[/green]")
                continue

            if command.lower() == 'help':
                help_table = Table(title="NOVA Commands", border_style="blue")
                help_table.add_column("Command", style="cyan")
                help_table.add_column("Description", style="white")
                help_table.add_row("status", "Refresh system dashboard")
                help_table.add_row("/model", "Switch AI model")
                help_table.add_row("/monitor", "Open live system dashboard")
                help_table.add_row("/identity", "View NOVA's identity card")
                help_table.add_row("/personality", "View NOVA's personality traits")
                help_table.add_row("/memory", "View memory statistics")
                help_table.add_row("clear", "Clear screen")
                help_table.add_row("exit", "Quit NOVA")
                help_table.add_row("[dim]any text[/dim]", "Chat or give system commands")
                console.print(help_table)
                continue

            # /monitor - Live Dashboard
            if command.lower() == '/monitor':
                try:
                    from nova.ui import run_dashboard
                    run_dashboard()
                except ImportError:
                    console.print("[red]UI module not found or incomplete.[/red]")
                except Exception as e:
                    console.print(f"[red]Dashboard error: {e}[/red]")
                # Clear screen after returning from dashboard
                render_header()
                continue

            # /identity - Show NOVA's identity card
            if command.lower() == '/identity':
                identity = nova_consciousness.get_identity_card()
                stats = identity["lifetime_stats"]
                personality = stats.get("personality", {})
                emotion = stats.get("current_emotion", {})
                
                id_table = Table(title="NOVA Identity Card", border_style="magenta")
                id_table.add_column("Attribute", style="cyan")
                id_table.add_column("Value", style="white")
                id_table.add_row("Name", identity["name"])
                id_table.add_row("Version", identity["version"])
                id_table.add_row("Total Messages", str(stats.get("total_messages", 0)))
                id_table.add_row("Total Sessions", str(stats.get("total_sessions", 0)))
                id_table.add_row("Facts Learned", str(stats.get("facts_learned", 0)))
                id_table.add_row("Current Emotion", emotion.get("emotion", "curious"))
                id_table.add_row("Active Goals", str(stats.get("active_goals", 0)))
                console.print(id_table)
                
                reflection = identity.get("self_reflection", "")
                if reflection:
                    console.print(Panel(reflection, title="[bold]Self-Reflection[/bold]", border_style="dim"))
                continue

            # /personality - Show personality traits
            if command.lower() == '/personality':
                personality = nova_consciousness.personality.traits
                p_table = Table(title="NOVA Personality Profile", border_style="green")
                p_table.add_column("Trait", style="cyan")
                p_table.add_column("Value", style="white")
                p_table.add_column("Bar", style="green")
                
                for trait, value in sorted(personality.items()):
                    bar_len = int(value * 20)
                    bar = "#" * bar_len + "." * (20 - bar_len)
                    p_table.add_row(trait.capitalize(), f"{value:.2f}", f"[{bar}]")
                console.print(p_table)
                console.print("[dim]Traits evolve with every interaction.[/dim]")
                continue

            # /memory - Show memory stats
            if command.lower() == '/memory':
                stats = nova_memory.get_lifetime_stats()
                m_table = Table(title="NOVA Memory", border_style="blue")
                m_table.add_column("Metric", style="cyan")
                m_table.add_column("Value", style="white")
                m_table.add_row("Total Messages Stored", str(stats["total_messages"]))
                m_table.add_row("Total Sessions", str(stats["total_sessions"]))
                m_table.add_row("Facts Learned", str(stats["facts_learned"]))
                m_table.add_row("Active Goals", str(stats["active_goals"]))

                # Show learned preferences
                prefs = nova_memory.get_all_preferences()
                if prefs:
                    console.print(m_table)
                    pref_table = Table(title="Learned Preferences", border_style="yellow")
                    pref_table.add_column("Preference", style="cyan")
                    pref_table.add_column("Value", style="white")
                    pref_table.add_column("Confidence", style="green")
                    for p in prefs[:10]:
                        pref_table.add_row(p["key"], p["value"], f"{p['confidence']:.1f}")
                    console.print(pref_table)
                else:
                    console.print(m_table)
                continue

            # Model Switching
            if command.lower().startswith('/model'):
                parts = command.split()
                available_models = {
                    "1": "google/gemma-2-2b-it",
                    "2": "meta-llama/Llama-3.2-3B-Instruct",
                    "3": "mistralai/Mistral-7B-Instruct-v0.3",
                    "4": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    "5": "Qwen/Qwen2.5-7B-Instruct",
                    "6": "NousResearch/Hermes-3-Llama-3.1-8B",
                    "7": "local",
                }
                
                if len(parts) == 1:
                    console.print("\n[bold yellow]Available Models:[/bold yellow]")
                    for k, v in available_models.items(): console.print(f"  {k}. {v}")
                    continue
                
                target = parts[1]
                if target in available_models:
                    model_id = available_models[target]
                    if model_id == "local": use_hf = False
                    else:
                        hf_chat_manager.set_model(model_id)
                        use_hf = True
                    console.print(f"[green]Switched to {model_id}[/green]")
                continue

            # ══════════════════════════════════════════════════
            # SMART ROUTING: NIE first, then LLM if needed
            # ══════════════════════════════════════════════════
            
            # Step 1: Try NIE classification (instant, <1ms)
            nie_result = nie_engine.process_command(command, session_id)
            
            intent = nie_result["intent"]
            confidence = nie_result["confidence"]
            consciousness = nie_result.get("consciousness", {})
            emotion = consciousness.get("emotional_state", {})
            
            # Show intent classification (subtle indicator)
            emo_indicator = emotion.get("emotion", "")
            conf_pct = f"{confidence*100:.0f}%"
            
            if nie_result["handled"]:
                # ── NIE handled it directly (no LLM needed!) ──
                result = nie_result["result"]
                ms = nie_result["inference_ms"]
                
                console.print(Panel(
                    f"[bold]{result}[/bold]",
                    title=f"[bold cyan]NIE > {intent}[/bold cyan] [dim]({conf_pct} confidence, {ms:.1f}ms)[/dim]",
                    border_style="cyan",
                    subtitle=f"[dim]{emo_indicator}[/dim]"
                ))
            else:
                # ── Fall through to LLM ──
                if intent != "UNKNOWN" and confidence > 0.2:
                    console.print(f"[dim]NIE: {intent} ({conf_pct}) - low confidence, asking AI...[/dim]")
                
                with console.status("[bold blue]Thinking...[/bold blue]", spinner="dots"):
                    try:
                        if use_hf:
                            reply = hf_chat_manager.stream_response(command)
                            # Legacy NIE action parsing
                            actions = nie_engine.parse_ai_response(reply)
                            action_results = []
                            for act in actions:
                                r = nie_engine.execute_action(act['type'], act['params'])
                                action_results.append(f"[bold cyan]NIE Action:[/bold cyan] {r}")
                            reply = re.sub(r"\[ACTION:.*?\]", "", reply).strip()
                        else:
                            reply = chat_manager.stream_response(command)
                            actions = nie_engine.parse_ai_response(reply)
                            action_results = []
                            for act in actions:
                                r = nie_engine.execute_action(act['type'], act['params'])
                                action_results.append(f"[bold cyan]NIE Action:[/bold cyan] {r}")
                            reply = re.sub(r"\[ACTION:.*?\]", "", reply).strip()
                    except Exception as e:
                        reply = f"Error: {e}"
                        action_results = []
                
                # Store response in consciousness
                nova_consciousness.store_response(reply)
                
                console.print(Panel(reply, title="[bold green]Nova AI[/bold green]", border_style="green"))
                
                # Show legacy NIE action results if any
                for res in action_results:
                    console.print(Panel(res, border_style="cyan"))
                
        except KeyboardInterrupt:
            nova_consciousness.end_session()
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(description="Nova CLI")
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--interactive', '-i', action='store_true')
    
    try:
        args = parser.parse_args()
        if len(sys.argv) == 1 or getattr(args, 'interactive', False):
            interactive_mode()
        else:
            # If arguments were passed but not handled by argparse (or help shown), exit
            pass
    except SystemExit:
        pass
    except Exception:
        interactive_mode()

if __name__ == "__main__":
    main()
