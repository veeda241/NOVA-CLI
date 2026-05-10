from __future__ import annotations

import os
import sys
import time
from typing import Any

import requests
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console

from ui.display import (
    calendar_table,
    console as rich_console,
    file_table,
    help_panel,
    show_ai_response,
    show_error,
    show_meta,
    show_startup,
    system_table,
    task_table,
    device_table,
)
from voice import listen

BACKEND_URL = os.getenv("WAYNE_BACKEND_URL", "http://localhost:8000")
SESSION_ID = os.getenv("WAYNE_SESSION_ID", "cli")
console = Console()
history: list[dict[str, str]] = []


def api(method: str, path: str, **kwargs: Any) -> Any:
    response = requests.request(method, f"{BACKEND_URL}{path}", timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


def backend_online() -> bool:
    try:
        api("GET", "/system/status")
        return True
    except requests.RequestException:
        return False


def approx_tokens(text: str) -> int:
    return max(1, len(text.split()) + len(text) // 20)


def send_chat(text: str) -> None:
    started = time.perf_counter()
    try:
        result = api("POST", "/chat", json={"messages": history, "query": text, "session_id": SESSION_ID})
        reply = result.get("reply", "")
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply})
        show_ai_response(reply)
        show_meta(approx_tokens(text + reply), time.perf_counter() - started)
        interaction_id = result.get("interaction_id")
        if interaction_id:
            show_feedback_prompt(interaction_id)
    except requests.RequestException as exc:
        show_error(f"Backend unavailable: {exc}")


def show_feedback_prompt(interaction_id: int) -> None:
    rich_console.print("\n[dim]Rate this response: [cyan]1[/]-[cyan]5[/] or Enter to skip[/dim] ", end="")
    try:
        rating_input = input().strip()
    except (KeyboardInterrupt, EOFError):
        return
    if rating_input not in {"1", "2", "3", "4", "5"}:
        return
    try:
        api("POST", "/feedback", json={"interaction_id": interaction_id, "score": int(rating_input), "session_id": SESSION_ID})
        stars = "★" * int(rating_input) + "☆" * (5 - int(rating_input))
        rich_console.print(f"[cyan]{stars}[/] [dim]Feedback recorded.[/]\n")
    except requests.RequestException as exc:
        show_error(f"Feedback failed: {exc}")


def enter_passive_mode() -> None:
    try:
        api("POST", "/wayne/sleep", json={"source": "cli_sleep", "session_id": SESSION_ID})
    except requests.RequestException:
        pass
    rich_console.print("[cyan][W.A.Y.N.E][/cyan] Returning to passive mode. Say 'WAYNE' to reinitialize.")


def show_tasks() -> None:
    rich_console.print(task_table(api("GET", "/tasks")))


def show_files(path: str | None = None) -> None:
    params = {"dir": path} if path else None
    data = api("GET", "/files", params=params)
    rich_console.print(file_table(data.get("files", [])))


def print_json_result(result: Any) -> None:
    import json

    rich_console.print_json(json.dumps(result, default=str))


def file_find(query: str) -> None:
    print_json_result(api("POST", "/files/search", json={"query": query, "max_results": 20}))


def file_read(path: str) -> None:
    result = api("POST", "/files/read", json={"path": path})
    content = result.get("content") or result.get("error") or result
    rich_console.print(str(content)[:6000])


def file_open(path: str) -> None:
    print_json_result(api("POST", "/files/open", json={"path": path}))


def file_operation(operation: str, path: str, **extra: Any) -> None:
    payload = {"operation": operation, "path": path, **extra}
    print_json_result(api("POST", "/files/operation", json=payload))


def pc_post(path: str, confirmed: bool = False, **extra: Any) -> None:
    payload = {"confirmed": confirmed, **extra}
    print_json_result(api("POST", path, json=payload))


def pc_get(path: str, **params: Any) -> None:
    print_json_result(api("GET", path, params=params or None))


def show_schedule() -> None:
    rich_console.print(calendar_table(api("GET", "/events/today")))


def show_status() -> None:
    rich_console.print(system_table(api("GET", "/system/status")))


def show_devices() -> None:
    data = api("GET", "/device/status")
    rich_console.print(device_table(data.get("devices", [])))


def resolve_device(alias: str) -> str:
    if alias.lower() in {"laptop", "macbook", "computer"}:
        return os.getenv("LAPTOP_DEVICE_ID", "laptop-001")
    if alias.lower() in {"iphone", "phone"}:
        return os.getenv("PHONE_DEVICE_ID", "iphone-001")
    return alias


def send_device_command(command: str, device_alias: str, require_yes: bool = True) -> None:
    device_id = resolve_device(device_alias)
    if require_yes:
        answer = input(f"Confirm: {command} on {device_id}? Reply YES to proceed: ")
        if answer.strip() != "YES":
            show_error("Command cancelled.")
            return
    result = api("POST", "/device/command", json={"device_id": device_id, "command": command, "confirmed": True, "issued_by": "cli"})
    show_ai_response(f"[DEVICE CONTROL] {result.get('status')} for {command} on {device_id}.")


def send_notification(message: str) -> None:
    phone_id = os.getenv("PHONE_DEVICE_ID", "iphone-001")
    result = api("POST", "/device/push", json={"device_id": phone_id, "title": "W.A.Y.N.E", "body": message})
    show_ai_response(f"[DEVICE CONTROL] Push notification status: {result.get('status')}.")


def show_history() -> None:
    recent = history[-20:]
    for item in recent:
        prefix = "[cyan]USER[/cyan]" if item["role"] == "user" else "[green]W.A.Y.N.E[/green]"
        rich_console.print(f"{prefix}: {item['content']}")


def handle_command(command: str) -> bool:
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None
    try:
        if cmd == "/tasks":
            show_tasks()
        elif cmd == "/files":
            show_files(arg)
        elif cmd == "/find":
            file_find(arg or "")
        elif cmd == "/ls":
            print_json_result(api("GET", "/files/list", params={"path": arg or os.getcwd()}))
        elif cmd == "/read":
            file_read(arg or "")
        elif cmd == "/open":
            file_open(arg or "")
        elif cmd == "/info":
            print_json_result(api("GET", "/files/info", params={"path": arg or ""}))
        elif cmd == "/mkdir":
            file_operation("create", arg or "", is_directory=True)
        elif cmd == "/rm":
            answer = input(f"Confirm delete {arg}? Reply YES: ")
            file_operation("delete", arg or "", confirmed=answer.strip() == "YES")
        elif cmd in {"/cp", "/mv"}:
            if not arg or len(arg.split(maxsplit=1)) < 2:
                show_error(f"Usage: {cmd} [src] [dest]")
            else:
                src, dest = arg.split(maxsplit=1)
                file_operation("copy" if cmd == "/cp" else "move", src, destination=dest)
        elif cmd == "/watch":
            print_json_result(api("POST", "/files/watch", json={"path": arg or os.getcwd()}))
        elif cmd == "/cache":
            if (arg or "").lower() == "browser":
                pc_post("/pc/cache/browser", confirmed=True)
            elif (arg or "").lower() == "temp":
                pc_post("/pc/cache/temp", confirmed=True)
            else:
                pc_post("/pc/cache/clear", confirmed=True)
        elif cmd == "/dns":
            print_json_result(api("POST", "/pc/dns/flush", json={}))
        elif cmd == "/ram":
            pc_post("/pc/memory/optimize", confirmed=True)
        elif cmd == "/disk":
            pc_post("/pc/disk/cleanup", confirmed=True)
        elif cmd == "/trash":
            pc_post("/pc/trash/empty", confirmed=True)
        elif cmd == "/ps":
            pc_get("/pc/processes", sort_by=arg or "cpu")
        elif cmd == "/kill":
            answer = input(f"Confirm kill {arg}? Reply YES: ")
            pc_post("/pc/processes/kill", confirmed=answer.strip() == "YES", name=arg)
        elif cmd == "/startup":
            pc_get("/pc/startup")
        elif cmd == "/perf":
            pc_post("/pc/performance", confirmed=True, mode=arg or "balanced")
        elif cmd == "/speedtest":
            pc_get("/pc/network/speedtest")
        elif cmd == "/net":
            pc_get("/pc/network")
        elif cmd == "/reg":
            key, _, value = (arg or "").partition(" ")
            pc_get("/pc/registry", key_path=key, value_name=value or None)
        elif cmd == "/schedule":
            show_schedule()
        elif cmd == "/status":
            show_status()
        elif cmd == "/devices":
            show_devices()
        elif cmd in {"/shutdown", "/restart", "/sleep", "/lock"}:
            send_device_command(cmd[1:], arg or "laptop", require_yes=cmd in {"/shutdown", "/restart", "/sleep", "/lock"})
        elif cmd == "/notify":
            send_notification((arg or "").strip().strip('"'))
        elif cmd == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
        elif cmd == "/history":
            show_history()
        elif cmd == "/help":
            help_panel()
        elif cmd == "/voice":
            text = listen()
            if text:
                rich_console.print(f"[dim]Voice:[/dim] {text}")
                send_chat(text)
        elif cmd == "/exit":
            rich_console.print("[cyan]W.A.Y.N.E session closed.[/cyan]")
            return False
        else:
            show_error(f"Unknown command: {cmd}. Type /help.")
    except requests.RequestException as exc:
        show_error(f"Command failed: {exc}")
    return True


def main() -> int:
    online = backend_online()
    try:
        status = api("GET", "/system/status") if online else {"cpu_percent": 0, "ram_percent": 0}
    except requests.RequestException:
        status = {"cpu_percent": 0, "ram_percent": 0}
    show_startup(online, status)
    help_panel()

    bindings = KeyBindings()

    @bindings.add("f2")
    def _(event: Any) -> None:
        text = listen()
        if text:
            event.app.current_buffer.text = text
            event.app.current_buffer.cursor_position = len(text)

    session = PromptSession(key_bindings=bindings)
    while True:
        try:
            text = session.prompt(HTML("<ansicyan>W.A.Y.N.E ❯ </ansicyan>")).strip()
        except (EOFError, KeyboardInterrupt):
            rich_console.print("\n[cyan]W.A.Y.N.E session closed.[/cyan]")
            return 0
        if not text:
            continue
        if text.lower() in {"sleep wayne", "goodbye wayne", "standby wayne", "goodnight wayne"}:
            enter_passive_mode()
            return 0
        if text.startswith("/"):
            if not handle_command(text):
                return 0
        else:
            send_chat(text)


if __name__ == "__main__":
    sys.exit(main())
