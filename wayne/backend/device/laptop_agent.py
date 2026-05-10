from __future__ import annotations

import asyncio
import json
import os
import platform
import socket
import subprocess
from typing import Any

import psutil
import requests
import websockets
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
WS_URL = BACKEND_URL.replace("http://", "ws://").replace("https://", "wss://")
DEVICE_ID = os.getenv("LAPTOP_DEVICE_ID", "laptop-001")
HOSTNAME = "WAYNE-Laptop"


def local_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def battery_percent() -> int:
    battery = psutil.sensors_battery()
    return int(battery.percent) if battery else 100


def register() -> None:
    payload = {"device_id": DEVICE_ID, "type": "laptop", "name": HOSTNAME, "ip": local_ip()}
    requests.post(f"{BACKEND_URL}/device/register", json=payload, timeout=10).raise_for_status()


def run(command: list[str] | str) -> None:
    if isinstance(command, str):
        subprocess.Popen(command, shell=True)
    else:
        subprocess.Popen(command)


def execute_volume(command: str) -> None:
    system = platform.system()
    if system == "Darwin":
        if command == "mute":
            run(["osascript", "-e", "set volume output muted true"])
        elif command == "unmute":
            run(["osascript", "-e", "set volume output muted false"])
        elif command == "volume_up":
            run(["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) + 10)"])
        elif command == "volume_down":
            run(["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) - 10)"])
    elif system == "Linux":
        linux_map = {"mute": "amixer set Master mute", "unmute": "amixer set Master unmute", "volume_up": "amixer set Master 10%+", "volume_down": "amixer set Master 10%-"}
        run(linux_map[command])
    elif system == "Windows":
        ps = {
            "mute": "(New-Object -ComObject WScript.Shell).SendKeys([char]173)",
            "unmute": "(New-Object -ComObject WScript.Shell).SendKeys([char]173)",
            "volume_up": "1..5 | % {(New-Object -ComObject WScript.Shell).SendKeys([char]175)}",
            "volume_down": "1..5 | % {(New-Object -ComObject WScript.Shell).SendKeys([char]174)}",
        }
        run(["powershell", "-NoProfile", "-Command", ps[command]])


def execute_command(payload: dict[str, Any]) -> None:
    if not payload.get("confirmed"):
        return
    command = payload.get("command")
    system = platform.system()
    if command == "shutdown":
        run("shutdown /s /t 1" if system == "Windows" else "sudo shutdown -h now")
    elif command == "restart":
        run("shutdown /r /t 1" if system == "Windows" else "sudo shutdown -r now" if system == "Darwin" else "sudo reboot")
    elif command == "sleep":
        run("rundll32.exe powrprof.dll,SetSuspendState 0,1,0" if system == "Windows" else "pmset sleepnow" if system == "Darwin" else "systemctl suspend")
    elif command == "lock":
        run("rundll32.exe user32.dll,LockWorkStation" if system == "Windows" else "pmset displaysleepnow" if system == "Darwin" else "loginctl lock-session")
    elif command == "logout":
        run("shutdown /l" if system == "Windows" else "osascript -e 'tell app \"System Events\" to log out'" if system == "Darwin" else "loginctl terminate-user $USER")
    elif command in {"mute", "unmute", "volume_up", "volume_down"}:
        execute_volume(command)
    elif command == "screen_off":
        run("powershell (Add-Type '[DllImport(\"user32.dll\")] public static extern int SendMessage(int hWnd, int hMsg, int wParam, int lParam);' -Name a -Pas)::SendMessage(-1,0x0112,0xF170,2)" if system == "Windows" else "pmset displaysleepnow")
    elif command == "screen_on":
        return


async def tracking_loop() -> None:
    while True:
        try:
            async with websockets.connect(f"{WS_URL}/ws/track/{DEVICE_ID}") as websocket:
                while True:
                    disk = psutil.disk_usage("/")
                    payload = {
                        "battery": battery_percent(),
                        "cpu_percent": psutil.cpu_percent(),
                        "ram_percent": psutil.virtual_memory().percent,
                        "disk_percent": disk.percent,
                        "ip_address": local_ip(),
                        "online": True,
                    }
                    await websocket.send(json.dumps(payload))
                    await asyncio.sleep(5)
        except Exception:
            await asyncio.sleep(3)


async def command_loop() -> None:
    while True:
        try:
            async with websockets.connect(f"{WS_URL}/ws/commands/{DEVICE_ID}") as websocket:
                async for message in websocket:
                    payload = json.loads(message)
                    execute_command(payload)
        except Exception:
            await asyncio.sleep(3)


async def main() -> None:
    print("W.A.Y.N.E Laptop Agent — Online")
    print(f"Device ID: {DEVICE_ID}")
    print(f"Reporting to: {BACKEND_URL}")
    while True:
        try:
            register()
            break
        except Exception:
            await asyncio.sleep(3)
    await asyncio.gather(tracking_loop(), command_loop())


if __name__ == "__main__":
    asyncio.run(main())
