from __future__ import annotations

import os
import platform
import subprocess
from typing import Any

import psutil


def system_status() -> dict[str, Any]:
    disk = psutil.disk_usage("/")
    memory = psutil.virtual_memory()
    battery = psutil.sensors_battery()
    return {
        "platform": platform.platform(),
        "cpu_percent": psutil.cpu_percent(interval=0.2),
        "ram_percent": memory.percent,
        "ram_used_gb": round(memory.used / (1024**3), 2),
        "ram_total_gb": round(memory.total / (1024**3), 2),
        "disk_percent": disk.percent,
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "battery_percent": battery.percent if battery else None,
        "battery_plugged": battery.power_plugged if battery else None,
    }


def open_app(command: str) -> dict[str, Any]:
    if os.name == "nt":
        subprocess.Popen(["powershell", "-NoProfile", "-Command", f"Start-Process {command}"])
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", "-a", command])
    else:
        subprocess.Popen([command])
    return {"opened": command}
