from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
import uuid
from datetime import datetime
from typing import Any

import httpx
import jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import SessionLocal
from models import DeviceCommandLog, DeviceStatus

POWER_COMMANDS = {
    "shutdown",
    "restart",
    "sleep",
    "lock",
    "logout",
    "screen_off",
    "screen_on",
    "volume_up",
    "volume_down",
    "mute",
    "unmute",
    "do_not_disturb_on",
    "do_not_disturb_off",
}


def normalize_device_id(device_id: str | None) -> str:
    raw = (device_id or "").strip()
    lowered = raw.lower().replace("_", "-")
    if lowered in {"", "laptop", "my laptop", "wayne-laptop", "computer", "pc", "windows"}:
        return os.getenv("LAPTOP_DEVICE_ID", "laptop-001")
    if lowered in {"phone", "iphone", "my phone", "wayne-phone"}:
        return os.getenv("PHONE_DEVICE_ID", "iphone-001")
    return raw


class DeviceConnectionManager:
    def __init__(self) -> None:
        self.command_sockets: dict[str, set[Any]] = {}
        self.tracking_sockets: set[Any] = set()
        self.chat_sockets: dict[str, set[Any]] = {}
        self.pending_commands: dict[str, list[dict[str, Any]]] = {}

    async def connect_command(self, device_id: str, websocket: Any) -> None:
        await websocket.accept()
        self.command_sockets.setdefault(device_id, set()).add(websocket)
        for command in self.pending_commands.pop(device_id, []):
            await websocket.send_json(command)

    def disconnect_command(self, device_id: str, websocket: Any) -> None:
        self.command_sockets.get(device_id, set()).discard(websocket)

    async def connect_tracking(self, websocket: Any) -> None:
        await websocket.accept()
        self.tracking_sockets.add(websocket)

    def disconnect_tracking(self, websocket: Any) -> None:
        self.tracking_sockets.discard(websocket)

    async def connect_chat(self, session_id: str, websocket: Any) -> None:
        await websocket.accept()
        self.chat_sockets.setdefault(session_id, set()).add(websocket)

    def disconnect_chat(self, session_id: str, websocket: Any) -> None:
        self.chat_sockets.get(session_id, set()).discard(websocket)

    async def send_command(self, device_id: str, payload: dict[str, Any]) -> bool:
        device_id = normalize_device_id(device_id)
        sockets = list(self.command_sockets.get(device_id, set()))
        if not sockets:
            self.pending_commands.setdefault(device_id, []).append(payload)
            return False
        for websocket in sockets:
            await websocket.send_json(payload)
        return True

    async def broadcast_tracking(self, payload: dict[str, Any]) -> None:
        for websocket in list(self.tracking_sockets):
            try:
                await websocket.send_json(payload)
            except Exception:
                self.tracking_sockets.discard(websocket)

    async def broadcast_chat(self, session_id: str, payload: dict[str, Any]) -> None:
        for websocket in list(self.chat_sockets.get(session_id, set())):
            try:
                await websocket.send_json(payload)
            except Exception:
                self.disconnect_chat(session_id, websocket)


manager = DeviceConnectionManager()


def serialize_device(device: DeviceStatus) -> dict[str, Any]:
    return {
        "id": device.id,
        "device_id": device.device_id,
        "device_name": device.device_name,
        "device_type": device.device_type,
        "battery_level": device.battery_level,
        "cpu_percent": device.cpu_percent,
        "ram_percent": device.ram_percent,
        "disk_percent": device.disk_percent,
        "is_online": device.is_online,
        "last_seen": device.last_seen.isoformat(timespec="seconds"),
        "ip_address": device.ip_address,
        "latitude": device.latitude,
        "longitude": device.longitude,
        "push_token": device.push_token,
    }


def register_device(
    device_id: str,
    device_type: str,
    name: str,
    push_token: str | None = None,
    ip_address: str | None = None,
    db: Session | None = None,
) -> dict[str, Any]:
    device_id = normalize_device_id(device_id)
    owns_session = db is None
    session = db or SessionLocal()
    try:
        device = session.scalar(select(DeviceStatus).where(DeviceStatus.device_id == device_id))
        if device is None:
            device = DeviceStatus(device_id=device_id, device_name=name, device_type=device_type)
            session.add(device)
        device.device_name = name
        device.device_type = device_type
        device.push_token = push_token or device.push_token
        device.ip_address = ip_address or device.ip_address
        device.is_online = True
        device.last_seen = datetime.now()
        session.commit()
        session.refresh(device)
        return serialize_device(device)
    finally:
        if owns_session:
            session.close()


def update_device_status(device_id: str, payload: dict[str, Any], db: Session | None = None) -> dict[str, Any]:
    device_id = normalize_device_id(device_id)
    owns_session = db is None
    session = db or SessionLocal()
    try:
        device = session.scalar(select(DeviceStatus).where(DeviceStatus.device_id == device_id))
        if device is None:
            device = DeviceStatus(device_id=device_id, device_name=device_id, device_type=payload.get("type", "unknown"))
            session.add(device)
        device.battery_level = int(payload.get("battery", payload.get("battery_level", device.battery_level or 0)) or 0)
        device.cpu_percent = float(payload.get("cpu", payload.get("cpu_percent", device.cpu_percent or 0)) or 0)
        device.ram_percent = float(payload.get("ram", payload.get("ram_percent", device.ram_percent or 0)) or 0)
        device.disk_percent = float(payload.get("disk", payload.get("disk_percent", device.disk_percent or 0)) or 0)
        device.is_online = bool(payload.get("online", True))
        device.ip_address = payload.get("ip", payload.get("ip_address", device.ip_address))
        device.latitude = payload.get("latitude", device.latitude)
        device.longitude = payload.get("longitude", device.longitude)
        device.last_seen = datetime.now()
        session.commit()
        session.refresh(device)
        return serialize_device(device)
    finally:
        if owns_session:
            session.close()


def get_device_status(device_id: str | None = None, db: Session | None = None) -> dict[str, Any]:
    device_id = normalize_device_id(device_id) if device_id else None
    owns_session = db is None
    session = db or SessionLocal()
    try:
        stmt = select(DeviceStatus).order_by(DeviceStatus.device_type.asc(), DeviceStatus.device_name.asc())
        if device_id:
            stmt = stmt.where(DeviceStatus.device_id == device_id)
        return {"devices": [serialize_device(device) for device in session.scalars(stmt).all()]}
    finally:
        if owns_session:
            session.close()


async def send_device_command_async(
    device_id: str,
    command: str,
    confirmed: bool = False,
    issued_by: str = "wayne",
    db: Session | None = None,
) -> dict[str, Any]:
    device_id = normalize_device_id(device_id)
    if command not in POWER_COMMANDS:
        raise ValueError(f"Unsupported command: {command}")
    if not confirmed:
        return {"needs_confirmation": True, "message": f"Confirm: should I {command} your {device_id}?"}

    owns_session = db is None
    session = db or SessionLocal()
    try:
        payload = {"device_id": device_id, "command": command, "confirmed": True, "timestamp": datetime.now().isoformat()}
        delivered = await manager.send_command(device_id, payload)
        local_result = None
        if not delivered and device_id == os.getenv("LAPTOP_DEVICE_ID", "laptop-001"):
            local_result = execute_local_laptop_command(command)
            delivered = local_result["status"] == "executed"
        log = DeviceCommandLog(
            device_id=device_id,
            command=command,
            issued_by=issued_by,
            confirmed=True,
            status="command_sent" if delivered else "queued",
        )
        session.add(log)
        session.commit()
        return {
            "status": "command_sent" if delivered else "queued",
            "device_id": device_id,
            "command": command,
            "timestamp": payload["timestamp"],
            "delivery": "websocket" if delivered and local_result is None else "local_backend" if local_result else "queued_until_agent_connects",
            "local_result": local_result,
        }
    finally:
        if owns_session:
            session.close()


def send_device_command(device_id: str, command: str, confirmed: bool = False, issued_by: str = "wayne", db: Session | None = None) -> dict[str, Any]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(send_device_command_async(device_id, command, confirmed, issued_by, db))
    raise RuntimeError("send_device_command_sync called inside an active loop; use send_device_command_async")


def execute_local_laptop_command(command: str) -> dict[str, Any]:
    system = platform.system()
    command_map: dict[str, list[str] | str] = {}
    if system == "Windows":
        command_map = {
            "shutdown": ["shutdown", "/s", "/t", "5"],
            "restart": ["shutdown", "/r", "/t", "5"],
            "sleep": ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
            "lock": ["rundll32.exe", "user32.dll,LockWorkStation"],
            "logout": ["shutdown", "/l"],
            "screen_off": [
                "powershell",
                "-NoProfile",
                "-WindowStyle",
                "Hidden",
                "-Command",
                "(Add-Type '[DllImport(\"user32.dll\")] public static extern int SendMessage(int hWnd, int hMsg, int wParam, int lParam);' -Name a -Pas)::SendMessage(-1,0x0112,0xF170,2)",
            ],
        }
    elif system == "Darwin":
        command_map = {
            "shutdown": ["osascript", "-e", 'tell app "System Events" to shut down'],
            "restart": ["osascript", "-e", 'tell app "System Events" to restart'],
            "sleep": ["pmset", "sleepnow"],
            "lock": ["pmset", "displaysleepnow"],
            "logout": ["osascript", "-e", 'tell app "System Events" to log out'],
            "screen_off": ["pmset", "displaysleepnow"],
        }
    elif system == "Linux":
        command_map = {
            "shutdown": ["systemctl", "poweroff"],
            "restart": ["systemctl", "reboot"],
            "sleep": ["systemctl", "suspend"],
            "lock": ["loginctl", "lock-session"],
            "logout": ["loginctl", "terminate-user", os.getenv("USER", "")],
            "screen_off": "xset dpms force off",
        }
    if command in {"mute", "unmute", "volume_up", "volume_down"}:
        return execute_local_volume_command(command)
    local_command = command_map.get(command)
    if not local_command:
        return {"status": "not_supported_locally", "command": command, "platform": system}
    try:
        if isinstance(local_command, str):
            subprocess.Popen(local_command, shell=True)
        else:
            subprocess.Popen(local_command)
        return {"status": "executed", "command": command, "platform": system}
    except Exception as exc:
        return {"status": "failed", "command": command, "platform": system, "error": str(exc)}


def execute_local_volume_command(command: str) -> dict[str, Any]:
    try:
        system = platform.system()
        if system == "Windows":
            script = {
                "mute": "(New-Object -ComObject WScript.Shell).SendKeys([char]173)",
                "unmute": "(New-Object -ComObject WScript.Shell).SendKeys([char]173)",
                "volume_up": "1..5 | % {(New-Object -ComObject WScript.Shell).SendKeys([char]175)}",
                "volume_down": "1..5 | % {(New-Object -ComObject WScript.Shell).SendKeys([char]174)}",
            }[command]
            subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script])
        elif system == "Darwin":
            scripts = {
                "mute": "set volume output muted true",
                "unmute": "set volume output muted false",
                "volume_up": "set volume output volume ((output volume of (get volume settings)) + 10)",
                "volume_down": "set volume output volume ((output volume of (get volume settings)) - 10)",
            }
            subprocess.Popen(["osascript", "-e", scripts[command]])
        else:
            scripts = {"mute": "amixer set Master mute", "unmute": "amixer set Master unmute", "volume_up": "amixer set Master 10%+", "volume_down": "amixer set Master 10%-"}
            subprocess.Popen(scripts[command], shell=True)
        return {"status": "executed", "command": command, "platform": system}
    except Exception as exc:
        return {"status": "failed", "command": command, "error": str(exc)}


def _apns_token() -> str:
    key_id = os.getenv("APNS_KEY_ID", "")
    team_id = os.getenv("APNS_TEAM_ID", "")
    cert_path = os.getenv("APNS_CERT_PATH", "./certs/apns.p8")
    if not key_id or not team_id or not os.path.exists(cert_path):
        raise RuntimeError("APNs credentials are not configured.")
    private_key = open(cert_path, "r", encoding="utf-8").read()
    return jwt.encode({"iss": team_id, "iat": int(datetime.now().timestamp())}, private_key, algorithm="ES256", headers={"alg": "ES256", "kid": key_id})


async def send_push_notification_async(title: str, body: str, device_id: str, db: Session | None = None) -> dict[str, Any]:
    owns_session = db is None
    session = db or SessionLocal()
    try:
        device = session.scalar(select(DeviceStatus).where(DeviceStatus.device_id == device_id))
        if device is None or not device.push_token:
            return {"status": "no_push_token", "message_id": None}
        bundle_id = os.getenv("APNS_BUNDLE_ID", "com.wayne.assistant")
        endpoint = f"https://api.sandbox.push.apple.com/3/device/{device.push_token}"
        payload = {"aps": {"alert": {"title": title, "body": body}, "sound": "default"}}
        headers = {"authorization": f"bearer {_apns_token()}", "apns-topic": bundle_id}
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
        return {"status": "sent" if response.status_code < 300 else "failed", "message_id": response.headers.get("apns-id", str(uuid.uuid4()))}
    finally:
        if owns_session:
            session.close()


def send_push_notification(title: str, body: str, device_id: str, db: Session | None = None) -> dict[str, Any]:
    return asyncio.run(send_push_notification_async(title, body, device_id, db))
