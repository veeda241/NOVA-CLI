from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

from fastapi import WebSocket

from database import SessionLocal
from models import ConnectionLog


class ConnectionManager:
    def __init__(self) -> None:
        self.active: dict[str, set[WebSocket]] = {
            "chat": set(),
            "voice": set(),
            "track": set(),
            "commands": set(),
            "system": set(),
        }
        self.devices: dict[str, WebSocket] = {}
        self.sessions: dict[str, WebSocket] = {}
        self.meta: dict[WebSocket, dict[str, Any]] = {}
        self.queue: dict[str, list[dict[str, Any]]] = {}
        self.last_ping: dict[WebSocket, float] = {}
        self._heartbeat_started = False

    async def connect(self, websocket: WebSocket, channel: str, identifier: str | None = None, accept: bool = True) -> None:
        if accept:
            await websocket.accept()
        self.active.setdefault(channel, set()).add(websocket)
        self.meta[websocket] = {
            "channel": channel,
            "identifier": identifier,
            "connected_at": datetime.utcnow().isoformat(),
            "last_seen": time.time(),
        }
        self.last_ping[websocket] = time.time()

        if identifier:
            if channel in {"track", "commands"}:
                self.devices[identifier] = websocket
            if channel in {"chat", "voice"}:
                self.sessions[identifier] = websocket

        await websocket.send_json(
            {
                "type": "connected",
                "channel": channel,
                "identifier": identifier,
                "timestamp": datetime.utcnow().isoformat(),
                "message": "W.A.Y.N.E connection established",
            }
        )
        await self._flush(identifier, websocket)
        self._log(channel, identifier, "connected")

    def disconnect(self, websocket: WebSocket) -> None:
        meta = self.meta.get(websocket, {})
        channel = meta.get("channel")
        identifier = meta.get("identifier")
        if channel:
            self.active.get(channel, set()).discard(websocket)
        if identifier and self.devices.get(identifier) is websocket:
            self.devices.pop(identifier, None)
        if identifier and self.sessions.get(identifier) is websocket:
            self.sessions.pop(identifier, None)
        self.meta.pop(websocket, None)
        self.last_ping.pop(websocket, None)
        self._log(channel or "unknown", identifier, "disconnected")

    async def send_to_session(self, session_id: str, message: dict[str, Any]) -> bool:
        return await self._send_to(self.sessions.get(session_id), session_id, message)

    async def send_to_device(self, device_id: str, message: dict[str, Any]) -> bool:
        return await self._send_to(self.devices.get(device_id), device_id, message)

    async def broadcast(self, channel: str, message: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for websocket in list(self.active.get(channel, set())):
            try:
                await websocket.send_json(message)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self.disconnect(websocket)

    async def heartbeat_loop(self) -> None:
        if self._heartbeat_started:
            return
        self._heartbeat_started = True
        while True:
            await asyncio.sleep(5)
            now = time.time()
            dead: list[WebSocket] = []
            for websocket, last_seen in list(self.last_ping.items()):
                if now - last_seen > 30:
                    dead.append(websocket)
                    continue
                try:
                    await websocket.send_json({"type": "ping", "timestamp": now})
                    self.last_ping[websocket] = now
                except Exception:
                    dead.append(websocket)
            for websocket in dead:
                self.disconnect(websocket)

    def update_ping(self, websocket: WebSocket) -> None:
        self.last_ping[websocket] = time.time()
        if websocket in self.meta:
            self.meta[websocket]["last_seen"] = time.time()

    def get_status(self) -> dict[str, Any]:
        return {
            "total_connections": sum(len(value) for value in self.active.values()),
            "by_channel": {key: len(value) for key, value in self.active.items()},
            "active_devices": sorted(self.devices.keys()),
            "active_sessions": sorted(self.sessions.keys()),
            "queued_messages": sum(len(value) for value in self.queue.values()),
        }

    async def _send_to(self, websocket: WebSocket | None, identifier: str, message: dict[str, Any]) -> bool:
        if not websocket:
            self.queue.setdefault(identifier, []).append(message)
            return False
        try:
            await websocket.send_json(message)
            return True
        except Exception:
            self.disconnect(websocket)
            self.queue.setdefault(identifier, []).append(message)
            return False

    async def _flush(self, identifier: str | None, websocket: WebSocket) -> None:
        if not identifier or identifier not in self.queue:
            return
        for message in self.queue.pop(identifier, []):
            try:
                await websocket.send_json(message)
            except Exception:
                self.queue.setdefault(identifier, []).append(message)
                break

    def _log(self, channel: str, identifier: str | None, event: str) -> None:
        db = SessionLocal()
        try:
            db.add(ConnectionLog(channel=channel, identifier=identifier or "unknown", event=event))
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()


connection_manager = ConnectionManager()
