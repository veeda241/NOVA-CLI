from __future__ import annotations

import asyncio
import base64
import uuid
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from core.brain import (
    OllamaUnavailable,
    chat_with_ollama,
    current_model,
    list_ollama_models,
    select_model,
    stream_chat_with_ollama,
)
from core.memory import add_message, get_history
from core.offline import check_ollama, offline_reply
from core.rl.agent import agent
from core.rl_engine import rl_engine
from core.voice_session import VoiceSession
from database import DATABASE_URL, SessionLocal, engine, get_db, init_db
from models import (
    ExperienceBuffer,
    FileAccessLog,
    GoldenResponse,
    Habit,
    ImprovementLog,
    Interaction,
    KnownContact,
    LanguagePattern,
    TopicTracker,
    UserPreference,
    UserState,
    WakeEvent,
)
from tools.calendar_tool import authorization_url, handle_callback, today_events
from tools.device_control import (
    get_device_status,
    manager,
    register_device,
    send_device_command_async,
    send_push_notification_async,
    update_device_status,
)
from tools.file_engine import file_engine
from tools.file_indexer import file_indexer
from tools.file_tool import list_files, open_file
from tools.file_watcher import file_watcher
from tools.pc_manager import pc_manager
from tools.system_tool import system_status
from tools.task_tool import create_task, delete_task, list_tasks, toggle_task

app = FastAPI(title="W.A.Y.N.E Shared Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

wake_event_sockets: set[WebSocket] = set()
pending_device_confirmations: dict[str, dict[str, str]] = {}


def _serialize_wake_event(event: WakeEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "source": event.source,
        "session_id": event.session_id,
        "event_type": event.event_type,
        "fired_at": event.fired_at.isoformat(timespec="seconds"),
    }


def _parse_power_command(text: str) -> dict[str, str] | None:
    lowered = text.lower().strip()
    command_aliases = {
        "shutdown": ["shutdown", "shut down", "power off", "turn off"],
        "restart": ["restart", "reboot"],
        "sleep": ["sleep", "suspend"],
        "lock": ["lock", "lock screen", "lock my screen"],
        "logout": ["logout", "log out", "sign out"],
    }
    device = "laptop-001"
    if "phone" in lowered or "iphone" in lowered:
        device = "iphone-001"
    elif "laptop" not in lowered and "computer" not in lowered and "pc" not in lowered:
        return None
    for command, phrases in command_aliases.items():
        if any(phrase in lowered for phrase in phrases):
            return {"device_id": device, "command": command}
    return None


async def _broadcast_wake_event(payload: dict[str, Any]) -> None:
    for websocket in list(wake_event_sockets):
        try:
            await websocket.send_json({"type": "wayne_event", "event": payload})
        except Exception:
            wake_event_sockets.discard(websocket)


async def _record_wake_event(db: Session, source: str, event_type: str, session_id: str | None = None) -> dict[str, Any]:
    event = WakeEvent(
        source=source or event_type,
        event_type=event_type,
        session_id=session_id or str(uuid.uuid4()),
        fired_at=datetime.now(),
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    payload = _serialize_wake_event(event)
    await _broadcast_wake_event(payload)
    return payload


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "W.A.Y.N.E",
        "full_name": "Wireless Artificial Yielding Network Engine",
        "status": "online",
        "engine": current_model(),
        "docs": "/docs",
        "health": "/system/status",
        "web": "http://localhost:3000",
    }


class ChatRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    query: str
    session_id: str = "default"
    prev_interaction_id: int | None = None


class TaskCreate(BaseModel):
    title: str
    priority: str = "medium"


class FileOpenRequest(BaseModel):
    path: str


class FileSearchRequest(BaseModel):
    query: str
    file_type: str | None = None
    directory: str | None = None
    include_restricted: bool = False
    max_results: int = 20


class FileReadRequest(BaseModel):
    path: str
    summarize: bool = False
    force_restricted: bool = False


class FileWriteRequest(BaseModel):
    path: str
    content: str
    mode: str = "overwrite"


class FileOperationRequest(BaseModel):
    operation: str
    path: str
    destination: str | None = None
    new_name: str | None = None
    content: str = ""
    is_directory: bool = False
    permanent: bool = False
    confirmed: bool = False


class FileWatchRequest(BaseModel):
    path: str


class PCConfirmRequest(BaseModel):
    confirmed: bool = False


class PCKillRequest(BaseModel):
    name: str | None = None
    pid: int | None = None
    confirmed: bool = False


class PCStartupRequest(BaseModel):
    name: str
    confirmed: bool = False


class PCDNSRequest(BaseModel):
    primary: str
    secondary: str = "8.8.4.4"
    confirmed: bool = False


class PCPerformanceRequest(BaseModel):
    mode: str = "balanced"
    confirmed: bool = False


class PCRegistryWriteRequest(BaseModel):
    key_path: str
    value_name: str
    value_data: str
    value_type: str = "REG_SZ"
    confirmed: bool = False


class DeviceRegisterRequest(BaseModel):
    device_id: str
    type: str
    name: str
    push_token: str | None = None
    ip: str | None = None


class DeviceCommandRequest(BaseModel):
    device_id: str
    command: str
    confirmed: bool = False
    issued_by: str = "wayne"


class PushRequest(BaseModel):
    title: str
    body: str
    device_id: str


class FeedbackRequest(BaseModel):
    interaction_id: int
    score: float
    feedback_text: str | None = None
    session_id: str = "default"
    voice_feedback: str | None = None


class ImplicitFeedbackRequest(BaseModel):
    interaction_id: int
    next_user_message: str
    session_id: str = "default"


class PreferencePatchRequest(BaseModel):
    key: str
    value: str


class VoiceFeedbackRequest(BaseModel):
    interaction_id: int
    transcript: str


@app.post("/chat")
async def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    if payload.prev_interaction_id:
        previous = db.get(Interaction, payload.prev_interaction_id)
        if previous:
            implicit = rl_engine.calculate_implicit_reward(db, previous.user_message, previous.wayne_response, payload.query)
            previous.implicit_score = implicit
            previous.final_reward = implicit if previous.explicit_score is None else rl_engine.calculate_final_reward(
                rl_engine.calculate_explicit_reward(previous.explicit_score),
                implicit,
            )
            db.commit()
            rl_engine.infer_preferences_from_interaction(db, previous.user_message, previous.wayne_response, previous.final_reward)

    lowered_query = payload.query.lower()
    if any(phrase in lowered_query for phrase in ("sleep wayne", "goodbye wayne", "standby wayne", "goodnight wayne")):
        event = await _record_wake_event(db, "sleep_command", "sleep", payload.session_id)
        reply = "[AI RESPONSE] W.A.Y.N.E standing by. Say my name when you need me, Sir."
        add_message(db, payload.session_id, "user", payload.query)
        add_message(db, payload.session_id, "assistant", reply)
        return {"reply": reply, "tool_used": "wayne_sleep", "messages": [{"role": "assistant", "content": reply}], "event": event}

    if lowered_query.strip() in {"yes", "y", "confirm", "confirmed", "proceed"} and payload.session_id in pending_device_confirmations:
        pending = pending_device_confirmations.pop(payload.session_id)
        result = await send_device_command_async(pending["device_id"], pending["command"], True, "wayne", db)
        reply = f"[DEVICE CONTROL] Done, Sir. {pending['command'].title()} command sent to {pending['device_id']}."
        if result.get("status") == "queued":
            reply = f"[DEVICE CONTROL] Command queued, Sir. Start the laptop agent if it does not execute."
        add_message(db, payload.session_id, "user", payload.query)
        add_message(db, payload.session_id, "assistant", reply)
        return {"reply": reply, "tool_used": "send_device_command", "device_result": result, "messages": [{"role": "assistant", "content": reply}]}

    parsed_power = _parse_power_command(payload.query)
    if parsed_power:
        pending_device_confirmations[payload.session_id] = parsed_power
        reply = f"[DEVICE CONTROL] Confirm: should I {parsed_power['command']} your {parsed_power['device_id']}? Reply YES to proceed."
        add_message(db, payload.session_id, "user", payload.query)
        add_message(db, payload.session_id, "assistant", reply)
        return {"reply": reply, "tool_used": "send_device_command_pending", "needs_confirmation": True, "messages": [{"role": "assistant", "content": reply}]}

    add_message(db, payload.session_id, "user", payload.query)
    messages = payload.messages or get_history(db, payload.session_id)

    if not await check_ollama():
        result = offline_reply(payload.query, db)
    else:
        try:
            result = await chat_with_ollama(messages, payload.query, db, session_id=payload.session_id)
        except OllamaUnavailable as exc:
            result = offline_reply(payload.query, db, str(exc))
            result["interaction_id"] = rl_engine.log_interaction(
                db,
                payload.session_id,
                payload.query,
                result["reply"],
                intent="offline",
                tool_used=result.get("tool_used"),
                token_count=len(result["reply"].split()),
                implicit_score=0.5,
            )

    add_message(db, payload.session_id, "assistant", result["reply"])
    await rl_engine.analyze_and_improve(db)
    return result


@app.post("/feedback")
async def feedback(payload: FeedbackRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    final_reward = rl_engine.update_interaction_reward(db, payload.interaction_id, payload.score, payload.feedback_text)
    if final_reward is None:
        raise HTTPException(status_code=404, detail="Interaction not found")
    interaction = db.get(Interaction, payload.interaction_id)
    if interaction:
        state = interaction.state_snapshot or {"intent": interaction.intent or "general", "emotion": "neutral"}
        final_reward = await agent.learn(
            interaction_id=payload.interaction_id,
            state=state,
            action=interaction.action_type or "feedback_update",
            wayne_response=interaction.wayne_response,
            user_message=interaction.user_message,
            explicit_score=payload.score,
            voice_feedback=payload.voice_feedback,
            response_time_ms=interaction.response_time_ms,
        )
    return {"status": "feedback_received", "final_reward": final_reward}


@app.post("/feedback/voice")
async def feedback_voice(payload: VoiceFeedbackRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    lowered = payload.transcript.lower()
    positive = ["good answer", "perfect", "correct", "well done", "that's right", "exactly", "great job"]
    negative = ["wrong", "incorrect", "try again", "that's wrong", "not right", "no that's not it"]
    if any(phrase in lowered for phrase in positive):
        score = 5.0
        sentiment = "positive"
    elif any(phrase in lowered for phrase in negative):
        score = 1.0
        sentiment = "negative"
    else:
        return {"reward_applied": False, "detected_sentiment": "none"}
    final_reward = rl_engine.update_interaction_reward(db, payload.interaction_id, score, payload.transcript)
    if final_reward is None:
        raise HTTPException(status_code=404, detail="Interaction not found")
    interaction = db.get(Interaction, payload.interaction_id)
    if interaction:
        final_reward = await agent.learn(
            interaction_id=payload.interaction_id,
            state=interaction.state_snapshot or {"intent": interaction.intent or "general", "emotion": "neutral"},
            action=interaction.action_type or "voice_feedback",
            wayne_response=interaction.wayne_response,
            user_message=interaction.user_message,
            explicit_score=score,
            voice_feedback=payload.transcript,
            response_time_ms=interaction.response_time_ms,
        )
    return {"reward_applied": True, "detected_sentiment": sentiment, "final_reward": final_reward}


@app.post("/feedback/implicit")
async def feedback_implicit(payload: ImplicitFeedbackRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    interaction = db.get(Interaction, payload.interaction_id)
    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")
    implicit = rl_engine.calculate_implicit_reward(db, interaction.user_message, interaction.wayne_response, payload.next_user_message)
    interaction.implicit_score = implicit
    interaction.final_reward = implicit
    db.commit()
    rl_engine.infer_preferences_from_interaction(db, interaction.user_message, interaction.wayne_response, implicit)
    return {"implicit_score": implicit}


@app.get("/learning/stats")
def learning_stats(db: Session = Depends(get_db)) -> dict[str, Any]:
    return rl_engine.get_learning_stats(db)


@app.get("/learning/preferences")
def learning_preferences(db: Session = Depends(get_db)) -> dict[str, Any]:
    return {"preferences": rl_engine.get_preference_rows(db)}


@app.patch("/learning/preferences")
def patch_learning_preference(payload: PreferencePatchRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    pref = db.scalar(select(UserPreference).where(UserPreference.preference_key == payload.key))
    if pref:
        pref.preference_value = payload.value
        pref.confidence = 1.0
        pref.sample_count += 1
        pref.last_updated = datetime.now()
    else:
        db.add(UserPreference(preference_key=payload.key, preference_value=payload.value, confidence=1.0, sample_count=1))
    db.commit()
    return {"status": "updated"}


@app.get("/learning/history")
def learning_history(limit: int = 20, min_reward: float = 0.0, db: Session = Depends(get_db)) -> dict[str, Any]:
    return {"interactions": rl_engine.get_history(db, limit=limit, min_reward=min_reward)}


@app.get("/learning/habits")
def learning_habits(db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.scalars(select(Habit).order_by(Habit.confidence.desc(), Habit.frequency.desc())).all()
    return {
        "habits": [
            {
                "id": row.id,
                "habit_type": row.habit_type,
                "pattern": row.pattern,
                "frequency": row.frequency,
                "confidence": row.confidence,
                "last_observed": row.last_observed.isoformat(timespec="seconds"),
                "metadata": row.metadata_json,
            }
            for row in rows
        ]
    }


@app.get("/learning/contacts")
def learning_contacts(db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.scalars(select(KnownContact).order_by(KnownContact.mention_count.desc())).all()
    return {
        "contacts": [
            {
                "id": row.id,
                "name": row.name,
                "email": row.email,
                "mention_count": row.mention_count,
                "relationship": row.relationship,
                "last_mentioned": row.last_mentioned.isoformat(timespec="seconds"),
            }
            for row in rows
        ]
    }


@app.get("/learning/improvements")
def learning_improvements(db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.scalars(select(ImprovementLog).order_by(ImprovementLog.applied_at.desc()).limit(50)).all()
    return {
        "improvements": [
            {
                "id": row.id,
                "improvement": row.improvement,
                "improvement_type": row.improvement_type,
                "old_behavior": row.old_behavior,
                "new_behavior": row.new_behavior,
                "trigger_reason": row.trigger_reason or row.triggered_by,
                "reward_delta": row.reward_delta,
                "applied_at": row.applied_at.isoformat(timespec="seconds"),
            }
            for row in rows
        ]
    }


@app.get("/learning/heatmap")
def learning_heatmap(db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = db.scalars(select(Interaction)).all()
    grid = [[0 for _ in range(24)] for _ in range(7)]
    for row in rows:
        when = row.timestamp or row.created_at
        grid[when.weekday()][when.hour] += 1
    return {"heatmap": grid}


@app.delete("/learning/reset")
def learning_reset(db: Session = Depends(get_db)) -> dict[str, Any]:
    for model in (
        ExperienceBuffer,
        GoldenResponse,
        Habit,
        LanguagePattern,
        KnownContact,
        FileAccessLog,
        ImprovementLog,
        TopicTracker,
        UserState,
        UserPreference,
    ):
        db.query(model).delete()
    db.commit()
    rl_engine.ensure_defaults(db)
    return {"status": "reset", "message": "W.A.Y.N.E learning data reset to defaults."}


@app.get("/tasks")
def tasks(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return list_tasks(db)


@app.post("/tasks")
def add_task(payload: TaskCreate, db: Session = Depends(get_db)) -> dict[str, Any]:
    return create_task(db, payload.title, payload.priority)


@app.patch("/tasks/{task_id}")
def patch_task(task_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        return toggle_task(db, task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/tasks/{task_id}")
def remove_task(task_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        return delete_task(db, task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/files")
def files(dir: str | None = Query(default=None)) -> dict[str, Any]:
    try:
        return list_files(dir)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/files/open")
async def open_file_route(payload: FileOpenRequest) -> dict[str, Any]:
    return await file_engine.open(payload.path)


@app.post("/files/read")
async def read_file_engine(payload: FileReadRequest) -> dict[str, Any]:
    return await file_engine.read(payload.path, payload.summarize, payload.force_restricted)


@app.post("/files/search")
async def search_files(payload: FileSearchRequest) -> dict[str, Any]:
    return await file_engine.search(payload.query, payload.file_type, payload.directory, payload.max_results, payload.include_restricted)


@app.post("/files/write")
async def write_file_engine(payload: FileWriteRequest) -> dict[str, Any]:
    return await file_engine.write(payload.path, payload.content, payload.mode)


@app.post("/files/operation")
async def file_operation(payload: FileOperationRequest) -> dict[str, Any]:
    operation = payload.operation.lower()
    if operation == "copy":
        if not payload.destination:
            raise HTTPException(status_code=400, detail="destination required")
        return await file_engine.copy(payload.path, payload.destination)
    if operation == "move":
        if not payload.destination:
            raise HTTPException(status_code=400, detail="destination required")
        return await file_engine.move(payload.path, payload.destination)
    if operation == "rename":
        if not payload.new_name:
            raise HTTPException(status_code=400, detail="new_name required")
        return await file_engine.rename(payload.path, payload.new_name)
    if operation == "delete":
        return await file_engine.delete(payload.path, payload.permanent, payload.confirmed)
    if operation == "create":
        return await file_engine.create(payload.path, payload.content, payload.is_directory)
    if operation == "extract":
        return await file_engine.extract_archive(payload.path, payload.destination)
    if operation == "info":
        return await file_engine.get_info(payload.path)
    raise HTTPException(status_code=400, detail=f"Unsupported file operation: {operation}")


@app.get("/files/list")
async def list_directory_route(path: str | None = None, show_hidden: bool = False) -> dict[str, Any]:
    return await file_engine.list_directory(path, show_hidden)


@app.get("/files/info")
async def file_info_route(path: str) -> dict[str, Any]:
    return await file_engine.get_info(path)


@app.post("/files/watch")
def watch_file_route(payload: FileWatchRequest) -> dict[str, Any]:
    try:
        return file_watcher.watch(payload.path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/files/watched")
def watched_files_route() -> dict[str, Any]:
    return {"watched": file_watcher.get_watched()}


@app.post("/pc/cache/clear")
async def pc_clear_cache(payload: PCConfirmRequest) -> dict[str, Any]:
    return await pc_manager.clear_all_cache(payload.confirmed)


@app.post("/pc/cache/browser")
async def pc_clear_browser(payload: PCConfirmRequest) -> dict[str, Any]:
    return await pc_manager.clear_browser_cache(payload.confirmed)


@app.post("/pc/cache/temp")
async def pc_clear_temp(payload: PCConfirmRequest) -> dict[str, Any]:
    return await pc_manager.clear_temp_files(payload.confirmed)


@app.post("/pc/dns/flush")
async def pc_flush_dns() -> dict[str, Any]:
    return await pc_manager.flush_dns()


@app.post("/pc/memory/optimize")
async def pc_optimize_memory(payload: PCConfirmRequest) -> dict[str, Any]:
    return await pc_manager.optimize_memory(payload.confirmed)


@app.get("/pc/memory")
async def pc_memory() -> dict[str, Any]:
    return await pc_manager.get_memory_info()


@app.post("/pc/disk/cleanup")
async def pc_disk_cleanup(payload: PCConfirmRequest) -> dict[str, Any]:
    return await pc_manager.disk_cleanup(payload.confirmed)


@app.post("/pc/trash/empty")
async def pc_empty_trash(payload: PCConfirmRequest) -> dict[str, Any]:
    return await pc_manager.empty_trash(payload.confirmed)


@app.get("/pc/status")
async def pc_status() -> dict[str, Any]:
    return await pc_manager.get_system_status()


@app.get("/pc/processes")
async def pc_processes(sort_by: str = "cpu") -> dict[str, Any]:
    return await pc_manager.list_processes(sort_by)


@app.post("/pc/processes/kill")
async def pc_kill_process(payload: PCKillRequest) -> dict[str, Any]:
    return await pc_manager.kill_process(payload.name, payload.pid, payload.confirmed)


@app.get("/pc/startup")
async def pc_startup() -> dict[str, Any]:
    return await pc_manager.get_startup_programs()


@app.post("/pc/startup/disable")
async def pc_startup_disable(payload: PCStartupRequest) -> dict[str, Any]:
    return await pc_manager.disable_startup_program(payload.name, payload.confirmed)


@app.get("/pc/disk")
async def pc_disk() -> dict[str, Any]:
    return await pc_manager.get_disk_info()


@app.get("/pc/network")
async def pc_network() -> dict[str, Any]:
    return await pc_manager.get_network_info()


@app.post("/pc/network/dns")
async def pc_set_dns(payload: PCDNSRequest) -> dict[str, Any]:
    return await pc_manager.set_dns(payload.primary, payload.secondary, payload.confirmed)


@app.get("/pc/network/speedtest")
async def pc_speedtest() -> dict[str, Any]:
    return await pc_manager.network_speed_test()


@app.post("/pc/performance")
async def pc_performance(payload: PCPerformanceRequest) -> dict[str, Any]:
    return await pc_manager.set_performance_mode(payload.mode, payload.confirmed)


@app.get("/pc/registry")
async def pc_registry(key_path: str, value_name: str | None = None) -> dict[str, Any]:
    return await pc_manager.read_registry(key_path, value_name)


@app.post("/pc/registry")
async def pc_registry_write(payload: PCRegistryWriteRequest) -> dict[str, Any]:
    return await pc_manager.write_registry(payload.key_path, payload.value_name, payload.value_data, payload.value_type, payload.confirmed)


@app.get("/indexer/status")
def indexer_status() -> dict[str, Any]:
    return {"indexed": file_indexer.indexed_count, "is_running": file_indexer.is_running, "last_error": file_indexer.last_error}


@app.post("/indexer/start")
def indexer_start(body: dict[str, Any] | None = None) -> dict[str, Any]:
    roots = body.get("roots") if body else None
    return file_indexer.start_background_index(roots)


@app.get("/events/today")
def events_today() -> list[dict[str, Any]]:
    try:
        return today_events()
    except Exception:
        return []


@app.get("/system/status")
def status() -> dict[str, Any]:
    return system_status()


@app.get("/database/status")
def database_status() -> dict[str, Any]:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    backend = "supabase-postgres" if DATABASE_URL.startswith("postgresql") else "sqlite"
    safe_url = DATABASE_URL
    if "@" in safe_url:
        prefix, suffix = safe_url.rsplit("@", 1)
        scheme = prefix.split("://", 1)[0]
        safe_url = f"{scheme}://***:***@{suffix}"
    return {"status": "connected", "backend": backend, "url": safe_url}


@app.get("/models")
async def models() -> dict[str, Any]:
    try:
        return await list_ollama_models()
    except Exception:
        return {"models": [], "current": current_model(), "error": "Ollama is not running"}


@app.post("/models/select")
async def models_select(body: dict[str, Any]) -> dict[str, str]:
    model = select_model(body.get("model", "qwen2.5:1.5b"))
    return {"status": "switched", "model": model}


@app.post("/wayne/wake")
async def wake_detected(body: dict[str, Any], db: Session = Depends(get_db)) -> dict[str, Any]:
    event = await _record_wake_event(db, body.get("source", "wake_word"), "wake", body.get("session_id"))
    return {
        "status": "wake_acknowledged",
        "session_id": event["session_id"],
        "message": "W.A.Y.N.E initializing",
        "event": event,
    }


@app.post("/wayne/sleep")
async def wayne_sleep(body: dict[str, Any], db: Session = Depends(get_db)) -> dict[str, Any]:
    event = await _record_wake_event(db, body.get("source", "sleep_command"), "sleep", body.get("session_id"))
    return {
        "status": "standby_acknowledged",
        "session_id": event["session_id"],
        "message": "W.A.Y.N.E standing by",
        "event": event,
    }


@app.get("/wayne/status")
async def wayne_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    latest = db.scalar(select(WakeEvent).order_by(WakeEvent.fired_at.desc(), WakeEvent.id.desc()).limit(1))
    return {"online": True, "last_wake": _serialize_wake_event(latest) if latest else None}


@app.post("/device/register")
def device_register(payload: DeviceRegisterRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    return register_device(payload.device_id, payload.type, payload.name, payload.push_token, payload.ip, db)


@app.get("/device/status")
def device_status(device_id: str | None = None, db: Session = Depends(get_db)) -> dict[str, Any]:
    return get_device_status(device_id, db)


@app.post("/device/command")
async def device_command(payload: DeviceCommandRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    return await send_device_command_async(payload.device_id, payload.command, payload.confirmed, payload.issued_by, db)


@app.post("/device/push")
async def device_push(payload: PushRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    return await send_push_notification_async(payload.title, payload.body, payload.device_id, db)


@app.get("/auth/google")
def auth_google() -> RedirectResponse:
    return RedirectResponse(authorization_url())


@app.get("/auth/callback")
def auth_callback(code: str) -> dict[str, str]:
    return handle_callback(code)


@app.websocket("/ws/track/{device_id}")
async def websocket_track(websocket: WebSocket, device_id: str) -> None:
    await manager.connect_tracking(websocket)
    try:
        while True:
            payload = await websocket.receive_json()
            updated = update_device_status(device_id, payload)
            await manager.broadcast_tracking({"type": "device_update", "device": updated})
    except WebSocketDisconnect:
        manager.disconnect_tracking(websocket)
        update_device_status(device_id, {"online": False})


@app.websocket("/ws/commands/{device_id}")
async def websocket_commands(websocket: WebSocket, device_id: str) -> None:
    await manager.connect_command(device_id, websocket)
    try:
        while True:
            message = await websocket.receive_text()
            await websocket.send_json({"type": "ack", "received": message})
    except WebSocketDisconnect:
        manager.disconnect_command(device_id, websocket)


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str) -> None:
    await manager.connect_chat(session_id, websocket)
    try:
        while True:
            payload = await websocket.receive_json()
            query = payload.get("query", "")
            db = SessionLocal()
            try:
                if not await check_ollama():
                    result = offline_reply(query, db)
                    for token in result["reply"].split():
                        await websocket.send_json({"type": "ai_token", "token": f"{token} "})
                    await websocket.send_json({"type": "ai_done", "full_text": result["reply"]})
                else:
                    try:
                        full_text = ""
                        async for token in stream_chat_with_ollama(payload.get("messages", []), query, db):
                            full_text += token
                            await websocket.send_json({"type": "ai_token", "token": token})
                        await websocket.send_json({"type": "ai_done", "full_text": full_text})
                        result = {"reply": full_text}
                    except OllamaUnavailable as exc:
                        result = offline_reply(query, db, str(exc))
                        for token in result["reply"].split():
                            await websocket.send_json({"type": "ai_token", "token": f"{token} "})
                        await websocket.send_json({"type": "ai_done", "full_text": result["reply"]})
            finally:
                db.close()
            await manager.broadcast_chat(session_id, {"type": "message", "reply": result["reply"]})
    except WebSocketDisconnect:
        manager.disconnect_chat(session_id, websocket)


@app.websocket("/ws/wayne/events")
async def websocket_wayne_events(websocket: WebSocket) -> None:
    await websocket.accept()
    wake_event_sockets.add(websocket)
    try:
        await websocket.send_json({"type": "ready", "message": "W.A.Y.N.E event stream ready"})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        wake_event_sockets.discard(websocket)


async def _run_voice_turn(websocket: WebSocket, session: VoiceSession, transcript: str) -> None:
    session.is_speaking = True
    session.is_listening = False
    db = SessionLocal()
    try:
        if not await check_ollama():
            result = offline_reply(transcript, db)
        else:
            try:
                full_text = ""
                async for token in stream_chat_with_ollama([], f"[VOICE] {transcript}", db, session_id=session.session_id):
                    if not session.is_speaking:
                        return
                    full_text += token
                    await websocket.send_json({"type": "ai_token", "token": token})
                await websocket.send_json({"type": "ai_done", "full_text": full_text})
                session.is_speaking = False
                session.is_listening = True
                return
            except OllamaUnavailable as exc:
                result = offline_reply(transcript, db, str(exc))
            except Exception as exc:
                full_text = f"[AI RESPONSE] I heard you, Sir, but the voice action failed: {exc}"
                await websocket.send_json({"type": "ai_token", "token": full_text})
                await websocket.send_json({"type": "ai_done", "full_text": full_text})
                session.is_speaking = False
                session.is_listening = True
                return
    except asyncio.CancelledError:
        raise
    finally:
        db.close()

    full_text = result["reply"]
    for token in full_text.split():
        if not session.is_speaking:
            return
        await websocket.send_json({"type": "ai_token", "token": f"{token} "})
        await asyncio.sleep(0.025)
    await websocket.send_json({"type": "ai_done", "full_text": full_text})
    session.is_speaking = False
    session.is_listening = True


async def _finish_voice_utterance(websocket: WebSocket, session: VoiceSession, transcript_override: str | None = None) -> None:
    await websocket.send_json({"type": "transcribing"})
    try:
        transcript = transcript_override.strip() if transcript_override else await session.transcribe()
    except Exception as exc:
        await websocket.send_json({
            "type": "error",
            "message": f"Voice transcription unavailable: {exc}. Install ffmpeg or use typed chat.",
        })
        await websocket.send_json({"type": "ai_done", "full_text": ""})
        return
    if not transcript:
        await websocket.send_json({"type": "transcript", "text": ""})
        await websocket.send_json({"type": "ai_done", "full_text": ""})
        return
    await websocket.send_json({"type": "transcript", "text": transcript})
    lowered = transcript.strip().lower()
    if lowered in {"stop", "that's enough", "thats enough"}:
        await websocket.send_json({"type": "ai_done", "full_text": "Voice session stopped."})
        await websocket.close()
        return
    if any(phrase in lowered for phrase in ("sleep wayne", "goodbye wayne", "standby wayne", "goodnight wayne")):
        db = SessionLocal()
        try:
            await _record_wake_event(db, "voice_sleep", "sleep", session.session_id)
        finally:
            db.close()
        full_text = "W.A.Y.N.E standing by. Say my name when you need me, Sir."
        await websocket.send_json({"type": "ai_token", "token": full_text})
        await websocket.send_json({"type": "ai_done", "full_text": full_text})
        return
    if lowered in {"yes", "y", "confirm", "confirmed", "proceed"} and session.session_id in pending_device_confirmations:
        pending = pending_device_confirmations.pop(session.session_id)
        db = SessionLocal()
        try:
            result = await send_device_command_async(pending["device_id"], pending["command"], True, "voice", db)
        finally:
            db.close()
        full_text = f"Done, Sir. {pending['command'].title()} command sent to your laptop."
        if result.get("status") == "queued":
            full_text = "Command queued, Sir. Start the laptop agent if it does not execute."
        await websocket.send_json({"type": "ai_token", "token": full_text})
        await websocket.send_json({"type": "ai_done", "full_text": full_text})
        return
    parsed_power = _parse_power_command(transcript)
    if parsed_power:
        pending_device_confirmations[session.session_id] = parsed_power
        full_text = f"Confirm: should I {parsed_power['command']} your laptop? Reply yes to proceed."
        await websocket.send_json({"type": "ai_token", "token": full_text})
        await websocket.send_json({"type": "ai_done", "full_text": full_text})
        return
    session.current_stream = asyncio.create_task(_run_voice_turn(websocket, session, transcript))


@app.websocket("/ws/voice/{session_id}")
async def websocket_voice(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = VoiceSession(session_id)
    await websocket.send_json({"type": "ready", "message": "W.A.Y.N.E voice ready"})
    try:
        while True:
            payload = await websocket.receive_json()
            message_type = payload.get("type")
            if message_type == "audio_chunk":
                try:
                    chunk = base64.b64decode(payload.get("data", ""))
                except Exception:
                    chunk = b""
                if chunk and session.process_audio_chunk(chunk):
                    session.cancel_stream()
                    await _finish_voice_utterance(websocket, session)
            elif message_type == "speech_end":
                transcript_text = str(payload.get("text", "")).strip()
                if transcript_text or session.audio_buffer:
                    session.cancel_stream()
                    await _finish_voice_utterance(websocket, session, transcript_text or None)
            elif message_type == "interrupt":
                session.cancel_stream()
                await websocket.send_json({"type": "interrupted", "message": "Listening..."})
            elif message_type == "device_confirm":
                command = str(payload.get("command", "")).strip().lower()
                device_id = str(payload.get("device_id", "laptop-001")).strip()
                if command:
                    pending_device_confirmations[session.session_id] = {"device_id": device_id, "command": command}
                    await websocket.send_json({"type": "ai_token", "token": f"Confirm: should I {command} your laptop? Reply yes to proceed."})
                    await websocket.send_json({"type": "ai_done", "full_text": f"Confirm: should I {command} your laptop? Reply yes to proceed."})
            elif message_type == "stop_voice":
                session.cancel_stream()
                await websocket.close()
                return
    except WebSocketDisconnect:
        session.cancel_stream()
