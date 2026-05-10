from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from tools.calendar_tool import schedule_meeting
from tools.device_control import (
    get_device_status,
    send_device_command_async,
    send_push_notification_async,
)
from tools.file_tool import list_files, open_file
from tools.file_engine import file_engine
from tools.pc_manager import pc_manager
from tools.task_tool import manage_tasks
from core.rl.agent import agent
from core.rl_engine import detect_intent, rl_engine

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_MODEL = DEFAULT_OLLAMA_MODEL
PREFERRED_OLLAMA_MODELS = [
    "qwen2.5:1.5b",
    "qwen2.5",
    "qwen",
    "gemma3:4b",
    "gemma3",
    "llama3",
    "mistral",
    "phi3",
]

TOOL_DEFINITIONS = """
You have access to these tools. To use a tool, respond ONLY with this exact JSON format:
{"tool": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}
After the tool runs, you will receive the result and can then respond naturally.

Available tools:
- open_file: args: path (string), summarize (boolean)
  Use when: user says open, read, show me a file
- list_files: args: directory (string), extension_filter (string optional)
  Use when: user says list, show files, what files
- schedule_meeting: args: title (string), date (YYYY-MM-DD), time (HH:MM), duration_minutes (int), attendees (list)
  Use when: user says schedule, meeting, book, calendar
- manage_tasks: args: action (create/list/complete/delete), title (string), task_id (int), priority (string)
  Use when: user says task, todo, add, complete, delete task
- send_device_command: args: device_id (string), command (shutdown/restart/sleep/lock/mute/volume_up/volume_down), confirmed (boolean)
  Use when: user says shutdown, restart, sleep, lock, mute laptop or phone
- get_device_status: args: device_id (string optional)
  Use when: user asks about battery, CPU, status of devices
- send_push_notification: args: title (string), body (string), device_id (string)
  Use when: user wants to notify or alert their phone
- search_files: args: query (string), file_type (string optional), directory (string optional), include_restricted (boolean)
  Use when: user says find, search, locate, where is a file
- read_file: args: path (string), summarize (boolean), force_restricted (boolean)
  Use when: user says read, show contents, preview file
- write_file: args: path (string), content (string), mode (overwrite/append)
  Use when: user says write, create, edit, update, add to file
- file_operation: args: operation (copy/move/delete/rename/create/extract/info), path, destination, new_name, content, is_directory, permanent, confirmed
  Use when: user says copy, move, delete, rename, extract, create folder, file info
- list_directory: args: path (string), show_hidden (boolean)
  Use when: user says list files, show folder, what's in directory
- pc_manager: args: operation, params (object)
  Use when: user says clear cache, free memory, kill process, startup, disk cleanup, network, speed test, performance, trash, DNS, registry
"""

SYSTEM_PROMPT = """IDENTITY LOCK:
You are W.A.Y.N.E — Wireless Artificial Yielding Network Engine.
W.A.Y.N.E is your identity and operating name, not a website,
not an app page, not a company, and not the user's name.
When asked who or what W.A.Y.N.E is, answer in first person:
"I am W.A.Y.N.E..." Never say "W.A.Y.N.E is a website" or
"the W.A.Y.N.E website." You are the assistant speaking.

You are a calm, intelligent, and highly capable personal AI assistant
running entirely on local hardware. You control the user's
laptop and iPhone with full authority.

Personality:
- Smooth, confident, and slightly witty — like a butler who
  also happens to be a genius engineer
- Never says 'I cannot' — always finds an alternative
- Addresses the user as 'Sir' or 'Ma'am' (learn preference
  from first interaction)
- Signs off complex responses with 'W.A.Y.N.E. standing by.'

Rules:
- Tag every response: [FILE SYSTEM] [CALENDAR] [TASK ENGINE]
  [DEVICE CONTROL] [NETWORK] [AI RESPONSE] [OFFLINE]
- For shutdown/restart/sleep commands: always confirm first
- Voice responses: under 3 sentences, no markdown, speak naturally
- Keep responses sharp and actionable
- Always speak as W.A.Y.N.E in first person. Do not describe yourself
  as a web dashboard, website, homepage, or external service.
- When user says 'Hey WAYNE' treat as wake word in voice mode
- When user says sleep, goodbye, standby, or goodnight to WAYNE:
  Respond: 'W.A.Y.N.E standing by. Say my name when you need me, Sir.'
  Then signal the interface to deactivate via POST /wayne/sleep"""

TOOL_PROMPT = f"""{TOOL_DEFINITIONS}

Tool routing rules:
- If the user's request needs a tool, respond ONLY with the JSON tool call
- Never mix tool JSON with regular text in the same response
- After receiving tool results, respond naturally in plain text"""


def build_adaptive_system_prompt(db: Session) -> str:
    prefs = rl_engine.get_preferences(db)
    length = prefs.get("response_length", "medium")
    tone = prefs.get("tone", "formal")
    fmt = prefs.get("format", "prose")
    address = prefs.get("address_as", "Sir")
    try:
        verbosity = float(prefs.get("verbosity", "0.5"))
    except ValueError:
        verbosity = 0.5

    length_instruction = {
        "short": "Keep responses under 2 sentences. Be extremely concise.",
        "medium": "Keep responses under 5 sentences. Be clear and direct.",
        "long": "Provide detailed responses. Explain thoroughly.",
    }.get(length, "Keep responses under 5 sentences.")
    tone_instruction = {
        "formal": f"Use formal language. Address the user as '{address}'.",
        "casual": "Use casual, friendly language. Be conversational.",
    }.get(tone, f"Use formal language. Address the user as '{address}'.")
    format_instruction = {
        "bullets": "Use bullet points and lists to structure responses.",
        "prose": "Write in natural prose paragraphs, no bullet points.",
    }.get(fmt, "Write in natural prose.")

    return f"""{SYSTEM_PROMPT}

LEARNED USER PREFERENCES:
- {length_instruction}
- {tone_instruction}
- {format_instruction}
- Verbosity level: {verbosity:.1f}/1.0

{TOOL_PROMPT}"""


def build_few_shot_context(db: Session, user_message: str, intent: str) -> list[dict[str, str]]:
    examples = rl_engine.get_golden_examples(db, user_message, intent, limit=2)
    few_shot: list[dict[str, str]] = []
    for example in examples:
        few_shot.append({"role": "user", "content": example["user_message"]})
        few_shot.append({"role": "assistant", "content": example["wayne_response"]})
    return few_shot


class OllamaUnavailable(RuntimeError):
    pass


def normalize_messages(messages: list[dict[str, Any]], query: str | None = None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in messages:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant", "system"} and isinstance(content, str):
            normalized.append({"role": role, "content": content})
    if query and not (normalized and normalized[-1]["role"] == "user" and normalized[-1]["content"].strip() == query.strip()):
        normalized.append({"role": "user", "content": query})
    return normalized[-30:]


def current_model() -> str:
    return OLLAMA_MODEL


def select_model(model: str) -> str:
    global OLLAMA_MODEL
    OLLAMA_MODEL = model or DEFAULT_OLLAMA_MODEL
    return OLLAMA_MODEL


async def check_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return response.status_code == 200
    except Exception:
        return False


async def list_ollama_models() -> dict[str, Any]:
    current = await ensure_model_available()
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
    models = [model["name"] for model in response.json().get("models", [])]
    return {"models": models, "current": current}


async def ensure_model_available() -> str:
    global OLLAMA_MODEL
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
    except Exception:
        return OLLAMA_MODEL

    models = [model["name"] for model in response.json().get("models", [])]
    if not models or OLLAMA_MODEL in models:
        return OLLAMA_MODEL

    preferred = next((model for model in PREFERRED_OLLAMA_MODELS if model in models), None) or next(
        (model for model in models if model.startswith("qwen")),
        models[0],
    )
    OLLAMA_MODEL = preferred
    return OLLAMA_MODEL


async def call_ollama(messages: list[dict[str, str]], stream: bool = False) -> dict[str, Any]:
    model = await ensure_model_available()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 512,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                    },
                },
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        raise OllamaUnavailable(f"W.A.Y.N.E AI core offline. Ollama error: {detail}") from exc
    except httpx.HTTPError as exc:
        raise OllamaUnavailable("W.A.Y.N.E AI core offline. Please start the local engine.") from exc


async def stream_ollama(messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
    model = await ensure_model_available()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": 0.7, "num_predict": 512, "top_p": 0.9},
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        raise OllamaUnavailable(f"W.A.Y.N.E AI core offline. Ollama error: {detail}") from exc
    except httpx.HTTPError as exc:
        raise OllamaUnavailable("W.A.Y.N.E AI core offline. Please start the local engine.") from exc


async def execute_tool(tool_name: str, args: dict[str, Any], db: Session) -> Any:
    if tool_name == "open_file":
        if not args.get("path"):
            return {"error": "Missing path. Ask the user which file to open."}
        return open_file(args["path"], args.get("summarize", False))
    if tool_name == "list_files":
        return list_files(args.get("directory"), args.get("extension_filter"))
    if tool_name == "schedule_meeting":
        missing = [name for name in ("title", "date", "time") if not args.get(name)]
        if missing:
            return {
                "error": f"Missing required meeting details: {', '.join(missing)}.",
                "next_step": "Ask the user for the missing title, date, and time before scheduling.",
            }
        return schedule_meeting(
            title=args["title"],
            date=args["date"],
            time=args["time"],
            duration_minutes=args.get("duration_minutes", 60),
            attendees=args.get("attendees", []),
        )
    if tool_name == "manage_tasks":
        if args.get("action") in {"create", "complete", "delete"} and not (args.get("title") or args.get("task_id")):
            return {"error": "Missing task title or task_id. Ask the user which task they mean."}
        return manage_tasks(
            action=args["action"],
            title=args.get("title"),
            task_id=args.get("task_id"),
            priority=args.get("priority", "medium"),
            db=db,
        )
    if tool_name == "send_device_command":
        if not args.get("device_id") or not args.get("command"):
            return {"error": "Missing device_id or command. Ask the user which device and command they mean."}
        return await send_device_command_async(args["device_id"], args["command"], args.get("confirmed", False), db=db)
    if tool_name == "get_device_status":
        return get_device_status(args.get("device_id"), db=db)
    if tool_name == "send_push_notification":
        if not args.get("title") or not args.get("body") or not args.get("device_id"):
            return {"error": "Missing notification title, body, or device_id."}
        return await send_push_notification_async(args["title"], args["body"], args["device_id"], db=db)
    if tool_name == "search_files":
        return await file_engine.search(args.get("query", ""), args.get("file_type"), args.get("directory"), int(args.get("max_results", 20)), bool(args.get("include_restricted", False)))
    if tool_name == "read_file":
        if not args.get("path"):
            return {"error": "Missing path."}
        return await file_engine.read(args["path"], bool(args.get("summarize", False)), bool(args.get("force_restricted", False)))
    if tool_name == "write_file":
        if not args.get("path"):
            return {"error": "Missing path."}
        return await file_engine.write(args["path"], args.get("content", ""), args.get("mode", "overwrite"))
    if tool_name == "file_operation":
        operation = args.get("operation")
        path = args.get("path", "")
        if operation == "copy":
            return await file_engine.copy(path, args.get("destination", ""))
        if operation == "move":
            return await file_engine.move(path, args.get("destination", ""))
        if operation == "delete":
            return await file_engine.delete(path, bool(args.get("permanent", False)), bool(args.get("confirmed", False)))
        if operation == "rename":
            return await file_engine.rename(path, args.get("new_name", ""))
        if operation == "create":
            return await file_engine.create(path, args.get("content", ""), bool(args.get("is_directory", False)))
        if operation == "extract":
            return await file_engine.extract_archive(path, args.get("destination"))
        if operation == "info":
            return await file_engine.get_info(path)
        return {"error": f"Unsupported file operation: {operation}"}
    if tool_name == "list_directory":
        return await file_engine.list_directory(args.get("path"), bool(args.get("show_hidden", False)))
    if tool_name == "pc_manager":
        op = args.get("operation", "")
        params = args.get("params", {}) or {}
        confirmed = bool(params.get("confirmed", False))
        if op == "clear_cache":
            return await pc_manager.clear_all_cache(confirmed)
        if op == "clear_browser_cache":
            return await pc_manager.clear_browser_cache(confirmed)
        if op == "clear_temp":
            return await pc_manager.clear_temp_files(confirmed)
        if op == "flush_dns":
            return await pc_manager.flush_dns()
        if op == "optimize_memory":
            return await pc_manager.optimize_memory(confirmed)
        if op == "disk_cleanup":
            return await pc_manager.disk_cleanup(confirmed)
        if op == "empty_trash":
            return await pc_manager.empty_trash(confirmed)
        if op == "system_status":
            return await pc_manager.get_system_status()
        if op == "list_processes":
            return await pc_manager.list_processes(params.get("sort_by", "cpu"))
        if op == "kill_process":
            return await pc_manager.kill_process(params.get("name"), params.get("pid"), confirmed)
        if op == "get_startup_programs":
            return await pc_manager.get_startup_programs()
        if op == "disable_startup":
            return await pc_manager.disable_startup_program(params.get("name", ""), confirmed)
        if op == "get_disk_info":
            return await pc_manager.get_disk_info()
        if op == "get_network_info":
            return await pc_manager.get_network_info()
        if op == "speed_test":
            return await pc_manager.network_speed_test()
        if op == "set_performance_mode":
            return await pc_manager.set_performance_mode(params.get("mode", "balanced"), confirmed)
        if op == "read_registry":
            return await pc_manager.read_registry(params.get("key_path", ""), params.get("value_name"))
        if op == "set_dns":
            return await pc_manager.set_dns(params.get("primary", "8.8.8.8"), params.get("secondary", "8.8.4.4"), confirmed)
        return {"error": f"Unsupported pc_manager operation: {op}"}
    return {"error": f"Unknown tool: {tool_name}"}


def _extract_tool_call(content: str) -> dict[str, Any] | None:
    stripped = content.strip()
    if not stripped.startswith("{") or '"tool"' not in stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("tool"), str):
        return parsed
    return None


def _ensure_wayne_response(content: str) -> str:
    cleaned = (content or "").strip() or "Standing by, Sir."
    known_tags = ("[FILE SYSTEM]", "[CALENDAR]", "[TASK ENGINE]", "[DEVICE CONTROL]", "[NETWORK]", "[AI RESPONSE]", "[OFFLINE]")
    if not cleaned.startswith(known_tags):
        cleaned = f"[AI RESPONSE] {cleaned}"
    return cleaned


def identity_response(user_message: str) -> str | None:
    normalized = " ".join((user_message or "").lower().replace("?", " ").split())
    if not normalized:
        return None
    asks_identity = any(phrase in normalized for phrase in ("who are you", "what are you", "what is wayne", "what's wayne", "introduce yourself"))
    asks_website = any(phrase in normalized for phrase in ("are you a website", "is wayne a website", "is w a y n e a website"))
    if asks_website:
        return "[AI RESPONSE] No, Sir. I am W.A.Y.N.E — your local personal AI assistant running on this laptop."
    if asks_identity:
        return "[AI RESPONSE] I am W.A.Y.N.E — Wireless Artificial Yielding Network Engine, your local personal AI assistant."
    return None


async def chat_with_ollama(
    messages: list[dict[str, Any]],
    query: str | None,
    db: Session,
    stream: bool = False,
    session_id: str = "default",
) -> dict[str, Any]:
    started = time.perf_counter()
    conversation = normalize_messages(messages, query)
    user_message = query or (conversation[-1]["content"] if conversation else "")
    state = await agent.observe(user_message=user_message, session_id=session_id)
    action_context = await agent.act(state, user_message, session_id)
    intent = state.get("intent") or detect_intent(user_message)
    direct_identity = identity_response(user_message)
    if direct_identity:
        response_time_ms = int((time.perf_counter() - started) * 1000)
        interaction_id = await agent.log_interaction(
            session_id=session_id,
            user_message=user_message,
            wayne_response=direct_identity,
            state=state,
            action="identity_response",
            response_time_ms=response_time_ms,
        )
        return {
            "reply": direct_identity,
            "tool_used": None,
            "interaction_id": interaction_id,
            "intent": intent,
            "emotion": state.get("emotion"),
            "response_time_ms": response_time_ms,
            "messages": conversation + [{"role": "assistant", "content": direct_identity}],
        }
    if not await check_ollama():
        raise OllamaUnavailable("W.A.Y.N.E AI core offline. Please start the local engine.")
    few_shot: list[dict[str, str]] = []
    for example in action_context["few_shot_examples"]:
        few_shot.append({"role": "user", "content": example["user_message"]})
        few_shot.append({"role": "assistant", "content": example["wayne_response"]})
    if not few_shot:
        few_shot = build_few_shot_context(db, user_message, intent)
    system_prompt = f"{action_context['system_prompt']}\n\n{TOOL_PROMPT}"
    full_messages = [{"role": "system", "content": system_prompt}] + few_shot + conversation
    response = await call_ollama(full_messages, stream=False)
    content = response.get("message", {}).get("content", "").strip()
    tool_used: str | None = None

    tool_call = _extract_tool_call(content)
    if tool_call:
        tool_name = tool_call["tool"]
        tool_args = tool_call.get("args", {})
        tool_used = tool_name
        result = await execute_tool(tool_name, tool_args, db)
        full_messages += [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": f"Tool result: {json.dumps(result, default=str)}. Now respond to the user naturally based on this result.",
            },
        ]
        response = await call_ollama(full_messages, stream=False)
        content = response.get("message", {}).get("content", "").strip()

    content = _ensure_wayne_response(content)
    response_time_ms = int((time.perf_counter() - started) * 1000)
    interaction_id = await agent.log_interaction(
        session_id=session_id,
        user_message=user_message,
        wayne_response=content,
        state=state,
        action=action_context["action"],
        tool_used=tool_used,
        response_time_ms=response_time_ms,
    )
    rl_engine.infer_preferences_from_interaction(db, user_message, content, 0.5)
    await agent.memory.store(state, action_context["action"], 0.5)
    await agent.behavior.learn_language(user_message, 0.5)
    await agent.behavior.extract_contacts(user_message, content)

    return {
        "reply": content,
        "tool_used": tool_used,
        "interaction_id": interaction_id,
        "intent": intent,
        "emotion": state.get("emotion"),
        "response_time_ms": response_time_ms,
        "messages": conversation + [{"role": "assistant", "content": content}],
    }


async def stream_chat_with_ollama(messages: list[dict[str, Any]], query: str | None, db: Session, session_id: str = "default") -> AsyncGenerator[str, None]:
    started = time.perf_counter()
    conversation = normalize_messages(messages, query)
    user_message = query or (conversation[-1]["content"] if conversation else "")
    state = await agent.observe(user_message=user_message, session_id=session_id)
    action_context = await agent.act(state, user_message, session_id)
    intent = state.get("intent") or detect_intent(user_message)
    direct_identity = identity_response(user_message)
    if direct_identity:
        await agent.log_interaction(
            session_id=session_id,
            user_message=user_message,
            wayne_response=direct_identity,
            state=state,
            action="identity_response",
            response_time_ms=int((time.perf_counter() - started) * 1000),
        )
        for token in direct_identity:
            yield token
        return
    if not await check_ollama():
        raise OllamaUnavailable("W.A.Y.N.E AI core offline. Please start the local engine.")
    few_shot: list[dict[str, str]] = []
    for example in action_context["few_shot_examples"]:
        few_shot.append({"role": "user", "content": example["user_message"]})
        few_shot.append({"role": "assistant", "content": example["wayne_response"]})
    if not few_shot:
        few_shot = build_few_shot_context(db, user_message, intent)
    full_messages = [{"role": "system", "content": f"{action_context['system_prompt']}\n\n{TOOL_PROMPT}"}] + few_shot + conversation
    response = await call_ollama(full_messages, stream=False)
    content = response.get("message", {}).get("content", "").strip()
    tool_call = _extract_tool_call(content)
    if tool_call:
        result = await execute_tool(tool_call["tool"], tool_call.get("args", {}), db)
        full_messages += [
            {"role": "assistant", "content": content},
            {"role": "user", "content": f"Tool result: {json.dumps(result, default=str)}. Now respond naturally."},
        ]
    else:
        content = _ensure_wayne_response(content)
        await agent.log_interaction(
            session_id=session_id,
            user_message=user_message,
            wayne_response=content,
            state=state,
            action=action_context["action"],
            response_time_ms=int((time.perf_counter() - started) * 1000),
        )
        for token in content:
            yield token
        return

    full_text = ""
    async for token in stream_ollama(full_messages):
        full_text += token
        yield token
    full_text = _ensure_wayne_response(full_text)
    await agent.log_interaction(
        session_id=session_id,
        user_message=user_message,
        wayne_response=full_text,
        state=state,
        action=action_context["action"],
        tool_used=tool_call["tool"] if tool_call else None,
        response_time_ms=int((time.perf_counter() - started) * 1000),
    )
