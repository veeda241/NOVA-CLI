from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import httpx
from sqlalchemy.orm import Session

from tools.file_tool import list_files
from tools.system_tool import system_status
from tools.task_tool import manage_tasks


async def check_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            return response.status_code == 200
    except Exception:
        return False



def offline_reply(query: str, db: Session, reason: str | None = None) -> dict[str, Any]:
    text = query.strip()
    lowered = text.lower()
    tool_used: str | None = None

    try:
        if "list files" in lowered:
            match = re.search(r"list files(?: in| from)?\s*(.*)", text, flags=re.IGNORECASE)
            directory = match.group(1).strip() if match and match.group(1).strip() else None
            result = list_files(directory)
            tool_used = "list_files"
            reply = f"[OFFLINE MODE] [FILE SYSTEM] Listed {len(result['files'])} item(s) in {result['directory']}."
        elif lowered.startswith("add task") or lowered.startswith("create task"):
            title = re.sub(r"^(add|create)\s+task\s*", "", text, flags=re.IGNORECASE).strip()
            priority = "high" if " high" in lowered else "low" if " low" in lowered else "medium"
            result = manage_tasks("create", title=title or "Untitled task", priority=priority, db=db)
            tool_used = "manage_tasks"
            reply = f"[OFFLINE MODE] [TASK ENGINE] Task added. You now have {len(result['tasks'])} task(s)."
        elif "what time" in lowered or lowered == "time" or "date" in lowered:
            reply = f"[OFFLINE MODE] [AI RESPONSE] Current local time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
        elif "system status" in lowered or lowered == "status":
            status = system_status()
            tool_used = "system_status"
            reply = f"[OFFLINE MODE] [SYSTEM] CPU {status['cpu_percent']}%, RAM {status['ram_percent']}%, disk {status['disk_percent']}%."
        elif reason:
            reply = f"[OFFLINE MODE] [AI RESPONSE] {reason}"
        else:
            reply = "[OFFLINE MODE] [AI RESPONSE] W.A.Y.N.E AI core offline. Please start the local engine."
        return {"reply": reply, "tool_used": tool_used, "messages": [{"role": "assistant", "content": reply}]}
    except Exception as exc:
        return {"reply": f"[OFFLINE MODE] [AI RESPONSE] Local tool failed: {exc}", "tool_used": tool_used, "messages": [{"role": "assistant", "content": str(exc)}]}
