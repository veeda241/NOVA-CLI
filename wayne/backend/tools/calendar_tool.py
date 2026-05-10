from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/calendar"]
TOKEN_FILE = Path("data/google_token.json")


def _client_config() -> dict[str, Any]:
    return {
        "web": {
            "client_id": os.getenv("GOOGLE_CALENDAR_CLIENT_ID", ""),
            "client_secret": os.getenv("GOOGLE_CALENDAR_CLIENT_SECRET", ""),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [os.getenv("GOOGLE_CALENDAR_REDIRECT_URI", "http://localhost:3000/auth/callback")],
        }
    }


def build_oauth_flow() -> Flow:
    return Flow.from_client_config(
        _client_config(),
        scopes=SCOPES,
        redirect_uri=os.getenv("GOOGLE_CALENDAR_REDIRECT_URI", "http://localhost:3000/auth/callback"),
    )


def authorization_url() -> str:
    flow = build_oauth_flow()
    url, _ = flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")
    return url


def handle_callback(code: str) -> dict[str, str]:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    flow = build_oauth_flow()
    flow.fetch_token(code=code)
    TOKEN_FILE.write_text(flow.credentials.to_json(), encoding="utf-8")
    return {"status": "connected"}


def _credentials() -> Credentials | None:
    if not TOKEN_FILE.exists():
        return None
    creds = Credentials.from_authorized_user_info(json.loads(TOKEN_FILE.read_text(encoding="utf-8")), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    return creds


def _service():
    creds = _credentials()
    if creds is None:
        raise RuntimeError("Google Calendar is not connected. Visit /auth/google first.")
    return build("calendar", "v3", credentials=creds)


def today_events() -> list[dict[str, Any]]:
    service = _service()
    now = datetime.now().astimezone()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    end = now.replace(hour=23, minute=59, second=59, microsecond=0).isoformat()
    events = service.events().list(calendarId="primary", timeMin=start, timeMax=end, singleEvents=True, orderBy="startTime").execute()
    results = []
    for event in events.get("items", []):
        start_value = event.get("start", {}).get("dateTime") or event.get("start", {}).get("date")
        end_value = event.get("end", {}).get("dateTime") or event.get("end", {}).get("date")
        results.append({"id": event.get("id"), "title": event.get("summary", "Untitled"), "start": start_value, "end": end_value})
    return results


def schedule_meeting(
    title: str,
    date: str,
    time: str,
    duration_minutes: int = 60,
    attendees: list[str] | None = None,
) -> dict[str, Any]:
    service = _service()
    local_start = datetime.fromisoformat(f"{date}T{time}:00").astimezone()
    local_end = local_start + timedelta(minutes=duration_minutes)
    event = {
        "summary": title,
        "start": {"dateTime": local_start.isoformat()},
        "end": {"dateTime": local_end.isoformat()},
        "attendees": [{"email": email} for email in attendees or []],
    }
    created = service.events().insert(calendarId="primary", body=event, sendUpdates="all").execute()
    return {
        "status": "created",
        "id": created.get("id"),
        "event_id": created.get("id"),
        "title": created.get("summary"),
        "start": created.get("start", {}).get("dateTime"),
        "end": created.get("end", {}).get("dateTime"),
        "link": created.get("htmlLink"),
        "event_link": created.get("htmlLink"),
    }


def offline_today_placeholder() -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    return [{"title": "Calendar offline or not connected", "start": now.isoformat(), "end": now.isoformat()}]
