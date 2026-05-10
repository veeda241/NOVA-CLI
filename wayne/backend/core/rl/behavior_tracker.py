from __future__ import annotations

import re
from datetime import datetime

from sqlalchemy import and_, select

from database import SessionLocal
from models import FileAccessLog, KnownContact, LanguagePattern


class BehaviorTracker:
    RESERVED_NAMES = {"WAYNE", "W.A.Y.N.E", "W", "AI", "SIR", "MAAM", "MA'AM"}

    async def learn_language(self, user_message: str, reward: float) -> None:
        lowered = user_message.lower()
        for pattern in ["please", "could you", "would you", "kindly", "i would like"]:
            if pattern in lowered:
                await self._log_pattern("formal", pattern)
        for pattern in ["hey", "yo", "gonna", "wanna", "lol", "btw"]:
            if pattern in lowered:
                await self._log_pattern("casual", pattern)

        words = user_message.split()
        avg_len = sum(len(word) for word in words) / max(len(words), 1)
        await self._log_pattern("complex_vocabulary" if avg_len > 7 else "simple_vocabulary", "high" if avg_len > 7 else "low")

        if words:
            first = words[0].lower().strip(".,!?")
            if first in {"open", "show", "find", "schedule", "add", "delete", "send", "check", "tell"}:
                await self._log_pattern("command_verb", first)

    async def extract_contacts(self, user_message: str, wayne_response: str) -> None:
        for email in re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", user_message):
            await self._update_contact(email.split("@")[0].replace(".", " ").title(), email)
        for index, word in enumerate(user_message.split()):
            clean = word.strip(".,!?")
            if index > 0 and len(clean) > 2 and clean.isalpha() and clean[0].isupper():
                await self._update_contact(clean)

    async def track_file_access(self, file_path: str) -> None:
        if not file_path:
            return
        file_name = file_path.replace("\\", "/").split("/")[-1]
        file_type = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "unknown"
        with SessionLocal() as db:
            row = db.scalar(select(FileAccessLog).where(FileAccessLog.file_path == file_path))
            if row:
                row.access_count += 1
                row.last_accessed = datetime.now()
            else:
                db.add(FileAccessLog(file_path=file_path, file_name=file_name, file_type=file_type, access_count=1))
            db.commit()

    async def _log_pattern(self, pattern_type: str, value: str) -> None:
        with SessionLocal() as db:
            row = db.scalar(
                select(LanguagePattern).where(
                    and_(LanguagePattern.pattern_type == pattern_type, LanguagePattern.pattern_value == value)
                )
            )
            if row:
                row.frequency += 1
                row.last_seen = datetime.now()
            else:
                db.add(LanguagePattern(pattern_type=pattern_type, pattern_value=value, frequency=1, context="user_message"))
            db.commit()

    async def _update_contact(self, name: str, email: str | None = None) -> None:
        if name.upper() in self.RESERVED_NAMES:
            return
        with SessionLocal() as db:
            row = db.scalar(select(KnownContact).where(KnownContact.name == name))
            if row:
                row.mention_count += 1
                row.last_mentioned = datetime.now()
                row.email = row.email or email
            else:
                db.add(KnownContact(name=name, email=email, mention_count=1))
            db.commit()
