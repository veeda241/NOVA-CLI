from __future__ import annotations

from collections import Counter
from datetime import datetime

from sqlalchemy import func, select

from database import SessionLocal
from models import (
    FileAccessLog,
    Habit,
    Interaction,
    KnownContact,
    LanguagePattern,
    Task,
    UserPreference,
)


class StateBuilder:
    async def build(self, user_message: str, session_id: str, voice_audio: bytes | None = None, device_context: dict | None = None) -> dict:
        now = datetime.now()
        prefs = self._preferences()
        interactions = self._interaction_stats()
        task_stats = self._task_stats()
        return {
            "hour": now.hour,
            "minute": now.minute,
            "day_of_week": now.weekday(),
            "is_weekend": now.weekday() >= 5,
            "is_morning": 6 <= now.hour < 12,
            "is_afternoon": 12 <= now.hour < 17,
            "is_evening": 17 <= now.hour < 22,
            "is_night": now.hour >= 22 or now.hour < 6,
            "current_message": user_message,
            "message_length": len(user_message.split()),
            "session_id": session_id,
            "recent_topics": self._recent_topics(session_id),
            "preferred_length": prefs.get("response_length", "medium"),
            "preferred_tone": prefs.get("tone", "formal"),
            "preferred_format": prefs.get("format", "prose"),
            "address_as": prefs.get("address_as", "Sir"),
            "verbosity": float(prefs.get("verbosity", "0.5")),
            "language_complexity": float(prefs.get("language_complexity", "0.5")),
            "proactivity_level": float(prefs.get("proactivity_level", "0.5")),
            "interruption_tolerance": float(prefs.get("interruption_tolerance", "0.5")),
            "total_interactions": interactions["total"],
            "avg_reward": interactions["avg_reward"],
            "peak_hour": interactions["peak_hour"],
            "pending_tasks": task_stats["pending"],
            "completed_today": task_stats["completed_today"],
            "completion_rate": task_stats["completion_rate"],
            "top_files": self._top_files(),
            "known_contacts": self._known_contacts(),
            "active_habits": self._active_habits(),
            "user_vocabulary_level": self._language_stats()["complexity"],
            "avg_message_length": self._language_stats()["avg_length"],
            "formality_score": self._language_stats()["formality"],
            "active_device": (device_context or {}).get("device_id", "web-dashboard"),
            "device_battery": (device_context or {}).get("battery", 100),
            "laptop_cpu": (device_context or {}).get("cpu", 0),
            "emotion": "neutral",
            "stress_level": 0.5,
            "focus_level": 0.5,
            "intent": "general",
            "habit_context": {},
        }

    def _preferences(self) -> dict[str, str]:
        with SessionLocal() as db:
            return {row.preference_key: row.preference_value for row in db.scalars(select(UserPreference)).all()}

    def _recent_topics(self, session_id: str) -> list[str]:
        with SessionLocal() as db:
            rows = db.scalars(select(Interaction).where(Interaction.session_id == session_id).order_by(Interaction.created_at.desc()).limit(5)).all()
            return [row.intent for row in rows if row.intent]

    def _active_habits(self) -> list[str]:
        with SessionLocal() as db:
            rows = db.scalars(select(Habit).where(Habit.confidence >= 0.6).order_by(Habit.frequency.desc()).limit(5)).all()
            return [row.pattern for row in rows]

    def _top_files(self) -> list[str]:
        with SessionLocal() as db:
            rows = db.scalars(select(FileAccessLog).order_by(FileAccessLog.access_count.desc()).limit(5)).all()
            return [row.file_name for row in rows if row.file_name]

    def _known_contacts(self) -> list[str]:
        with SessionLocal() as db:
            rows = db.scalars(select(KnownContact).order_by(KnownContact.mention_count.desc()).limit(10)).all()
            return [row.name for row in rows]

    def _task_stats(self) -> dict:
        with SessionLocal() as db:
            tasks = db.scalars(select(Task)).all()
            pending = sum(1 for task in tasks if not task.completed)
            completed_today = sum(1 for task in tasks if task.completed and task.completed_at and task.completed_at.date() == datetime.now().date())
            total = len(tasks)
            return {"pending": pending, "completed_today": completed_today, "completion_rate": (total - pending) / total if total else 0.5}

    def _interaction_stats(self) -> dict:
        with SessionLocal() as db:
            rows = db.scalars(select(Interaction)).all()
            if not rows:
                return {"total": 0, "avg_reward": 0.5, "peak_hour": 9}
            rewards = [row.final_reward for row in rows if row.final_reward is not None]
            hours = [getattr(row, "timestamp", row.created_at).hour for row in rows]
            return {
                "total": len(rows),
                "avg_reward": sum(rewards) / len(rewards) if rewards else 0.5,
                "peak_hour": Counter(hours).most_common(1)[0][0] if hours else 9,
            }

    def _language_stats(self) -> dict:
        with SessionLocal() as db:
            rows = db.scalars(select(LanguagePattern)).all()
            if not rows:
                return {"complexity": 0.5, "avg_length": 10, "formality": 0.5}
            formal = sum(1 for row in rows if row.pattern_type == "formal")
            complexity = 0.75 if any(row.pattern_type == "complex_vocabulary" for row in rows) else 0.5
            return {"complexity": complexity, "avg_length": 10, "formality": formal / max(len(rows), 1)}
