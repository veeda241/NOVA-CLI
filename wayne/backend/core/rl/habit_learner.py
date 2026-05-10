from __future__ import annotations

from datetime import datetime

from sqlalchemy import and_, select

from database import SessionLocal
from models import Habit


class HabitLearner:
    async def observe(self, state: dict, action: str, reward: float) -> None:
        hour = state.get("hour", 12)
        day = state.get("day_of_week", 0)
        intent = state.get("intent", "general")
        await self._record("time_intent", f"{intent}_at_hour_{hour}", {"hour": hour, "intent": intent, "reward": reward})
        await self._record("day_pattern", f"{intent}_on_day_{day}", {"day": day, "intent": intent})
        if 6 <= hour <= 10 and reward > 0.6:
            await self._record("morning_routine", f"{intent}_in_morning", {"hour": hour})
        if 17 <= hour <= 21 and reward > 0.6:
            await self._record("evening_routine", f"{intent}_in_evening", {"hour": hour})

    async def get_context(self, state: dict) -> dict:
        hour = state.get("hour", 12)
        with SessionLocal() as db:
            rows = db.scalars(
                select(Habit).where(Habit.habit_type == "time_intent", Habit.confidence >= 0.6).order_by(Habit.frequency.desc()).limit(10)
            ).all()
        relevant = [row for row in rows if f"hour_{hour}" in row.pattern]
        return {
            "expected_intents": [row.pattern.split("_at_")[0] for row in relevant],
            "is_routine_time": bool(relevant),
            "routine_confidence": max((row.confidence for row in relevant), default=0.0),
        }

    async def _record(self, habit_type: str, pattern: str, metadata: dict) -> None:
        with SessionLocal() as db:
            row = db.scalar(select(Habit).where(and_(Habit.habit_type == habit_type, Habit.pattern == pattern)))
            if row:
                row.frequency += 1
                row.confidence = round(min(0.99, row.confidence + (1 / (row.frequency + 10))), 4)
                row.last_observed = datetime.now()
                row.metadata_json = metadata
            else:
                db.add(Habit(habit_type=habit_type, pattern=pattern, frequency=1, confidence=0.1, metadata_json=metadata))
            db.commit()
