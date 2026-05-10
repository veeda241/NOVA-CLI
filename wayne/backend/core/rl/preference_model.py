from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from database import SessionLocal
from models import UserPreference


class PreferenceModel:
    async def update_from_interaction(self, user_message: str, wayne_response: str, state: dict, reward: float) -> None:
        words = len(wayne_response.split())
        if reward > 0.6:
            length = "short" if words < 40 else "long" if words > 120 else "medium"
            await self._update("response_length", length, reward)
            await self._update("format", "bullets" if "- " in wayne_response or "* " in wayne_response else "prose", reward)
            formal = any(word in wayne_response for word in ["Sir", "certainly", "indeed", "shall"])
            await self._update("tone", "formal" if formal else "casual", reward)
            await self._update("verbosity", str(round(min(1.0, words / 200), 2)), reward)

        avg_word_len = sum(len(word) for word in user_message.split()) / max(len(user_message.split()), 1)
        await self._update("language_complexity", str(round(min(1.0, avg_word_len / 10), 2)), reward)
        await self._update("preferred_wake_hour", str(state.get("hour", 12)), reward)

        completion = float(state.get("completion_rate", 0.5))
        style = "aggressive" if completion < 0.3 else "gentle" if completion > 0.7 else "moderate"
        await self._update("task_reminder_style", style, max(0.1, reward * completion))

    async def _update(self, key: str, value: str, reward: float) -> None:
        with SessionLocal() as db:
            row = db.scalar(select(UserPreference).where(UserPreference.preference_key == key))
            if row:
                count = row.sample_count
                old_sum = row.reward_sum or (row.confidence * max(count, 1))
                new_count = count + 1
                new_sum = old_sum + reward
                running_avg = old_sum / max(count, 1)
                row.preference_value = value if reward >= running_avg else row.preference_value
                row.confidence = round(new_sum / new_count, 4)
                row.sample_count = new_count
                row.reward_sum = new_sum
                row.last_updated = datetime.now()
            else:
                db.add(UserPreference(preference_key=key, preference_value=value, confidence=reward, sample_count=1, reward_sum=reward))
            db.commit()

    def get_all(self) -> dict[str, str]:
        with SessionLocal() as db:
            rows = db.scalars(select(UserPreference)).all()
            return {row.preference_key: row.preference_value for row in rows}
