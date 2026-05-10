from __future__ import annotations

from sqlalchemy import select

from database import SessionLocal
from models import ImprovementLog, Interaction, Task, UserPreference


class SelfImprover:
    async def analyze_and_improve(self, interaction_count: int) -> list[str]:
        improvements: list[str] = []
        with SessionLocal() as db:
            recent = db.scalars(select(Interaction).order_by(Interaction.created_at.desc()).limit(50)).all()
            if not recent:
                return improvements
            rewards = [row.final_reward for row in recent if row.final_reward is not None]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.5
            avg_time = sum(row.response_time_ms or 0 for row in recent) / len(recent)

            if avg_reward < 0.35:
                improvements.append("Switched to shorter responses because recent rewards were low.")
                self._set_pref(db, "response_length", "short", 0.85)
                self._log(db, "simplify_responses", "Complex responses", "Short simple responses", "Average reward below 0.35", avg_reward)
            elif avg_reward > 0.85:
                improvements.append("Maintained current response style because satisfaction is high.")
                self._log(db, "maintain_style", "Current style", "Current style", "Average reward above 0.85", avg_reward)

            if avg_time > 10000:
                improvements.append("Prioritizing speed because responses are taking too long.")
                self._set_pref(db, "response_length", "short", 0.75)
                self._log(db, "optimize_speed", "Slow responses", "Prioritize short replies", f"Average response time {avg_time:.0f}ms", avg_reward)

            tasks = db.scalars(select(Task)).all()
            if tasks:
                completion = sum(1 for task in tasks if task.completed) / len(tasks)
                if completion < 0.3:
                    improvements.append("Increasing task reminder firmness because task completion is low.")
                    self._set_pref(db, "task_reminder_style", "aggressive", 0.8)
                    self._log(db, "increase_task_reminders", "Gentle reminders", "More proactive reminders", "Task completion below 30%", completion)
            db.commit()
        return improvements

    def _set_pref(self, db, key: str, value: str, confidence: float) -> None:
        pref = db.scalar(select(UserPreference).where(UserPreference.preference_key == key))
        if pref:
            pref.preference_value = value
            pref.confidence = confidence

    def _log(self, db, kind: str, old: str, new: str, reason: str, delta: float) -> None:
        db.add(
            ImprovementLog(
                improvement=f"{kind}: {reason}",
                improvement_type=kind,
                old_behavior=old,
                new_behavior=new,
                trigger_reason=reason,
                reward_delta=delta,
            )
        )
