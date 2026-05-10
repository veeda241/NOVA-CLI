from __future__ import annotations

import random

from sqlalchemy import func, select

from database import SessionLocal
from models import ExperienceBuffer, ImprovementLog

BUFFER_SIZE = 10000


class MemoryBank:
    async def store(self, state: dict, action: str, reward: float, next_state: dict | None = None) -> None:
        with SessionLocal() as db:
            count = db.scalar(select(func.count(ExperienceBuffer.id))) or 0
            if count >= BUFFER_SIZE:
                oldest = db.scalars(select(ExperienceBuffer).order_by(ExperienceBuffer.timestamp).limit(100)).all()
                for row in oldest:
                    db.delete(row)
            db.add(ExperienceBuffer(state=state, action=action, reward=reward, next_state=next_state, done=next_state is None))
            db.commit()

    async def replay_and_update(self, preference_model) -> None:
        with SessionLocal() as db:
            rows = db.scalars(select(ExperienceBuffer).order_by(ExperienceBuffer.timestamp.desc()).limit(1000)).all()
            if len(rows) < 10:
                return
            batch = random.sample(rows, min(32, len(rows)))
            high_reward = [row for row in batch if row.reward > 0.7]
            avg = sum(row.reward for row in high_reward) / max(len(high_reward), 1)
            db.add(
                ImprovementLog(
                    improvement="Experience replay reinforced high-reward response patterns.",
                    improvement_type="experience_replay",
                    new_behavior=f"Replayed {len(batch)} experiences; {len(high_reward)} high reward.",
                    trigger_reason="scheduled_replay",
                    reward_delta=avg,
                )
            )
            db.commit()

        for row in high_reward:
            await preference_model._update("response_length", row.state.get("preferred_length", "medium"), row.reward)
