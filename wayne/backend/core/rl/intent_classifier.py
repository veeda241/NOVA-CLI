from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from database import SessionLocal
from models import TopicTracker


class IntentClassifier:
    PRIMARY_INTENTS = {
        "file": ["file", "open", "read", "folder", "document", "pdf", "find", "search", "show me"],
        "calendar": ["meeting", "schedule", "event", "calendar", "book", "appointment", "free time"],
        "task": ["task", "todo", "add", "complete", "done", "finish", "remind me", "delete task"],
        "device": ["laptop", "phone", "shutdown", "restart", "sleep", "lock", "battery", "mute", "volume"],
        "system": ["cpu", "ram", "memory", "disk", "performance", "status", "storage"],
        "voice": ["listen", "speak", "voice", "talk", "say", "read aloud"],
        "learn": ["remember", "last time", "how often", "usually", "stats", "learning"],
        "general": [],
    }

    async def classify(self, message: str, state: dict) -> str:
        lowered = message.lower()
        scores: dict[str, float] = {}
        for intent, keywords in self.PRIMARY_INTENTS.items():
            score = sum(1 for keyword in keywords if keyword in lowered)
            if score:
                scores[intent] = float(score)

        if not scores:
            expected = state.get("habit_context", {}).get("expected_intents", [])
            intent = expected[0] if expected else "general"
        else:
            for expected in state.get("habit_context", {}).get("expected_intents", []):
                if expected in scores:
                    scores[expected] *= 1.5
            intent = max(scores, key=scores.get)

        await self._update_topic(intent)
        return intent

    async def _update_topic(self, topic: str) -> None:
        with SessionLocal() as db:
            row = db.scalar(select(TopicTracker).where(TopicTracker.topic == topic))
            if row:
                row.frequency += 1
                row.last_asked = datetime.now()
            else:
                db.add(TopicTracker(topic=topic, frequency=1, avg_score=0.5, avg_reward=0.5))
            db.commit()
