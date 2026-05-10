from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from models import (
    FileAccessLog,
    GoldenResponse,
    Habit,
    ImprovementLog,
    Interaction,
    KnownContact,
    Task,
    TopicTracker,
    UserPreference,
)

DEFAULT_PREFERENCES = {
    "response_length": ("medium", 0.5),
    "tone": ("formal", 0.5),
    "format": ("prose", 0.5),
    "address_as": ("Sir", 0.5),
    "preferred_wake_time": ("unknown", 0.1),
    "primary_language": ("english", 0.9),
    "verbosity": ("0.5", 0.5),
    "detail_level": ("medium", 0.5),
    "preferred_wake_hour": ("unknown", 0.1),
    "peak_productivity_hour": ("unknown", 0.1),
    "interruption_tolerance": ("0.5", 0.5),
    "proactivity_level": ("0.5", 0.5),
    "emotion_sensitivity": ("0.5", 0.5),
    "task_reminder_style": ("gentle", 0.5),
    "meeting_buffer_mins": ("10", 0.5),
    "preferred_file_types": ("unknown", 0.1),
    "language_complexity": ("0.5", 0.5),
}


class RLEngine:
    positive_words = {
        "thanks",
        "perfect",
        "great",
        "good",
        "correct",
        "exactly",
        "yes",
        "nice",
        "awesome",
        "helpful",
        "that's right",
        "got it",
        "makes sense",
    }
    negative_words = {
        "wrong",
        "incorrect",
        "not what",
        "that's not",
        "again",
        "repeat",
        "didn't understand",
        "what do you mean",
        "clarify",
        "confused",
    }

    def ensure_defaults(self, db: Session) -> None:
        for key, (value, confidence) in DEFAULT_PREFERENCES.items():
            existing = db.scalar(select(UserPreference).where(UserPreference.preference_key == key))
            if existing is None:
                db.add(UserPreference(preference_key=key, preference_value=value, confidence=confidence, sample_count=1))
        db.commit()

    def calculate_implicit_reward(
        self,
        db: Session,
        user_message: str,
        wayne_response: str,
        next_user_message: str | None = None,
    ) -> float:
        reward = 0.5
        next_lower = (next_user_message or "").lower()
        if any(word in next_lower for word in self.positive_words):
            reward += 0.2
        if any(word in next_lower for word in self.negative_words):
            reward -= 0.3
        if next_user_message and self._is_similar(user_message, next_user_message):
            reward -= 0.25

        prefs = self.get_preferences(db)
        preferred_length = prefs.get("response_length", "medium")
        response_words = len(wayne_response.split())
        if preferred_length == "short" and response_words > 100:
            reward -= 0.1
        elif preferred_length == "long" and response_words < 30:
            reward -= 0.1
        else:
            reward += 0.05
        return max(0.0, min(1.0, reward))

    def calculate_explicit_reward(self, score: float) -> float:
        if score <= 1:
            return 0.0
        if score <= 2:
            return 0.25
        if score <= 3:
            return 0.5
        if score <= 4:
            return 0.75
        return 1.0

    def calculate_final_reward(self, explicit: float | None = None, implicit: float = 0.5) -> float:
        if explicit is not None:
            return (explicit * 0.7) + (implicit * 0.3)
        return implicit

    def _is_similar(self, msg1: str, msg2: str) -> bool:
        words1 = set(msg1.lower().split())
        words2 = set(msg2.lower().split())
        if not words1 or not words2:
            return False
        return len(words1 & words2) / len(words1 | words2) > 0.6

    def update_preference(self, db: Session, key: str, value: str, reward: float) -> None:
        pref = db.scalar(select(UserPreference).where(UserPreference.preference_key == key))
        if pref:
            old_count = pref.sample_count
            old_conf = pref.confidence
            pref.confidence = round(((old_conf * old_count) + reward) / (old_count + 1), 4)
            pref.sample_count = old_count + 1
            if reward > old_conf + 0.1:
                pref.preference_value = value
            pref.last_updated = datetime.now()
        else:
            db.add(UserPreference(preference_key=key, preference_value=value, confidence=reward, sample_count=1))
        db.commit()

    def get_preferences(self, db: Session) -> dict[str, str]:
        self.ensure_defaults(db)
        rows = db.scalars(select(UserPreference)).all()
        return {row.preference_key: row.preference_value for row in rows}

    def get_preference_rows(self, db: Session) -> list[dict[str, Any]]:
        self.ensure_defaults(db)
        return [
            {
                "key": row.preference_key,
                "value": row.preference_value,
                "confidence": row.confidence,
                "sample_count": row.sample_count,
                "last_updated": row.last_updated.isoformat(timespec="seconds"),
            }
            for row in db.scalars(select(UserPreference).order_by(UserPreference.preference_key)).all()
        ]

    def infer_preferences_from_interaction(self, db: Session, user_msg: str, wayne_response: str, reward: float) -> None:
        words = len(wayne_response.split())
        if reward > 0.7:
            if words < 40:
                self.update_preference(db, "response_length", "short", reward)
            elif words < 120:
                self.update_preference(db, "response_length", "medium", reward)
            else:
                self.update_preference(db, "response_length", "long", reward)

            has_bullets = bool(re.search(r"^\s*[-*•]", wayne_response, re.MULTILINE))
            self.update_preference(db, "format", "bullets" if has_bullets else "prose", reward)
            is_formal = "Sir" in wayne_response or "certainly" in wayne_response.lower()
            self.update_preference(db, "tone", "formal" if is_formal else "casual", reward)
        self._track_topic(db, user_msg, reward)

    def _track_topic(self, db: Session, user_message: str, reward: float) -> None:
        topics = {
            "files": ["file", "open", "read", "folder", "document"],
            "calendar": ["meeting", "schedule", "event", "calendar", "book"],
            "tasks": ["task", "todo", "remind", "add", "complete"],
            "devices": ["laptop", "phone", "shutdown", "restart", "battery"],
            "system": ["cpu", "ram", "status", "performance", "memory"],
            "voice": ["listen", "speak", "voice", "talk", "say"],
        }
        msg_lower = user_message.lower()
        topic = next((name for name, words in topics.items() if any(word in msg_lower for word in words)), "general")
        row = db.scalar(select(TopicTracker).where(TopicTracker.topic == topic))
        if row:
            row.avg_score = round(((row.avg_score * row.frequency) + reward) / (row.frequency + 1), 4)
            row.frequency += 1
            row.last_asked = datetime.now()
        else:
            db.add(TopicTracker(topic=topic, frequency=1, avg_score=reward))
        db.commit()

    def store_golden_response(self, db: Session, user_msg: str, wayne_response: str, intent: str | None, reward: float) -> None:
        if reward >= 0.75:
            db.add(GoldenResponse(user_message=user_msg, wayne_response=wayne_response, intent=intent, reward_score=reward))
            db.commit()

    def get_golden_examples(self, db: Session, user_message: str, intent: str | None = None, limit: int = 3) -> list[dict[str, Any]]:
        query = select(GoldenResponse).order_by(GoldenResponse.reward_score.desc())
        if intent:
            query = query.where(GoldenResponse.intent == intent)
        rows = db.scalars(query.limit(limit * 3)).all()
        examples = []
        for row in rows:
            if self._is_similar(user_message, row.user_message) or not examples:
                row.use_count += 1
                examples.append({"user_message": row.user_message, "wayne_response": row.wayne_response, "reward_score": row.reward_score})
                if len(examples) >= limit:
                    break
        db.commit()
        return examples

    def log_interaction(
        self,
        db: Session,
        session_id: str,
        user_message: str,
        wayne_response: str,
        intent: str | None = None,
        tool_used: str | None = None,
        response_time_ms: int = 0,
        token_count: int = 0,
        implicit_score: float = 0.5,
    ) -> int:
        interaction = Interaction(
            session_id=session_id,
            user_message=user_message,
            wayne_response=wayne_response,
            intent=intent,
            tool_used=tool_used,
            response_time_ms=response_time_ms,
            token_count=token_count,
            implicit_score=implicit_score,
            final_reward=implicit_score,
        )
        db.add(interaction)
        db.commit()
        db.refresh(interaction)
        return interaction.id

    def update_interaction_reward(self, db: Session, interaction_id: int, explicit_score: float, feedback_text: str | None = None) -> float | None:
        interaction = db.get(Interaction, interaction_id)
        if not interaction:
            return None
        explicit_reward = self.calculate_explicit_reward(explicit_score)
        final = self.calculate_final_reward(explicit_reward, interaction.implicit_score)
        interaction.explicit_score = explicit_score
        interaction.final_reward = round(final, 4)
        interaction.feedback_text = feedback_text
        db.commit()
        self.infer_preferences_from_interaction(db, interaction.user_message, interaction.wayne_response, final)
        self.store_golden_response(db, interaction.user_message, interaction.wayne_response, interaction.intent, final)
        return final

    def get_learning_stats(self, db: Session) -> dict[str, Any]:
        self.ensure_defaults(db)
        interactions = db.scalars(select(Interaction)).all()
        rewards = [row.final_reward for row in interactions if row.final_reward is not None]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        topics = db.scalars(select(TopicTracker).order_by(TopicTracker.frequency.desc()).limit(6)).all()
        golden_count = db.scalar(select(func.count(GoldenResponse.id))) or 0
        contacts_count = db.scalar(select(func.count(KnownContact.id))) or 0
        habit_count = db.scalar(select(func.count(Habit.id))) or 0
        improvement_count = db.scalar(select(func.count(ImprovementLog.id))) or 0
        emotion_rows: dict[str, int] = {}
        heatmap = [[0 for _ in range(24)] for _ in range(7)]
        for row in interactions:
            snapshot = row.state_snapshot or {}
            emotion = snapshot.get("emotion") or "neutral"
            emotion_rows[emotion] = emotion_rows.get(emotion, 0) + 1
            when = row.timestamp or row.created_at
            heatmap[when.weekday()][when.hour] += 1
        files = db.scalars(select(FileAccessLog).order_by(FileAccessLog.access_count.desc()).limit(5)).all()
        tasks = db.scalars(select(Task)).all()
        task_completion_rate = (sum(1 for task in tasks if task.completed) / len(tasks)) if tasks else 0.0
        return {
            "total_interactions": len(interactions),
            "average_reward": round(avg_reward, 3),
            "golden_responses_stored": golden_count,
            "top_topics": [{"topic": t.topic, "frequency": t.frequency, "avg_score": t.avg_score} for t in topics],
            "learned_preferences": self.get_preference_rows(db),
            "known_contacts_count": contacts_count,
            "habit_count": habit_count,
            "golden_responses_count": golden_count,
            "improvement_count": improvement_count,
            "emotion_distribution": emotion_rows,
            "interaction_heatmap": heatmap,
            "top_files": [{"file_name": f.file_name, "access_count": f.access_count, "file_type": f.file_type} for f in files],
            "task_completion_rate": round(task_completion_rate, 3),
            "learning_score": round(min(avg_reward * 100, 100), 1),
        }

    def get_history(self, db: Session, limit: int = 20, min_reward: float = 0.0) -> list[dict[str, Any]]:
        rows = db.scalars(
            select(Interaction)
            .where(Interaction.final_reward >= min_reward)
            .order_by(Interaction.created_at.desc())
            .limit(limit)
        ).all()
        return [
            {
                "id": row.id,
                "created_at": row.created_at.isoformat(timespec="seconds"),
                "user_message": row.user_message,
                "wayne_response": row.wayne_response,
                "intent": row.intent,
                "final_reward": row.final_reward,
            }
            for row in rows
        ]

    async def analyze_and_improve(self, db: Session) -> list[str]:
        stats = self.get_learning_stats(db)
        improvements: list[str] = []
        if stats["total_interactions"] == 0 or stats["total_interactions"] % 50 != 0:
            return improvements
        if stats["average_reward"] < 0.4:
            improvements.append("Low average reward detected. Shortening responses and simplifying language.")
            self.update_preference(db, "response_length", "short", 0.8)
            self.update_preference(db, "verbosity", "0.3", 0.8)
        elif stats["average_reward"] > 0.8:
            improvements.append("High satisfaction detected. Maintaining current response style.")
        if stats["top_topics"]:
            top = stats["top_topics"][0]["topic"]
            improvements.append(f"User frequently asks about {top}. Prioritizing {top} context in responses.")
        for improvement in improvements:
            db.add(ImprovementLog(improvement=improvement, triggered_by=f"auto_analysis_{stats['total_interactions']}_interactions"))
        db.commit()
        return improvements


rl_engine = RLEngine()


def detect_intent(message: str) -> str:
    msg = message.lower()
    if any(w in msg for w in ["file", "open", "read", "folder"]):
        return "file"
    if any(w in msg for w in ["meeting", "schedule", "calendar"]):
        return "calendar"
    if any(w in msg for w in ["task", "todo", "remind"]):
        return "task"
    if any(w in msg for w in ["shutdown", "restart", "sleep", "battery"]):
        return "device"
    if any(w in msg for w in ["cpu", "ram", "status", "system"]):
        return "system"
    if any(w in msg for w in ["voice", "listen", "speak", "hear"]):
        return "voice"
    return "general"
