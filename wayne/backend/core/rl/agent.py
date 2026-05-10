from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select

from database import SessionLocal
from models import GoldenResponse, Interaction, UserState

from .behavior_tracker import BehaviorTracker
from .emotion_detector import EmotionDetector
from .habit_learner import HabitLearner
from .intent_classifier import IntentClassifier
from .memory_bank import MemoryBank
from .policy import Policy
from .preference_model import PreferenceModel
from .reward_engine import RewardEngine
from .self_improver import SelfImprover
from .state_builder import StateBuilder


class WAYNEAgent:
    def __init__(self) -> None:
        self.state_builder = StateBuilder()
        self.reward_engine = RewardEngine()
        self.policy = Policy()
        self.memory = MemoryBank()
        self.preferences = PreferenceModel()
        self.behavior = BehaviorTracker()
        self.emotion = EmotionDetector()
        self.habits = HabitLearner()
        self.intent = IntentClassifier()
        self.improver = SelfImprover()

    async def observe(
        self,
        user_message: str,
        session_id: str,
        voice_audio: bytes | None = None,
        device_context: dict | None = None,
    ) -> dict:
        state = await self.state_builder.build(user_message, session_id, voice_audio, device_context)
        state["emotion"] = await (self.emotion.from_audio(voice_audio) if voice_audio else self.emotion.from_text(user_message))
        state["habit_context"] = await self.habits.get_context(state)
        state["intent"] = await self.intent.classify(user_message, state)
        await self._save_state(state)
        return state

    async def act(self, state: dict, user_message: str, session_id: str) -> dict:
        action = await self.policy.select_action(state, user_message)
        return {
            "action": action,
            "system_prompt": await self.policy.build_system_prompt(state),
            "few_shot_examples": await self.policy.get_best_examples(user_message, state.get("intent", "general"), state),
            "state": state,
        }

    async def learn(
        self,
        interaction_id: int,
        state: dict,
        action: str,
        wayne_response: str,
        user_message: str,
        next_user_message: str | None = None,
        explicit_score: float | None = None,
        voice_feedback: str | None = None,
        response_time_ms: int = 0,
    ) -> float:
        implicit = await self.reward_engine.implicit(user_message, wayne_response, next_user_message, state)
        emotion = await self.reward_engine.from_emotion(state.get("emotion"), next_user_message)
        timing = await self.reward_engine.from_timing(response_time_ms, state)
        explicit = await self.reward_engine.from_explicit(explicit_score) if explicit_score is not None else None
        voice = await self.reward_engine.from_voice_feedback(voice_feedback) if voice_feedback else None
        final = await self.reward_engine.combine(implicit, emotion, timing, explicit, voice)

        with SessionLocal() as db:
            row = db.get(Interaction, interaction_id)
            if row:
                row.implicit_score = implicit
                row.emotion_score = emotion
                row.explicit_score = explicit_score
                row.final_reward = final
                row.action_type = action
            db.commit()

        await self.preferences.update_from_interaction(user_message, wayne_response, state, final)
        await self.memory.store(state, action, final)
        await self.habits.observe(state, action, final)
        await self.behavior.learn_language(user_message, final)
        await self.behavior.extract_contacts(user_message, wayne_response)

        if final >= 0.75:
            with SessionLocal() as db:
                db.add(GoldenResponse(user_message=user_message, wayne_response=wayne_response, intent=state.get("intent"), state_context=state, reward_score=final))
                db.commit()

        with SessionLocal() as db:
            count = db.scalar(select(func.count(Interaction.id))) or 0
        if count and count % 50 == 0:
            await self.improver.analyze_and_improve(count)
        if count and count % 100 == 0:
            await self.memory.replay_and_update(self.preferences)
        return final

    async def log_interaction(
        self,
        session_id: str,
        user_message: str,
        wayne_response: str,
        state: dict,
        action: str,
        tool_used: str | None = None,
        response_time_ms: int = 0,
    ) -> int:
        with SessionLocal() as db:
            row = Interaction(
                session_id=session_id,
                timestamp=datetime.now(),
                user_message=user_message,
                wayne_response=wayne_response,
                intent=state.get("intent"),
                tool_used=tool_used,
                action_type=action,
                state_snapshot=state,
                response_time_ms=response_time_ms,
                token_count=len(wayne_response.split()),
                implicit_score=0.5,
                final_reward=0.5,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return row.id

    async def _save_state(self, state: dict) -> None:
        with SessionLocal() as db:
            db.add(
                UserState(
                    active_hour=state.get("hour"),
                    day_of_week=state.get("day_of_week"),
                    dominant_intent=state.get("intent"),
                    emotion=state.get("emotion"),
                    device_active=state.get("active_device"),
                    recent_topics=state.get("recent_topics", []),
                    stress_level=state.get("stress_level", 0.5),
                    focus_level=state.get("focus_level", 0.5),
                    productivity_score=state.get("completion_rate", 0.5),
                )
            )
            db.commit()


agent = WAYNEAgent()
