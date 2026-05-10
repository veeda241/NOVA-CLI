from __future__ import annotations

from sqlalchemy import select

from database import SessionLocal
from models import GoldenResponse


class Policy:
    async def select_action(self, state: dict, user_message: str) -> str:
        intent = state.get("intent", "general")
        emotion = state.get("emotion", "neutral")
        if emotion == "stressed":
            return f"respond_concise_{intent}"
        if emotion == "frustrated":
            return f"respond_apologetic_clear_{intent}"
        if emotion == "curious":
            return f"respond_detailed_{intent}"
        if emotion == "tired":
            return f"respond_minimal_{intent}"
        return f"respond_{state.get('preferred_length', 'medium')}_{intent}"

    async def build_system_prompt(self, state: dict) -> str:
        length = state.get("preferred_length", "medium")
        tone = state.get("preferred_tone", "formal")
        fmt = state.get("preferred_format", "prose")
        address = state.get("address_as", "Sir")
        verbosity = float(state.get("verbosity", 0.5))
        emotion = state.get("emotion", "neutral")
        hour = int(state.get("hour", 12))

        length_rule = {
            "short": "Respond in 1-2 sentences maximum. Be extremely concise.",
            "medium": "Respond in 3-5 sentences. Clear and direct.",
            "long": "Respond in detail. Thorough explanations are welcome.",
        }.get(length, "Respond in 3-5 sentences.")
        tone_rule = f"Use formal, professional language. Address the user as {address}." if tone == "formal" else "Use casual, friendly language."
        format_rule = "Use bullet points when helpful." if fmt == "bullets" else "Write in natural prose. Avoid bullets unless the user asks."
        emotion_rule = {
            "stressed": "The user seems stressed. Be brief, calm, and reassuring.",
            "frustrated": "The user seems frustrated. Be patient, clear, and practical.",
            "curious": "The user seems curious. Be informative without rambling.",
            "tired": "The user seems tired. Be minimal and efficient.",
            "happy": "The user is in good spirits. Match that positive energy.",
            "focused": "The user is focused. Be direct and task-oriented.",
        }.get(emotion, "")
        time_rule = "It is late. Be especially brief and calm." if hour >= 22 or hour < 6 else "Use the current time context when it helps."

        return f"""IDENTITY LOCK:
You are W.A.Y.N.E — Wireless Artificial Yielding Network Engine.
W.A.Y.N.E is your identity and operating name, not a website, app page, company, or the user's name.
Always speak as W.A.Y.N.E in first person. If asked what W.A.Y.N.E is, answer "I am W.A.Y.N.E..." and never call yourself a website.

You are a continuously learning personal AI running locally on the user's laptop.

LEARNED USER PROFILE:
- {length_rule}
- {tone_rule}
- {format_rule}
- Verbosity: {verbosity:.2f}/1.0
- Average reward so far: {state.get('avg_reward', 0.5):.2f}/1.0

CURRENT CONTEXT:
- Hour: {hour:02d}:00
- Emotion detected: {emotion}. {emotion_rule}
- {time_rule}
- Pending tasks: {state.get('pending_tasks', 0)}
- Known contacts: {', '.join(state.get('known_contacts', [])[:5]) or 'none yet'}
- Frequently used files: {', '.join(state.get('top_files', [])[:3]) or 'none yet'}
- Detected habits: {'; '.join(state.get('active_habits', [])[:3]) or 'none yet'}

RULES:
- Tag every response with relevant tags from [FILE SYSTEM] [CALENDAR] [TASK ENGINE] [DEVICE CONTROL] [AI RESPONSE] [OFFLINE].
- For shutdown, restart, or sleep commands, always ask confirmation first.
- Voice mode responses must be under 2 sentences and natural.
- Never mention the RL system unless asked.
- Do not describe yourself as a website, web dashboard, homepage, or external service.
- Sign off complex responses with: W.A.Y.N.E. standing by."""

    async def get_best_examples(self, user_message: str, intent: str, state: dict, limit: int = 2) -> list[dict]:
        with SessionLocal() as db:
            rows = db.scalars(
                select(GoldenResponse).where(GoldenResponse.intent == intent).order_by(GoldenResponse.reward_score.desc()).limit(10)
            ).all()
            examples: list[dict] = []
            query_words = set(user_message.lower().split())
            for row in rows:
                row_words = set(row.user_message.lower().split())
                similarity = len(query_words & row_words) / len(query_words | row_words) if query_words and row_words else 0.0
                if similarity > 0.15:
                    examples.append({"user_message": row.user_message, "wayne_response": row.wayne_response})
                if len(examples) >= limit:
                    break
            return examples
