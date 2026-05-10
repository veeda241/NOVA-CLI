from __future__ import annotations


class RewardEngine:
    async def implicit(
        self,
        user_message: str,
        wayne_response: str,
        next_message: str | None,
        state: dict,
    ) -> float:
        reward = 0.5
        if next_message:
            msg = next_message.lower()
            if any(word in msg for word in ["perfect", "exactly", "great", "excellent", "thanks", "helpful", "correct", "yes"]):
                reward += 0.3
            if any(word in msg for word in ["ok", "got it", "sure", "makes sense", "understood"]):
                reward += 0.1
            if any(word in msg for word in ["wrong", "incorrect", "not what", "terrible", "useless", "nevermind"]):
                reward -= 0.35
            if any(word in msg for word in ["again", "repeat", "clarify", "confused", "didn't get"]):
                reward -= 0.2
            if self._similarity(user_message, next_message) > 0.65:
                reward -= 0.25

        words = len(wayne_response.split())
        preferred = state.get("preferred_length", "medium")
        if preferred == "short" and words > 80:
            reward -= 0.1
        elif preferred == "long" and words < 30:
            reward -= 0.1
        elif preferred == "medium" and 20 <= words <= 120:
            reward += 0.05

        emotion = state.get("emotion", "neutral")
        if emotion == "stressed" and words < 50:
            reward += 0.05
        if emotion == "curious" and words > 50:
            reward += 0.05
        if emotion == "frustrated" and words > 100:
            reward -= 0.1
        return self._clamp(reward)

    async def from_emotion(self, emotion: str | None, next_message: str | None) -> float:
        if not emotion or not next_message:
            return 0.5
        msg = next_message.lower()
        if emotion == "stressed":
            return 0.75 if any(word in msg for word in ["ok", "thanks", "clear", "better"]) else 0.45
        if emotion == "frustrated":
            return 0.8 if any(word in msg for word in ["thanks", "got it", "perfect"]) else 0.3
        if emotion == "happy":
            return 0.7
        return 0.5

    async def from_timing(self, response_time_ms: int, state: dict) -> float:
        if response_time_ms <= 0:
            return 0.5
        if response_time_ms < 1000:
            return 0.9
        if response_time_ms < 3000:
            return 0.75
        if response_time_ms < 8000:
            return 0.6
        if response_time_ms < 15000:
            return 0.4
        return 0.2

    async def from_explicit(self, score: float | None) -> float | None:
        if score is None:
            return None
        return self._clamp((score - 1) / 4)

    async def from_voice_feedback(self, voice_text: str | None) -> float | None:
        if not voice_text:
            return None
        text = voice_text.lower()
        if any(word in text for word in ["good", "perfect", "correct", "right", "well done", "exactly", "yes"]):
            return 1.0
        if any(word in text for word in ["wrong", "bad", "incorrect", "terrible", "again"]):
            return 0.0
        return None

    async def combine(
        self,
        implicit: float,
        emotion: float = 0.5,
        timing: float = 0.5,
        explicit: float | None = None,
        voice: float | None = None,
    ) -> float:
        if explicit is not None:
            value = explicit * 0.6 + implicit * 0.2 + emotion * 0.1 + timing * 0.1
        elif voice is not None:
            value = voice * 0.55 + implicit * 0.25 + emotion * 0.1 + timing * 0.1
        else:
            value = implicit * 0.5 + emotion * 0.3 + timing * 0.2
        return round(self._clamp(value), 4)

    def _similarity(self, first: str, second: str) -> float:
        one = set(first.lower().split())
        two = set(second.lower().split())
        if not one or not two:
            return 0.0
        return len(one & two) / len(one | two)

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))
