from __future__ import annotations


class EmotionDetector:
    KEYWORDS = {
        "stressed": ["urgent", "asap", "deadline", "hurry", "quick", "panic", "stuck", "broken", "crash", "error", "help"],
        "frustrated": ["again", "still", "not working", "wrong", "useless", "annoying", "seriously"],
        "happy": ["thanks", "great", "awesome", "perfect", "excellent", "good"],
        "curious": ["how", "what", "why", "explain", "tell me", "show me"],
        "tired": ["tired", "sleep", "rest", "break", "long day", "enough"],
        "focused": ["start", "begin", "continue", "next", "finish", "complete", "focus"],
    }

    async def from_text(self, text: str) -> str:
        lowered = text.lower()
        scores: dict[str, int] = {}
        for emotion, words in self.KEYWORDS.items():
            score = sum(1 for word in words if word in lowered)
            if score:
                scores[emotion] = score
        if text.count("!") > 2:
            scores["stressed"] = scores.get("stressed", 0) + 2
        if text.count("?") > 2:
            scores["curious"] = scores.get("curious", 0) + 1
        if text.isupper() and len(text) > 5:
            scores["frustrated"] = scores.get("frustrated", 0) + 3
        return max(scores, key=scores.get) if scores else "neutral"

    async def from_audio(self, audio_bytes: bytes) -> str:
        try:
            import numpy as np

            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            rms = float(np.sqrt(np.mean(audio.astype(float) ** 2)))
            if rms > 8000:
                return "stressed"
            if rms < 2000:
                return "tired"
        except Exception:
            pass
        return "neutral"
