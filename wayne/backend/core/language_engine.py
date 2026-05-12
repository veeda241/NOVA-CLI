from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

import numpy as np
import whisper

try:
    import torch
except Exception:  # pragma: no cover - optional acceleration check
    torch = None

from database import SessionLocal
from models import UserPreference


SUPPORTED_LANGUAGES: dict[str, dict[str, str]] = {
    "auto": {"name": "Auto Detect", "whisper": "auto", "greeting": "W.A.Y.N.E online. Speak in any language."},
    "en": {"name": "English", "whisper": "en", "greeting": "W.A.Y.N.E online. How can I assist you?"},
    "ta": {"name": "Tamil", "whisper": "ta", "greeting": "Wayne இயங்குகிறது. நான் உங்களுக்கு எப்படி உதவலாம்?"},
    "hi": {"name": "Hindi", "whisper": "hi", "greeting": "Wayne ऑनलाइन है। मैं आपकी कैसे मदद कर सकता हूं?"},
    "te": {"name": "Telugu", "whisper": "te", "greeting": "Wayne ఆన్‌లైన్‌లో ఉంది. నేను మీకు ఎలా సహాయం చేయగలను?"},
    "kn": {"name": "Kannada", "whisper": "kn", "greeting": "Wayne ಆನ್‌ಲೈನ್‌ನಲ್ಲಿದೆ. ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?"},
    "ml": {"name": "Malayalam", "whisper": "ml", "greeting": "Wayne ഓൺലൈനിലാണ്. ഞാൻ നിങ്ങൾക്ക് എങ്ങനെ സഹായിക്കാം?"},
    "fr": {"name": "French", "whisper": "fr", "greeting": "Wayne en ligne. Comment puis-je vous aider ?"},
    "es": {"name": "Spanish", "whisper": "es", "greeting": "Wayne en línea. ¿Cómo puedo ayudarte?"},
    "de": {"name": "German", "whisper": "de", "greeting": "Wayne online. Wie kann ich Ihnen helfen?"},
    "ja": {"name": "Japanese", "whisper": "ja", "greeting": "Wayne オンラインです。どのようにお手伝いできますか？"},
    "zh": {"name": "Chinese", "whisper": "zh", "greeting": "Wayne 在线。我能帮您什么？"},
    "ar": {"name": "Arabic", "whisper": "ar", "greeting": "Wayne متصل. كيف يمكنني مساعدتك؟"},
    "pt": {"name": "Portuguese", "whisper": "pt", "greeting": "Wayne online. Como posso ajudá-lo?"},
    "ru": {"name": "Russian", "whisper": "ru", "greeting": "Wayne онлайн. Чем могу помочь?"},
    "ko": {"name": "Korean", "whisper": "ko", "greeting": "Wayne 온라인. 어떻게 도와드릴까요?"},
}


class LanguageEngine:
    def __init__(self) -> None:
        self.current_language = "auto"
        self.auto_detect = True
        self.user_name: str | None = None
        self.custom_vocabulary: list[str] = ["W.A.Y.N.E", "Wayne", "Spiderboy"]
        self.language_history: list[str] = []
        self._model: Any | None = None

    def _model_name(self) -> str:
        configured = os.getenv("WAYNE_WHISPER_MODEL")
        if configured:
            return configured
        return "medium" if torch is not None and getattr(torch.cuda, "is_available", lambda: False)() else "small"

    def model(self):
        if self._model is None:
            model_name = self._model_name()
            print(f"[W.A.Y.N.E] Loading Whisper {model_name} model...")
            self._model = whisper.load_model(model_name)
            print("[W.A.Y.N.E] Whisper ready.")
        return self._model

    def transcribe(self, audio: bytes | np.ndarray | str, language: str | None = None, context_hint: str | None = None) -> dict:
        audio_input: np.ndarray | str
        if isinstance(audio, bytes):
            audio_input = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_input = audio
        options: dict[str, Any] = {
            "task": "transcribe",
            "best_of": 5,
            "beam_size": 5,
            "temperature": (0.0, 0.2, 0.4),
            "condition_on_previous_text": True,
            "word_timestamps": True,
            "no_speech_threshold": 0.6,
            "logprob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,
            "fp16": False,
        }
        selected_language = language or (None if self.auto_detect else self.current_language)
        if selected_language and selected_language != "auto":
            options["language"] = SUPPORTED_LANGUAGES.get(selected_language, {}).get("whisper", selected_language)
        prompt = self._context_prompt(context_hint)
        if prompt:
            options["initial_prompt"] = prompt

        try:
            result = self.model().transcribe(audio_input, **options)
            text = str(result.get("text", "")).strip()
            detected = str(result.get("language") or selected_language or "en")
            if self.auto_detect and detected:
                self.current_language = detected
                self.language_history.append(detected)
            self._extract_user_info(text)
            return {
                "text": text,
                "language": detected,
                "language_name": SUPPORTED_LANGUAGES.get(detected, {}).get("name", detected),
                "confidence": self._confidence(result),
                "segments": result.get("segments", []),
            }
        except Exception as exc:
            return {"text": "", "language": "en", "language_name": "English", "confidence": 0.0, "error": str(exc)}

    def detect_language(self, text: str) -> str:
        if not text.strip():
            return self.current_language if self.current_language != "auto" else "en"
        try:
            from langdetect import detect

            detected = detect(text)
            return "zh" if detected.startswith("zh") else detected
        except Exception:
            if re.search(r"[\u0b80-\u0bff]", text):
                return "ta"
            if re.search(r"[\u0900-\u097f]", text):
                return "hi"
            if re.search(r"[\u0c00-\u0c7f]", text):
                return "te"
            return self.current_language if self.current_language != "auto" else "en"

    def add_vocabulary(self, words: list[str]) -> list[str]:
        for word in words:
            cleaned = str(word).strip()
            if cleaned and cleaned.lower() not in {item.lower() for item in self.custom_vocabulary}:
                self.custom_vocabulary.append(cleaned)
        self.custom_vocabulary = self.custom_vocabulary[:80]
        return self.custom_vocabulary

    def set_language(self, language: str) -> None:
        if language not in SUPPORTED_LANGUAGES:
            language = "auto"
        self.current_language = language
        self.auto_detect = language == "auto"
        self._save_preference("language", language)

    def get_greeting(self, language: str | None = None) -> str:
        code = language or self.current_language
        return SUPPORTED_LANGUAGES.get(code, SUPPORTED_LANGUAGES["en"])["greeting"]

    def supported(self) -> list[dict[str, str]]:
        return [{"code": code, "name": info["name"]} for code, info in SUPPORTED_LANGUAGES.items()]

    def _context_prompt(self, context_hint: str | None = None) -> str:
        parts: list[str] = []
        if self.user_name:
            parts.append(f"The user's name is {self.user_name}.")
        if self.custom_vocabulary:
            parts.append("Common words and names: " + ", ".join(self.custom_vocabulary[:30]) + ".")
        if context_hint:
            parts.append(context_hint)
        return " ".join(parts)

    def _extract_user_info(self, text: str) -> None:
        lowered = text.lower()
        for pattern in ("my name is", "i am", "i'm", "call me", "you can call me"):
            if pattern in lowered:
                after = text[lowered.index(pattern) + len(pattern) :].strip()
                match = re.match(r"([A-Za-z0-9_.-]{2,40})", after)
                if match:
                    self.user_name = match.group(1)
                    self.add_vocabulary([self.user_name])
                    self._save_preference("user_name", self.user_name)
                return
        correction = re.search(r"i said\s+(.+?)\s+not\s+(.+)", lowered)
        if correction:
            self.add_vocabulary([correction.group(1).strip().title()])

    def _confidence(self, result: dict) -> float:
        segments = result.get("segments", [])
        if not segments:
            return 0.5
        avg = sum(float(item.get("avg_logprob", -1.0)) for item in segments) / len(segments)
        return max(0.0, min(1.0, (avg + 1.2) / 1.2))

    def _save_preference(self, key: str, value: str) -> None:
        try:
            with SessionLocal() as db:
                pref = db.query(UserPreference).filter(UserPreference.preference_key == key).one_or_none()
                if pref:
                    pref.preference_value = value
                    pref.confidence = 1.0
                    pref.sample_count += 1
                else:
                    db.add(UserPreference(preference_key=key, preference_value=value, confidence=1.0, sample_count=1))
                db.commit()
        except Exception:
            pass


@lru_cache(maxsize=1)
def get_language_engine() -> LanguageEngine:
    return LanguageEngine()


language_engine = get_language_engine()
