from __future__ import annotations

from core.language_engine import SUPPORTED_LANGUAGES, language_engine


class TranslationTool:
    async def translate(self, text: str, target_language: str, source_language: str = "auto") -> dict:
        if target_language == source_language or target_language == "auto":
            return {"original": text, "translated": text, "source_language": source_language, "target_language": target_language, "success": True}
        try:
            from deep_translator import GoogleTranslator

            translated = GoogleTranslator(source=source_language, target=target_language).translate(text)
            return {
                "original": text,
                "translated": translated,
                "source_language": source_language,
                "target_language": target_language,
                "success": True,
            }
        except Exception as exc:
            return {"original": text, "translated": text, "source_language": source_language, "target_language": target_language, "error": str(exc), "success": False}

    async def detect_and_translate_to_english(self, text: str) -> dict:
        detected = language_engine.detect_language(text)
        if detected == "en":
            return {"text": text, "original_language": "en", "was_translated": False}
        translated = await self.translate(text, "en", detected)
        return {
            "text": translated.get("translated", text),
            "original_language": detected,
            "original_text": text,
            "was_translated": translated.get("success", False),
        }

    async def translate_response(self, english_text: str, target_language: str) -> str:
        if target_language in {"en", "auto"}:
            return english_text
        result = await self.translate(english_text, target_language, "en")
        return result.get("translated", english_text)

    def get_supported_languages(self) -> list[dict[str, str]]:
        return [{"code": code, "name": info["name"]} for code, info in SUPPORTED_LANGUAGES.items()]


translation_tool = TranslationTool()
