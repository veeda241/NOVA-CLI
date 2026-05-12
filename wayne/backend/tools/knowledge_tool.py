from __future__ import annotations

from tools.datetime_tool import datetime_tool
from tools.special_days import special_days_engine
from tools.web_search_tool import web_search
from tools.wikipedia_tool import wikipedia


class KnowledgeTool:
    WIKI_TRIGGERS = (
        "who is",
        "who was",
        "what is",
        "what was",
        "tell me about",
        "explain",
        "history of",
        "biography",
        "definition",
        "capital of",
        "where is",
        "invented",
        "discovered",
    )
    WEB_TRIGGERS = (
        "latest",
        "recent",
        "today",
        "current",
        "news",
        "2026",
        "2025",
        "price",
        "weather",
        "score",
        "result",
        "winner",
        "now",
        "live",
    )

    async def answer(self, query: str, source: str = "auto") -> dict:
        lowered = query.lower()
        if source == "local" or self._is_datetime_query(lowered):
            return await self._answer_datetime(query)
        if self._is_special_day_query(lowered):
            return await self._answer_special_days(query)
        if source == "web" or any(trigger in lowered for trigger in self.WEB_TRIGGERS):
            return await self._answer_web(query)
        if source == "wikipedia" or any(trigger in lowered for trigger in self.WIKI_TRIGGERS):
            wiki = await self._answer_wikipedia(query)
            if wiki.get("answer"):
                return wiki
        wiki = await self._answer_wikipedia(query)
        if wiki.get("answer"):
            return wiki
        return await self._answer_web(query)

    def _is_datetime_query(self, lowered: str) -> bool:
        return any(phrase in lowered for phrase in ("what time", "current time", "today date", "today's date", "what date", "what day", "which day", "current date"))

    def _is_special_day_query(self, lowered: str) -> bool:
        return any(word in lowered for word in ("holiday", "festival", "special day", "observance", "celebration", "republic day", "independence day", "christmas", "upcoming days"))

    async def _answer_datetime(self, query: str) -> dict:
        lowered = query.lower()
        if "time in" in lowered:
            result = datetime_tool.get_time_in_timezone(self._extract_timezone(query))
        else:
            result = datetime_tool.get_current()
        answer = (
            f"Today is {result.get('day_name')}, {result.get('month_name')} {result.get('day_number')}, "
            f"{result.get('year')}. The current time is {result.get('time_12h')}."
        )
        return {"answer": answer, "data": result, "source": "system_clock"}

    async def _answer_special_days(self, query: str) -> dict:
        lowered = query.lower()
        if "upcoming" in lowered or "next" in lowered:
            upcoming = special_days_engine.get_upcoming(30)
            if not upcoming:
                return {"answer": "No major upcoming special days found in the next 30 days.", "source": "special_days_db"}
            answer = "Upcoming special days: " + "; ".join(
                f"{item['name']} in {item['days_away']} days ({item['formatted_date']})" for item in upcoming[:5]
            )
            return {"answer": answer, "data": upcoming[:5], "source": "special_days_db"}
        today = special_days_engine.get_today_specials()
        if today.get("has_special_day"):
            names = ", ".join(item["name"] for item in today["special_days"])
            answer = f"Today ({today.get('formatted') or today.get('today')}) is {names}."
        else:
            answer = f"Today ({today.get('formatted') or today.get('today')}) has no major special observance in the local database."
        return {"answer": answer, "data": today, "source": "special_days_db"}

    async def _answer_wikipedia(self, query: str) -> dict:
        summary = await wikipedia.get_summary(query)
        if summary.get("summary"):
            return {"answer": summary["summary"], "title": summary.get("title"), "url": summary.get("url"), "source": "wikipedia"}
        return {"answer": None, "source": "wikipedia"}

    async def _answer_web(self, query: str) -> dict:
        result = await web_search.search(query)
        results = result.get("results", [])
        if not results:
            return {"answer": "I could not find reliable current web results for that.", "source": "web_search", "data": result}
        answer = " ".join(item["snippet"] for item in results[:2] if item.get("snippet"))[:1200]
        return {"answer": answer, "sources": [item.get("url") for item in results[:2] if item.get("url")], "source": "web_search", "data": results[:2]}

    def _extract_timezone(self, query: str) -> str:
        mapping = {
            "new york": "America/New_York",
            "london": "Europe/London",
            "paris": "Europe/Paris",
            "tokyo": "Asia/Tokyo",
            "dubai": "Asia/Dubai",
            "singapore": "Asia/Singapore",
            "sydney": "Australia/Sydney",
            "india": "Asia/Kolkata",
            "chennai": "Asia/Kolkata",
            "mumbai": "Asia/Kolkata",
            "delhi": "Asia/Kolkata",
            "kolkata": "Asia/Kolkata",
            "utc": "UTC",
            "gmt": "GMT",
            "est": "America/New_York",
            "pst": "America/Los_Angeles",
        }
        lowered = query.lower()
        for key, zone in mapping.items():
            if key in lowered:
                return zone
        return "Asia/Kolkata"


knowledge_tool = KnowledgeTool()
