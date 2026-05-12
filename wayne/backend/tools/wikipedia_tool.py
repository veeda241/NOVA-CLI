from __future__ import annotations

import re

import httpx


class WikipediaTool:
    API_URL = "https://en.wikipedia.org/w/api.php"
    SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"

    async def search(self, query: str, limit: int = 3) -> dict:
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                response = await client.get(
                    self.API_URL,
                    params={"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json"},
                    headers={"User-Agent": "WAYNE/1.0 local assistant"},
                )
                response.raise_for_status()
            results = [
                {"title": item["title"], "snippet": re.sub("<.*?>", "", item.get("snippet", "")), "pageid": item["pageid"]}
                for item in response.json().get("query", {}).get("search", [])
            ]
            return {"query": query, "results": results}
        except Exception as exc:
            return {"query": query, "results": [], "error": str(exc)}

    async def get_summary(self, title_or_query: str) -> dict:
        title = title_or_query
        if " " in title_or_query or "?" in title_or_query:
            search = await self.search(title_or_query, limit=1)
            if search.get("results"):
                title = search["results"][0]["title"]
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                response = await client.get(
                    f"{self.SUMMARY_URL}/{title.replace(' ', '_')}",
                    headers={"Accept": "application/json", "User-Agent": "WAYNE/1.0 local assistant"},
                )
                response.raise_for_status()
            data = response.json()
            return {
                "title": data.get("title", title),
                "summary": data.get("extract", "")[:2000],
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "thumbnail": data.get("thumbnail", {}).get("source", ""),
                "description": data.get("description", ""),
                "source": "wikipedia",
            }
        except Exception as exc:
            return {"title": title, "summary": "", "error": str(exc), "source": "wikipedia"}

    async def get_sections(self, title: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(
                    self.API_URL,
                    params={"action": "parse", "page": title, "prop": "sections", "format": "json"},
                    headers={"User-Agent": "WAYNE/1.0 local assistant"},
                )
                response.raise_for_status()
            sections = response.json().get("parse", {}).get("sections", [])
            return {"title": title, "sections": [section["line"] for section in sections[:10]]}
        except Exception as exc:
            return {"title": title, "sections": [], "error": str(exc)}


wikipedia = WikipediaTool()
