from __future__ import annotations

import httpx
from bs4 import BeautifulSoup


class WebSearchTool:
    async def search(self, query: str, max_results: int = 5) -> dict:
        instant = await self._duckduckgo_instant(query, max_results)
        if instant.get("results"):
            return instant
        return await self._duckduckgo_html(query, max_results)

    async def _duckduckgo_instant(self, query: str, max_results: int) -> dict:
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_redirect": "1", "no_html": "1", "skip_disambig": "1"},
                )
                response.raise_for_status()
                data = response.json()
            results: list[dict] = []
            if data.get("Answer"):
                results.append({"title": "Direct Answer", "snippet": data["Answer"], "url": data.get("AbstractURL", ""), "source": "DuckDuckGo"})
            if data.get("AbstractText"):
                results.append({"title": data.get("Heading") or query, "snippet": data["AbstractText"], "url": data.get("AbstractURL", ""), "source": "DuckDuckGo"})
            for topic in data.get("RelatedTopics", []):
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({"title": topic.get("Text", "")[:90], "snippet": topic["Text"], "url": topic.get("FirstURL", ""), "source": "DuckDuckGo"})
                if len(results) >= max_results:
                    break
            return {"query": query, "results": results[:max_results], "source": "duckduckgo"}
        except Exception as exc:
            return {"query": query, "results": [], "error": str(exc), "source": "duckduckgo"}

    async def _duckduckgo_html(self, query: str, max_results: int) -> dict:
        try:
            async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True) as client:
                response = await client.get("https://html.duckduckgo.com/html/", params={"q": query})
                response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            results: list[dict] = []
            for item in soup.select(".result__body")[:max_results]:
                title = item.select_one(".result__title")
                snippet = item.select_one(".result__snippet")
                link = item.select_one(".result__a")
                if title and snippet:
                    results.append(
                        {
                            "title": title.get_text(" ", strip=True),
                            "snippet": snippet.get_text(" ", strip=True),
                            "url": link.get("href", "") if link else "",
                            "source": "DuckDuckGo",
                        }
                    )
            return {"query": query, "results": results, "source": "duckduckgo_html"}
        except Exception as exc:
            return {"query": query, "results": [], "error": str(exc), "source": "duckduckgo_html"}

    async def get_page_content(self, url: str, max_chars: int = 3000) -> dict:
        try:
            async with httpx.AsyncClient(timeout=15.0, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = " ".join(soup.get_text(" ", strip=True).split())
            return {"url": url, "content": text[:max_chars], "success": True}
        except Exception as exc:
            return {"url": url, "error": str(exc), "success": False}


web_search = WebSearchTool()
