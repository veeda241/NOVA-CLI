from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from typing import Any

import httpx

from database import SessionLocal
from models import ResponseCache, SpeedMetric

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")


class SpeedEngine:
    def __init__(self) -> None:
        self.cache: dict[str, dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.model_warm = False
        self.first_token_times: list[float] = []
        self.client = httpx.AsyncClient(timeout=120.0)
        self._keep_alive_started = False

    async def warm_model(self) -> None:
        try:
            await self.client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": "hi", "stream": False, "keep_alive": "24h"},
                timeout=30.0,
            )
            self.model_warm = True
            print("[W.A.Y.N.E SPEED] Model warmed and kept in RAM.")
        except Exception as exc:
            self.model_warm = False
            print(f"[W.A.Y.N.E SPEED] Warm failed: {exc}")

    async def keep_model_alive(self) -> None:
        if self._keep_alive_started:
            return
        self._keep_alive_started = True
        while True:
            await asyncio.sleep(600)
            try:
                await self.client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": "", "stream": False, "keep_alive": "24h"},
                    timeout=20.0,
                )
                self.model_warm = True
            except Exception:
                self.model_warm = False

    def get_cache_key(self, messages: list[dict[str, Any]], system_prompt: str = "") -> str:
        payload = json.dumps({"messages": messages[-3:], "system": system_prompt[:240]}, sort_keys=True, default=str)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    async def get_cached(self, key: str) -> str | None:
        entry = self.cache.get(key)
        if entry and time.time() - float(entry["timestamp"]) < 300:
            self.cache_hits += 1
            entry["hits"] = int(entry.get("hits", 0)) + 1
            self._bump_cache_row(key)
            return str(entry["response"])
        if entry:
            self.cache.pop(key, None)

        db = SessionLocal()
        try:
            row = db.query(ResponseCache).filter(ResponseCache.cache_key == key).first()
            if row and (row.expires_at is None or row.expires_at > datetime.now()):
                row.hit_count += 1
                db.commit()
                self.cache_hits += 1
                self.cache[key] = {"response": row.response, "timestamp": time.time(), "hits": row.hit_count}
                return row.response
        except Exception:
            db.rollback()
        finally:
            db.close()

        self.cache_misses += 1
        return None

    def set_cache(self, key: str, response: str, user_message: str | None = None) -> None:
        if len(self.cache) >= 1000:
            oldest_key = min(self.cache, key=lambda item: float(self.cache[item]["timestamp"]))
            self.cache.pop(oldest_key, None)
        self.cache[key] = {"response": response, "timestamp": time.time(), "hits": 0}
        db = SessionLocal()
        try:
            row = db.query(ResponseCache).filter(ResponseCache.cache_key == key).first()
            if row:
                row.response = response
                row.user_message = user_message
                row.expires_at = datetime.now() + timedelta(minutes=5)
            else:
                db.add(
                    ResponseCache(
                        cache_key=key,
                        user_message=user_message,
                        response=response,
                        expires_at=datetime.now() + timedelta(minutes=5),
                    )
                )
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()

    async def stream_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> AsyncGenerator[dict[str, Any], None]:
        start = time.time()
        first_token_sent = False
        full_response = ""
        full_messages = [{"role": "system", "content": system_prompt}] + messages[-10:]
        async with self.client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": full_messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_ctx": 4096,
                },
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    if not first_token_sent:
                        first_token_ms = (time.time() - start) * 1000
                        self.first_token_times.append(first_token_ms)
                        first_token_sent = True
                        yield {"type": "first_token", "first_token_ms": first_token_ms}
                    full_response += token
                    yield {"type": "token", "token": token}
                if chunk.get("done"):
                    yield {"type": "done", "response": full_response}
                    return

    async def predict_and_prefetch(self, current_message: str, state: dict[str, Any], system_prompt: str) -> None:
        for query in self._predict_next_query(current_message, state):
            key = self.get_cache_key([{"role": "user", "content": query}], system_prompt)
            if key not in self.cache:
                asyncio.create_task(self._prefetch(query, system_prompt, key))

    def log_metric(
        self,
        session_id: str,
        first_token_ms: float | None,
        total_response_ms: float | None,
        tokens_per_second: float | None,
        cache_hit: bool,
    ) -> None:
        db = SessionLocal()
        try:
            db.add(
                SpeedMetric(
                    session_id=session_id,
                    first_token_ms=first_token_ms,
                    total_response_ms=total_response_ms,
                    tokens_per_second=tokens_per_second,
                    cache_hit=cache_hit,
                )
            )
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()

    def get_avg_first_token_ms(self) -> float:
        recent = self.first_token_times[-20:]
        return sum(recent) / len(recent) if recent else 0.0

    def get_cache_stats(self) -> dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round((self.cache_hits / total) * 100, 1) if total else 0.0,
            "avg_first_token_ms": round(self.get_avg_first_token_ms(), 1),
            "model_warm": self.model_warm,
        }

    async def _prefetch(self, query: str, system_prompt: str, key: str) -> None:
        try:
            response = await self.client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 128, "keep_alive": "24h"},
                },
                timeout=60.0,
            )
            response.raise_for_status()
            text = response.json().get("message", {}).get("content", "")
            if text:
                self.set_cache(key, text, query)
        except Exception:
            pass

    def _predict_next_query(self, message: str, state: dict[str, Any]) -> list[str]:
        msg = (message or "").lower()
        hour = int(state.get("hour") or state.get("active_hour") or 12)
        predictions: list[str] = []
        if "task" in msg:
            predictions.append("show me all my tasks")
        if "file" in msg or "open" in msg:
            predictions.append("list my recent files")
        if "schedule" in msg or "meeting" in msg:
            predictions.append("what is on my calendar today")
        if "cache" in msg or "clean" in msg:
            predictions.append("how much disk space is free")
        if "shutdown" in msg:
            predictions.append("save my work first")
        if 8 <= hour <= 10:
            predictions.append("what is my schedule today")
        if 17 <= hour <= 19:
            predictions.append("what did I finish today")
        return predictions[:2]

    def _bump_cache_row(self, key: str) -> None:
        db = SessionLocal()
        try:
            row = db.query(ResponseCache).filter(ResponseCache.cache_key == key).first()
            if row:
                row.hit_count += 1
                db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()


speed_engine = SpeedEngine()
