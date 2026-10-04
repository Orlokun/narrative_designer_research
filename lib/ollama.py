"""
Thin async client for a local Ollama server.

Deliberately small: three verbs (generate, chat, embed) plus model listing.
If you find yourself wanting more, add narrowly-scoped methods here rather
than reaching into httpx from caller code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from lib.config import settings
from lib.llm_errors import LLMError


class OllamaError(LLMError):
    """Raised when Ollama returns a non-2xx or malformed response."""


class OllamaClient:
    """
    Usage:
        async with OllamaClient() as ol:
            reply = await ol.chat(
                model="gemma4:e4b",
                messages=[{"role": "user", "content": "Hola"}],
            )
    """

    def __init__(
        self,
        host: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.host = host or settings.ollama_host
        self.timeout = timeout or settings.ollama_timeout_s
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OllamaClient:
        self._client = httpx.AsyncClient(base_url=self.host, timeout=self.timeout)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- Internals --

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("OllamaClient must be used inside `async with`")
        return self._client

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._require_client()
        r = await client.post(path, json=payload)
        if r.status_code != 200:
            raise OllamaError(f"{path} -> {r.status_code}: {r.text[:500]}")
        try:
            return r.json()
        except ValueError as e:
            raise OllamaError(f"{path}: non-JSON response: {r.text[:500]}") from e

    async def _get_json(self, path: str) -> dict[str, Any]:
        client = self._require_client()
        r = await client.get(path)
        if r.status_code != 200:
            raise OllamaError(f"{path} -> {r.status_code}: {r.text[:500]}")
        try:
            return r.json()
        except ValueError as e:
            raise OllamaError(f"{path}: non-JSON response: {r.text[:500]}") from e

    # -- Public API --

    async def list_models(self) -> list[dict[str, Any]]:
        data = await self._get_json("/api/tags")
        return data.get("models", [])

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_predict: int = 512,
        stop: list[str] | None = None,
    ) -> str:
        """Single-shot completion. Use `chat` for multi-turn."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
            },
        }
        if system is not None:
            payload["system"] = system
        if stop:
            payload["options"]["stop"] = stop
        data = await self._post_json("/api/generate", payload)
        return data.get("response", "")

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_predict: int = 512,
        think: bool | None = None,
    ) -> str:
        """Multi-turn chat. messages is a list of {role, content}.

        Pass think=False to disable the thinking/reasoning phase on models that
        support it (e.g. Gemma4). Without this, thinking tokens consume the
        num_predict budget before the model outputs any visible content.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
            },
        }
        if think is not None:
            payload["think"] = think
        data = await self._post_json("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_predict: int = 512,
    ) -> AsyncIterator[str]:
        """
        Stream tokens for the Unity typewriter effect.
        Yields content deltas; caller concatenates if they need the full reply.
        """
        client = self._require_client()
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
            },
        }
        async with client.stream("POST", "/api/chat", json=payload) as r:
            if r.status_code != 200:
                body = await r.aread()
                raise OllamaError(f"/api/chat -> {r.status_code}: {body[:500]!r}")
            import json

            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    yield delta
                if chunk.get("done"):
                    break

    async def embed(self, model: str, text: str | list[str]) -> list[list[float]]:
        """
        Return embeddings. Accepts a single string or a list of strings;
        always returns a list of vectors for consistency.
        """
        payload = {"model": model, "input": text if isinstance(text, list) else [text]}
        data = await self._post_json("/api/embed", payload)
        return data.get("embeddings", [])
