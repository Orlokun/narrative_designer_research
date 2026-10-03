"""
LLM provider layer — one factory, two backends.

The agents never care *where* the model runs. They ask for ``LLMClient()`` and
get an async context manager with the same verbs the pipeline has always used
(``generate``, ``chat``, ``list_models``):

* ``ollama`` — a local Ollama server (``lib.ollama.OllamaClient``). The original
  Cybersyn setup.
* ``gemini`` — Google AI Studio's Gemini API (``GeminiClient`` below), which
  serves the open Gemma models (e.g. ``gemma-3-27b-it``) on a free tier. Used
  when no local GPU is available.

Selection: ``settings.llm_provider`` (env ``LLM_PROVIDER``). Model names the
agents pass (``settings.ollama_model_npc`` …) are Ollama tags; the Gemini backend
maps every request onto ``settings.gemini_model`` so the agents need no changes.

All backends raise ``LLMError`` (``OllamaError`` is a subclass) so callers catch
one exception type.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from lib.config import settings
from lib.llm_errors import LLMError

__all__ = ["GEMINI_BASE_URL", "GeminiClient", "LLMClient", "LLMError", "build_gemini_payload"]


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Free-tier 429s are common; back off and retry a few times before giving up.
_GEMINI_MAX_ATTEMPTS: int = 4
_GEMINI_BACKOFF_S: tuple[float, ...] = (2.0, 5.0, 15.0)


class GeminiClient:
    """Async client for the Gemini API (Google AI Studio), Gemma models included.

    Mirrors ``OllamaClient``'s ``generate`` / ``chat`` / ``list_models`` so the
    agents can swap backends through ``LLMClient()``. Requests are paced to
    ``settings.gemini_rpm`` and retried on HTTP 429 / 5xx with a short backoff.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        base_url: str = GEMINI_BASE_URL,
        rpm: int | None = None,
    ) -> None:
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.timeout = timeout or settings.ollama_timeout_s
        self.base_url = base_url
        self.min_interval_s = 60.0 / max(1, rpm or settings.gemini_rpm)
        self._client: httpx.AsyncClient | None = None
        self._last_request_at: float = 0.0

    async def __aenter__(self) -> GeminiClient:
        if not self.api_key:
            raise LLMError(
                "GEMINI_API_KEY is not set (LLM_PROVIDER=gemini needs a Google AI Studio key)"
            )
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Internals ─────────────────────────────────────────────────────────────

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("GeminiClient must be used inside `async with`")
        return self._client

    def resolve_model(self, requested: str) -> str:
        """Map an Ollama-style tag (``gemma4:e4b``) onto the configured Gemini model.

        A name without a colon is assumed to already be a Gemini model id and is
        passed through, so callers can address a specific Gemini model directly.
        """
        if ":" in requested or not requested:
            return self.model
        return requested

    async def _pace(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait = self.min_interval_s - elapsed
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request_at = time.monotonic()

    async def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        client = self._require_client()
        last_error: str = ""
        for attempt in range(_GEMINI_MAX_ATTEMPTS):
            await self._pace()
            try:
                response = await client.request(
                    method, path, params={"key": self.api_key}, json=payload
                )
            except httpx.HTTPError as exc:
                raise LLMError(f"Gemini request failed: {exc}") from exc
            if response.status_code < 300:
                try:
                    return response.json()
                except ValueError as exc:
                    raise LLMError("Gemini returned non-JSON body") from exc
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < _GEMINI_MAX_ATTEMPTS - 1:
                    await asyncio.sleep(_GEMINI_BACKOFF_S[min(attempt, len(_GEMINI_BACKOFF_S) - 1)])
                    continue
            break
        raise LLMError(f"Gemini error after {_GEMINI_MAX_ATTEMPTS} attempts — {last_error}")

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            feedback = data.get("promptFeedback", {})
            if feedback.get("blockReason"):
                raise LLMError(f"Gemini blocked the prompt: {feedback['blockReason']}")
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)

    # ── Public verbs (same shape as OllamaClient) ─────────────────────────────

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models, as ``{"name": ...}`` dicts.

        Besides the Gemini listing, the configured Ollama tags are reported as
        aliases of ``self.model`` when it is available, so the agents' "is my model
        present?" checks (written against Ollama tags) keep working.
        """
        data = await self._request("GET", "/models")
        models = [
            {"name": m.get("name", "").removeprefix("models/")} for m in data.get("models", [])
        ]
        if any(m["name"] == self.model for m in models):
            for alias in (settings.ollama_model_npc, settings.ollama_model_director):
                models.append({"name": alias, "alias_of": self.model})
        return models

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
        """Single-shot completion."""
        messages = [{"role": "user", "content": prompt}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        return await self.chat(
            model,
            messages,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
            stop=stop,
        )

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_predict: int = 512,
        think: bool | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Multi-turn chat. ``think`` is accepted for signature parity and ignored."""
        payload = build_gemini_payload(
            messages, temperature=temperature, top_p=top_p, num_predict=num_predict, stop=stop
        )
        resolved = self.resolve_model(model)
        data = await self._request("POST", f"/models/{resolved}:generateContent", payload)
        return self._extract_text(data)


def build_gemini_payload(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    top_p: float,
    num_predict: int,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    """Translate OpenAI/Ollama-style messages into a ``generateContent`` body.

    System messages become ``systemInstruction``; ``assistant`` maps to Gemini's
    ``model`` role. Pure function — unit-tested without the network.
    """
    system_parts: list[dict[str, str]] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        text = message.get("content", "")
        if role == "system":
            system_parts.append({"text": text})
            continue
        contents.append(
            {"role": "model" if role == "assistant" else "user", "parts": [{"text": text}]}
        )
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": num_predict,
        },
    }
    if stop:
        payload["generationConfig"]["stopSequences"] = stop
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    return payload


def LLMClient(**kwargs: Any):  # noqa: N802 — used like a class by the agents
    """Return the configured backend client (an async context manager).

    ``settings.llm_provider``: ``"ollama"`` (default) or ``"gemini"``.
    """
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        return GeminiClient(**kwargs)
    if provider == "ollama":
        from lib.ollama import OllamaClient

        return OllamaClient(**kwargs)
    raise LLMError(
        f"Unknown LLM_PROVIDER '{settings.llm_provider}' (expected 'ollama' or 'gemini')"
    )
