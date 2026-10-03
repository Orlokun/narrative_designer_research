"""Tests for lib/llm.py — the provider factory and the Gemini (Google AI Studio) backend."""

from __future__ import annotations

import httpx
import pytest
import respx

import lib.llm as llm
from lib.llm import GEMINI_BASE_URL, GeminiClient, LLMClient, LLMError, build_gemini_payload
from lib.ollama import OllamaClient, OllamaError

# ── Payload translation (pure) ────────────────────────────────────────────────


def test_build_payload_maps_roles_and_system_instruction():
    payload = build_gemini_payload(
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "Hi"},
        ],
        temperature=0.3,
        top_p=0.9,
        num_predict=64,
        stop=["\n\n"],
    )
    assert payload["systemInstruction"] == {"parts": [{"text": "Be terse."}]}
    assert [c["role"] for c in payload["contents"]] == ["user", "model"]
    assert payload["generationConfig"] == {
        "temperature": 0.3,
        "topP": 0.9,
        "maxOutputTokens": 64,
        "stopSequences": ["\n\n"],
    }


def test_build_payload_without_system_has_no_instruction():
    payload = build_gemini_payload(
        [{"role": "user", "content": "x"}], temperature=0.5, top_p=0.9, num_predict=8
    )
    assert "systemInstruction" not in payload


# ── Model resolution ──────────────────────────────────────────────────────────


def test_resolve_model_maps_ollama_tags_to_configured_gemini_model():
    client = GeminiClient(api_key="k", model="gemma-3-27b-it", rpm=6000)
    assert client.resolve_model("gemma4:e4b") == "gemma-3-27b-it"
    assert client.resolve_model("") == "gemma-3-27b-it"
    assert client.resolve_model("gemini-2.5-flash") == "gemini-2.5-flash"


# ── Network behaviour (respx) ─────────────────────────────────────────────────


def _reply(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


@pytest.mark.asyncio
@respx.mock
async def test_chat_posts_generate_content_and_returns_text():
    route = respx.post(f"{GEMINI_BASE_URL}/models/gemma-3-27b-it:generateContent").mock(
        return_value=httpx.Response(200, json=_reply("0.85"))
    )
    async with GeminiClient(api_key="k", model="gemma-3-27b-it", rpm=6000) as client:
        out = await client.chat(
            "gemma4:e4b", [{"role": "user", "content": "rate"}], num_predict=8, think=False
        )
    assert out == "0.85"
    request = route.calls.last.request
    assert request.url.params["key"] == "k"
    assert b'"maxOutputTokens": 8' in request.content or b'"maxOutputTokens":8' in request.content


@pytest.mark.asyncio
@respx.mock
async def test_generate_wraps_system_and_prompt():
    route = respx.post(f"{GEMINI_BASE_URL}/models/gemma-3-27b-it:generateContent").mock(
        return_value=httpx.Response(200, json=_reply("[]"))
    )
    async with GeminiClient(api_key="k", model="gemma-3-27b-it", rpm=6000) as client:
        await client.generate("gemma4:e4b", "p", system="s")
    body = route.calls.last.request.content
    assert b"systemInstruction" in body


@pytest.mark.asyncio
@respx.mock
async def test_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm, "_GEMINI_BACKOFF_S", (0.0, 0.0, 0.0))
    route = respx.post(f"{GEMINI_BASE_URL}/models/gemma-3-27b-it:generateContent").mock(
        side_effect=[
            httpx.Response(429, json={"error": "quota"}),
            httpx.Response(200, json=_reply("ok")),
        ]
    )
    async with GeminiClient(api_key="k", model="gemma-3-27b-it", rpm=6000) as client:
        assert await client.chat("gemma4:e4b", [{"role": "user", "content": "x"}]) == "ok"
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(llm, "_GEMINI_BACKOFF_S", (0.0, 0.0, 0.0))
    respx.post(f"{GEMINI_BASE_URL}/models/gemma-3-27b-it:generateContent").mock(
        return_value=httpx.Response(503, text="down")
    )
    async with GeminiClient(api_key="k", model="gemma-3-27b-it", rpm=6000) as client:
        with pytest.raises(LLMError, match="HTTP 503"):
            await client.chat("gemma4:e4b", [{"role": "user", "content": "x"}])


@pytest.mark.asyncio
@respx.mock
async def test_client_error_is_not_retried():
    route = respx.post(f"{GEMINI_BASE_URL}/models/gemma-3-27b-it:generateContent").mock(
        return_value=httpx.Response(400, text="bad request")
    )
    async with GeminiClient(api_key="k", model="gemma-3-27b-it", rpm=6000) as client:
        with pytest.raises(LLMError):
            await client.chat("gemma4:e4b", [{"role": "user", "content": "x"}])
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_blocked_prompt_raises():
    respx.post(f"{GEMINI_BASE_URL}/models/gemma-3-27b-it:generateContent").mock(
        return_value=httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    )
    async with GeminiClient(api_key="k", model="gemma-3-27b-it", rpm=6000) as client:
        with pytest.raises(LLMError, match="SAFETY"):
            await client.chat("gemma4:e4b", [{"role": "user", "content": "x"}])


@pytest.mark.asyncio
@respx.mock
async def test_list_models_reports_ollama_tags_as_aliases(monkeypatch):
    monkeypatch.setattr(llm.settings, "ollama_model_npc", "gemma4:e4b")
    respx.get(f"{GEMINI_BASE_URL}/models").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "models/gemma-3-27b-it"}]})
    )
    async with GeminiClient(api_key="k", model="gemma-3-27b-it", rpm=6000) as client:
        models = await client.list_models()
    names = [m["name"] for m in models]
    assert "gemma-3-27b-it" in names
    assert "gemma4:e4b" in names  # the agents' presence check passes


@pytest.mark.asyncio
async def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.setattr(llm.settings, "gemini_api_key", None)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        async with GeminiClient(model="gemma-3-27b-it"):
            pass


# ── Factory ───────────────────────────────────────────────────────────────────


def test_factory_defaults_to_ollama(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "ollama")
    assert isinstance(LLMClient(), OllamaClient)


def test_factory_selects_gemini(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "gemini")
    assert isinstance(LLMClient(), GeminiClient)


def test_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "skynet")
    with pytest.raises(LLMError):
        LLMClient()


def test_ollama_error_is_an_llm_error():
    assert issubclass(OllamaError, LLMError)
