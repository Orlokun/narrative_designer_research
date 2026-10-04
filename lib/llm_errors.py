"""Shared exception type for every LLM backend (see lib/llm.py)."""


class LLMError(RuntimeError):
    """Raised by any LLM backend on a non-2xx, malformed or exhausted response."""
