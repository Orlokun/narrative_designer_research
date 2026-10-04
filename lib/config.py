"""
Central configuration. Reads from environment variables (and .env if present).
Import `settings` from here anywhere you need config — don't read env vars ad-hoc.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM provider: "ollama" (local server) or "gemini" (Google AI Studio API) ----
    llm_provider: str = "ollama"

    # ---- Gemini / Google AI Studio (serves Gemma models on a free tier) ----
    gemini_api_key: str | None = None
    gemini_model: str = "gemma-3-27b-it"
    gemini_rpm: int = 25  # stay under the free-tier 30 requests/minute

    # ---- Ollama ----
    ollama_host: str = "http://localhost:11434"
    ollama_model_npc: str = "gemma4:e4b"
    ollama_model_director: str = "gemma4:26b"
    ollama_model_embed: str = "bge-m3"
    ollama_model_npc_adapter: str | None = None
    ollama_timeout_s: float = 120.0

    # ---- Research project (which historical moment the pipeline researches) ----
    # A slug under projects/<slug>/project.yaml or a path to a YAML spec.
    research_project: str = "cybersyn"

    # ---- Paths (resolved against the `agents/` package root) ----
    data_dir: Path = Path("./data")
    archive_db: Path = Path("./data/archivo/archivo.sqlite")
    chroma_dir: Path = Path("./data/chroma")
    datasets_dir: Path = Path("./data/datasets")
    logs_dir: Path = Path("./data/logs")
    checkpoints_dir: Path = Path("./data/checkpoints")

    # ---- Source API keys (optional; source skipped/unauthenticated when unset) ----
    semantic_scholar_api_key: str | None = None

    # ---- Runtime ----
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"

    # ---- Admin Dashboard (port 8080) ----
    admin_host: str = "127.0.0.1"
    admin_port: int = 8080

    # ---- Gatekeeper (Research Engine, port 8001, offline only) ----
    gatekeeper_host: str = "127.0.0.1"
    gatekeeper_port: int = 8001
    gatekeeper_cache_dir: Path = Path("./data/gatekeeper_cache")
    gatekeeper_cache_ttl_days: int = 7
    gatekeeper_circuit_breaker_threshold: int = 3
    gatekeeper_cooldown_minutes: float = 15.0
    gatekeeper_rate_limits_path: Path = Path("./research_engine/gatekeeper/rate_limits.yaml")
    gatekeeper_fetch_log: Path = Path("./data/logs/fetch_log.jsonl")

    # ---- Experiment ----
    experiment_condition: str = Field(
        default="baseline",
        description="'baseline' (Gemma 4 E4B) or 'floresian' (E4B + adapter)",
    )

    def active_npc_model(self) -> str:
        """Return the model tag NPCs should use given the experiment condition."""
        if self.experiment_condition == "floresian" and self.ollama_model_npc_adapter:
            return self.ollama_model_npc_adapter
        return self.ollama_model_npc

    def ensure_dirs(self) -> None:
        """Create all data directories if missing."""
        for p in (
            self.data_dir,
            self.archive_db.parent,
            self.chroma_dir,
            self.datasets_dir,
            self.logs_dir,
            self.checkpoints_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
