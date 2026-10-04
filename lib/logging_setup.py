"""Structured logging for the agent system. Import `get_logger` everywhere."""

from __future__ import annotations

import atexit
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from lib.config import settings

_configured = False
_log_file: Path | None = None


def _make_file_tee(log_file: Path) -> Any:
    """
    Structlog processor that appends each event as a JSON line to a file.
    File is created lazily on first write so empty-crash files never appear.
    """
    resolved = log_file.resolve()
    state: dict[str, Any] = {}  # holds {"fh": file_handle} once opened

    def _tee(logger: Any, method: str, event_dict: dict) -> dict:
        try:
            if "fh" not in state:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                fh = resolved.open("a", encoding="utf-8")
                state["fh"] = fh
                atexit.register(fh.close)
            state["fh"].write(json.dumps(event_dict, default=str) + "\n")
            state["fh"].flush()
        except Exception:
            pass
        return event_dict

    return _tee


def configure_logging(level: str | None = None) -> None:
    """Call once at process start. Idempotent."""
    global _configured, _log_file
    if _configured:
        return

    lvl = getattr(logging, (level or settings.log_level).upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=lvl,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file = Path(settings.logs_dir).resolve() / f"session_{ts}.log"

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _make_file_tee(_log_file),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(lvl),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def current_log_file() -> Path | None:
    """Return the path of this session's log file, or None if not yet configured."""
    return _log_file


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
