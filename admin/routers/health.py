"""Proxy endpoints to the Gatekeeper (port 8001) and summary of all RE services."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Query

from lib.config import settings

router = APIRouter(tags=["health"])

_GK_BASE = f"http://{settings.gatekeeper_host}:{settings.gatekeeper_port}"
_TIMEOUT = 3.0


async def _gk_get(path: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(f"{_GK_BASE}{path}")
            return r.json()
    except Exception as exc:
        return {"error": str(exc), "available": False}


@router.get("/health/services")
async def all_services() -> dict:
    """Live status of every Research Engine service."""
    gk = await _gk_get("/health")
    return {
        "gatekeeper": {
            "status": "ok" if gk.get("status") == "ok" else "unavailable",
            "port": settings.gatekeeper_port,
            "detail": gk,
        },
        "propositor": {"status": "not_implemented", "port": None},
        "chronicler":  {"status": "not_implemented", "port": None},
    }


@router.get("/health/gatekeeper/status")
async def gatekeeper_status(domain: str | None = Query(default=None)) -> dict:
    path = "/status" + (f"?domain={domain}" if domain else "")
    return await _gk_get(path)


@router.get("/health/gatekeeper/log")
async def gatekeeper_log(n: int = Query(default=20, ge=1, le=200)) -> dict:
    return await _gk_get(f"/log?n={n}")
