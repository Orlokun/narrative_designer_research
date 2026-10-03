"""
CyberSyn Research Engine — Admin Dashboard (port 8080).

Serves a single-page HTML dashboard that aggregates data from:
  - Gatekeeper API (port 8001)  → health, domain states, fetch log
  - archivo.sqlite               → document counts, coverage matrix
  - research_engine state        → mission queue, heatmap

Run:
    uv run uvicorn admin.app:app --port 8080 --reload
"""

from __future__ import annotations

import contextlib
import pathlib

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from lib.config import settings
from lib.logging_setup import configure_logging, get_logger
from admin.routers import (
    health,
    heatmap,
    database,
    pipeline,
    documents,
    agents,
    performance,
    cast,
    locations,
)

log = get_logger("admin")

_STATIC = pathlib.Path(__file__).parent / "static"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    configure_logging()
    log.info(
        "admin.started", port=settings.admin_port, ui=f"http://localhost:{settings.admin_port}"
    )
    yield
    log.info("admin.stopped")


app = FastAPI(
    title="CyberSyn Admin",
    description="Research Engine control panel",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
)

app.include_router(health.router, prefix="/api")
app.include_router(heatmap.router, prefix="/api")
app.include_router(database.router, prefix="/api")
app.include_router(pipeline.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(agents.router, prefix="/api")
app.include_router(performance.router, prefix="/api")
app.include_router(cast.router, prefix="/api")
app.include_router(locations.router, prefix="/api")

# Shared assets (navbar script, future css/js) — pages reference /static/...
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/propositor", include_in_schema=False)
async def propositor_page() -> FileResponse:
    return FileResponse(_STATIC / "propositor.html")


@app.get("/archivero", include_in_schema=False)
async def archivero_page() -> FileResponse:
    return FileResponse(_STATIC / "archivero.html")


@app.get("/verificator", include_in_schema=False)
async def verificator_page() -> FileResponse:
    return FileResponse(_STATIC / "verificator.html")


@app.get("/mapper", include_in_schema=False)
async def mapper_page() -> FileResponse:
    return FileResponse(_STATIC / "mapper.html")


@app.get("/cast-manager", include_in_schema=False)
async def cast_manager_page() -> FileResponse:
    return FileResponse(_STATIC / "cast-manager.html")


@app.get("/cast-director", include_in_schema=False)
async def cast_director_page() -> FileResponse:
    return FileResponse(_STATIC / "cast-director.html")


@app.get("/performance", include_in_schema=False)
async def performance_page() -> FileResponse:
    return FileResponse(_STATIC / "performance.html")


@app.get("/locations", include_in_schema=False)
async def locations_page() -> FileResponse:
    return FileResponse(_STATIC / "locations.html")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "admin.app:app",
        host=settings.admin_host,
        port=settings.admin_port,
        reload=False,
        log_config=None,
    )


if __name__ == "__main__":
    main()
