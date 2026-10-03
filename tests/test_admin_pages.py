"""Tests for the admin static pages — every screen carries the shared navbar.

Fully offline: httpx ASGITransport against the FastAPI app.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from admin.app import app

PAGES = [
    "/",
    "/propositor",
    "/archivero",
    "/verificator",
    "/mapper",
    "/cast-manager",
    "/cast-director",
    "/performance",
]


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


class TestSharedNavbar:
    @pytest.mark.parametrize("path", PAGES)
    async def test_page_includes_nav_placeholder_and_script(self, client, path):
        r = await client.get(path)
        assert r.status_code == 200
        assert 'id="main-nav"' in r.text
        assert "/static/nav.js" in r.text

    async def test_nav_script_served_and_lists_all_pages(self, client):
        r = await client.get("/static/nav.js")
        assert r.status_code == 200
        for path in PAGES:
            assert f"'{path}'" in r.text
