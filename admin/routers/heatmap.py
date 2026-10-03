"""Coverage matrix endpoints — theme×month and theme×genre projections of the
theme × genre × month cube, read from archivo.sqlite (zeros when the DB is absent)."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Query

from lib.config import settings
from lib.genres import GENRES, genre_by_id
from lib.genres import plausibility as genre_plausibility
from lib.project import project

router = APIRouter(tags=["heatmap"])

_PROJECT = project()

# Themes in canonical (id) order — index i ↔ category_id i+1.
CATEGORIES: list[str] = [t.name for t in sorted(_PROJECT.themes, key=lambda t: t.id)]

# Relevance weight per category for the Propositor's priority formula
CATEGORY_WEIGHTS: list[float] = _PROJECT.theme_weights()

# Key events — highlighted on the heatmap X-axis
CRITICAL_MONTHS: list[str] = [cm.month_iso for cm in _PROJECT.critical_months]
CRITICAL_LABELS: dict[str, str] = _PROJECT.critical_month_labels()

MONTHS: list[str] = _PROJECT.months()


# Verified complete documents needed for a cell to reach 100% coverage.
# With ~100 docs/cell as target for deep research, 20 complete docs = 20% — gives
# useful visual feedback while keeping overall % realistic for large corpora.
_TARGET_COMPLETE_DOCS: int = 20

# Incomplete docs (short abstracts/snippets) count fractionally toward coverage.
_INCOMPLETE_WEIGHT: float = 0.2


def _read_coverage() -> dict[tuple[int, str], float]:
    """
    Calculate per-cell coverage from VERIFIED documents only.

    Formula: min(1.0, (complete_count + incomplete_count × 0.2) / 20)

    This bypasses the cached `coverage` table (which is based on raw fetch counts)
    and uses actual verification state. A cell with only short snippets will show
    much lower coverage than one with complete primary sources.
    """
    db_path = settings.archive_db
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("""
            SELECT COALESCE(d.mapped_category_id, m.category_id) AS cat_id,
                   COALESCE(d.mapped_month_iso,   m.month_iso)   AS mon,
                   SUM(CASE WHEN d.is_complete IS NULL OR d.is_complete != 0 THEN 1 ELSE 0 END) AS complete_n,
                   SUM(CASE WHEN d.is_complete = 0 THEN 1 ELSE 0 END) AS incomplete_n
            FROM documents d
            LEFT JOIN missions m
              ON m.mission_id = json_extract(d.provenance, '$.mission_id')
            WHERE d.verified_at IS NOT NULL
              AND (d.doc_tag IS NULL OR d.doc_tag != 'context')
              AND COALESCE(d.mapped_category_id, m.category_id) IS NOT NULL
              AND COALESCE(d.mapped_month_iso,   m.month_iso)   IS NOT NULL
            GROUP BY cat_id, mon
        """).fetchall()
        conn.close()
        result = {}
        for r in rows:
            effective = r[2] + r[3] * _INCOMPLETE_WEIGHT
            score = min(1.0, round(effective / _TARGET_COMPLETE_DOCS, 4))
            result[(int(r[0]), str(r[1]))] = score
        return result
    except Exception:
        return {}


def _read_genre_counts() -> dict[tuple[int, int], tuple[int, int]] | None:
    """Per (category_id, genre_id): (complete_count, incomplete_count) from
    verified, in-scope documents. NULL genres count under genre 0. None when the
    database does not exist or cannot be read."""
    db_path = settings.archive_db
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("""
            SELECT mapped_category_id,
                   COALESCE(mapped_genre_id, 0) AS genre_id,
                   SUM(CASE WHEN is_complete IS NULL OR is_complete != 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN is_complete = 0 THEN 1 ELSE 0 END)
            FROM documents
            WHERE verified_at IS NOT NULL
              AND (doc_tag IS NULL OR doc_tag != 'context')
              AND mapped_category_id IS NOT NULL
            GROUP BY mapped_category_id, genre_id
        """).fetchall()
        conn.close()
        return {(int(r[0]), int(r[1])): (int(r[2]), int(r[3])) for r in rows}
    except Exception:
        return None


@router.get("/heatmap/genres")
async def heatmap_genres() -> dict:
    """Theme × genre projection of the matrix (aggregated over all 48 months).

    Includes the plausibility prior per cell (for shading structurally-empty
    pairings) and a plausibility-weighted coverage summary. Documents without a
    genre are reported per category in ``unclassified``.
    """
    counts = _read_genre_counts()

    n_categories = len(CATEGORIES)
    cells:        list[list[float]] = []
    doc_counts:   list[list[int]] = []
    plausibility: list[list[float]] = []
    unclassified: list[int] = []

    for row in range(n_categories):
        category_id = row + 1
        score_row, count_row, plaus_row = [], [], []
        for genre in GENRES:
            complete, incomplete = (counts or {}).get((category_id, genre.id), (0, 0))
            effective = complete + incomplete * _INCOMPLETE_WEIGHT
            score_row.append(min(1.0, round(effective / _TARGET_COMPLETE_DOCS, 4)))
            count_row.append(complete + incomplete)
            plaus_row.append(genre_plausibility(category_id, genre.id))
        cells.append(score_row)
        doc_counts.append(count_row)
        plausibility.append(plaus_row)
        unclass_complete, unclass_incomplete = (counts or {}).get((category_id, 0), (0, 0))
        unclassified.append(unclass_complete + unclass_incomplete)

    filled = sum(1 for row in doc_counts for n in row if n > 0)
    total = n_categories * len(GENRES)
    # Weighted by plausibility: implausible cells barely count toward the goal.
    plaus_sum = sum(p for row in plausibility for p in row)
    weighted = sum(
        cells[i][j] * plausibility[i][j]
        for i in range(n_categories) for j in range(len(GENRES))
    )

    return {
        "available": counts is not None,
        "categories": CATEGORIES,
        "genres": [{"id": genre.id, "name": genre.name} for genre in GENRES],
        "cells": cells,
        "doc_counts": doc_counts,
        "plausibility": plausibility,
        "unclassified": unclassified,
        "summary": {
            "total_cells": total,
            "filled_cells": filled,
            "coverage_pct": round(weighted / plaus_sum * 100, 2) if plaus_sum else 0.0,
        },
    }


def _read_month_counts(category_id: int, genre_id: int) -> dict[str, tuple[int, int]] | None:
    """Per month_iso: (complete, incomplete) for one (theme, genre) cell.

    genre_id 0 selects the unclassified bucket (NULL mapped_genre_id). None when
    the database does not exist or cannot be read.
    """
    db_path = settings.archive_db
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("""
            SELECT mapped_month_iso,
                   SUM(CASE WHEN is_complete IS NULL OR is_complete != 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN is_complete = 0 THEN 1 ELSE 0 END)
            FROM documents
            WHERE verified_at IS NOT NULL
              AND (doc_tag IS NULL OR doc_tag != 'context')
              AND mapped_category_id = ?
              AND COALESCE(mapped_genre_id, 0) = ?
              AND mapped_month_iso IS NOT NULL
            GROUP BY mapped_month_iso
        """, (category_id, genre_id)).fetchall()
        conn.close()
        return {str(r[0]): (int(r[1]), int(r[2])) for r in rows}
    except Exception:
        return None


@router.get("/heatmap/genres/months")
async def heatmap_genre_months(
    category_id: int = Query(ge=1, le=16),
    genre_id: int = Query(ge=0, le=13),
) -> dict:
    """The 48-month strip for one (theme, genre) cell — the drill-down shown when
    a square of the genre matrix is clicked. genre_id 0 = unclassified bucket."""
    counts = _read_month_counts(category_id, genre_id)

    cells, doc_counts = [], []
    for month in MONTHS:
        complete, incomplete = (counts or {}).get(month, (0, 0))
        effective = complete + incomplete * _INCOMPLETE_WEIGHT
        cells.append(min(1.0, round(effective / _TARGET_COMPLETE_DOCS, 4)))
        doc_counts.append(complete + incomplete)

    genre = genre_by_id(genre_id)
    return {
        "available": counts is not None,
        "category": CATEGORIES[category_id - 1],
        "genre": genre.name if genre else "Sin clasificar",
        "months": MONTHS,
        "cells": cells,
        "doc_counts": doc_counts,
        "critical_months": CRITICAL_MONTHS,
        "critical_labels": CRITICAL_LABELS,
    }


@router.get("/heatmap")
async def heatmap() -> dict:
    """Return the 16×48 theme×month coverage projection plus metadata."""
    coverage = _read_coverage()

    cells: list[list[float]] = []
    for i in range(len(CATEGORIES)):
        row = [round(coverage.get((i + 1, m), 0.0), 4) for m in MONTHS]
        cells.append(row)

    # Cells with any verified docs (used for "N cells touched" label)
    filled = sum(1 for row in cells for s in row if s > 0.0)
    total  = len(CATEGORIES) * len(MONTHS)
    # Weighted coverage: sum of all scores / total cells × 100.
    # With TARGET=20, a corpus of 58 docs gives ~0.1-0.5% — realistic for early stage.
    total_score = sum(s for row in cells for s in row)

    return {
        "categories": CATEGORIES,
        "category_weights": CATEGORY_WEIGHTS,
        "months": MONTHS,
        "cells": cells,
        "critical_months": CRITICAL_MONTHS,
        "critical_labels": CRITICAL_LABELS,
        "summary": {
            "total_cells":   total,
            "filled_cells":  filled,
            # Weighted average of all cell scores gives a realistic research progress %.
            # e.g. 58 mostly-incomplete docs across 768 cells ≈ 0.1–0.5%
            "coverage_pct":  round(total_score / total * 100, 2),
        },
    }
