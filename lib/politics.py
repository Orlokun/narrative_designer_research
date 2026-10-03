"""
Political compass — a 2-axis position per character and its effect on relations.

Each character can be placed on a two-axis compass:

  - economic:  −1 = izquierda (left)      … +1 = derecha (right)
  - social:    −1 = libertario (liberal)  … +1 = autoritario (authoritarian)

The Cast Director uses ideological distance as a sanity check on extracted
relations: two figures who sit far apart on the compass are *not* políticamente
``afiliados`` — Kissinger and Allende are counterparts (``contraparte``), even
adversaries — even if a document merely co-mentions them. This module is pure
(no I/O) so both the Cast Manager (which stores positions) and the Cast Director
(which refines relations) can share it and it stays fully offline-testable.
"""

from __future__ import annotations

import json
import math

# Ideological distance beyond which a claimed "afiliacion" is downgraded to
# "contraparte". Max possible distance is sqrt(8) ≈ 2.83 (opposite corners).
FAR_THRESHOLD: float = 1.2

# Dead zone around the origin that reads as centre / moderate.
_DEAD_ZONE: float = 0.15


def clamp_axis(value: object) -> float | None:
    """Clamp a numeric axis value to [-1, 1]; None when not a real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(-1.0, min(1.0, float(value)))


def political_distance(
    econ_a: float | None, soc_a: float | None,
    econ_b: float | None, soc_b: float | None,
) -> float | None:
    """Euclidean distance between two compass positions, or None if any is unknown."""
    if None in (econ_a, soc_a, econ_b, soc_b):
        return None
    return math.hypot(econ_a - econ_b, soc_a - soc_b)


def compass_quadrant(econ: float | None, soc: float | None) -> str:
    """Human label for a position, e.g. "izquierda · libertario". "" if unknown."""
    if econ is None or soc is None:
        return ""
    if econ < -_DEAD_ZONE:
        lr = "izquierda"
    elif econ > _DEAD_ZONE:
        lr = "derecha"
    else:
        lr = "centro"
    if soc > _DEAD_ZONE:
        la = "autoritario"
    elif soc < -_DEAD_ZONE:
        la = "libertario"
    else:
        la = "moderado"
    return f"{lr} · {la}"


def refine_relation_kind(
    kind: str,
    econ_a: float | None, soc_a: float | None,
    econ_b: float | None, soc_b: float | None,
) -> str:
    """Second-guess a relation kind using the pair's ideological distance.

    Conservative by design: only a claimed ``afiliacion`` (shared political
    alignment) between two figures who sit far apart on the compass is corrected
    — to ``contraparte``. Evidence-based kinds (enemigo, colega, familia, …) are
    never changed, and an unknown position leaves the kind untouched.
    """
    distance = political_distance(econ_a, soc_a, econ_b, soc_b)
    if distance is None:
        return kind
    if kind == "afiliacion" and distance > FAR_THRESHOLD:
        return "contraparte"
    return kind


def parse_political_response(response: str) -> tuple[float, float, str] | None:
    """Parse an LLM compass placement into (economic, social, label).

    Expected shape: ``{"economic": -1..1, "social": -1..1, "label": "..."}``.
    Axes are clamped; a missing/invalid axis returns None (unusable placement).
    """
    try:
        data = json.loads(response.strip())
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    if not isinstance(data, dict):
        return None
    econ = clamp_axis(data.get("economic"))
    soc = clamp_axis(data.get("social"))
    if econ is None or soc is None:
        return None
    label = data.get("label")
    label = label.strip() if isinstance(label, str) else ""
    return econ, soc, label
