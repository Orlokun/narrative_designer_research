"""
RE-2 Chronicler — Módulo C: Mapa de Lugares e Infraestructura.

Georreferencia fábricas, cordones industriales, puertos y nodos CORFO.
Registra el estado técnico y político de cada lugar por mes (operativo,
en huelga, dañado, intervenido).

Produce `data/places.geojson` exportado a Unity StreamingAssets,
consumido por el Level Designer (Ag-6) para el estado inicial de los escenarios.

Inputs:  tabla `documents` procesada por el Archivero
Outputs: tabla `places` en archivo.sqlite + `data/places.geojson`
Stack:   spacy (NER), geopy + Nominatim para geocoding
"""

from __future__ import annotations

# TODO: implementar en Fase 1
