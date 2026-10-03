"""
RE-2 Chronicler — Módulo B: Perfiles de Personajes.

Extrae actores históricos mediante NER + correferencia y construye fichas
dinámicas acumulativas con afiliaciones, red de compromisos y estilo de habla.

Produce `data/characters.jsonl`, consumido directamente por el NPC Pool (Ag-10)
para inicializar las persona cards de los personajes históricos.

Inputs:  tabla `documents` procesada por el Archivero
Outputs: tabla `characters` en archivo.sqlite + `data/characters.jsonl`
Stack:   spacy (es_core_news_lg), Gemma 4 E4B para resolución de ambigüedad
"""

from __future__ import annotations

# TODO: implementar en Fase 1
