"""
RE-2 Chronicler — Módulo A: Actos de Habla Históricos.

Clasifica pasajes de documentos históricos (discursos, actas, editoriales)
en la tríada floresiana: Pedido | Promesa | Quiebre.

Produce el campo `speech_act_primary` en la tabla `documents` de archivo.sqlite,
que el Curador Floresiano (Ag-4) usa como señal de priorización para el dataset QLoRA.

Inputs:  tabla `documents` (doc_id, text, lang, date_iso, source_kind)
Outputs: campo `speech_act_primary` + `speech_act_confidence` en `documents`
Model:   Gemma 4 E4B vía Ollama (zero-shot classification)
"""

from __future__ import annotations

# TODO: implementar en Fase 1
