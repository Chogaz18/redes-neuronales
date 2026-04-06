from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from src.data.metadata import RecordRef
from src.data.wfdb_hea_fix import _sanitize_hea_text, rdheader_from_string


_SNOMED_RE = re.compile(r"\b\d{9}\b")


def read_wfdb_header(record_ref: RecordRef):
    """
    Lee el header WFDB asociado al registro (`.hea` + `sig_name`, `fs`, etc.).
    Aplica saneamiento temporal si la fecha/hora de la línea 0 es inválida (dataset Chapman).

    El parse se hace en memoria (`rdheader_from_string`) para no depender de escribir
    el `.hea` ni de que wfdb/fsspec abran exactamente la misma ruta que Path.
    """
    raw = record_ref.hea_path.read_text(encoding="ascii", errors="ignore")
    fixed = _sanitize_hea_text(raw)
    return rdheader_from_string(fixed)


def extract_snomed_ct_codes_from_hea(hea_path: Path) -> List[str]:
    """
    Extrae códigos SNOMED CT (heurística: 9 dígitos) desde el `.hea`.
    """
    text = hea_path.read_text(encoding="utf-8", errors="ignore")
    return sorted(set(_SNOMED_RE.findall(text)))


def header_to_metadata(record_ref: RecordRef) -> dict:
    header = read_wfdb_header(record_ref)
    snomed_codes = extract_snomed_ct_codes_from_hea(record_ref.hea_path)
    return {
        "record_id": record_ref.record_id,
        "fs": float(header.fs),
        "lead_names": list(header.sig_name),
        "units": list(header.units) if getattr(header, "units", None) is not None else None,
        "snomed_ct_codes": snomed_codes,
    }


def choose_rpeak_lead(lead_names: List[str], preferred: str) -> str:
    """
    Selecciona el lead para detección de picos R.
    Fallback a primer lead si no existe el preferido.
    """
    for name in lead_names:
        if name.strip().upper() == preferred.strip().upper():
            return name
    # Fallback: algunos datasets usan nombres tipo 'II' o 'Lead II'
    preferred_upper = preferred.strip().upper()
    for name in lead_names:
        if preferred_upper in name.strip().upper():
            return name
    return lead_names[0] if lead_names else preferred

