from __future__ import annotations

from pathlib import Path

from src.data.metadata import RecordRef


def wfdb_local_record_name(record_ref: RecordRef) -> str:
    """
    Ruta local completa al registro WFDB (carpeta + nombre base sin extensión).

    `wfdb.rdheader` / `wfdb.rdrecord` con **pn_dir distinto de None** asumen descarga
    remota desde PhysioNet. Para archivos locales hay que usar **pn_dir=None** y
    pasar aquí el path completo en `record_name` (documentación oficial de wfdb).
    """
    return str((record_ref.record_dir / record_ref.record_id).expanduser().resolve())
