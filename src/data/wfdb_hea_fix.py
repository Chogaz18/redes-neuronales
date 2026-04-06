"""
Algunos registros del dataset Chapman/PhysioNet tienen la línea 0 del `.hea` con
fecha/hora inválidas (p. ej. token `/` en lugar de dd/mm/yyyy). La librería `wfdb`
falla al hacer strptime. Aquí se sanea una copia en memoria y se restaura el
archivo original tras la lectura (sin dejar el dataset modificado).
"""

from __future__ import annotations

import re
from contextlib import contextmanager

from wfdb.io import _header as wfdb__header
from wfdb.io._header import _parse_record_line
from wfdb.io.header import parse_header_content
from wfdb.io.record import MultiRecord, Record

from src.data.metadata import RecordRef


def rdheader_from_string(header_content: str, *, rd_segments: bool = False) -> Record | MultiRecord:
    """
    Equivalente a `wfdb.rdheader` con `pn_dir=None` pero leyendo el texto del `.hea`
    ya en memoria (sin depender de rutas en disco ni de fsspec).

    Evita desincronizar saneamiento ↔ lectura en Windows cuando el archivo temporal
    y la ruta pasada a wfdb no coinciden exactamente.
    """
    header_lines, comment_lines = parse_header_content(header_content)
    if not header_lines:
        raise ValueError("WFDB header vacío o sin líneas de registro/señal")
    record_fields = wfdb__header._parse_record_line(header_lines[0])
    if record_fields["n_seg"] is None:
        record = Record()
        if len(header_lines) > 1:
            signal_fields = wfdb__header._parse_signal_lines(header_lines[1:])
            for field in signal_fields:
                setattr(record, field, signal_fields[field])
        for field in record_fields:
            if field == "n_seg":
                continue
            setattr(record, field, record_fields[field])
    else:
        record = MultiRecord()
        segment_fields = wfdb__header._read_segment_lines(header_lines[1:])
        for field in segment_fields:
            setattr(record, field, segment_fields[field])
        for field in record_fields:
            setattr(record, field, record_fields[field])
        if record.seg_len[0] == 0:
            record.layout = "variable"
        else:
            record.layout = "fixed"
        if rd_segments:
            raise NotImplementedError(
                "rd_segments=True requiere rutas en disco; use wfdb.rdheader con pn_dir"
            )
    record.comments = [line.strip(" \t#") for line in comment_lines]
    return record


def _first_header_line_text(text: str) -> str:
    """Igual que `wfdb.io.record.rdheader`: primera línea no vacía que no sea `#...`."""
    hl, _ = parse_header_content(text)
    return hl[0] if hl else ""


def _index_first_record_line(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        st = line.strip()
        if st and not st.startswith("#"):
            return i
    return None


def _fallback_minimal_record_line(line: str) -> str | None:
    """
    Algunos `.hea` (p. ej. Chapman) tienen la línea 0 corrupta: el 4º campo lleva
    `500000/mV` (wfdb interpreta `/mV` como base_date `/`) o se pegó el inicio de
    la línea de señal (`16 0 15 ...`). Se reconstruye solo `nombre n_sig fs sig_len`.
    """
    parts = line.split()
    if len(parts) < 4:
        return None
    line_st = line.strip()
    has_mV = bool(re.search(r"/(?:mV|uV|mv|uv)(?:\b|$)", parts[3], re.I))
    p3 = re.sub(r"/(?:mV|uV|mv|uv)$", "", parts[3], flags=re.IGNORECASE)
    if "/" in p3 and not re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", p3):
        p3 = p3.split("/")[0]
    cand = f"{parts[0]} {parts[1]} {parts[2]} {p3}"
    if not _record_line_parses_in_wfdb(cand):
        m = re.match(r"^(\d+)", parts[3])
        if not m:
            return None
        cand = f"{parts[0]} {parts[1]} {parts[2]} {m.group(1)}"
        if not _record_line_parses_in_wfdb(cand):
            return None
    if len(parts) > 4 or has_mV or cand != line_st:
        return cand
    return None


def _record_line_parses_in_wfdb(line: str) -> bool:
    """
    Valida con el mismo parser que `wfdb.rdheader` (no basta con strptime del grupo
    base_date: un único token `dd/mm/yyyy` se parte mal en base_time/base_date).
    """
    try:
        _parse_record_line(line.rstrip("\r\n"))
        return True
    except Exception:
        return False


def _sanitize_first_record_line(line: str) -> str:
    """Reemplaza tokens de fecha/hora rotos que rompen `wfdb.io._header._parse_record_line`."""
    parts = line.split()
    if not parts:
        return line
    fixed: list[str] = []
    for p in parts:
        if p in ("/", "//", "///"):
            fixed.append("01/01/2000")
        elif p in (":", "::", ":::"):
            fixed.append("12:00:00")
        else:
            fixed.append(p)
    s = " ".join(fixed)
    # Casos donde `/` queda pegado o como único carácter entre espacios
    s = re.sub(r"(?<=\s)/(?=\s|$)", "01/01/2000", s)
    s = re.sub(r"^/(?=\s)", "01/01/2000", s)
    # '/' pegado a hora (p. ej. 12:00:00/): el split no separa el token '/'
    s = re.sub(r"(\d{1,2}:\d{1,2}(?::\d{1,2}(?:\.\d+)?)?)/+\s*$", r"\1", s)

    for _ in range(32):
        if _record_line_parses_in_wfdb(s):
            return s
        s2 = re.sub(r"/+\s*$", "", s)
        if s2 != s:
            s = s2
            continue
        parts = s.split()
        if len(parts) <= 4:
            p3 = re.sub(r"/(?:mV|uV|mv|uv)$", "", parts[3], flags=re.IGNORECASE)
            if p3 != parts[3]:
                s = f"{parts[0]} {parts[1]} {parts[2]} {p3}"
                continue
            break
        s = " ".join(parts[:-1])

    if _record_line_parses_in_wfdb(s):
        return s
    fb = _fallback_minimal_record_line(s)
    if fb is not None:
        return fb
    return s


def _sanitize_hea_text(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    idx = _index_first_record_line(lines)
    if idx is None:
        return text
    rec = _sanitize_first_record_line(lines[idx])
    if not _record_line_parses_in_wfdb(rec.strip()):
        fb = _fallback_minimal_record_line(rec.strip())
        if fb is not None:
            rec = fb
    lines[idx] = rec
    out = "\n".join(lines)
    if text.endswith("\n"):
        out += "\n"
    if not _record_line_parses_in_wfdb(_first_header_line_text(out)):
        fb = _fallback_minimal_record_line(_first_header_line_text(out))
        if fb is not None:
            lines[idx] = fb
            out = "\n".join(lines)
            if text.endswith("\n"):
                out += "\n"
    return out


@contextmanager
def wfdb_hea_safe_read(record_ref: RecordRef):
    """
    Si el `.hea` necesita saneamiento, escribe la versión corregida, ejecuta el bloque
    `with` y restaura el contenido original al salir.
    """
    path = record_ref.hea_path
    raw = path.read_text(encoding="ascii", errors="ignore")
    fixed = _sanitize_hea_text(raw)
    fixed_line = _first_header_line_text(fixed)
    fixed_ok = _record_line_parses_in_wfdb(fixed_line) if fixed_line else False
    if not fixed_ok:
        raise ValueError(
            f"No se pudo corregir la línea de registro WFDB en {path}: {fixed_line!r}"
        )
    raw_line = _first_header_line_text(raw)
    raw_ok = _record_line_parses_in_wfdb(raw_line) if raw_line else False
    if fixed == raw and raw_ok:
        yield
        return
    path.write_text(fixed, encoding="ascii")
    try:
        yield
    finally:
        path.write_text(raw, encoding="ascii")
