from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from src.config.settings import Settings


@dataclass(frozen=True)
class RecordRef:
    record_id: str
    hea_path: Path

    @property
    def record_dir(self) -> Path:
        return self.hea_path.parent


def _discover_hea_files(wfdb_records_dir: Path) -> Iterable[Path]:
    if not wfdb_records_dir.exists():
        return []
    return wfdb_records_dir.rglob("*.hea")


def discover_record_refs(
    settings: Settings,
    *,
    limit: Optional[int] = None,
) -> List[RecordRef]:
    """
    Recorre `WFDBRecords/` y devuelve references por `.hea`.
    """
    heas = list(_discover_hea_files(settings.wfdb_records_dir))
    heas_sorted = sorted(heas, key=lambda p: str(p))

    refs: List[RecordRef] = []
    seen = set()
    for hea_path in heas_sorted:
        record_id = hea_path.stem
        if record_id in seen:
            continue
        refs.append(RecordRef(record_id=record_id, hea_path=hea_path))
        seen.add(record_id)
        if limit is not None and len(refs) >= limit:
            break

    return refs


def safe_list(records: Sequence[RecordRef]) -> List[RecordRef]:
    return list(records)

