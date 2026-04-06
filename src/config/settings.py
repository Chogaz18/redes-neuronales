from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict

from src.utils.paths import project_root


def _default_classifier_path(root: Path) -> Path:
    """Prefiere `ecg_mlp_pipeline.joblib` si existe; si no, el bundle clásico."""
    candidates = [
        root / "artifacts" / "models" / "ecg_mlp_pipeline.joblib",
        root / "artifacts" / "models" / "ecg_mlp_4class.joblib",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return candidates[0].resolve()


def _resolve_project_path(raw: str, root: Path) -> Path:
    """
    Rutas en .env suelen ser relativas (p.ej. data/raw/WFDBRecords).
    Deben resolverse respecto a la raíz del repo, no al cwd de Jupyter/terminal.
    """
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def _load_dotenv(dotenv_path: Path) -> Dict[str, str]:
    """Carga variables KEY=VALUE desde un .env simple (sin dependencias externas)."""
    if not dotenv_path.exists():
        return {}

    values: Dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    dotenv_path = project_root() / ".env"
    dotenv_values = _load_dotenv(dotenv_path)

    def getenv(name: str, default: str) -> str:
        return os.getenv(name, dotenv_values.get(name, default))

    root = project_root()

    data_raw_wfdb_dir = _resolve_project_path(
        getenv("WFDB_RECORDS_DIR", str(root / "data" / "raw" / "WFDBRecords")),
        root,
    )
    sample_count = int(getenv("SAMPLE_RECORD_COUNT", "100"))
    default_window_s = float(getenv("DEFAULT_WINDOW_DURATION_S", "10.0"))
    min_window_s = float(getenv("MIN_WINDOW_DURATION_S", "1.0"))
    max_window_s = float(getenv("MAX_WINDOW_DURATION_S", "15.0"))

    # Rangos HR
    hr_min_bpm = float(getenv("HR_MIN_BPM", "60"))
    hr_max_bpm = float(getenv("HR_MAX_BPM", "100"))

    # Para detección de picos por default
    default_rpeak_lead = getenv("DEFAULT_RPEAK_LEAD", "II")

    classifier_model_path = _resolve_project_path(
        getenv("CLASSIFIER_MODEL_PATH", str(_default_classifier_path(root))),
        root,
    )

    return Settings(
        wfdb_records_dir=data_raw_wfdb_dir,
        sample_record_count=sample_count,
        default_window_duration_s=default_window_s,
        min_window_duration_s=min_window_s,
        max_window_duration_s=max_window_s,
        hr_min_bpm=hr_min_bpm,
        hr_max_bpm=hr_max_bpm,
        default_rpeak_lead=default_rpeak_lead,
        classifier_model_path=classifier_model_path,
        project_root=root,
    )


@dataclass(frozen=True)
class Settings:
    wfdb_records_dir: Path
    sample_record_count: int
    default_window_duration_s: float
    min_window_duration_s: float
    max_window_duration_s: float
    hr_min_bpm: float
    hr_max_bpm: float
    default_rpeak_lead: str
    classifier_model_path: Path
    project_root: Path

