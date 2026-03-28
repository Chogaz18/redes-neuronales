from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

from src.config.settings import Settings
from src.data.exceptions import DatasetNotFoundError
from src.utils.logger import get_logger
from src.utils.paths import raw_dir

logger = get_logger(__name__)


@dataclass(frozen=True)
class DownloadPlan:
    url: str
    zip_path: Path
    extract_to: Path


def _has_any_hea(heas_root: Path) -> bool:
    if not heas_root.exists():
        return False
    for p in heas_root.rglob("*.hea"):
        return True
    return False


def get_default_download_plan(settings: Settings) -> DownloadPlan:
    # PhysioNet ofrece un ZIP con el dataset completo.
    url = "https://physionet.org/content/ecg-arrhythmia/get-zip/1.0.0/"
    zip_path = raw_dir() / "ecg-arrhythmia-1.0.0.zip"
    extract_to = raw_dir()
    return DownloadPlan(url=url, zip_path=zip_path, extract_to=extract_to)


def ensure_dataset(
    settings: Settings,
    *,
    download_if_missing: bool = False,
) -> Path:
    """
    Valida que exista el directorio WFDB y opcionalmente descarga el ZIP completo.
    """
    wfdb_records_dir = settings.wfdb_records_dir
    if _has_any_hea(wfdb_records_dir):
        return wfdb_records_dir

    if not download_if_missing:
        raise DatasetNotFoundError(
            f"No se encontró dataset en: {wfdb_records_dir}. "
            "Descarga el dataset (ZIP) de PhysioNet y asegúrate de que exista la estructura WFDBRecords/."
        )

    plan = get_default_download_plan(settings)
    raw_dir().mkdir(parents=True, exist_ok=True)

    logger.info("Descargando dataset ZIP (puede tardar)...")
    if plan.zip_path.exists():
        plan.zip_path.unlink()
    urlretrieve(plan.url, str(plan.zip_path))

    logger.info("Extrayendo ZIP...")
    with zipfile.ZipFile(plan.zip_path, "r") as zf:
        zf.extractall(plan.extract_to)

    # Algunos ZIP pueden incluir una carpeta extra; hacemos un fallback:
    # Si aparece WFDBRecords en otro nivel, buscamos.
    if _has_any_hea(settings.wfdb_records_dir):
        return settings.wfdb_records_dir

    # Busca bajo raw_dir y ajusta a la carpeta candidata.
    for candidate in raw_dir().rglob("WFDBRecords"):
        if _has_any_hea(candidate):
            # No mutamos settings (frozen en dataclass). Se devuelve candidate.
            return candidate

    raise DatasetNotFoundError("Descarga/extracción completada, pero no se encontró WFDBRecords con .hea.")

