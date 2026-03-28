from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    # src/utils/paths.py -> src/utils -> src -> project_root
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return project_root() / "data"


def raw_dir() -> Path:
    return data_dir() / "raw"


def interim_dir() -> Path:
    return data_dir() / "interim"


def processed_dir() -> Path:
    return data_dir() / "processed"


def artifacts_dir() -> Path:
    return project_root() / "artifacts"


def models_dir() -> Path:
    return artifacts_dir() / "models"


def sample_records_dir() -> Path:
    return artifacts_dir() / "sample_records"


def physionet_dataset_root() -> Path:
    """Raíz local del extracto PhysioNet (misma carpeta que `WFDBRecords/`)."""
    return raw_dir()


def condition_names_snomed_csv() -> Path:
    """Mapeo acrónimo / nombre / SNOMED CT del dataset."""
    return raw_dir() / "ConditionNames_SNOMED-CT.csv"


def physionet_records_file() -> Path:
    """Lista de rutas relativas de registros (archivo `RECORDS`)."""
    return raw_dir() / "RECORDS"


def dataset_license_txt() -> Path:
    """Licencia del dataset en PhysioNet (`LICENSE.txt`)."""
    return raw_dir() / "LICENSE.txt"


def sha256sums_txt() -> Path:
    """Checksums de integridad (`SHA256SUMS.txt`)."""
    return raw_dir() / "SHA256SUMS.txt"

