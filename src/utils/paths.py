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

