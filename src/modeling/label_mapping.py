from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class RhythmClass:
    key: str
    snomed_ct_code: Optional[str]  # depende del mapeo desde el dataset


FOUR_CLASS_KEYS = [
    "Sinus Bradycardia",
    "Sinus Rhythm",
    "Atrial Fibrillation",
    "Sinus Tachycardia",
]


def map_snomed_codes_to_four_classes(snomed_codes: list[str]) -> Dict[str, str]:
    """
    Mapea SNOMED CT codes a las 4 clases objetivo (heurística / placeholder).

    Referencia de nombres/códigos: `src.utils.paths.condition_names_snomed_csv()`
    (`data/raw/ConditionNames_SNOMED-CT.csv`).
    """
    raise NotImplementedError("Pendiente: implementar el mapeo SNOMED->clases desde el CSV.")

