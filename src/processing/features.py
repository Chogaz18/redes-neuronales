from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ECGFeatures:
    features: Dict[str, float]


def extract_basic_features(signal: np.ndarray, fs: float) -> ECGFeatures:
    """
    Placeholder para features adicionales (p.ej. para clasificación).
    """
    raise NotImplementedError

