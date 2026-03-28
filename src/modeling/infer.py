from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ClassificationPrediction:
    predicted_class: str
    probabilities: dict[str, float]


def load_model_and_infer(
    *,
    signal_mV: np.ndarray,
    fs: float,
    model_path: Optional[str] = None,
) -> ClassificationPrediction:
    """
    Inferencia desde la app (pendiente).
    """
    raise NotImplementedError("Pendiente: cargar el modelo entrenado y predecir clase/probabilidades.")

