from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class HeartRateResult:
    bpm_mean: float
    rr_mean_s: float
    n_rpeaks: int
    duration_s: float
    rpeak_indices: np.ndarray
    cleaned_signal: np.ndarray


HeartRateAlert = Literal["bradycardia", "normal", "taquicardia"]


def estimate_heart_rate(
    signal: np.ndarray,
    fs: float,
    *,
    min_bpm: float,
    max_bpm: float,
) -> HeartRateResult:
    """
    Estima frecuencia cardiaca usando NeuroKit2:
    - limpieza de señal si aplica
    - detección de picos R
    - RR/bpm
    """
    raise NotImplementedError(
        "Pendiente: implementar limpieza, detección de picos R y cálculo de bpm con NeuroKit2."
    )


def heart_rate_alert_class(bpm_mean: float, *, min_bpm: float, max_bpm: float) -> HeartRateAlert:
    if bpm_mean < min_bpm:
        return "bradycardia"
    if bpm_mean > max_bpm:
        return "taquicardia"
    return "normal"

