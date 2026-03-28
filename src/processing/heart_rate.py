from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import neurokit2 as nk

from src.processing.ecg_units import to_millivolts
from src.utils.helpers import clamp


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
    Limpieza + picos R + RR/bpm con NeuroKit2 (`ecg_process`).

    Lead II suele ser preferible para QRS; la elección del lead ocurre antes (loader).
    `min_bpm` / `max_bpm` se reservan para la capa de alertas (`heart_rate_alert_class`).
    """
    _ = (min_bpm, max_bpm)
    sig_mv = to_millivolts(signal)
    if sig_mv.size < 8:
        raise ValueError("Segmento demasiado corto para estimar frecuencia cardiaca.")

    signals, info = nk.ecg_process(sig_mv, sampling_rate=float(fs))
    cleaned = signals["ECG_Clean"].to_numpy(dtype=float)
    if "ECG_R_Peaks" in signals.columns:
        r_mask = signals["ECG_R_Peaks"].to_numpy()
        rpeaks = np.flatnonzero(np.asarray(r_mask) > 0).astype(np.int64)
    else:
        rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64).ravel()

    duration_s = float(sig_mv.size / float(fs))

    if rpeaks.size >= 2:
        rr_s = np.diff(rpeaks.astype(np.float64)) / float(fs)
        rr_mean_s = float(np.clip(np.mean(rr_s), 1e-6, None))
        bpm_mean = float(60.0 / rr_mean_s)
    elif rpeaks.size == 1:
        rr_mean_s = float("nan")
        bpm_mean = float("nan")
    else:
        rr_mean_s = float("nan")
        bpm_mean = float("nan")

    bpm_mean = float(clamp(bpm_mean, 20.0, 300.0)) if np.isfinite(bpm_mean) else bpm_mean

    return HeartRateResult(
        bpm_mean=bpm_mean,
        rr_mean_s=rr_mean_s if np.isfinite(rr_mean_s) else float("nan"),
        n_rpeaks=int(rpeaks.size),
        duration_s=duration_s,
        rpeak_indices=rpeaks,
        cleaned_signal=cleaned,
    )


def heart_rate_alert_class(bpm_mean: float, *, min_bpm: float, max_bpm: float) -> HeartRateAlert:
    if not np.isfinite(bpm_mean):
        return "normal"
    if bpm_mean < min_bpm:
        return "bradycardia"
    if bpm_mean > max_bpm:
        return "taquicardia"
    return "normal"
