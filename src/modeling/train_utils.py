from __future__ import annotations

from typing import List, Sequence, Tuple
import warnings

import neurokit2 as nk
import numpy as np
from scipy.signal import resample

from src.config.settings import Settings
from src.data.loaders import load_ecg_record
from src.data.metadata import RecordRef
from src.processing.ecg_units import to_millivolts

# Orden estándar 12 derivaciones (visualización y matriz fija)
STANDARD_LEADS: List[str] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def normalize_lead_name(name: str) -> str:
    return str(name).strip().upper().replace("LEAD", "").replace(" ", "")


def align_leads_12(signals: np.ndarray, lead_names: Sequence[str]) -> np.ndarray:
    """
    Reordena a (12, T) según STANDARD_LEADS; derivaciones ausentes quedan en cero.
    """
    sig = np.asarray(signals, dtype=float)
    n_samples = sig.shape[1]
    out = np.zeros((12, n_samples), dtype=float)
    name_to_row = {}
    for i, n in enumerate(lead_names):
        key = normalize_lead_name(n)
        if key not in name_to_row:
            name_to_row[key] = i
    for j, lead in enumerate(STANDARD_LEADS):
        key = normalize_lead_name(lead)
        if key in name_to_row:
            out[j] = sig[name_to_row[key]]
    return out


def resample_12_leads(signals_12: np.ndarray, fs: float, target_len: int) -> np.ndarray:
    """Re-muestrea cada derivación a `target_len` muestras."""
    s = np.asarray(signals_12, dtype=float)
    out = np.zeros((12, target_len), dtype=float)
    for i in range(min(12, s.shape[0])):
        out[i] = resample(s[i], target_len)
    return out


def load_12lead_aligned(
    settings: Settings,
    record_ref: RecordRef,
    *,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> Tuple[np.ndarray, float, List[str]]:
    rec = load_ecg_record(settings, record_ref, leads=None, start_s=start_s, duration_s=duration_s)
    aligned = align_leads_12(rec.signals, rec.lead_names)
    return aligned, rec.fs, list(STANDARD_LEADS)


def _lead_stats(x: np.ndarray) -> list[float]:
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size == 0:
        return [0.0] * 15

    dx = np.diff(x) if x.size > 1 else np.array([0.0], dtype=np.float32)
    q10, q25, q50, q75, q90 = np.percentile(x, [10, 25, 50, 75, 90])
    zc = float(np.mean((x[:-1] * x[1:]) < 0)) if x.size > 1 else 0.0

    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(np.ptp(x)),
        float(np.sqrt(np.mean(x ** 2))),
        float(np.mean(np.abs(x))),
        float(np.mean(np.abs(dx))),
        float(np.std(dx)),
        float(q10),
        float(q25),
        float(q50),
        float(q75),
        float(q90),
        zc,
    ]


def _rhythm_features(signal: np.ndarray, fs: float) -> list[float]:
    sig_mv = to_millivolts(signal)
    sig_mv = np.asarray(sig_mv, dtype=np.float32).ravel()

    if sig_mv.size < max(int(fs * 2), 50):
        return [0.0] * 8

    sig_mv = np.nan_to_num(sig_mv, nan=0.0, posinf=0.0, neginf=0.0)

    if float(np.std(sig_mv)) < 1e-6:
        return [0.0] * 8

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, info = nk.ecg_peaks(sig_mv, sampling_rate=float(fs))

        rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64).ravel()
        duration_s = float(len(sig_mv) / float(fs))

        if rpeaks.size < 2:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(rpeaks.size), float(rpeaks.size / max(duration_s, 1e-6))]

        rr = np.diff(rpeaks).astype(np.float32) / float(fs)
        rr = rr[np.isfinite(rr)]

        if rr.size == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(rpeaks.size), float(rpeaks.size / max(duration_s, 1e-6))]

        rr_mean = float(np.mean(rr))
        rr_std = float(np.std(rr))
        rr_cv = float(rr_std / rr_mean) if rr_mean > 0 else 0.0
        bpm = float(60.0 / rr_mean) if rr_mean > 0 else 0.0

        if rr.size >= 2:
            rr_diff = np.diff(rr)
            rr_diff = rr_diff[np.isfinite(rr_diff)]
            if rr_diff.size > 0:
                rmssd = float(np.sqrt(np.mean(rr_diff ** 2)))
                pnn50 = float(np.mean(np.abs(rr_diff) > 0.05))
            else:
                rmssd = 0.0
                pnn50 = 0.0
        else:
            rmssd = 0.0
            pnn50 = 0.0

        return [
            bpm,
            rr_mean,
            rr_std,
            rr_cv,
            rmssd,
            pnn50,
            float(rpeaks.size),
            float(rpeaks.size / max(duration_s, 1e-6)),
        ]
    except Exception:
        return [0.0] * 8


def features_for_mlp(signals_12: np.ndarray, fs: float, target_len: int) -> np.ndarray:
    """
    Devuelve un vector fila (1, 188):
    - 12 derivaciones x 15 estadísticas = 180
    - 8 rasgos de ritmo = 8
    """
    r = resample_12_leads(signals_12, fs, target_len)

    feats = []
    for i in range(r.shape[0]):
        feats.extend(_lead_stats(r[i]))

    lead_for_rhythm = signals_12[1] if signals_12.shape[0] > 1 else signals_12[0]
    feats.extend(_rhythm_features(lead_for_rhythm, fs))

    out = np.asarray(feats, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(1, -1)