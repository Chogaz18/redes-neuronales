from __future__ import annotations

import neurokit2 as nk

from src.processing.ecg_units import to_millivolts


def clean_ecg_signal(signal: np.ndarray, fs: float) -> np.ndarray:
    """Limpieza NeuroKit2 (`ecg_clean`), coherente con `heart_rate.estimate_heart_rate`."""
    sig_mv = to_millivolts(signal)
    return np.asarray(nk.ecg_clean(sig_mv, sampling_rate=float(fs)), dtype=float)
