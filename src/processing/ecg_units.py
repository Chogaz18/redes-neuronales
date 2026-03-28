from __future__ import annotations

import numpy as np


def to_millivolts(signal: np.ndarray) -> np.ndarray:
    """NeuroKit2 asume mV; amplitudes muy altas suelen ser µV (p. ej. WFDB)."""
    sig = np.asarray(signal, dtype=float).ravel()
    if sig.size == 0:
        return sig
    if float(np.nanmax(np.abs(sig))) > 50.0:
        return sig * 0.001
    return sig
