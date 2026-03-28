from __future__ import annotations

from typing import Optional

import numpy as np

from src.utils.helpers import clamp


def segment_signal(
    signals: np.ndarray,  # shape: [n_leads, n_samples]
    fs: float,
    *,
    start_s: float,
    duration_s: float,
) -> np.ndarray:
    start_idx = int(round(start_s * fs))
    end_idx = int(round((start_s + duration_s) * fs))
    start_idx = max(0, start_idx)
    end_idx = min(signals.shape[1], end_idx)
    return signals[:, start_idx:end_idx]


def compute_time_axis(fs: float, n_samples: int) -> np.ndarray:
    return np.arange(n_samples, dtype=float) / float(fs)

