from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go


def make_ecg_figure(
    *,
    time_s: np.ndarray,
    signal_mV: np.ndarray,
    fs: float,
    rpeaks_indices: Optional[np.ndarray] = None,
    grid_shapes: Optional[list] = None,
    title: str = "ECG",
) -> go.Figure:
    """
    Construye la figura ECG (señal + grid + opcional R peaks).
    """
    raise NotImplementedError

