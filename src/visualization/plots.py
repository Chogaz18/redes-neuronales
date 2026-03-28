from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go


def make_hr_ecg_figure(
    time_s: np.ndarray,
    signal_mV: np.ndarray,
    rpeak_indices: np.ndarray,
    *,
    title: str = "ECG y picos R",
) -> go.Figure:
    """Señal limpia + marcadores en picos R (análisis de frecuencia cardiaca)."""
    t = np.asarray(time_s, dtype=float).ravel()
    y = np.asarray(signal_mV, dtype=float).ravel()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            mode="lines",
            name="ECG (limpio)",
            line=dict(color="#2563eb", width=1.2),
        )
    )
    rp = np.asarray(rpeak_indices, dtype=np.int64).ravel()
    rp = rp[(rp >= 0) & (rp < y.size)]
    if rp.size:
        fig.add_trace(
            go.Scatter(
                x=t[rp],
                y=y[rp],
                mode="markers",
                name="Picos R",
                marker=dict(color="#dc2626", size=9, symbol="circle", line=dict(width=0)),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Tiempo (s)",
        yaxis_title="Amplitud (mV)",
        hovermode="x unified",
        template="plotly_white",
        height=440,
        margin=dict(l=56, r=24, t=52, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


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

