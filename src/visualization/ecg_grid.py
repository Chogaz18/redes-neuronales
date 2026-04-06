from __future__ import annotations

from typing import List, Literal, Tuple

import numpy as np

# Papel estándar: 25 mm/s horizontal; 10 mm = 1 mV vertical
MM_PER_SECOND = 25.0
MM_PER_MV = 10.0
SEC_PER_MM_H = 1.0 / MM_PER_SECOND  # 0.04 s
MV_PER_MM_V = 1.0 / MM_PER_MV  # 0.1 mV

GridDetail = Literal["full", "major"]


def build_ecg_paper_grid(
    *,
    x_min_sec: float,
    x_max_sec: float,
    y_min_mv: float,
    y_max_mv: float,
    detail: GridDetail = "full",
    color_minor: str = "rgba(239,68,68,0.18)",
    color_major: str = "rgba(220,38,38,0.35)",
) -> List[dict]:
    """
    Cuadrícula tipo papel ECG: líneas finas cada 1 mm (0.04 s, 0.1 mV) y gruesas cada 5 mm.
    """
    shapes: List[dict] = []

    def vline(x: float, major: bool) -> None:
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": x,
                "x1": x,
                "y0": y_min_mv,
                "y1": y_max_mv,
                "line": {
                    "color": color_major if major else color_minor,
                    "width": 1.4 if major else 0.7,
                },
                "layer": "below",
            }
        )

    def hline(y: float, major: bool) -> None:
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": x_min_sec,
                "x1": x_max_sec,
                "y0": y,
                "y1": y,
                "line": {
                    "color": color_major if major else color_minor,
                    "width": 1.4 if major else 0.7,
                },
                "layer": "below",
            }
        )

    if detail == "major":
        dt = SEC_PER_MM_H * 5  # 0.2 s
        dv = MV_PER_MM_V * 5  # 0.5 mV
    else:
        dt = SEC_PER_MM_H
        dv = MV_PER_MM_V

    x = x_min_sec
    k = 0
    while x <= x_max_sec + 1e-9:
        major = detail == "major" or (k % 5 == 0)
        vline(x, major)
        x += dt
        k += 1

    y = y_min_mv
    m = 0
    while y <= y_max_mv + 1e-9:
        major = detail == "major" or (m % 5 == 0)
        hline(y, major)
        y += dv
        m += 1

    return shapes


def y_limits_for_signal(signal_mV: List[float] | np.ndarray, *, pad_mv: float = 0.5) -> Tuple[float, float]:
    s = np.asarray(signal_mV, dtype=float).ravel()
    if s.size == 0:
        return (-1.0, 1.0)
    lo = float(np.nanmin(s)) - pad_mv
    hi = float(np.nanmax(s)) + pad_mv
    if lo >= hi:
        lo, hi = -1.0, 1.0
    return (lo, hi)
