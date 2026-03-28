from __future__ import annotations

from typing import List, Optional

import numpy as np


def build_ecg_paper_grid(
    *,
    duration_s: float,
    fs: float,
    amplitude_mV_range: float,
    x_major_mm: int = 5,
    x_minor_mm: int = 1,
    time_step_s_per_mm: float = 0.04,
    shapes_color_major: str = "rgba(0,0,0,0.25)",
    shapes_color_minor: str = "rgba(0,0,0,0.10)",
) -> List[dict]:
    """
    Retorna shapes de Plotly para emular el papel ECG.
    """
    raise NotImplementedError

