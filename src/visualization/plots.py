from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.visualization.ecg_grid import GridDetail, build_ecg_paper_grid, y_limits_for_signal


def _apply_ecg_layout(
    fig: go.Figure,
    *,
    x_title: str = "Tiempo (s)",
    y_title: str = "mV",
    height: int | None = None,
) -> None:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#fffbeb",
        plot_bgcolor="#fffbeb",
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        margin=dict(l=56, r=24, t=48, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if height:
        fig.update_layout(height=height)


def make_hr_ecg_figure(
    time_s: np.ndarray,
    signal_mV: np.ndarray,
    rpeak_indices: np.ndarray,
    *,
    title: str = "ECG y picos R",
    show_paper_grid: bool = True,
    grid_detail: GridDetail = "full",
) -> go.Figure:
    """Señal limpia + marcadores en picos R; opcionalmente cuadrícula papel ECG."""
    t = np.asarray(time_s, dtype=float).ravel()
    y = np.asarray(signal_mV, dtype=float).ravel()
    y0, y1 = y_limits_for_signal(y, pad_mv=0.35)
    shapes: List[dict] = []
    if show_paper_grid and t.size:
        shapes = build_ecg_paper_grid(
            x_min_sec=float(t[0]),
            x_max_sec=float(t[-1]),
            y_min_mv=y0,
            y_max_mv=y1,
            detail=grid_detail,
        )

    fig = go.Figure()
    if shapes:
        fig.update_layout(shapes=shapes)
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            mode="lines",
            name="ECG (limpio)",
            line=dict(color="#1e3a5f", width=1.4),
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
                marker=dict(color="#b91c1c", size=9, symbol="circle", line=dict(width=0)),
            )
        )
    fig.update_layout(title=title, height=440)
    _apply_ecg_layout(fig)
    return fig


def make_ecg_figure(
    *,
    time_s: np.ndarray,
    signal_mV: np.ndarray,
    fs: float,
    rpeaks_indices: Optional[np.ndarray] = None,
    grid_shapes: Optional[list] = None,
    title: str = "ECG",
    grid_detail: GridDetail = "full",
) -> go.Figure:
    """Una derivación: señal + cuadrícula papel ECG + picos R opcionales."""
    t = np.asarray(time_s, dtype=float).ravel()
    y = np.asarray(signal_mV, dtype=float).ravel()
    y0, y1 = y_limits_for_signal(y, pad_mv=0.35)
    shapes = grid_shapes
    if shapes is None and t.size:
        shapes = build_ecg_paper_grid(
            x_min_sec=float(t[0]),
            x_max_sec=float(t[-1]),
            y_min_mv=y0,
            y_max_mv=y1,
            detail=grid_detail,
        )
    fig = go.Figure()
    if shapes:
        fig.update_layout(shapes=shapes)
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            mode="lines",
            name="ECG",
            line=dict(color="#1e3a5f", width=1.4),
        )
    )
    if rpeaks_indices is not None and rpeaks_indices.size:
        rp = np.asarray(rpeaks_indices, dtype=np.int64).ravel()
        rp = rp[(rp >= 0) & (rp < y.size)]
        if rp.size:
            fig.add_trace(
                go.Scatter(
                    x=t[rp],
                    y=y[rp],
                    mode="markers",
                    name="Picos R",
                    marker=dict(color="#b91c1c", size=8),
                )
            )
    fig.update_layout(title=title, height=420)
    _apply_ecg_layout(fig)
    return fig


def make_twelve_lead_ecg_figure(
    time_s: np.ndarray,
    signals_mV: np.ndarray,
    lead_names: Sequence[str],
    *,
    title: str = "ECG 12 derivaciones",
    grid_detail: GridDetail = "major",
) -> go.Figure:
    """
    Vista estilo monitor: hasta 12 filas, eje X compartido.
    Cuadrícula por shapes con `yref` por fila (y, y2, …). `major` = menos líneas (más fluido).
    """
    sig = np.asarray(signals_mV, dtype=float)
    n_leads = min(12, sig.shape[0])
    t = np.asarray(time_s, dtype=float).ravel()
    x0 = float(t[0]) if t.size else 0.0
    x1 = float(t[-1]) if t.size else 1.0

    fig = make_subplots(
        rows=n_leads,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.012,
        subplot_titles=[str(lead_names[i]) for i in range(n_leads)],
    )

    step_x = 0.2 if grid_detail == "major" else 0.04
    step_y = 0.5 if grid_detail == "major" else 0.1
    all_shapes: List[dict] = []

    for i in range(n_leads):
        row = i + 1
        y = sig[i]
        y0, y1 = y_limits_for_signal(y, pad_mv=0.25)
        yref = "y" if row == 1 else f"y{row}"
        xref = "x"

        xv = x0
        k = 0
        while xv <= x1 + 1e-9:
            major = grid_detail == "major" or (k % 5 == 0)
            all_shapes.append(
                {
                    "type": "line",
                    "xref": xref,
                    "yref": yref,
                    "x0": xv,
                    "x1": xv,
                    "y0": y0,
                    "y1": y1,
                    "line": {
                        "color": "rgba(220,38,38,0.35)" if major else "rgba(239,68,68,0.18)",
                        "width": 1.2 if major else 0.65,
                    },
                    "layer": "below",
                }
            )
            xv += step_x
            k += 1

        yv = y0
        m = 0
        while yv <= y1 + 1e-9:
            major = grid_detail == "major" or (m % 5 == 0)
            all_shapes.append(
                {
                    "type": "line",
                    "xref": xref,
                    "yref": yref,
                    "x0": x0,
                    "x1": x1,
                    "y0": yv,
                    "y1": yv,
                    "line": {
                        "color": "rgba(220,38,38,0.35)" if major else "rgba(239,68,68,0.18)",
                        "width": 1.2 if major else 0.65,
                    },
                    "layer": "below",
                }
            )
            yv += step_y
            m += 1

        fig.add_trace(
            go.Scatter(
                x=t,
                y=y,
                mode="lines",
                line=dict(color="#1e3a5f", width=1.05),
                showlegend=False,
            ),
            row=row,
            col=1,
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="#fffbeb",
        plot_bgcolor="#fffbeb",
        height=min(110 * n_leads + 100, 1400),
        margin=dict(l=52, r=20, t=72, b=44),
        shapes=all_shapes,
    )
    fig.update_xaxes(title_text="Tiempo (s)", row=n_leads, col=1)
    return fig


CLASSIFICATION_PLOTLY_CONFIG: dict = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
}


def make_classification_probability_figure(
    probabilities: dict[str, float],
    predicted_class: str,
) -> go.Figure:
    """
    Una sola figura: barras horizontales ordenadas (mayor arriba), porcentajes legibles,
    borde suave en la categoría elegida por el modelo. Sin subplots ni leyenda duplicada.
    """
    from src.modeling.label_mapping import educational_blurb_for_class

    palette = {
        "Sinus Bradycardia": "#2563eb",
        "Sinus Rhythm": "#16a34a",
        "Atrial Fibrillation": "#b91c1c",
        "Sinus Tachycardia": "#d97706",
    }

    items = sorted(probabilities.items(), key=lambda x: -x[1])
    labels = [educational_blurb_for_class(k)["titulo"] for k, _ in items]
    keys = [k for k, _ in items]
    xs = [float(p) * 100.0 for _, p in items]
    colors = [palette.get(k, "#64748b") for k in keys]
    line_w = [2.0 if k == predicted_class else 0.0 for k in keys]
    line_c = ["#334155" if k == predicted_class else "rgba(0,0,0,0)" for k in keys]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            orientation="h",
            x=xs,
            y=labels,
            marker=dict(color=colors, line=dict(width=line_w, color=line_c)),
            text=[f"{x:.1f} %" for x in xs],
            textposition="outside",
            textfont=dict(size=13, color="#1e293b"),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x:.2f} % del total<extra></extra>",
            showlegend=False,
        )
    )

    xmax = max(100.0, max(xs) * 1.08) if xs else 100.0
    fig.update_layout(
        title=dict(
            text="Confianza del modelo por categoría",
            font=dict(size=17, color="#0f172a"),
            x=0.5,
            xanchor="center",
            pad=dict(t=8, b=12),
        ),
        template="plotly_white",
        paper_bgcolor="#fafaf9",
        plot_bgcolor="#ffffff",
        height=max(260, 68 * len(labels) + 120),
        margin=dict(l=8, r=88, t=88, b=56),
        xaxis=dict(
            title=dict(text="Porcentaje (%)", font=dict(size=12)),
            range=[0, xmax],
            showgrid=True,
            gridcolor="rgba(15,23,42,0.07)",
            zeroline=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            tickfont=dict(size=12),
        ),
        font=dict(family="Segoe UI, system-ui, sans-serif", color="#334155"),
    )
    return fig
