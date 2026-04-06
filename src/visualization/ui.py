"""Estilos globales reutilizables en páginas Streamlit."""

from __future__ import annotations

import streamlit as st


def inject_global_styles() -> None:
    st.markdown(
        """
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
    h1 { font-weight: 700 !important; letter-spacing: -0.02em; color: #0f172a !important; }
    [data-testid="stSidebarNav"] { padding-top: 0.5rem; }
    [data-testid="stSidebarNav"] ul li:first-child { display: none; }
    [data-testid="stSidebarNav"] a { text-transform: lowercase; }
    div[data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 600; }
</style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.sidebar.markdown(
        """
<div style="padding: 0.2rem 0 1rem 0;">
  <div style="font-size: 1.1rem; font-weight: 700; color: #1e3a8a;">ECG · Arrhythmia Lab</div>
  <div style="font-size: 0.78rem; opacity: 0.85; margin-top: 0.2rem;">PhysioNet ECG-Arrhythmia v1.0</div>
</div>
        """,
        unsafe_allow_html=True,
    )


RECORD_STATE_KEY = "ecg_selected_record_id"


def record_select_sync(
    record_ids: list[str],
    *,
    label: str = "Registro",
    sidebar: bool = False,
    widget_key: str = "ecg_record_select",
) -> str:
    """
    Select de registro enlazado a `st.session_state[RECORD_STATE_KEY]` para que el
    mismo registro se mantenga al cambiar de página (explorar registros, frecuencia, clasificación).
    Usa un `widget_key` distinto por página para evitar colisiones de widgets.
    """
    if not record_ids:
        return ""
    st.session_state.setdefault(RECORD_STATE_KEY, record_ids[0])
    if st.session_state[RECORD_STATE_KEY] not in record_ids:
        st.session_state[RECORD_STATE_KEY] = record_ids[0]
    idx = record_ids.index(st.session_state[RECORD_STATE_KEY])
    container = st.sidebar if sidebar else st
    sel = container.selectbox(label, record_ids, index=idx, key=widget_key)
    st.session_state[RECORD_STATE_KEY] = sel
    return sel
