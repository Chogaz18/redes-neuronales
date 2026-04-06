from __future__ import annotations

import numpy as np
import streamlit as st

from src.config.settings import get_settings
from src.data.downloader import ensure_dataset
from src.data.exceptions import DatasetNotFoundError
from src.data.loaders import load_ecg_record
from src.data.metadata import discover_record_refs
from src.visualization.plots import make_ecg_figure, make_twelve_lead_ecg_figure
from src.visualization.ui import (
    inject_global_styles,
    record_select_sync,
    sidebar_brand,
)

inject_global_styles()
sidebar_brand()

st.title("Explorar registros y señal ECG")
st.markdown(
    "Elige un registro y mira la señal con **cuadrícula tipo papel**: "
    "25 mm/s (1 mm = 0,04 s); **10 mm = 1 mV** (1 mm = 0,1 mV)."
)

settings = get_settings()
wfdb_dir_str = str(settings.wfdb_records_dir)

try:
    ensure_dataset(settings, download_if_missing=False)
except DatasetNotFoundError:
    st.warning(
        "Dataset no detectado. Asegúrate de que exista `WFDBRecords/` bajo la ruta configurada."
    )
    st.info(f"Ruta esperada: `{wfdb_dir_str}`")
    st.stop()


@st.cache_data(show_spinner=False)
def _discover(limit: int):
    return discover_record_refs(settings, limit=limit)


refs = _discover(settings.sample_record_count)

record_ids = [r.record_id for r in refs]
if not record_ids:
    st.warning("No se encontraron archivos `.hea`. Verifica la instalación del dataset.")
    st.stop()

ref_by_id = {r.record_id: r for r in refs}

with st.container(border=True):
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("**Análisis frecuencia**")
        st.caption("NeuroKit2, picos R, bpm y alertas de ritmo.")
        st.page_link(
            "pages/3_analisis_frecuencia.py",
            label="Abrir análisis de frecuencia",
            icon="📈",
        )
    with g2:
        st.markdown("**Clasificación**")
        st.caption("Segmentación MLP")
        st.page_link(
            "pages/4_clasificacion.py",
            label="Abrir clasificación",
            icon="🧠",
        )

st.subheader("Registros disponibles")
selected_record_id = record_select_sync(
    record_ids,
    label="Selecciona un registro",
    sidebar=False,
    widget_key="ecg_record_dataset_main",
)
selected_ref = ref_by_id[selected_record_id]

st.caption(
    "El registro elegido se conserva al ir a **Análisis frecuencia** o **Clasificación** (barra lateral)."
)

view_mode = st.sidebar.radio("Vista", ["Una derivación", "12 derivaciones"], horizontal=False)

st.sidebar.subheader("Ventana temporal")
start_s = st.sidebar.slider(
    "Inicio (s)",
    min_value=0.0,
    max_value=max(0.0, settings.default_window_duration_s),
    value=0.0,
    step=0.5,
)
duration_s = st.sidebar.slider(
    "Duración visible (s)",
    min_value=settings.min_window_duration_s,
    max_value=settings.max_window_duration_s,
    value=settings.default_window_duration_s,
    step=0.5,
)

st.divider()
st.subheader("Visualización")

try:
    ecg = load_ecg_record(
        settings,
        selected_ref,
        leads=None,
        start_s=float(start_s),
        duration_s=float(duration_s),
    )
except Exception as e:
    st.error(f"No se pudo cargar el registro: {e}")
    st.stop()

fs = float(ecg.fs)
n = ecg.signals.shape[1]
t = np.arange(n, dtype=float) / fs + float(start_s)

if view_mode == "Una derivación":
    lead = st.sidebar.selectbox("Derivación", ecg.lead_names, index=0)
    li = ecg.lead_names.index(lead)
    fig = make_ecg_figure(
        time_s=t,
        signal_mV=ecg.signals[li],
        fs=fs,
        title=f"{selected_record_id} · {lead}",
        grid_detail="full",
    )
else:
    fig = make_twelve_lead_ecg_figure(
        t,
        ecg.signals,
        ecg.lead_names,
        title=f"{selected_record_id} · 12 derivaciones",
        grid_detail="major",
    )

st.plotly_chart(fig, use_container_width=True)
