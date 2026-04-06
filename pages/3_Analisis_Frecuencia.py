from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from src.config.settings import Settings, get_settings
from src.data.downloader import ensure_dataset
from src.data.exceptions import DatasetNotFoundError, ECGReadError
from src.data.loaders import load_ecg_record
from src.data.metadata import RecordRef, discover_record_refs
from src.data.parsers import choose_rpeak_lead, header_to_metadata
from src.processing.heart_rate import estimate_heart_rate, heart_rate_alert_class
from src.visualization.plots import make_hr_ecg_figure
from src.visualization.ui import (
    inject_global_styles,
    record_select_sync,
    sidebar_brand,
)

inject_global_styles()
sidebar_brand()

st.title("Análisis de frecuencia cardiaca")

settings = get_settings()

try:
    ensure_dataset(settings, download_if_missing=False)
except DatasetNotFoundError:
    st.warning(
        "Dataset no detectado. Asegúrate de que exista `WFDBRecords/` bajo la ruta configurada."
    )
    st.stop()


@st.cache_data(show_spinner=False)
def _discover(limit: int):
    return discover_record_refs(settings, limit=limit)


@st.cache_data(show_spinner="Cargando segmento ECG…")
def _load_hr_segment(
    wfdb_records_dir: str,
    record_id: str,
    hea_path_str: str,
    start_s: float,
    duration_s: float,
    rpeak_lead_pref: str,
):
    _ = wfdb_records_dir
    cfg: Settings = get_settings()
    ref = RecordRef(record_id=record_id, hea_path=Path(hea_path_str))
    meta = header_to_metadata(ref)
    lead = choose_rpeak_lead(meta["lead_names"], rpeak_lead_pref)
    ecg = load_ecg_record(cfg, ref, leads=[lead], start_s=start_s, duration_s=duration_s)
    return ecg, lead


refs = _discover(settings.sample_record_count)
record_ids = [r.record_id for r in refs]
if not record_ids:
    st.warning("No se detectaron registros. Verifica la ruta del dataset.")
    st.stop()

ref_by_id = {r.record_id: r for r in refs}

selected_record_id = record_select_sync(
    record_ids,
    label="Registro",
    sidebar=True,
    widget_key="ecg_record_hr_page",
)
start_s = st.sidebar.slider(
    "Inicio (s)",
    min_value=0.0,
    max_value=max(0.0, settings.default_window_duration_s),
    value=0.0,
    step=0.5,
)
duration_s = st.sidebar.slider(
    "Duración analizada (s)",
    min_value=settings.min_window_duration_s,
    max_value=settings.max_window_duration_s,
    value=settings.default_window_duration_s,
    step=0.5,
)

st.caption(
    "Picos R y bpm con NeuroKit2 sobre el tramo elegido. "
    f"Derivación para detección: preferencia **{settings.default_rpeak_lead}** (p. ej. lead II si existe)."
)

selected_ref = ref_by_id[selected_record_id]

try:
    ecg, lead_used = _load_hr_segment(
        str(settings.wfdb_records_dir),
        selected_ref.record_id,
        str(selected_ref.hea_path),
        float(start_s),
        float(duration_s),
        settings.default_rpeak_lead,
    )
except ECGReadError as e:
    st.error(str(e))
    st.stop()

signal_1d = ecg.signals[0]
fs = float(ecg.fs)

try:
    hr = estimate_heart_rate(
        signal_1d,
        fs,
        min_bpm=settings.hr_min_bpm,
        max_bpm=settings.hr_max_bpm,
    )
except ValueError as e:
    st.error(str(e))
    st.stop()

time_s = np.arange(hr.cleaned_signal.size, dtype=float) / fs + float(start_s)
fig = make_hr_ecg_figure(
    time_s,
    hr.cleaned_signal,
    hr.rpeak_indices,
    title=f"{selected_record_id} · derivación {lead_used}",
)

c1, c2, c3, c4 = st.columns(4)
bpm_disp = hr.bpm_mean
rr_disp = hr.rr_mean_s * 1000.0 if np.isfinite(hr.rr_mean_s) else float("nan")

with c1:
    st.metric(
        "Frecuencia cardiaca (prom.)",
        f"{bpm_disp:.1f} bpm" if np.isfinite(bpm_disp) else "—",
    )
with c2:
    st.metric(
        "RR (prom.)",
        f"{rr_disp:.0f} ms" if np.isfinite(rr_disp) else "—",
    )
with c3:
    st.metric("Picos R detectados", f"{hr.n_rpeaks}")
with c4:
    st.metric("Duración analizada", f"{hr.duration_s:.2f} s")

alert = heart_rate_alert_class(
    hr.bpm_mean,
    min_bpm=settings.hr_min_bpm,
    max_bpm=settings.hr_max_bpm,
)
if not np.isfinite(hr.bpm_mean):
    st.warning("No se pudo estimar bpm (menos de dos picos R en el segmento).")
elif alert == "normal":
    st.success(
        f"Ritmo dentro del rango habitual de la app ({settings.hr_min_bpm:.0f}–{settings.hr_max_bpm:.0f} bpm)."
    )
elif alert == "bradycardia":
    st.warning(
        f"**Bradicardia** (bpm inferior a {settings.hr_min_bpm:.0f}). "
        "Valoración clínica solo con contexto profesional."
    )
else:
    st.warning(
        f"**Taquicardia** (bpm superior a {settings.hr_max_bpm:.0f}). "
        "Valoración clínica solo con contexto profesional."
    )

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Las alertas usan el rango configurado en `.env` (por defecto 60–100 lpm); no sustituyen diagnóstico médico."
)
