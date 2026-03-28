import streamlit as st

from src.config.settings import get_settings
from src.data.downloader import ensure_dataset
from src.data.exceptions import DatasetNotFoundError
from src.data.metadata import discover_record_refs


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


refs = _discover(settings.sample_record_count)
record_ids = [r.record_id for r in refs]
if not record_ids:
    st.warning("No se detectaron registros. Verifica la ruta del dataset.")
    st.stop()

selected_record_id = st.sidebar.selectbox("Registro", record_ids, index=0)
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

st.subheader("Controles")
st.caption("Se detectarán picos R y se estimará bpm sobre el tramo seleccionado.")

st.write(
    {
        "record_id": selected_record_id,
        "start_s": start_s,
        "duration_s": duration_s,
        "hr_range_bpm": (settings.hr_min_bpm, settings.hr_max_bpm),
    }
)

