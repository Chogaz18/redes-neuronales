import streamlit as st

from src.config.settings import get_settings
from src.data.downloader import ensure_dataset
from src.data.exceptions import DatasetNotFoundError
from src.data.metadata import discover_record_refs


st.title("Exploración del dataset")

settings = get_settings()

has_dataset = True
wfdb_dir_str = str(settings.wfdb_records_dir)
try:
    ensure_dataset(settings, download_if_missing=False)
except DatasetNotFoundError:
    has_dataset = False

if not has_dataset:
    st.warning(
        "Dataset no detectado. "
        "Asegúrate de que exista `WFDBRecords/` bajo la ruta configurada."
    )
    st.info(f"Ruta esperada: `{wfdb_dir_str}`")
    st.stop()


@st.cache_data(show_spinner=False)
def _discover(limit: int):
    return discover_record_refs(settings, limit=limit)


refs = _discover(settings.sample_record_count)
st.success(f"Registros detectados (muestra): {len(refs)}")

record_ids = [r.record_id for r in refs]
default_record = record_ids[0] if record_ids else None

st.subheader("Registros disponibles")
if not record_ids:
    st.warning("No se encontraron archivos `.hea`. Verifica la instalación del dataset.")
    st.stop()

selected = st.selectbox("Selecciona un registro", record_ids, index=0)

st.caption("En las siguientes páginas podrás visualizar el ECG y analizar frecuencia cardiaca.")

st.divider()
st.markdown(
    """
Siguientes pasos:
- Visualización ECG (derivación + ventana temporal) -> `2_Visualizacion_ECG.py`
- Análisis HR con NeuroKit2 -> `3_Analisis_Frecuencia.py`
- Bonus clasificación -> `4_Clasificacion.py`
"""
)

