"""
Chat contextual en español sobre el registro ECG seleccionado y el clasificador MLP.
"""

from __future__ import annotations

import streamlit as st

from src.assistant.chat_ui import inject_assistant_page_styles, render_chat_core
from src.config.settings import get_settings
from src.data.downloader import ensure_dataset
from src.data.exceptions import DatasetNotFoundError
from src.visualization.ui import inject_global_styles, sidebar_brand

inject_global_styles()
sidebar_brand()
inject_assistant_page_styles()

st.title("Asistente de contexto ECG")
st.markdown(
    "Pregunta sobre el registro activo: muestreo, derivaciones, etiquetas en el archivo "
    "y **predicción del modelo** (misma lógica que en la página de clasificación). "
)


c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Qué puedes preguntar**")
    st.caption("Hz, derivaciones, SNOMED, ventana, predicción del modelo, ruta del `.joblib`.")
with c2:
    st.markdown("**Repreguntas**")
    st.caption("Tras una respuesta, escribe **«qué significa»** para ampliar en lenguaje sencillo.")
with c3:
    st.markdown("**Ayuda**")
    st.caption("Escribe **ayuda** para ver ejemplos de frases en cualquier momento.")

st.divider()

settings = get_settings()

try:
    ensure_dataset(settings, download_if_missing=False)
except DatasetNotFoundError:
    st.warning("Dataset no detectado.")
    st.stop()

render_chat_core(
    settings,
    container_sidebar=True,
    widget_prefix="page",
    show_clear_button=True,
    assistant_layout=True,
)
