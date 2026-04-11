import streamlit as st

from src.visualization.ui import inject_global_styles

st.set_page_config(
    page_title="ECG · Arrhythmia Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_styles()

st.title("ECG · Arrhythmia Lab")
st.markdown(
    """
Aplicación para el trabajo final (**visualización tipo papel ECG**, **frecuencia cardiaca con NeuroKit2** y **clasificación opcional**)
sobre el dataset [**PhysioNet ECG-Arrhythmia v1.0.0**](https://physionet.org/content/ecg-arrhythmia/1.0.0/).
"""
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Explorar registros** — Lista WFDB + visualización ECG (papel 25 mm/s).")
with c2:
    st.markdown("**Frecuencia** — Picos R, bpm y alertas con NeuroKit2.")
with c3:
    st.markdown("**Clasificación** — MLP entrenado en el notebook incluido.")

st.divider()
st.markdown(
    """
| Sección | Contenido |
|--------|-----------|
| **Explorar registros** | Registros + ECG (1 o 12 derivaciones); registro sincronizado con el resto de páginas |
| **Análisis frecuencia** | Picos R, bpm, alertas 60–100 lpm |
| **Clasificación** | Inferencia con el `.joblib` del MLP (`CLASSIFIER_MODEL_PATH`) |
| **Asistente (chat)** | Preguntas en español sobre el registro y el modelo (sin LLM externo) |

"""
)
