import streamlit as st


st.set_page_config(
    page_title="ECG Arrhythmia Explorer",
    layout="wide",
)

st.title("ECG Arrhythmia Explorer")
st.markdown(
    """
Aplicación para visualizar y analizar ECGs del dataset **PhysioNet: ECG-Arrhythmia (v1.0.0)**.

Usa las páginas del menú para:
- Explorar el dataset
- Visualizar ECG con estilo de papel electrocardiográfico
- Analizar frecuencia cardiaca (picos R y bpm)
- (Bonus) Clasificación por clases de arritmias
"""
)

st.divider()

st.subheader("Requisitos de datos")
st.markdown(
    """
Coloca el dataset en la ruta configurada (ver `README.md`). La app funcionará mejor si el directorio contiene el formato WFDB con la carpeta `WFDBRecords/` bajo `data/raw/`.
"""
)

