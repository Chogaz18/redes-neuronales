import streamlit as st

from src.config.settings import get_settings


st.title("Bonus: clasificación de arritmias")

settings = get_settings()

st.subheader("Estado")
st.info("Página preparada para clasificación baseline y posterior inferencia.")

st.markdown(
    """
Requerimientos del bonus:
- Entrenamiento offline y guardado de artefactos (`artifacts/models/`).
- Inferencia desde la app (probabilidades y clase predicha).
- Mapeo de etiquetas vía SNOMED CT hacia:
  - Sinus Bradycardia
  - Sinus Rhythm
  - Atrial Fibrillation
  - Sinus Tachycardia
"""
)

