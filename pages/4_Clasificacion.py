from __future__ import annotations

import streamlit as st

from src.config.settings import get_settings
from src.data.downloader import ensure_dataset
from src.data.exceptions import DatasetNotFoundError
from src.data.metadata import discover_record_refs
from src.data.parsers import header_to_metadata
from src.modeling.infer import load_model_and_infer
from src.modeling.label_mapping import (
    FOUR_CLASS_KEYS,
    educational_blurb_for_class,
    map_snomed_codes_to_four_classes,
)
from src.modeling.train_utils import load_12lead_aligned
from src.visualization.plots import (
    CLASSIFICATION_PLOTLY_CONFIG,
    make_classification_probability_figure,
)
from src.visualization.ui import (
    inject_global_styles,
    record_select_sync,
    sidebar_brand,
)

inject_global_styles()
sidebar_brand()

st.title("Clasificación de ritmo")

settings = get_settings()

try:
    ensure_dataset(settings, download_if_missing=False)
except DatasetNotFoundError:
    st.warning("Dataset no detectado.")
    st.stop()


@st.cache_data(show_spinner=False)
def _discover(limit: int):
    return discover_record_refs(settings, limit=limit)


refs = _discover(settings.sample_record_count)
record_ids = [r.record_id for r in refs]
if not record_ids:
    st.warning("No se detectaron registros.")
    st.stop()

ref_by_id = {r.record_id: r for r in refs}

selected_record_id = record_select_sync(
    record_ids,
    label="Registro",
    sidebar=True,
    widget_key="ecg_record_cls_page",
)
selected_ref = ref_by_id[selected_record_id]

meta = header_to_metadata(selected_ref)
snomed = meta.get("snomed_ct_codes", [])
mapped = map_snomed_codes_to_four_classes(snomed)

try:
    aligned, fs, _ = load_12lead_aligned(settings, selected_ref, start_s=0.0, duration_s=None)
except Exception as e:
    st.error(f"No se pudo cargar la señal: {e}")
    st.stop()

st.subheader("Resultado del modelo")
try:
    pred = load_model_and_infer(signals_12_mv=aligned, fs=fs, settings=settings)
except FileNotFoundError as e:
    st.warning(str(e))
    st.info(
        "Por defecto se busca `artifacts/models/ecg_mlp_pipeline.joblib` o `ecg_mlp_4class.joblib`. "
        "Puedes fijar la ruta en `.env` con `CLASSIFIER_MODEL_PATH=...`"
    )
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

main = educational_blurb_for_class(pred.predicted_class)
top_p = float(pred.probabilities.get(pred.predicted_class, 0.0))

with st.container(border=True):
    st.markdown(f"### {main['titulo']}")
    st.caption(f"**{pred.predicted_class}**")
    st.info(main["texto"])
    st.metric(
        label="Confianza del modelo en esta categoría",
        value=f"{top_p * 100:.1f} %",
        help=(
            "Indica qué tan fuerte es la respuesta del modelo para la categoría elegida. "
            "Valores altos suelen indicar que el modelo se inclina claramente por esa opción; "
            "no miden riesgo clínico ni certeza médica."
        ),
    )

st.markdown("##### ¿Qué tan probable es cada categoría?")
st.caption(
    "Orden **de mayor a menor** probabilidad. Cada valor es la **parte del 100 %** que el modelo asigna a ese ritmo; "
    "las cuatro suman 100 %. El **contorno fino** indica la categoría que el modelo toma como resultado principal."
)

with st.container(border=True):
    fig_probs = make_classification_probability_figure(pred.probabilities, pred.predicted_class)
    st.plotly_chart(fig_probs, use_container_width=True, config=CLASSIFICATION_PLOTLY_CONFIG)

with st.expander("Qué significa cada categoría (guía sencilla)"):
    for k in FOUR_CLASS_KEYS:
        b = educational_blurb_for_class(k)
        st.markdown(f"**{b['titulo']}** (`{k}`)")
        st.markdown(b["texto"])
        st.divider()