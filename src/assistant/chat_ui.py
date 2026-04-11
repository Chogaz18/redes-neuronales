"""
UI del asistente: mensajes en sesión y ventana temporal (solo página dedicada).
"""

from __future__ import annotations

import streamlit as st

from src.assistant.ecg_chat import AnalysisSnapshot, answer_question, build_snapshot
from src.config.settings import Settings
from src.data.downloader import ensure_dataset
from src.data.exceptions import DatasetNotFoundError
from src.data.metadata import discover_record_refs
from src.visualization.ui import record_select_sync

CHAT_MESSAGES_KEY = "ecg_chat_messages"
CHAT_LAST_TOPIC_KEY = "ecg_chat_last_topic"
CHAT_WIN_START = "ecg_chat_window_start_s"
CHAT_WIN_DUR = "ecg_chat_window_duration_s"


def inject_assistant_page_styles() -> None:
    st.markdown(
        """
<style>
    section.main > div > div.block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 920px;
    }
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 0.35rem;
    }
    [data-testid="stChatInput"] {
        border-radius: 12px;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def _slider_keys(widget_prefix: str) -> tuple[str, str]:
    return f"{widget_prefix}_win_start_s", f"{widget_prefix}_win_dur_s"


def _default_greeting() -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": (
                "Hola. Respondo sobre el **registro** que elijas, la **ventana temporal**, "
                "**Hz**, **derivaciones**, **códigos del archivo** y **predicción del modelo**. "
                "Escribe **ayuda** para ver ejemplos. Tras una respuesta puedes decir **«qué significa»** "
                "para una explicación sencilla (mayúsculas y tildes no importan)."
            ),
        }
    ]


def _init_chat_window_state(settings: Settings) -> None:
    st.session_state.setdefault(CHAT_WIN_START, 0.0)
    default_d = min(
        10.0,
        float(settings.default_window_duration_s),
        float(settings.max_window_duration_s),
    )
    st.session_state.setdefault(CHAT_WIN_DUR, default_d)


def _ensure_messages() -> None:
    if CHAT_MESSAGES_KEY not in st.session_state:
        st.session_state[CHAT_MESSAGES_KEY] = _default_greeting()


@st.cache_data(show_spinner=False)
def _discover_refs(limit: int):
    from src.config.settings import get_settings

    return discover_record_refs(get_settings(), limit=limit)


def render_chat_core(
    settings: Settings,
    *,
    container_sidebar: bool,
    widget_prefix: str,
    show_clear_button: bool = True,
    assistant_layout: bool = False,
) -> None:
    """Chat + select registro + ventana. `assistant_layout`: métricas visibles encima del hilo."""
    _ensure_messages()

    try:
        ensure_dataset(settings, download_if_missing=False)
    except DatasetNotFoundError:
        st.warning("Dataset no detectado. Coloca los datos en la ruta configurada para usar el asistente.")
        return

    refs = _discover_refs(settings.sample_record_count)
    record_ids = [r.record_id for r in refs]
    if not record_ids:
        st.warning("No se detectaron registros.")
        return

    ref_by_id = {r.record_id: r for r in refs}
    select_key = f"ecg_record_{widget_prefix}_chat"
    selected_record_id = record_select_sync(
        record_ids,
        label="Registro",
        sidebar=container_sidebar,
        widget_key=select_key,
    )
    selected_ref = ref_by_id[selected_record_id]

    ks, kd = _slider_keys(widget_prefix)
    _init_chat_window_state(settings)
    st.session_state.setdefault(ks, float(st.session_state[CHAT_WIN_START]))
    st.session_state.setdefault(kd, float(st.session_state[CHAT_WIN_DUR]))

    place = st.sidebar if container_sidebar else st
    place.subheader("Registro y ventana")
    place.slider(
        "Inicio (s)",
        min_value=0.0,
        max_value=max(0.0, settings.default_window_duration_s),
        step=0.5,
        key=ks,
    )
    place.slider(
        "Duración (s)",
        min_value=settings.min_window_duration_s,
        max_value=settings.max_window_duration_s,
        step=0.5,
        key=kd,
    )
    start_s = float(st.session_state[ks])
    duration_s = float(st.session_state[kd])
    st.session_state[CHAT_WIN_START] = start_s
    st.session_state[CHAT_WIN_DUR] = duration_s

    if show_clear_button and place.button("Limpiar historial del chat", key=f"{widget_prefix}_chat_clear"):
        st.session_state[CHAT_MESSAGES_KEY] = _default_greeting()
        st.session_state.pop(CHAT_LAST_TOPIC_KEY, None)
        st.rerun()

    try:
        snap: AnalysisSnapshot = build_snapshot(
            settings, selected_ref, start_s=start_s, duration_s=duration_s
        )
    except Exception as e:
        st.error(f"No se pudo cargar el contexto del registro: {e}")
        return

    if assistant_layout:
        st.markdown("###### Contexto del análisis")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Registro", snap.record_id if len(snap.record_id) <= 18 else snap.record_id[:16] + "…")
        with k2:
            st.metric("Muestreo", f"{snap.fs:.0f} Hz")
        with k3:
            st.metric("Muestras (ventana)", str(snap.n_samples))
        with k4:
            st.metric("Modelo", "Cargado" if snap.model_file_ok else "Falta .joblib")
        st.caption(f"Ruta del clasificador: `{snap.model_path}`")
    else:
        st.caption(
            f"Contexto: **{snap.record_id}** · {snap.fs:.0f} Hz · {snap.n_samples} muestras · "
            f"modelo: `{'OK' if snap.model_file_ok else 'falta'}`"
        )

    st.markdown("###### Conversación")
    for msg in st.session_state[CHAT_MESSAGES_KEY]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    last_topic = st.session_state.get(CHAT_LAST_TOPIC_KEY)
    if prompt := st.chat_input(
        "Escribe aquí (saludos y preguntas sin tildes también funcionan)…",
        key=f"{widget_prefix}_chat_input",
    ):
        st.session_state[CHAT_MESSAGES_KEY].append({"role": "user", "content": prompt})
        reply, topic = answer_question(
            prompt,
            snap,
            settings=settings,
            record_ref=selected_ref,
            start_s=start_s,
            duration_s=duration_s,
            last_topic=last_topic if isinstance(last_topic, str) else None,
        )
        if topic is not None:
            st.session_state[CHAT_LAST_TOPIC_KEY] = topic
        st.session_state[CHAT_MESSAGES_KEY].append({"role": "assistant", "content": reply})
        st.rerun()