"""
Respuestas en español para el asistente de chat: metadatos del registro,
ruta del clasificador e inferencia MLP. Incluye repreguntas tipo «¿qué significa?».
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

from src.config.settings import Settings
from src.data.parsers import header_to_metadata
from src.modeling.infer import ClassificationPrediction, load_model_and_infer
from src.modeling.label_mapping import map_snomed_codes_to_four_classes
from src.modeling.train_utils import load_12lead_aligned

# Temas para seguimiento («explica lo anterior»)
TOPIC_LEADS = "leads"
TOPIC_HZ = "hz"
TOPIC_SNOMED = "snomed"
TOPIC_MLP = "mlp"
TOPIC_WINDOW = "window"
TOPIC_RECORD = "record"
TOPIC_DATASET = "dataset"
TOPIC_MODEL_PATH = "model_path"
TOPIC_HELP = "help"
TOPIC_SAMPLES = "samples"


def _asks_about_sample_count(q_norm: str) -> bool:
    """Preguntas tipo «cuántas muestras», «número de muestras» (texto ya normalizado)."""
    if "muestras por segundo" in q_norm:
        return False
    if re.search(r"\b(cuantas|cuantos)\s+muestras\b", q_norm):
        return True
    if re.search(r"\b(numero|cantidad)\s+de\s+muestras\b", q_norm):
        return True
    if "muestras" in q_norm and re.search(
        r"\b(cuantas|cuantos|cuanto|cuanta|tenemos|tiene|hay)\b", q_norm
    ):
        return True
    if re.search(r"\bmuestras\b", q_norm) and re.search(
        r"\b(cuantas|cuantos|numero|cantidad)\b", q_norm
    ):
        return True
    return False


def _strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _normalize_question(question: str) -> str:
    """Minúsculas, sin tildes y espacios normalizados (mayúsculas y acentos no afectan la intención)."""
    q = question.strip().lower()
    q = _strip_accents(q)
    return re.sub(r"[\s¿?¡!.,;:]+", " ", q).strip()


def _is_greeting_only(q_norm: str) -> bool:
    if _has_strong_new_intent(q_norm):
        return False
    t = q_norm.strip()
    if not t or len(t) > 100:
        return False
    if len(t.split()) > 10:
        return False
    one_line = (
        "hola",
        "hey",
        "hi",
        "hello",
        "saludos",
        "buenas",
        "muy buenas",
        "buenos dias",
        "buenas tardes",
        "buenas noches",
        "buen dia",
        "que tal",
    )
    t2 = re.sub(r"[!?.]+$", "", t).strip()
    if t2 in ("hola", "hey", "hi", "hello", "saludos", "buenas", "muy buenas"):
        return True
    if t2 in ("buenos dias", "buenas tardes", "buenas noches", "buen dia", "que tal"):
        return True
    if re.match(
        r"^(hola|hey|hi|hello|saludos|buenas|muy buenas|buenos dias|buenas tardes|buenas noches|buen dia|que tal)\b",
        t2,
    ) and len(t2) < 70:
        return True
    if re.match(r"^(hola|hey)\s*[,]?\s*(buenos dias|buenas tardes|buenas noches|buenas)\b", t2) and len(t2) < 80:
        return True
    return False


def _is_thanks_only(q_norm: str) -> bool:
    if _has_strong_new_intent(q_norm):
        return False
    t = q_norm.strip()
    if not t or len(t) > 90 or len(t.split()) > 10:
        return False
    if re.match(r"^(gracias|muchas gracias|mil gracias|te lo agradezco|muy amable)\b", t):
        return True
    return False


def _greeting_reply() -> str:
    return (
        "¡Hola! Puedo orientarte sobre el **registro ECG** que tengas seleccionado: "
        "**frecuencia de muestreo (Hz)**, **derivaciones**, **códigos en el archivo**, "
        "**ventana temporal** y **predicción del modelo**. Escribe **ayuda** para ver ejemplos concretos."
    )


def _thanks_reply() -> str:
    return (
        "¡Con gusto! Si quieres seguir, pregunta por **Hz**, **derivaciones**, **SNOMED**, "
        "**ventana** o **predicción del modelo**, o escribe **ayuda**."
    )


_FOLLOWUP_RE = re.compile(
    r"(qué|que)\s+significa|explica(?:me)?|no\s+entiendo|"
    r"para\s+qué\s+sirve|más\s+detalle|mas\s+detalle|elabora|"
    r"por\s+qué\s+(?:dice|es|importa)|por\s+que\s+(?:dice|es|importa)|"
    r"^por\s+qué\s*\??$|^por\s+que\s*\??$|"
    r"^qué\s+es\s+eso\s*\??$|^que\s+es\s+eso\s*\??$|"
    r"^y\s+eso\s+qué\s+es\s*\??$"
)


def _is_followup_question(q_norm: str) -> bool:
    q2 = q_norm.strip()
    if len(q2) > 120:
        return False
    return bool(_FOLLOWUP_RE.search(q2)) or q2 in (
        "qué significa",
        "que significa",
        "explica",
        "por qué",
        "por que",
    )


def _has_strong_new_intent(q_norm: str) -> bool:
    """Si hay palabras que indican una pregunta nueva (no solo aclaración)."""
    keys = (
        "derivación",
        "derivaciones",
        "lead",
        "canales",
        "muestreo",
        " hz",
        "hz ",
        "hercio",
        "snomed",
        "modelo",
        "mlp",
        "clasificación",
        "clasificacion",
        "predicción",
        "prediccion",
        "probabilidad",
        "ventana",
        "duración",
        "duracion",
        "registro",
        "dataset",
        "physionet",
        "joblib",
        "artefacto",
        "frecuencia",
        "muestras",
        "muestra",
    )
    return any(k in q_norm for k in keys)


@dataclass
class AnalysisSnapshot:
    record_id: str
    fs: float
    n_samples: int
    window_s: float
    lead_names: list[str]
    snomed_codes: list[str]
    snomed_four_class: dict[str, str]
    model_path: str
    model_file_ok: bool


def build_snapshot(
    settings: Settings,
    record_ref: Any,
    *,
    start_s: float = 0.0,
    duration_s: float = 10.0,
) -> AnalysisSnapshot:
    meta = header_to_metadata(record_ref)
    aligned, fs, leads = load_12lead_aligned(
        settings, record_ref, start_s=start_s, duration_s=duration_s
    )
    snomed = meta.get("snomed_ct_codes", [])
    mapped = map_snomed_codes_to_four_classes(snomed)
    path = settings.classifier_model_path
    return AnalysisSnapshot(
        record_id=record_ref.record_id,
        fs=float(fs),
        n_samples=int(aligned.shape[1]),
        window_s=float(duration_s),
        lead_names=list(meta.get("lead_names") or leads),
        snomed_codes=list(snomed),
        snomed_four_class=mapped,
        model_path=str(path),
        model_file_ok=path.is_file(),
    )


def _try_predict(
    settings: Settings,
    record_ref: Any,
    *,
    start_s: float,
    duration_s: float,
) -> Optional[ClassificationPrediction]:
    try:
        aligned, fs, _ = load_12lead_aligned(
            settings, record_ref, start_s=start_s, duration_s=duration_s
        )
        return load_model_and_infer(signals_12_mv=aligned, fs=fs, settings=settings)
    except Exception:
        return None


def _explain_topic(
    topic: str,
    snap: AnalysisSnapshot,
    *,
    settings: Settings,
    record_ref: Any,
    start_s: float,
    duration_s: float,
) -> str:
    if topic == TOPIC_LEADS:
        return (
            "**Derivaciones ECG** son vistas distintas del mismo latido: cada una coloca los electrodos "
            "de forma que la señal resalte una dirección del corazón.\n\n"
            "- **I, II, III**: derivaciones **bipolares de las extremidades** (entre brazos y piernas).\n"
            "- **aVR, aVL, aVF**: **unipolares aumentadas** (combinan referencias para ver el frente y los lados).\n"
            "- **V1–V6**: **precordiales** sobre el pecho; suelen usarse para ver zonas del ventrículo y aurículas.\n\n"
            "En conjunto (12 derivaciones) se obtiene una **visión espacial** del ritmo; en esta app se listan "
            f"las que vienen en el archivo (`{len(snap.lead_names)}` señales) en el orden del header."
        )
    if topic == TOPIC_HZ:
        return (
            "**Hertz (Hz)** indica **cuántas mediciones por segundo** se guardan del ECG. "
            f"Si el registro está a **{snap.fs:.0f} Hz**, cada segundo hay **{snap.fs:.0f}** números por canal "
            "(muestras). A mayor Hz, la forma de onda se dibuja con **más detalle temporal**; "
            "también aumenta el tamaño del archivo. Para ver forma de onda y ritmo suele bastar un muestreo "
            "moderado; lo importante es que sea **coherente** con el modelo (entrenado con la misma lógica de ventana y características)."
        )
    if topic == TOPIC_SNOMED:
        return (
            "**SNOMED CT** es un vocabulario clínico internacional: **códigos numéricos** que identifican "
            "diagnósticos, hallazgos o situaciones. En el **header** del registro WFDB a veces aparecen "
            "códigos de **9 dígitos** en comentarios; la app intenta **mapearlos** a las **4 clases** del trabajo "
            "(ritmos de interés).\n\n"
            "Eso **no sustituye** un informe médico: aquí solo se **leen etiquetas del archivo** para contexto y comparación con la predicción del modelo."
        )
    if topic == TOPIC_MLP:
        pred = _try_predict(settings, record_ref, start_s=start_s, duration_s=duration_s)
        extra = ""
        if snap.model_file_ok and pred is not None:
            top = max(pred.probabilities.items(), key=lambda x: x[1])
            extra = (
                f"\n\nEn **tu ventana actual**, el modelo asigna más peso a **`{top[0]}`** "
                f"({top[1] * 100:.1f} %). Las demás clases son las **alternativas** que el MLP considera."
            )
        return (
            "Un **MLP** (perceptrón multicapa) es una **red neuronal** que recibe un **vector de números** "
            "(características calculadas del ECG en la ventana elegida) y devuelve **probabilidades** entre "
            "varias **clases de ritmo**. **No** “lee” la imagen como un humano: resume la señal en estadísticas "
            "(por derivación y en conjunto) y compara con lo aprendido en el entrenamiento.\n\n"
            "Las **probabilidades** suman 100 % entre las clases mostradas: la **clase predicha** es la de mayor valor. "
            "Valores parecidos indican **incertidumbre**." + extra
        )
    if topic == TOPIC_WINDOW:
        return (
            "La **ventana temporal** es el **trozo de señal** (en segundos) que la app usa para análisis: "
            f"desde **{start_s:.1f} s** durante **{duration_s:.1f} s**. "
            f"Eso da **{snap.n_samples}** muestras por canal a **{snap.fs:.0f} Hz**.\n\n"
            "Si acortas o desplazas la ventana, cambia el **fragmento del latido** que ven el modelo y las métricas; "
            "por eso conviene comparar siempre con la **misma ventana** si quieres repetir resultados."
        )
    if topic == TOPIC_SAMPLES:
        return (
            "Cada **muestra** es un valor de la señal en un instante; por canal se guardan tantas como "
            f"**duración × frecuencia** (en tu ventana: **{duration_s:.1f} s** × **{snap.fs:.0f} Hz** "
            f"≈ **{snap.n_samples}** puntos por canal). "
            "Si cambias la **ventana** en la barra lateral, cambia también el número de muestras."
        )
    if topic == TOPIC_RECORD:
        return (
            "Un **registro** es **un archivo de ECG** (en este proyecto, formato **WFDB**: `.hea` + datos). "
            f"El **identificador** `**{snap.record_id}**` distingue ese caso dentro de la muestra. "
            "Al elegir registro en la barra lateral, **todas las páginas** que usan el mismo selector "
            "ven **el mismo caso** para que compares visualización, frecuencia y clasificación."
        )
    if topic == TOPIC_DATASET:
        return (
            "El **dataset** es la **colección de registros** descargada de **PhysioNet ECG-Arrhythmia**. "
            "En la aplicación solo se **listan algunos** archivos (tamaño de muestra configurable) para que "
            "la demo sea ligera. Los datos son **reales** en origen, pero el uso aquí es **académico y exploratorio**, "
            "no asistencia clínica."
        )
    if topic == TOPIC_MODEL_PATH:
        ok = "encontrado" if snap.model_file_ok else "no encontrado"
        return (
            "La **ruta del modelo** es el archivo **`.joblib`** donde está guardado el **pipeline entrenado** "
            "(preprocesado + MLP). La variable de entorno **`CLASSIFIER_MODEL_PATH`** (o la ruta por defecto en "
            f"`artifacts/models/`) indica dónde buscarlo. Estado actual: `{snap.model_path}` (**{ok}**).\n\n"
            "Si falta el archivo, hay que **entrenar** el modelo con el notebook incluido o copiar el artefacto."
        )
    if topic == TOPIC_HELP:
        return (
            "La **ayuda** de este asistente es una **lista de temas** que entiende sin usar internet: "
            "frecuencia, derivaciones, códigos en el archivo, ventana y predicción del modelo. "
            "Escribe palabras como en los ejemplos o una **repregunta** (“¿qué significa?”) **después** de una respuesta "
            "para obtener una explicación **sencilla** del último tema."
        )
    return (
        "No tengo una explicación guardada para ese tema. Prueba **ayuda** o pregunta de nuevo por "
        "**derivaciones**, **Hz**, **SNOMED**, **ventana** o **modelo**."
    )


def answer_question(
    question: str,
    snap: AnalysisSnapshot,
    *,
    settings: Settings,
    record_ref: Any,
    start_s: float,
    duration_s: float,
    last_topic: str | None = None,
) -> tuple[str, str | None]:
    """
    Devuelve (texto_markdown, tema_para_siguiente_repregunta o None si no aplica).
    """
    q_norm = _normalize_question(question)

    if any(k in q_norm for k in ("ayuda", "que puedes", "help")):
        return _help_text(), TOPIC_HELP

    if _is_greeting_only(q_norm):
        return _greeting_reply(), None

    if _is_thanks_only(q_norm):
        return _thanks_reply(), None

    if _is_followup_question(q_norm) and last_topic and not _has_strong_new_intent(q_norm):
        return (
            _explain_topic(
                last_topic,
                snap,
                settings=settings,
                record_ref=record_ref,
                start_s=start_s,
                duration_s=duration_s,
            ),
            last_topic,
        )

    if any(k in q_norm for k in ("dataset", "physionet", "de dónde vienen", "de donde vienen", "origen de los datos")):
        return _dataset_blurb(), TOPIC_DATASET

    if any(k in q_norm for k in ("registro", "archivo", "id ", "cual es el id")):
        return (
            f"El registro activo es **`{snap.record_id}`**. "
            f"Ventana analizada: **{start_s:.1f} s** + **{duration_s:.1f} s** de duración.",
            TOPIC_RECORD,
        )

    if any(
        k in q_norm
        for k in (
            "frecuencia de muestreo",
            "muestreo",
            "hercios",
            " hz",
            "hz ",
            "muestras por segundo",
            "sample",
            "sampling",
        )
    ) or (re.search(r"\bfs\b", q_norm) and "snomed" not in q_norm):
        return (
            f"La frecuencia de muestreo del registro es **{snap.fs:.0f} Hz** "
            f"(aprox. **{snap.n_samples}** muestras en la ventana de **{snap.window_s:.1f} s**).",
            TOPIC_HZ,
        )

    if _asks_about_sample_count(q_norm):
        return (
            f"En la **ventana actual** (desde **{start_s:.1f} s**, duración **{duration_s:.1f} s**) hay "
            f"**{snap.n_samples}** muestras **por canal** a **{snap.fs:.0f} Hz** "
            f"(≈ {duration_s:.1f} s × {snap.fs:.0f} Hz puntos por canal).",
            TOPIC_SAMPLES,
        )

    if any(k in q_norm for k in ("derivación", "derivaciones", "lead", "canales", "electrodos")):
        leads = ", ".join(f"`{x}`" for x in snap.lead_names[:12])
        return (
            f"Hay **{len(snap.lead_names)}** señales en el header; orden de referencia: {leads}.",
            TOPIC_LEADS,
        )

    if any(
        k in q_norm
        for k in (
            "snomed",
            "codigo clinico",
            "diagnostico en cabecera",
            "etiqueta snomed",
        )
    ):
        if not snap.snomed_codes:
            return (
                "En el `.hea` de este registro **no aparecen códigos SNOMED de 9 dígitos** en comentarios.",
                TOPIC_SNOMED,
            )
        codes = ", ".join(snap.snomed_codes)
        four = snap.snomed_four_class
        if four:
            lines = [f"- `{c}` → **{four[c]}**" for c in four]
            return (
                "Códigos SNOMED detectados en el header: "
                f"{codes}.\n\nMapeo a las 4 clases del trabajo:\n"
                + "\n".join(lines),
                TOPIC_SNOMED,
            )
        return (
            f"Códigos en el header: **{codes}** (ninguno mapea a las 4 clases del enunciado).",
            TOPIC_SNOMED,
        )

    if any(
        k in q_norm
        for k in (
            "modelo",
            "mlp",
            "clasificación",
            "clasificacion",
            "clasificar",
            "predicción",
            "prediccion",
            "probabilidad",
            "ritmo predicho",
        )
    ):
        if not snap.model_file_ok:
            return (
                f"No está el artefacto en **`{snap.model_path}`**. "
                "Entrena el modelo o define `CLASSIFIER_MODEL_PATH` en `.env`.",
                TOPIC_MODEL_PATH,
            )
        pred = _try_predict(settings, record_ref, start_s=start_s, duration_s=duration_s)
        if pred is None:
            return (
                "No pude ejecutar la inferencia (revisa que el joblib coincida con `features_for_mlp` "
                "y `target_len` / `expected_n_features`).",
                TOPIC_MLP,
            )
        rows = sorted(pred.probabilities.items(), key=lambda x: -x[1])
        body = "\n".join(f"- **{name}**: {p * 100:.1f} %" for name, p in rows)
        return (
            f"**Clase predicha:** `{pred.predicted_class}`.\n\n"
            f"Probabilidades estimadas por el MLP:\n{body}",
            TOPIC_MLP,
        )

    if any(k in q_norm for k in ("artefacto", "joblib", "ruta del modelo")):
        ok = "sí existe" if snap.model_file_ok else "no se encontró"
        return (f"Ruta configurada: `{snap.model_path}` ({ok}).", TOPIC_MODEL_PATH)

    if any(k in q_norm for k in ("ventana", "duración", "duracion", "segmento", "muestras en la ventana")):
        return (
            f"Ventana: inicio **{start_s:.1f} s**, duración **{duration_s:.1f} s** → "
            f"**{snap.n_samples}** muestras a **{snap.fs:.0f} Hz**.",
            TOPIC_WINDOW,
        )

    if _is_followup_question(q_norm) and not last_topic:
        return (
            "Para explicarte **qué significa** algo, primero conviene que pregunte por un tema concreto "
            "(por ejemplo **derivaciones**, **Hz**, **SNOMED**, **ventana** o **predicción del modelo**). "
            "Después puedes escribir **«¿qué significa?»** o **«explícame»** y te amplío con palabras sencillas.",
            None,
        )

    return (
        "No tengo una respuesta automática para eso. Prueba: **registro**, **frecuencia de muestreo**, "
        "**cuántas muestras**, **derivaciones**, **SNOMED**, **predicción del modelo**, **ventana temporal**, "
        "**dataset** o **ayuda**.",
        None,
    )


def _help_text() -> str:
    return """Puedes preguntar, por ejemplo:

- *¿Cuál es la frecuencia de muestreo?*
- *¿Cuántas muestras hay?* / *número de muestras*
- *¿Qué registro estoy viendo?*
- *¿Qué derivaciones hay?*
- *¿Qué códigos SNOMED aparecen?*
- *¿Qué predice el modelo MLP?* / *probabilidades de clasificación*
- *¿Dónde está el archivo del modelo?*

Tras una respuesta, puedes escribir **«¿qué significa?»** o **«explícame»** para una explicación sencilla del último tema.

Las respuestas usan el **registro seleccionado** y la **ventana** configurada en la barra lateral (o en el panel del asistente)."""


def _dataset_blurb() -> str:
    return (
        "Los datos provienen del dataset **PhysioNet ECG-Arrhythmia v1.0.0** (registros WFDB). "
        "En esta app solo se listan una **muestra** de archivos `.hea` según `SAMPLE_RECORD_COUNT` en `.env`."
    )
