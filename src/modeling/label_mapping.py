from __future__ import annotations

from typing import Dict, List, Optional

# Códigos SNOMED CT según `ConditionNames_SNOMED-CT.csv` (filas SB, SR, AFIB, ST)
SNOMED_TO_FOUR_CLASS: Dict[str, str] = {
    "426177001": "Sinus Bradycardia",
    "426783006": "Sinus Rhythm",
    "164889003": "Atrial Fibrillation",
    "427084000": "Sinus Tachycardia",
}

# Prioridad si varias etiquetas coinciden (ritmo más específico / clínico primero)
PRIORITY_ORDER: tuple[str, ...] = (
    "164889003",  # AFIB
    "426177001",  # SB
    "427084000",  # ST
    "426783006",  # SR
)


FOUR_CLASS_KEYS = [
    "Sinus Bradycardia",
    "Sinus Rhythm",
    "Atrial Fibrillation",
    "Sinus Tachycardia",
]

# Textos didácticos para la interfaz (no constituyen diagnóstico clínico).
CLASS_EDUCATIONAL_ES: dict[str, dict[str, str]] = {
    "Sinus Bradycardia": {
        "titulo": "Ritmo sinusal lento (bradicardia)",
        "texto": (
            "Frecuencia cardiaca **por debajo** del rango que la app considera habitual. "
            "Puede ser normal en reposo o en personas muy entrenadas; en otros casos conviene valoración médica."
        ),
    },
    "Sinus Rhythm": {
        "titulo": "Ritmo sinusal",
        "texto": (
            "Latidos **regulares** producidos por el marcapasos natural del corazón. "
            "Es el patrón más habitual en reposo cuando no hay otra alteración."
        ),
    },
    "Atrial Fibrillation": {
        "titulo": "Fibrilación auricular",
        "texto": (
            "Arritmia con latidos **irregulares** (las aurículas no se contraen de forma ordenada). "
        ),
    },
    "Sinus Tachycardia": {
        "titulo": "Ritmo sinusal rápido (taquicardia)",
        "texto": (
            "Frecuencia cardiaca **elevada** pero con ritmo **regular**. "
            "Puede aparecer con esfuerzo, estrés, dolor, fiebre u otras causas según el contexto."
        ),
    },
}


def educational_blurb_for_class(class_name: str) -> dict[str, str]:
    """Devuelve `titulo` y `texto` en español; si no hay entrada, usa el nombre original."""
    if class_name in CLASS_EDUCATIONAL_ES:
        return CLASS_EDUCATIONAL_ES[class_name]
    return {"titulo": class_name, "texto": "Categoría predicha por el modelo."}


def map_snomed_codes_to_four_classes(snomed_codes: List[str]) -> Dict[str, str]:
    """Mapea cada código SNOMED presente a nombre de clase (solo las 4 objetivo)."""
    out: Dict[str, str] = {}
    for code in snomed_codes:
        c = str(code).strip()
        if c in SNOMED_TO_FOUR_CLASS:
            out[c] = SNOMED_TO_FOUR_CLASS[c]
    return out


def pick_primary_class(snomed_codes: List[str]) -> Optional[str]:
    """Una etiqueta por registro para entrenamiento supervisado (prioridad fija)."""
    codes = {str(c).strip() for c in snomed_codes}
    for snomed in PRIORITY_ORDER:
        if snomed in codes:
            return SNOMED_TO_FOUR_CLASS[snomed]
    return None
