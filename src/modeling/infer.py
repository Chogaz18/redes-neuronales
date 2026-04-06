from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

from src.config.settings import Settings, get_settings
from src.modeling.train_utils import features_for_mlp


@dataclass(frozen=True)
class ClassificationPrediction:
    predicted_class: str
    probabilities: dict[str, float]


def load_classifier_bundle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado: {path}. "
            "Ejecuta el notebook para generarlo."
        )
    return joblib.load(path)


def load_model_and_infer(
    *,
    signals_12_mv: np.ndarray,
    fs: float,
    model_path: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> ClassificationPrediction:  
    cfg = settings or get_settings()
    path = Path(model_path) if model_path else cfg.classifier_model_path
    bundle = load_classifier_bundle(path)

    s = np.asarray(signals_12_mv, dtype=float)
    if s.ndim != 2 or s.shape[0] != 12:
        raise ValueError("Se espera `signals_12_mv` con forma (12, n_samples).")

    model = bundle["model"]
    class_names: list[str] = list(bundle["class_names"])
    target_len = int(bundle.get("target_len", 500))
    expected_n_features = int(bundle.get("expected_n_features", 188))

    X = features_for_mlp(s, fs, target_len).astype(np.float32)

    if X.shape[1] != expected_n_features:
        raise ValueError(
            f"El modelo espera {expected_n_features} características por muestra, "
            f"pero se generaron {X.shape[1]}."
        )

    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    probs = {class_names[i]: float(proba[i]) for i in range(len(class_names))}

    return ClassificationPrediction(
        predicted_class=class_names[pred_idx],
        probabilities=probs,
    )