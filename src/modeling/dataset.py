from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ClassificationDataset:
    X: np.ndarray  # [n_samples, n_timesteps, n_channels]
    y: np.ndarray  # int labels


def build_classification_dataset(*args, **kwargs) -> ClassificationDataset:
    """
    Construye un dataset para clasificación.
    """
    raise NotImplementedError("Pendiente: implementar el pipeline de dataset para clasificación.")

